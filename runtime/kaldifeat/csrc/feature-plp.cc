// kaldifeat/csrc/feature-plp.cc
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

// This file is copied/modified from kaldi/src/feat/feature-plp.cc

#include "kaldifeat/csrc/feature-plp.h"

#include "kaldifeat/csrc/feature-functions.h"

namespace kaldifeat {

std::ostream &operator<<(std::ostream &os, const PlpOptions &opts) {
  os << opts.ToString();
  return os;
}

PlpComputer::PlpComputer(const PlpOptions &opts) : opts_(opts) {
  // our num-ceps includes C0.
  KALDIFEAT_ASSERT(opts_.num_ceps <= opts_.lpc_order + 1);

  if (opts.cepstral_lifter != 0.0) {
    lifter_coeffs_ = torch::empty({1, opts.num_ceps}, torch::kFloat32);
    ComputeLifterCoeffs(opts.cepstral_lifter, &lifter_coeffs_);
    lifter_coeffs_ = lifter_coeffs_.to(opts.device);
  }

  InitIdftBases(opts_.lpc_order + 1, opts_.mel_opts.num_bins + 2, &idft_bases_);

  // CAUTION: we save a transposed version of idft_bases_
  idft_bases_ = idft_bases_.to(opts.device).t();

  if (opts.energy_floor > 0.0) log_energy_floor_ = logf(opts.energy_floor);

  // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
  // [note: this call caches it.]
  GetMelBanks(1.0);
}

PlpComputer::~PlpComputer() {
  for (auto iter = mel_banks_.begin(); iter != mel_banks_.end(); ++iter)
    delete iter->second;

  for (auto iter = equal_loudness_.begin(); iter != equal_loudness_.end();
       ++iter)
    delete iter->second;
}

const MelBanks *PlpComputer::GetMelBanks(float vtln_warp) {
  MelBanks *this_mel_banks = nullptr;

  // std::map<float, MelBanks *>::iterator iter = mel_banks_.find(vtln_warp);
  auto iter = mel_banks_.find(vtln_warp);
  if (iter == mel_banks_.end()) {
    this_mel_banks =
        new MelBanks(opts_.mel_opts, opts_.frame_opts, vtln_warp, opts_.device);
    mel_banks_[vtln_warp] = this_mel_banks;
  } else {
    this_mel_banks = iter->second;
  }
  return this_mel_banks;
}

const torch::Tensor *PlpComputer::GetEqualLoudness(float vtln_warp) {
  const MelBanks *this_mel_banks = GetMelBanks(vtln_warp);
  torch::Tensor *ans = NULL;
  auto iter = equal_loudness_.find(vtln_warp);
  if (iter == equal_loudness_.end()) {
    ans = new torch::Tensor;
    GetEqualLoudnessVector(*this_mel_banks, ans);
    *ans = ans->to(opts_.device);
    equal_loudness_[vtln_warp] = ans;
  } else {
    ans = iter->second;
  }
  return ans;
}

// ans.shape [signal_frame.size(0), this->Dim()]
torch::Tensor PlpComputer::Compute(torch::Tensor signal_raw_log_energy,
                                   float vtln_warp,
                                   const torch::Tensor &signal_frame) {
  KALDIFEAT_ASSERT(signal_frame.dim() == 2);
  KALDIFEAT_ASSERT(signal_frame.size(1) == opts_.frame_opts.PaddedWindowSize());

  const MelBanks &mel_banks = *GetMelBanks(vtln_warp);
  const torch::Tensor &equal_loudness = *GetEqualLoudness(vtln_warp);

  // torch.finfo(torch.float32).eps
  constexpr float kEps = 1.1920928955078125e-07f;

  // Compute energy after window function (not the raw one).
  if (opts_.use_energy && !opts_.raw_energy) {
    signal_raw_log_energy =
        torch::clamp_min(signal_frame.pow(2).sum(1), kEps).log();
  }

  // note spectrum is in magnitude, not power, because of `abs()`
#if defined(KALDIFEAT_HAS_FFT_NAMESPACE)
  // signal_frame shape: [x, 512]
  // spectrum shape [x, 257
  torch::Tensor spectrum = torch::fft::rfft(signal_frame).abs();
#else
  // signal_frame shape [x, 512]
  // real_imag shape [x, 257, 2],
  //   where [..., 0] is the real part
  //         [..., 1] is the imaginary part
  torch::Tensor real_imag = torch::rfft(signal_frame, 1);
  torch::Tensor real = real_imag.index({"...", 0});
  torch::Tensor imag = real_imag.index({"...", 1});
  torch::Tensor spectrum = (real.square() + imag.square()).sqrt();
#endif

  // remove the last column, i.e., the highest fft bin
  spectrum = spectrum.index(
      {"...", torch::indexing::Slice(0, -1, torch::indexing::None)});

  // Use power instead of magnitude
  spectrum = spectrum.pow(2);

  torch::Tensor mel_energies = mel_banks.Compute(spectrum);

  mel_energies = torch::mul(mel_energies, equal_loudness);
  mel_energies = mel_energies.pow(opts_.compress_factor);

  // duplicate first and last elements
  //
  // left_padding = wave[:num_left_padding].flip(dims=(0,))
  // first = mel_energies[:, 0]
  // first.shape [num_frames, 1]
  torch::Tensor first = mel_energies.index({"...", 0}).unsqueeze(-1);
  // last = mel_energies[:, -1]
  // last.shape [num_frames, 1]
  torch::Tensor last = mel_energies.index({"...", -1}).unsqueeze(-1);

  mel_energies = torch::cat({first, mel_energies, last}, 1);

  torch::Tensor autocorr_coeffs = torch::mm(mel_energies, idft_bases_);

  torch::Tensor lpc_coeffs;
  torch::Tensor residual_log_energy = ComputeLpc(autocorr_coeffs, &lpc_coeffs);

  residual_log_energy = torch::clamp_min(residual_log_energy, kEps);

  torch::Tensor raw_cepstrum = Lpc2Cepstrum(lpc_coeffs);

  // torch.cat((residual_log_energy.unsqueeze(-1),
  // raw_cepstrum[:opts.num_ceps-1]), 1)
  //
  using namespace torch::indexing;  // It imports: Slice, None // NOLINT
  torch::Tensor features = torch::cat(
      {residual_log_energy.unsqueeze(-1),
       raw_cepstrum.index({"...", Slice(0, opts_.num_ceps - 1, None)})},
      1);

  if (opts_.cepstral_lifter != 0.0) {
    features = torch::mul(features, lifter_coeffs_);
  }

  if (opts_.cepstral_scale != 1.0) {
    features = features * opts_.cepstral_scale;
  }

  if (opts_.use_energy) {
    if (opts_.energy_floor > 0.0f) {
      signal_raw_log_energy =
          torch::clamp_min(signal_raw_log_energy, log_energy_floor_);
    }
    // column 0 is replaced by signal_raw_log_energy
    //
    // features[:, 0] = signal_raw_log_energy
    //
    features.index({"...", 0}) = signal_raw_log_energy;
  }

  if (opts_.htk_compat) {  // reorder the features.
    // shift left, so the original 0th column
    // becomes the last column;
    // the original first column becomes the 0th column
    features = torch::roll(features, -1, 1);
  }
  return features;
}

}  // namespace kaldifeat
