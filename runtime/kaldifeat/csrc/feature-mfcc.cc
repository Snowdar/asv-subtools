// kaldifeat/csrc/feature-mfcc.cc
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

// This file is copied/modified from kaldi/src/feat/feature-mfcc.cc

#include "kaldifeat/csrc/feature-mfcc.h"

#include "kaldifeat/csrc/matrix-functions.h"

namespace kaldifeat {

std::ostream &operator<<(std::ostream &os, const MfccOptions &opts) {
  os << opts.ToString();
  return os;
}

MfccComputer::MfccComputer(const MfccOptions &opts) : opts_(opts) {
  int32_t num_bins = opts.mel_opts.num_bins;

  if (opts.num_ceps > num_bins) {
    KALDIFEAT_ERR << "num-ceps cannot be larger than num-mel-bins."
                  << " It should be smaller or equal. You provided num-ceps: "
                  << opts.num_ceps << "  and num-mel-bins: " << num_bins;
  }

  torch::Tensor dct_matrix = torch::empty({num_bins, num_bins}, torch::kFloat);

  ComputeDctMatrix(&dct_matrix);
  // Note that we include zeroth dct in either case.  If using the
  // energy we replace this with the energy.  This means a different
  // ordering of features than HTK.

  using namespace torch::indexing;  // It imports: Slice, None  // NOLINT

  // dct_matrix[:opts.num_cepts, :]
  torch::Tensor dct_rows =
      dct_matrix.index({Slice(0, opts.num_ceps, None), "..."});

  dct_matrix_ = dct_rows.clone().t().to(opts.device);

  if (opts.cepstral_lifter != 0.0) {
    lifter_coeffs_ = torch::empty({1, opts.num_ceps}, torch::kFloat32);
    ComputeLifterCoeffs(opts.cepstral_lifter, &lifter_coeffs_);
    lifter_coeffs_ = lifter_coeffs_.to(opts.device);
  }
  if (opts.energy_floor > 0.0) log_energy_floor_ = logf(opts.energy_floor);

  // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
  // [note: this call caches it.]
  GetMelBanks(1.0);
}

const MelBanks *MfccComputer::GetMelBanks(float vtln_warp) {
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

MfccComputer::~MfccComputer() {
  for (auto iter = mel_banks_.begin(); iter != mel_banks_.end(); ++iter)
    delete iter->second;
}

// ans.shape [signal_frame.size(0), this->Dim()]
torch::Tensor MfccComputer::Compute(torch::Tensor signal_raw_log_energy,
                                    float vtln_warp,
                                    const torch::Tensor &signal_frame) {
  const MelBanks &mel_banks = *(GetMelBanks(vtln_warp));

  KALDIFEAT_ASSERT(signal_frame.dim() == 2);

  KALDIFEAT_ASSERT(signal_frame.size(1) == opts_.frame_opts.PaddedWindowSize());

  // torch.finfo(torch.float32).eps
  constexpr float kEps = 1.1920928955078125e-07f;

  // Compute energy after window function (not the raw one).
  if (opts_.use_energy && !opts_.raw_energy) {
    signal_raw_log_energy =
        torch::clamp_min(signal_frame.pow(2).sum(1), kEps).log();
  }

 


  torch::Tensor spectrum = torch::fft::rfft(signal_frame).abs();


  // remove the last column, i.e., the highest fft bin
  spectrum = spectrum.index(
      {"...", torch::indexing::Slice(0, -1, torch::indexing::None)});

  // Use power instead of magnitude
  spectrum = spectrum.pow(2);

  torch::Tensor mel_energies = mel_banks.Compute(spectrum);

  // Avoid log of zero (which should be prevented anyway by dithering).
  mel_energies = torch::clamp_min(mel_energies, kEps).log();

  torch::Tensor features = torch::mm(mel_energies, dct_matrix_);

  if (opts_.cepstral_lifter != 0.0) {
    features = torch::mul(features, lifter_coeffs_);
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

  if (opts_.htk_compat) {
    // energy = features[:, 0]
    // features[:, :-1] = features[:, 1:]
    // features[:, -1] = energy *sqrt(2)
    //
    // shift left, so the original 0th column
    // becomes the last column;
    // the original first column becomes the 0th column
    features = torch::roll(features, -1, 1);

    if (!opts_.use_energy) {
      // TODO(fangjun): change the DCT matrix so that we don't need
      // to do an extra multiplication here.
      //
      // scale on C0 (actually removing a scale
      // we previously added that's part of one common definition of
      // the cosine transform.)
      features.index({"...", -1}) *= M_SQRT2;
    }
  }

  return features;
}

}  // namespace kaldifeat
