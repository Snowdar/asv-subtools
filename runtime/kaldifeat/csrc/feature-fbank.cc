// kaldifeat/csrc/feature-fbank.cc
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

// This file is copied/modified from kaldi/src/feat/feature-fbank.cc

#include "kaldifeat/csrc/feature-fbank.h"

#include <cmath>

namespace kaldifeat {

std::ostream &operator<<(std::ostream &os, const FbankOptions &opts) {
  os << opts.ToString();
  return os;
}

FbankComputer::FbankComputer(const FbankOptions &opts) : opts_(opts) {
  if (opts.energy_floor > 0.0f) log_energy_floor_ = logf(opts.energy_floor);

  // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
  // [note: this call caches it.]
  GetMelBanks(1.0f);
}

FbankComputer::~FbankComputer() {
  for (auto iter = mel_banks_.begin(); iter != mel_banks_.end(); ++iter)
    delete iter->second;
}

const MelBanks *FbankComputer::GetMelBanks(float vtln_warp) {
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

// ans.shape [signal_frame.size(0), this->Dim()]
torch::Tensor FbankComputer::Compute(torch::Tensor signal_raw_log_energy,
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

  // Use power instead of magnitude if requested.
  if (opts_.use_power) {
    spectrum = spectrum.pow(2);
  }

  torch::Tensor mel_energies = mel_banks.Compute(spectrum);
  if (opts_.use_log_fbank) {
    // Avoid log of zero (which should be prevented anyway by dithering).
    mel_energies = torch::clamp_min(mel_energies, kEps).log();
  }

  // if use_energy is true, then we get an extra bin. That is,
  // if num_mel_bins is 23, the feature will contain 24 bins.
  //
  // if htk_compat is false, then the 0th bin is the log energy
  // if htk_compat is true, then the last bin is the log energy

  // Copy energy as first value (or the last, if htk_compat == true).
  if (opts_.use_energy) {
    if (opts_.energy_floor > 0.0f) {
      signal_raw_log_energy =
          torch::clamp_min(signal_raw_log_energy, log_energy_floor_);
    }

    signal_raw_log_energy.unsqueeze_(1);

    if (opts_.htk_compat) {
      mel_energies = torch::cat({mel_energies, signal_raw_log_energy}, 1);
    } else {
      mel_energies = torch::cat({signal_raw_log_energy, mel_energies}, 1);
    }
  }

  return mel_energies;
}

}  // namespace kaldifeat
