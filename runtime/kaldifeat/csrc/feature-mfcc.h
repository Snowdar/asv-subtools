// kaldifeat/csrc/feature-mfcc.h
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

// This file is copied/modified from kaldi/src/feat/feature-mfcc.h

#ifndef KALDIFEAT_CSRC_FEATURE_MFCC_H_
#define KALDIFEAT_CSRC_FEATURE_MFCC_H_

#include <map>
#include <string>

#include "kaldifeat/csrc/feature-common.h"
#include "kaldifeat/csrc/feature-window.h"
#include "kaldifeat/csrc/mel-computations.h"
#include "torch/script.h"

namespace kaldifeat {

/// MfccOptions contains basic options for computing MFCC features.
// (this class is copied from kaldi)
struct MfccOptions {
  FrameExtractionOptions frame_opts;
  MelBanksOptions mel_opts;

  // Number of cepstra in MFCC computation (including C0)
  int32_t num_ceps = 13;

  // Use energy (not C0) in MFCC computation
  bool use_energy = true;

  // Floor on energy (absolute, not relative) in MFCC
  // computation. Only makes a difference if use_energy=true;
  // only necessary if dither=0.0.
  // Suggested values: 0.1 or 1.0
  float energy_floor = 0.0;

  // If true, compute energy before preemphasis and windowing
  bool raw_energy = true;

  // Constant that controls scaling of MFCCs
  float cepstral_lifter = 22.0;

  // If true, put energy or C0 last and use a factor of
  // sqrt(2) on C0.
  // Warning: not sufficient to get HTK compatible features
  // (need to change other parameters)
  bool htk_compat = false;

  torch::Device device{"cpu"};

  MfccOptions() { mel_opts.num_bins = 23; }

  std::string ToString() const {
    std::ostringstream os;
    os << "frame_opts: \n";
    os << frame_opts << "\n";
    os << "\n";

    os << "mel_opts: \n";
    os << mel_opts << "\n";

    os << "num_ceps: " << num_ceps << "\n";
    os << "use_energy: " << use_energy << "\n";
    os << "energy_floor: " << energy_floor << "\n";
    os << "raw_energy: " << raw_energy << "\n";
    os << "cepstral_lifter: " << cepstral_lifter << "\n";
    os << "htk_compat: " << htk_compat << "\n";
    os << "device: " << device << "\n";
    return os.str();
  }
};

std::ostream &operator<<(std::ostream &os, const MfccOptions &opts);

class MfccComputer {
 public:
  using Options = MfccOptions;

  explicit MfccComputer(const MfccOptions &opts);
  ~MfccComputer();

  MfccComputer &operator=(const MfccComputer &) = delete;
  MfccComputer(const MfccComputer &) = delete;

  int32_t Dim() const { return opts_.num_ceps; }

  bool NeedRawLogEnergy() const { return opts_.use_energy && opts_.raw_energy; }

  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  const MfccOptions &GetOptions() const { return opts_; }

  // signal_raw_log_energy is log_energy_pre_window, which is not empty
  // iff NeedRawLogEnergy() returns true.
  torch::Tensor Compute(torch::Tensor signal_raw_log_energy, float vtln_warp,
                        const torch::Tensor &signal_frame);

 private:
  const MelBanks *GetMelBanks(float vtln_warp);

  MfccOptions opts_;
  torch::Tensor lifter_coeffs_;  // 1-D tensor

  // Note we save a transposed version of dct_matrix_
  // dct_matrix_.rows is num_mel_bins
  // dct_matrix_.cols is num_ceps
  torch::Tensor dct_matrix_;  // matrix we right-multiply by to perform DCT.
  float log_energy_floor_;
  std::map<float, MelBanks *> mel_banks_;  // float is VTLN coefficient.
};

using Mfcc = OfflineFeatureTpl<MfccComputer>;

}  // namespace kaldifeat

#endif  // KALDIFEAT_CSRC_FEATURE_MFCC_H_
