// kaldifeat/csrc/feature-plp.h
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

// This file is copied/modified from kaldi/src/feat/feature-plp.h

#ifndef KALDIFEAT_CSRC_FEATURE_PLP_H_
#define KALDIFEAT_CSRC_FEATURE_PLP_H_

#include <map>
#include <string>

#include "kaldifeat/csrc/feature-common.h"
#include "kaldifeat/csrc/feature-window.h"
#include "kaldifeat/csrc/mel-computations.h"
#include "torch/script.h"

namespace kaldifeat {

/// PlpOptions contains basic options for computing PLP features.
/// It only includes things that can be done in a "stateless" way, i.e.
/// it does not include energy max-normalization.
/// It does not include delta computation.
struct PlpOptions {
  FrameExtractionOptions frame_opts;
  MelBanksOptions mel_opts;

  // Order of LPC analysis in PLP computation
  //
  // 12 seems to be common for 16kHz-sampled data. For 8kHz-sampled
  // data, 15 may be better.
  int32_t lpc_order = 12;

  // Number of cepstra in PLP computation (including C0)
  int32_t num_ceps = 13;
  bool use_energy = true;  // use energy; else C0

  // Floor on energy (absolute, not relative) in PLP computation.
  // Only makes a difference if --use-energy=true; only necessary if
  // dither is 0.0.  Suggested values: 0.1 or 1.0
  float energy_floor = 0.0;

  // If true, compute energy before preemphasis and windowing
  bool raw_energy = true;

  // Compression factor in PLP computation
  float compress_factor = 0.33333;

  // Constant that controls scaling of PLPs
  int32_t cepstral_lifter = 22;

  // Scaling constant in PLP computation
  float cepstral_scale = 1.0;

  bool htk_compat = false;  // if true, put energy/C0 last and introduce a
                            // factor of sqrt(2) on C0 to be the same as HTK.
                            //
  torch::Device device{"cpu"};

  PlpOptions() { mel_opts.num_bins = 23; }

  std::string ToString() const {
    std::ostringstream os;
    os << "frame_opts: \n";
    os << frame_opts << "\n";
    os << "\n";

    os << "mel_opts: \n";
    os << mel_opts << "\n";

    os << "lpc_order: " << lpc_order << "\n";
    os << "num_ceps: " << num_ceps << "\n";
    os << "use_energy: " << use_energy << "\n";
    os << "energy_floor: " << energy_floor << "\n";
    os << "raw_energy: " << raw_energy << "\n";
    os << "compress_factor: " << compress_factor << "\n";
    os << "cepstral_lifter: " << cepstral_lifter << "\n";
    os << "cepstral_scale: " << cepstral_scale << "\n";
    os << "htk_compat: " << htk_compat << "\n";
    os << "device: " << device << "\n";
    return os.str();
  }
};

std::ostream &operator<<(std::ostream &os, const PlpOptions &opts);

class PlpComputer {
 public:
  using Options = PlpOptions;

  explicit PlpComputer(const PlpOptions &opts);
  ~PlpComputer();

  PlpComputer &operator=(const PlpComputer &) = delete;
  PlpComputer(const PlpComputer &) = delete;

  int32_t Dim() const { return opts_.num_ceps; }

  bool NeedRawLogEnergy() const { return opts_.use_energy && opts_.raw_energy; }

  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  const PlpOptions &GetOptions() const { return opts_; }

  // signal_raw_log_energy is log_energy_pre_window, which is not empty
  // iff NeedRawLogEnergy() returns true.
  torch::Tensor Compute(torch::Tensor signal_raw_log_energy, float vtln_warp,
                        const torch::Tensor &signal_frame);

 private:
  const MelBanks *GetMelBanks(float vtln_warp);

  const torch::Tensor *GetEqualLoudness(float vtln_warp);

  PlpOptions opts_;
  torch::Tensor lifter_coeffs_;
  torch::Tensor idft_bases_;  // 2-D tensor, kFloat. Caution: it is transposed
  float log_energy_floor_;
  std::map<float, MelBanks *> mel_banks_;  // float is VTLN coefficient.

  // value is a 1-D torch.Tensor
  std::map<float, torch::Tensor *> equal_loudness_;
};

using Plp = OfflineFeatureTpl<PlpComputer>;

}  // namespace kaldifeat

#endif  // KALDIFEAT_CSRC_FEATURE_PLP_H_
