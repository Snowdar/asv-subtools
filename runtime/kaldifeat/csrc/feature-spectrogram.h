// kaldifeat/csrc/feature-spectrogram.h
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

// This file is copied/modified from kaldi/src/feat/feature-spectrogram.h

#ifndef KALDIFEAT_CSRC_FEATURE_SPECTROGRAM_H_
#define KALDIFEAT_CSRC_FEATURE_SPECTROGRAM_H_

#include <string>

#include "kaldifeat/csrc/feature-common.h"
#include "kaldifeat/csrc/feature-window.h"
#include "torch/script.h"

namespace kaldifeat {

struct SpectrogramOptions {
  FrameExtractionOptions frame_opts;

  // Floor on energy (absolute, not relative) in Spectrogram
  // computation.  Caution: this floor is applied to the
  // zeroth component, representing the total signal energy.
  // The floor on the individual spectrogram elements is fixed at
  // std::numeric_limits<float>::epsilon()
  float energy_floor = 0.0;

  // If true, compute energy before preemphasis and windowing
  bool raw_energy = true;

  // If true, return raw FFT complex numbers instead of log magnitudes
  // Not implemented yet
  bool return_raw_fft = false;

  torch::Device device{"cpu"};

  std::string ToString() const {
    std::ostringstream os;
    os << "frame_opts: \n";
    os << frame_opts << "\n";

    os << "energy_floor: " << energy_floor << "\n";
    os << "raw_energy: " << raw_energy << "\n";
    // os << "return_raw_fft: " << return_raw_fft << "\n";
    os << "device: " << device << "\n";
    return os.str();
  }
};

std::ostream &operator<<(std::ostream &os, const SpectrogramOptions &opts);

class SpectrogramComputer {
 public:
  using Options = SpectrogramOptions;

  explicit SpectrogramComputer(const SpectrogramOptions &opts);

  ~SpectrogramComputer() = default;

  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  const SpectrogramOptions &GetOptions() const { return opts_; }

  int32_t Dim() const {
    if (opts_.return_raw_fft) {
      return opts_.frame_opts.PaddedWindowSize();
    } else {
      return opts_.frame_opts.PaddedWindowSize() / 2 + 1;
    }
  }

  bool NeedRawLogEnergy() const { return opts_.raw_energy; }

  // signal_raw_log_energy is log_energy_pre_window, which is not empty
  // iff NeedRawLogEnergy() returns true.
  //
  // vtln_warp is ignored by this function, it's only
  // needed for interface compatibility.
  torch::Tensor Compute(torch::Tensor signal_raw_log_energy, float vtln_warp,
                        const torch::Tensor &signal_frame);

 private:
  SpectrogramOptions opts_;
  float log_energy_floor_;
};

using Spectrogram = OfflineFeatureTpl<SpectrogramComputer>;

}  // namespace kaldifeat

#endif  // KALDIFEAT_CSRC_FEATURE_SPECTROGRAM_H_
