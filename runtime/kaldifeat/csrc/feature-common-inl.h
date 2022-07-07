// kaldifeat/csrc/feature-common-inl.h
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

// This file is copied/modified from kaldi/src/feat/feature-common-inl.h

#ifndef KALDIFEAT_CSRC_FEATURE_COMMON_INL_H_
#define KALDIFEAT_CSRC_FEATURE_COMMON_INL_H_

#include "kaldifeat/csrc/feature-window.h"

namespace kaldifeat {

template <class F>
torch::Tensor OfflineFeatureTpl<F>::ComputeFeatures(const torch::Tensor &wave,
                                                    float vtln_warp) {
  const FrameExtractionOptions &frame_opts = computer_.GetFrameOptions();

  torch::Tensor strided_input;
  if (wave.dim() == 1) {
    strided_input = GetStrided(wave, frame_opts);
  } else {
    KALDIFEAT_ASSERT(wave.dim() == 2);
    KALDIFEAT_ASSERT(wave.size(1) == frame_opts.WindowSize());
    strided_input = wave;
  }

  if (frame_opts.dither != 0.0f) {
    strided_input = Dither(strided_input, frame_opts.dither);
  }

  if (frame_opts.remove_dc_offset) {
    torch::Tensor row_means = strided_input.mean(1).unsqueeze(1);
    strided_input = strided_input - row_means;
  }

  bool use_raw_log_energy = computer_.NeedRawLogEnergy();
  torch::Tensor log_energy_pre_window;

  // torch.finfo(torch.float32).eps
  constexpr float kEps = 1.1920928955078125e-07f;

  if (use_raw_log_energy) {
    // it is true iff use_energy==true and row_energy==true
    log_energy_pre_window =
        torch::clamp_min(strided_input.pow(2).sum(1), kEps).log();
  }

  if (frame_opts.preemph_coeff != 0.0f) {
    strided_input = Preemphasize(frame_opts.preemph_coeff, strided_input);
  }

  strided_input = feature_window_function_.Apply(strided_input);

  int32_t padding = frame_opts.PaddedWindowSize() - strided_input.size(1);

  if (padding > 0) {
    strided_input = torch::nn::functional::pad(
        strided_input, torch::nn::functional::PadFuncOptions({0, padding})
                           .mode(torch::kConstant)
                           .value(0));
  }

  return computer_.Compute(log_energy_pre_window, vtln_warp, strided_input);
}

}  // namespace kaldifeat

#endif  // KALDIFEAT_CSRC_FEATURE_COMMON_INL_H_
