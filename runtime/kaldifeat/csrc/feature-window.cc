// kaldifeat/csrc/feature-window.cc
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

// This file is copied/modified from kaldi/src/feat/feature-window.cc

#include "kaldifeat/csrc/feature-window.h"

#include <cmath>
#include <vector>

#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559005
#endif

namespace kaldifeat {

std::ostream &operator<<(std::ostream &os, const FrameExtractionOptions &opts) {
  os << opts.ToString();
  return os;
}

FeatureWindowFunction::FeatureWindowFunction(const FrameExtractionOptions &opts,
                                             torch::Device device) {
  int32_t frame_length = opts.WindowSize();
  KALDIFEAT_ASSERT(frame_length > 0);

  window = torch::empty({frame_length}, torch::kFloat32);
  float *window_data = window.data_ptr<float>();

  double a = M_2PI / (frame_length - 1);
  for (int32_t i = 0; i < frame_length; i++) {
    double i_fl = static_cast<double>(i);
    if (opts.window_type == "hanning") {
      window_data[i] = 0.5 - 0.5 * cos(a * i_fl);
    } else if (opts.window_type == "sine") {
      // when you are checking ws wikipedia, please
      // note that 0.5 * a = M_PI/(frame_length-1)
      window_data[i] = sin(0.5 * a * i_fl);
    } else if (opts.window_type == "hamming") {
      window_data[i] = 0.54 - 0.46 * cos(a * i_fl);
    } else if (opts.window_type ==
               "povey") {  // like hamming but goes to zero at edges.
      window_data[i] = pow(0.5 - 0.5 * cos(a * i_fl), 0.85);
    } else if (opts.window_type == "rectangular") {
      window_data[i] = 1.0;
    } else if (opts.window_type == "blackman") {
      window_data[i] = opts.blackman_coeff - 0.5 * cos(a * i_fl) +
                       (0.5 - opts.blackman_coeff) * cos(2 * a * i_fl);
    } else {
      KALDIFEAT_ERR << "Invalid window type " << opts.window_type;
    }
  }

  window = window.unsqueeze(0);
  if (window.device() != device) {
    window = window.to(device);
  }
}

torch::Tensor FeatureWindowFunction::Apply(const torch::Tensor &wave) const {
  KALDIFEAT_ASSERT(wave.dim() == 2);
  KALDIFEAT_ASSERT(wave.size(1) == window.size(1));
  return wave.mul(window);
}

int64_t FirstSampleOfFrame(int32_t frame, const FrameExtractionOptions &opts) {
  int64_t frame_shift = opts.WindowShift();
  if (opts.snip_edges) {
    return frame * frame_shift;
  } else {
    int64_t midpoint_of_frame = frame_shift * frame + frame_shift / 2,
            beginning_of_frame = midpoint_of_frame - opts.WindowSize() / 2;
    return beginning_of_frame;
  }
}

int32_t NumFrames(int64_t num_samples, const FrameExtractionOptions &opts,
                  bool flush /*= true*/) {
  int64_t frame_shift = opts.WindowShift();
  int64_t frame_length = opts.WindowSize();
  if (opts.snip_edges) {
    // with --snip-edges=true (the default), we use a HTK-like approach to
    // determining the number of frames-- all frames have to fit completely into
    // the waveform, and the first frame begins at sample zero.
    if (num_samples < frame_length)
      return 0;
    else
      return (1 + ((num_samples - frame_length) / frame_shift));
    // You can understand the expression above as follows: 'num_samples -
    // frame_length' is how much room we have to shift the frame within the
    // waveform; 'frame_shift' is how much we shift it each time; and the ratio
    // is how many times we can shift it (integer arithmetic rounds down).
  } else {
    // if --snip-edges=false, the number of frames is determined by rounding the
    // (file-length / frame-shift) to the nearest integer.  The point of this
    // formula is to make the number of frames an obvious and predictable
    // function of the frame shift and signal length, which makes many
    // segmentation-related questions simpler.
    //
    // Because integer division in C++ rounds toward zero, we add (half the
    // frame-shift minus epsilon) before dividing, to have the effect of
    // rounding towards the closest integer.
    int32_t num_frames = (num_samples + (frame_shift / 2)) / frame_shift;

    if (flush) return num_frames;

    // note: 'end' always means the last plus one, i.e. one past the last.
    int64_t end_sample_of_last_frame =
        FirstSampleOfFrame(num_frames - 1, opts) + frame_length;

    // the following code is optimized more for clarity than efficiency.
    // If flush == false, we can't output frames that extend past the end
    // of the signal.
    while (num_frames > 0 && end_sample_of_last_frame > num_samples) {
      num_frames--;
      end_sample_of_last_frame -= frame_shift;
    }
    return num_frames;
  }
}

torch::Tensor GetStrided(const torch::Tensor &wave,
                         const FrameExtractionOptions &opts) {
  KALDIFEAT_ASSERT(wave.dim() == 1);

  std::vector<int64_t> strides = {opts.WindowShift() * wave.strides()[0],
                                  wave.strides()[0]};

  int64_t num_samples = wave.size(0);
  int32_t num_frames = NumFrames(num_samples, opts);
  std::vector<int64_t> sizes = {num_frames, opts.WindowSize()};
  if (opts.snip_edges) {
    return wave.as_strided(sizes, strides);
  }

  int32_t frame_length = opts.samp_freq / 1000 * opts.frame_length_ms;
  int32_t frame_shift = opts.samp_freq / 1000 * opts.frame_shift_ms;
  int64_t num_new_samples = (num_frames - 1) * frame_shift + frame_length;
  int32_t num_padding = num_new_samples - num_samples;
  int32_t num_left_padding = (frame_length - frame_shift) / 2;
  int32_t num_right_padding = num_padding - num_left_padding;
  // left_padding = wave[:num_left_padding].flip(dims=(0,))
  torch::Tensor left_padding =
      wave.index({torch::indexing::Slice(0, num_left_padding, 1)}).flip({0});

  // right_padding = wave[-num_righ_padding:].flip(dims=(0,))
  torch::Tensor right_padding =
      wave.index({torch::indexing::Slice(-num_right_padding,
                                         torch::indexing::None, 1)})
          .flip({0});

  torch::Tensor new_wave = torch::cat({left_padding, wave, right_padding}, 0);
  return new_wave.as_strided(sizes, strides);
}

torch::Tensor Dither(const torch::Tensor &wave, float dither_value) {
  if (dither_value == 0.0f) return wave;

  torch::Tensor rand_gauss = torch::randn_like(wave);
#if 1
  return wave + rand_gauss * dither_value;
#else
  // use in-place version of wave and change its to pointer type
  wave_->add_(rand_gauss, dither_value);
#endif
}

torch::Tensor Preemphasize(float preemph_coeff, const torch::Tensor &wave) {
  using namespace torch::indexing;  // It imports: Slice, None  // NOLINT
  if (preemph_coeff == 0.0f) return wave;

  KALDIFEAT_ASSERT(preemph_coeff >= 0.0f && preemph_coeff <= 1.0f);

  torch::Tensor ans = torch::empty_like(wave);

  // right = wave[:, 1:]
  torch::Tensor right = wave.index({"...", Slice(1, None, None)});

  // current = wave[:, 0:-1]
  torch::Tensor current = wave.index({"...", Slice(0, -1, None)});

  // ans[1:, :] = wave[:, 1:] - preemph_coeff * wave[:, 0:-1]
  ans.index({"...", Slice(1, None, None)}) = right - preemph_coeff * current;

  ans.index({"...", 0}) = wave.index({"...", 0}) * (1 - preemph_coeff);

  return ans;
}

}  // namespace kaldifeat
