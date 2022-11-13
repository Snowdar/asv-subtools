// kaldifeat/csrc/pitch-functions.h
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

// This file is copied/modified from kaldi/src/feat/pitch-functions.h

#ifndef KALDIFEAT_CSRC_PITCH_FUNCTIONS_H_
#define KALDIFEAT_CSRC_PITCH_FUNCTIONS_H_

// References
//
// Talkin, David, and W. Bastiaan Kleijn. "A robust algorithm for pitch
// tracking (RAPT)." coding and synthesis 495 (1995): 518.
// (https://www.ee.columbia.edu/~dpwe/papers/Talkin95-rapt.pdf)
//
// Ghahremani, Pegah, et al. "A pitch extraction algorithm tuned for
// automatic speech recognition." 2014 IEEE international conference on
// acoustics, speech and signal processing (ICASSP). IEEE, 2014.
// (http://danielpovey.com/files/2014_icassp_pitch.pdf)

#include <string>

#include "torch/script.h"

namespace kaldifeat {

struct PitchExtractionOptions {
  // sample frequency in hertz
  // must match the waveform file
  float samp_freq = 16000;
  float frame_shift_ms = 10.0;   // in milliseconds.
  float frame_length_ms = 25.0;  // in milliseconds.

  // Preemphasis coefficient. [use is deprecated.]
  float preemph_coeff = 0.0;

  float min_f0 = 50;            // min f0 to search (Hz)
  float max_f0 = 400;           // max f0 to search (Hz)
  float soft_min_f0 = 10.0;     // Minimum f0, applied in soft way, must not
                                // exceed min-f0
  float penalty_factor = 0.1;   // cost factor for FO change
  float lowpass_cutoff = 1000;  // cutoff frequency for Low pass filter (Hz)

  // Integer that determines filter width when
  // upsampling NCCF
  // Frequency that we down-sample the signal to.  Must be
  // more than twice lowpass-cutoff
  float resample_freq = 4000;

  float delta_pitch = 0.005;          // the pitch tolerance in pruning lags
  float nccf_ballast = 7000;          // Increasing this factor reduces NCCF for
                                      // quiet frames, helping ensure pitch
                                      // continuity in unvoiced region
  int32_t lowpass_filter_width = 1;   // Integer that determines filter width of
                                      // lowpass filter
  int32_t upsample_filter_width = 5;  // Integer that determines filter width
                                      // when upsampling NCCF

  // Below are newer config variables, not present in the original paper,
  // that relate to the online pitch extraction algorithm.

  // The maximum number of frames of latency that we allow the pitch-processing
  // to introduce, for online operation. If you set this to a large value,
  // there would be no inaccuracy from the Viterbi traceback (but it might make
  // you wait to see the pitch). This is not very relevant for the online
  // operation: normalization-right-context is more relevant, you
  // can just leave this value at zero.
  int32_t max_frames_latency = 0;

  // Only relevant for the function ComputeKaldiPitch which is called by
  // compute-kaldi-pitch-feats. If nonzero, we provide the input as chunks of
  // this size. This affects the energy normalization which has a small effect
  // on the resulting features, especially at the beginning of a file. For best
  // compatibility with online operation (e.g. if you plan to train models for
  // the online-deocding setup), you might want to set this to a small value,
  // like one frame.
  int32_t frames_per_chunk = 0;

  // Only relevant for the function ComputeKaldiPitch which is called by
  // compute-kaldi-pitch-feats, and only relevant if frames_per_chunk is
  // nonzero. If true, it will query the features as soon as they are
  // available, which simulates the first-pass features you would get in online
  // decoding. If false, the features you will get will be the same as those
  // available at the end of the utterance, after InputFinished() has been
  // called: e.g. during lattice rescoring.
  bool simulate_first_pass_online = false;

  // Only relevant for online operation or when emulating online operation
  // (e.g. when setting frames_per_chunk). This is the frame-index on which we
  // recompute the NCCF (e.g. frame-index 500 = after 5 seconds); if the
  // segment ends before this we do it when the segment ends. We do this by
  // re-computing the signal average energy, which affects the NCCF via the
  // "ballast term", scaling the resampled NCCF by a factor derived from the
  // average change in the "ballast term", and re-doing the backtrace
  // computation. Making this infinity would be the most exact, but would
  // introduce unwanted latency at the end of long utterances, for little
  // benefit.
  int32_t recompute_frame = 500;

  // This is a "hidden config" used only for testing the online pitch
  // extraction. If true, we compute the signal root-mean-squared for the
  // ballast term, only up to the current frame, rather than the end of the
  // current chunk of signal. This makes the output insensitive to the
  // chunking, which is useful for testing purposes.
  bool nccf_ballast_online = false;
  bool snip_edges = true;

  torch::Device device{"cpu"};

  PitchExtractionOptions() = default;

  /// Returns the window-size in samples, after resampling.  This is the
  /// "basic window size", not the full window size after extending by max-lag.
  // Because of floating point representation, it is more reliable to divide
  // by 1000 instead of multiplying by 0.001, but it is a bit slower.
  int32_t NccfWindowSize() const {
    return static_cast<int32_t>(resample_freq * frame_length_ms / 1000.0);
  }
  /// Returns the window-shift in samples, after resampling.
  int32_t NccfWindowShift() const {
    return static_cast<int32_t>(resample_freq * frame_shift_ms / 1000.0);
  }

  std::string ToString() const {
    std::ostringstream os;
    os << "samp_freq: " << samp_freq << "\n";
    os << "frame_shift_ms: " << frame_shift_ms << "\n";
    os << "frame_length_ms: " << frame_length_ms << "\n";
    os << "preemph_coeff: " << preemph_coeff << "\n";
    os << "min_f0: " << min_f0 << "\n";
    os << "max_f0: " << max_f0 << "\n";
    os << "soft_min_f0: " << soft_min_f0 << "\n";
    os << "penalty_factor: " << penalty_factor << "\n";
    os << "lowpass_cutoff: " << lowpass_cutoff << "\n";
    os << "resample_freq: " << resample_freq << "\n";
    os << "delta_pitch: " << delta_pitch << "\n";
    os << "nccf_ballast: " << nccf_ballast << "\n";
    os << "lowpass_filter_width: " << lowpass_filter_width << "\n";
    os << "upsample_filter_width: " << upsample_filter_width << "\n";
    os << "max_frames_latency: " << max_frames_latency << "\n";
    os << "frames_per_chunk: " << frames_per_chunk << "\n";
    os << "simulate_first_pass_online: " << simulate_first_pass_online << "\n";
    os << "recompute_frame: " << recompute_frame << "\n";
    os << "nccf_ballast_online: " << nccf_ballast_online << "\n";
    os << "snip_edges: " << snip_edges << "\n";
    os << "device: " << device << "\n";
  }
};

// TODO(fangjun): Implement it

}  // namespace kaldifeat

#endif  // KALDIFEAT_CSRC_PITCH_FUNCTIONS_H_
