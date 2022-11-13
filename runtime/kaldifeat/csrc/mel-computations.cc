// kaldifeat/csrc/mel-computations.cc
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
//
// This file is copied/modified from kaldi/src/feat/mel-computations.cc

#include "kaldifeat/csrc/mel-computations.h"

#include <algorithm>

#include "kaldifeat/csrc/feature-window.h"

namespace kaldifeat {

std::ostream &operator<<(std::ostream &os, const MelBanksOptions &opts) {
  os << opts.ToString();
  return os;
}

float MelBanks::VtlnWarpFreq(
    float vtln_low_cutoff,  // upper+lower frequency cutoffs for VTLN.
    float vtln_high_cutoff,
    float low_freq,  // upper+lower frequency cutoffs in mel computation
    float high_freq, float vtln_warp_factor, float freq) {
  /// This computes a VTLN warping function that is not the same as HTK's one,
  /// but has similar inputs (this function has the advantage of never producing
  /// empty bins).

  /// This function computes a warp function F(freq), defined between low_freq
  /// and high_freq inclusive, with the following properties:
  ///  F(low_freq) == low_freq
  ///  F(high_freq) == high_freq
  /// The function is continuous and piecewise linear with two inflection
  ///   points.
  /// The lower inflection point (measured in terms of the unwarped
  ///  frequency) is at frequency l, determined as described below.
  /// The higher inflection point is at a frequency h, determined as
  ///   described below.
  /// If l <= f <= h, then F(f) = f/vtln_warp_factor.
  /// If the higher inflection point (measured in terms of the unwarped
  ///   frequency) is at h, then max(h, F(h)) == vtln_high_cutoff.
  ///   Since (by the last point) F(h) == h/vtln_warp_factor, then
  ///   max(h, h/vtln_warp_factor) == vtln_high_cutoff, so
  ///   h = vtln_high_cutoff / max(1, 1/vtln_warp_factor).
  ///     = vtln_high_cutoff * min(1, vtln_warp_factor).
  /// If the lower inflection point (measured in terms of the unwarped
  ///   frequency) is at l, then min(l, F(l)) == vtln_low_cutoff
  ///   This implies that l = vtln_low_cutoff / min(1, 1/vtln_warp_factor)
  ///                       = vtln_low_cutoff * max(1, vtln_warp_factor)

  if (freq < low_freq || freq > high_freq)
    return freq;  // in case this gets called
  // for out-of-range frequencies, just return the freq.

  KALDIFEAT_ASSERT(vtln_low_cutoff > low_freq);
  KALDIFEAT_ASSERT(vtln_high_cutoff < high_freq);

  float one = 1.0f;
  float l = vtln_low_cutoff * std::max(one, vtln_warp_factor);
  float h = vtln_high_cutoff * std::min(one, vtln_warp_factor);
  float scale = 1.0f / vtln_warp_factor;
  float Fl = scale * l;  // F(l);
  float Fh = scale * h;  // F(h);
  KALDIFEAT_ASSERT(l > low_freq && h < high_freq);
  // slope of left part of the 3-piece linear function
  float scale_left = (Fl - low_freq) / (l - low_freq);
  // [slope of center part is just "scale"]

  // slope of right part of the 3-piece linear function
  float scale_right = (high_freq - Fh) / (high_freq - h);

  if (freq < l) {
    return low_freq + scale_left * (freq - low_freq);
  } else if (freq < h) {
    return scale * freq;
  } else {  // freq >= h
    return high_freq + scale_right * (freq - high_freq);
  }
}

float MelBanks::VtlnWarpMelFreq(
    float vtln_low_cutoff,  // upper+lower frequency cutoffs for VTLN.
    float vtln_high_cutoff,
    float low_freq,  // upper+lower frequency cutoffs in mel computation
    float high_freq, float vtln_warp_factor, float mel_freq) {
  return MelScale(VtlnWarpFreq(vtln_low_cutoff, vtln_high_cutoff, low_freq,
                               high_freq, vtln_warp_factor,
                               InverseMelScale(mel_freq)));
}

MelBanks::MelBanks(const MelBanksOptions &opts,
                   const FrameExtractionOptions &frame_opts,
                   float vtln_warp_factor, torch::Device device)
    : htk_mode_(opts.htk_mode) {
  int32_t num_bins = opts.num_bins;
  if (num_bins < 3) KALDIFEAT_ERR << "Must have at least 3 mel bins";

  float sample_freq = frame_opts.samp_freq;
  int32_t window_length_padded = frame_opts.PaddedWindowSize();
  KALDIFEAT_ASSERT(window_length_padded % 2 == 0);

  int32_t num_fft_bins = window_length_padded / 2;
  float nyquist = 0.5f * sample_freq;

  float low_freq = opts.low_freq, high_freq;
  if (opts.high_freq > 0.0f)
    high_freq = opts.high_freq;
  else
    high_freq = nyquist + opts.high_freq;

  if (low_freq < 0.0f || low_freq >= nyquist || high_freq <= 0.0f ||
      high_freq > nyquist || high_freq <= low_freq)
    KALDIFEAT_ERR << "Bad values in options: low-freq " << low_freq
                  << " and high-freq " << high_freq << " vs. nyquist "
                  << nyquist;

  float fft_bin_width = sample_freq / window_length_padded;
  // fft-bin width [think of it as Nyquist-freq / half-window-length]

  float mel_low_freq = MelScale(low_freq);
  float mel_high_freq = MelScale(high_freq);

  debug_ = opts.debug_mel;

  // divide by num_bins+1 in next line because of end-effects where the bins
  // spread out to the sides.
  float mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1);

  float vtln_low = opts.vtln_low, vtln_high = opts.vtln_high;
  if (vtln_high < 0.0f) {
    vtln_high += nyquist;
  }

  if (vtln_warp_factor != 1.0f &&
      (vtln_low < 0.0f || vtln_low <= low_freq || vtln_low >= high_freq ||
       vtln_high <= 0.0f || vtln_high >= high_freq || vtln_high <= vtln_low))
    KALDIFEAT_ERR << "Bad values in options: vtln-low " << vtln_low
                  << " and vtln-high " << vtln_high << ", versus "
                  << "low-freq " << low_freq << " and high-freq " << high_freq;

  // we will transpose bins_mat_ at the end of this funciton
  bins_mat_ = torch::zeros({num_bins, num_fft_bins}, torch::kFloat);

  int32_t stride = bins_mat_.strides()[0];

  center_freqs_ = torch::empty({num_bins}, torch::kFloat);
  float *center_freqs_data = center_freqs_.data_ptr<float>();

  for (int32_t bin = 0; bin < num_bins; ++bin) {
    float left_mel = mel_low_freq + bin * mel_freq_delta,
          center_mel = mel_low_freq + (bin + 1) * mel_freq_delta,
          right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

    if (vtln_warp_factor != 1.0f) {
      left_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                 vtln_warp_factor, left_mel);
      center_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                   vtln_warp_factor, center_mel);
      right_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                  vtln_warp_factor, right_mel);
    }
    center_freqs_data[bin] = InverseMelScale(center_mel);
    // this_bin will be a vector of coefficients that is only
    // nonzero where this mel bin is active.
    float *this_bin = bins_mat_.data_ptr<float>() + bin * stride;
    int32_t first_index = -1, last_index = -1;
    for (int32_t i = 0; i < num_fft_bins; ++i) {
      float freq = (fft_bin_width * i);  // Center frequency of this fft
                                         // bin.
      float mel = MelScale(freq);
      if (mel > left_mel && mel < right_mel) {
        float weight;
        if (mel <= center_mel)
          weight = (mel - left_mel) / (center_mel - left_mel);
        else
          weight = (right_mel - mel) / (right_mel - center_mel);
        this_bin[i] = weight;
        if (first_index == -1) first_index = i;
        last_index = i;
      }
    }
    KALDIFEAT_ASSERT(first_index != -1 && last_index >= first_index &&
                     "You may have set num_mel_bins too large.");

    // Replicate a bug in HTK, for testing purposes.
    if (opts.htk_mode && bin == 0 && mel_low_freq != 0.0f)
      this_bin[first_index] = 0.0f;
  }

  if (debug_) KALDIFEAT_LOG << bins_mat_;

  bins_mat_.t_();

  if (bins_mat_.device() != device) {
    bins_mat_ = bins_mat_.to(device);
  }
}

torch::Tensor MelBanks::Compute(const torch::Tensor &spectrum) const {
  return torch::mm(spectrum, bins_mat_);
}

void ComputeLifterCoeffs(float Q, torch::Tensor *coeffs) {
  // Compute liftering coefficients (scaling on cepstral coeffs)
  // coeffs are numbered slightly differently from HTK: the zeroth
  // index is C0, which is not affected.
  float *data = coeffs->data_ptr<float>();
  int32_t n = coeffs->numel();
  for (int32_t i = 0; i < n; ++i) {
    data[i] = 1.0 + 0.5 * Q * sin(M_PI * i / Q);
  }
}

void GetEqualLoudnessVector(const MelBanks &mel_banks, torch::Tensor *ans) {
  int32_t n = mel_banks.NumBins();
  // Central frequency of each mel bin.
  const torch::Tensor &f0 = mel_banks.GetCenterFreqs();
  const float *f0_data = f0.data_ptr<float>();

  *ans = torch::empty({1, n}, torch::kFloat);
  float *ans_data = ans->data_ptr<float>();
  for (int32_t i = 0; i < n; ++i) {
    float fsq = f0_data[i] * f0_data[i];
    float fsub = fsq / (fsq + 1.6e5);
    ans_data[i] = fsub * fsub * ((fsq + 1.44e6) / (fsq + 9.61e6));
  }
}

// Durbin's recursion - converts autocorrelation coefficients to the LPC
// pTmp - temporal place [n]
// pAC - autocorrelation coefficients [n + 1]
// pLP - linear prediction coefficients [n]
//       (predicted_sn = sum_1^P{a[i-1] * s[n-i]}})
//       F(z) = 1 / (1 - A(z)), 1 is not stored in the denominator
static float Durbin(int n, const float *pAC, float *pLP, float *pTmp) {
  float ki;  // reflection coefficient
  int i;
  int j;

  float E = pAC[0];

  for (i = 0; i < n; ++i) {
    // next reflection coefficient
    ki = pAC[i + 1];

    for (j = 0; j < i; ++j) ki += pLP[j] * pAC[i - j];

    ki = ki / E;

    // new error
    float c = 1 - ki * ki;
    if (c < 1.0e-5)  // remove NaNs for constant signal
      c = 1.0e-5;

    E *= c;

    // new LP coefficients
    pTmp[i] = -ki;

    for (j = 0; j < i; ++j) pTmp[j] = pLP[j] - ki * pLP[i - j - 1];

    for (j = 0; j <= i; ++j) pLP[j] = pTmp[j];
  }

  return E;
}

// Compute LP coefficients from autocorrelation coefficients.
torch::Tensor ComputeLpc(const torch::Tensor &autocorr_in,
                         torch::Tensor *lpc_out) {
  KALDIFEAT_ASSERT(autocorr_in.dim() == 2);

  int32_t num_frames = autocorr_in.size(0);
  int32_t lpc_order = autocorr_in.size(1) - 1;

  *lpc_out = torch::empty({num_frames, lpc_order}, torch::kFloat);
  torch::Tensor ans = torch::empty({num_frames}, torch::kFloat);

  // TODO(fangjun): Durbin runs only on CPU. Implement a CUDA version
  torch::Device saved_device = autocorr_in.device();
  torch::Device cpu("cpu");
  torch::Tensor in_cpu = autocorr_in.to(cpu);

  torch::Tensor tmp = torch::empty_like(*lpc_out);

  int32_t in_stride = in_cpu.stride(0);
  int32_t ans_stride = ans.stride(0);
  int32_t tmp_stride = tmp.stride(0);
  int32_t lpc_stride = lpc_out->stride(0);

  const float *in_data = in_cpu.data_ptr<float>();
  float *ans_data = ans.data_ptr<float>();
  float *tmp_data = tmp.data_ptr<float>();
  float *lpc_data = lpc_out->data_ptr<float>();

  // see
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Parallel.h#L58
  at::parallel_for(0, num_frames, 1, [&](int32_t begin, int32_t end) -> void {
    for (int32_t i = begin; i != end; ++i) {
      float ret = Durbin(lpc_order, in_data + i * in_stride,
                         lpc_data + i * lpc_stride, tmp_data + i * tmp_stride);

      if (ret <= 0.0) KALDIFEAT_WARN << "Zero energy in LPC computation";

      ans_data[i] = -logf(1.0 / ret);  // forms the C0 value
    }
  });

  *lpc_out = lpc_out->to(saved_device);
  return ans.to(saved_device);
}

static void Lpc2CepstrumInternal(int n, const float *pLPC, float *pCepst) {
  for (int32_t i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int32_t j = 0; j < i; ++j) {
      sum += (i - j) * pLPC[j] * pCepst[i - j - 1];
    }
    pCepst[i] = -pLPC[i] - sum / (i + 1);
  }
}

torch::Tensor Lpc2Cepstrum(const torch::Tensor &lpc) {
  KALDIFEAT_ASSERT(lpc.dim() == 2);
  torch::Device cpu("cpu");
  torch::Device saved_device = lpc.device();

  // TODO(fangjun): support cuda
  torch::Tensor in_cpu = lpc.to(cpu);

  int32_t num_frames = in_cpu.size(0);
  int32_t lpc_order = in_cpu.size(1);

  const float *in_data = in_cpu.data_ptr<float>();
  int32_t in_stride = in_cpu.stride(0);

  torch::Tensor ans = torch::zeros({num_frames, lpc_order}, torch::kFloat);
  int32_t ans_stride = ans.stride(0);
  float *ans_data = ans.data_ptr<float>();

  at::parallel_for(0, num_frames, 1, [&](int32_t begin, int32_t end) -> void {
    for (int32_t i = begin; i != end; ++i) {
      Lpc2CepstrumInternal(lpc_order, in_data + i * in_stride,
                           ans_data + i * ans_stride);
    }
  });

  return ans.to(saved_device);
}

}  // namespace kaldifeat
