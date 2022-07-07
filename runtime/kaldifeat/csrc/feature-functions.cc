// kaldifeat/csrc/feature-functions.cc
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

// This file is copied/modified from kaldi/src/feat/feature-functions.cc

#include "kaldifeat/csrc/feature-functions.h"

#include <cmath>

namespace kaldifeat {

void InitIdftBases(int32_t n_bases, int32_t dimension, torch::Tensor *mat_out) {
  float angle = M_PI / (dimension - 1);
  float scale = 1.0f / (2 * (dimension - 1));

  *mat_out = torch::empty({n_bases, dimension}, torch::kFloat);
  float *data = mat_out->data_ptr<float>();

  int32_t stride = mat_out->stride(0);

  for (int32_t i = 0; i < n_bases; ++i) {
    float *this_row = data + i * stride;
    this_row[0] = scale;
    for (int32_t j = 1; j < dimension - 1; ++j) {
      this_row[j] = 2 * scale * std::cos(angle * i * j);
    }

    this_row[dimension - 1] = scale * std::cos(angle * i * (dimension - 1));
  }
}

}  // namespace kaldifeat
