// kaldifeat/csrc/matrix-functions.cc
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

// This file is copied/modified from kaldi/src/matrix/matrix-functions.cc

#include "kaldifeat/csrc/matrix-functions.h"

#include <cmath>

#include "kaldifeat/csrc/log.h"

namespace kaldifeat {

void ComputeDctMatrix(torch::Tensor *mat) {
  KALDIFEAT_ASSERT(mat->dim() == 2);

  int32_t num_rows = mat->size(0);
  int32_t num_cols = mat->size(1);

  KALDIFEAT_ASSERT(num_rows == num_cols);
  KALDIFEAT_ASSERT(num_rows > 0);

  int32_t stride = mat->stride(0);

  // normalizer for X_0
  float normalizer = std::sqrt(1.0f / num_cols);

  // mat[0, :] = normalizer
  mat->index({0, "..."}) = normalizer;

  // normalizer for other elements
  normalizer = std::sqrt(2.0f / num_cols);

  float *data = mat->data_ptr<float>();
  for (int32_t r = 1; r < num_rows; ++r) {
    float *this_row = data + r * stride;
    for (int32_t c = 0; c < num_cols; ++c) {
      float v = std::cos(static_cast<double>(M_PI) / num_cols * (c + 0.5) * r);
      this_row[c] = normalizer * v;
    }
  }
}

}  // namespace kaldifeat
