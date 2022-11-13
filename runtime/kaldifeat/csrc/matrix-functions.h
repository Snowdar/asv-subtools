// kaldifeat/csrc/matrix-functions.h
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

// This file is copied/modified from kaldi/src/matrix/matrix-functions.h

#ifndef KALDIFEAT_CSRC_MATRIX_FUNCTIONS_H_
#define KALDIFEAT_CSRC_MATRIX_FUNCTIONS_H_

#include "torch/script.h"

namespace kaldifeat {

/// ComputeDctMatrix computes a matrix corresponding to the DCT, such that
/// M * v equals the DCT of vector v.  M must be square at input.
/// This is the type = II DCT with normalization, corresponding to the
/// following equations, where x is the signal and X is the DCT:
/// X_0 = sqrt(1/N) \sum_{n = 0}^{N-1} x_n
/// X_k = sqrt(2/N) \sum_{n = 0}^{N-1} x_n cos( \pi/N (n + 1/2) k )
/// See also
/// https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
void ComputeDctMatrix(torch::Tensor *M);

}  // namespace kaldifeat

#endif  // KALDIFEAT_CSRC_MATRIX_FUNCTIONS_H_
