// kaldifeat/csrc/feature-functions.h
//
// Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

// This file is copied/modified from kaldi/src/feat/feature-functions.h

#ifndef KALDIFEAT_CSRC_FEATURE_FUNCTIONS_H_
#define KALDIFEAT_CSRC_FEATURE_FUNCTIONS_H_

#include "torch/script.h"

namespace kaldifeat {

void InitIdftBases(int32_t n_bases, int32_t dimension, torch::Tensor *mat_out);

}

#endif  // KALDIFEAT_CSRC_FEATURE_FUNCTIONS_H_
