/**
 * Copyright      xmuspeech (Leo 2021-12-23)
 */

#ifndef FRONTEND_FEATURES_H_
#define FRONTEND_FEATURES_H_
#include <string>
#include <vector>

#include "frontend/feature-itf.h"
#include "kaldifeat/csrc/feature-fbank.h"
#include "kaldifeat/csrc/feature-mfcc.h"
#include "utils/options.h"

namespace subtools {

template<class C>
class GenericBaseFeature: public Basefeature {
public:
  // Constructor from options class
  explicit GenericBaseFeature(const typename C::Options &opts);

  virtual int32_t Dim() const { return feat_c.Dim(); }

  virtual torch::Tensor ComputeFeatures(const std::vector<float>& wav, float vtln_warp);
  virtual int ComputeFeatures(const std::vector<float>& wav,std::vector<torch::Tensor> *feats, float vtln_warp);

private:
    C feat_c;
};

typedef GenericBaseFeature<kaldifeat::Mfcc> Mfcc;

typedef GenericBaseFeature<kaldifeat::Fbank> Fbank;

}  // namespace subtools

#endif  // FRONTEND_FEATURES_H_
