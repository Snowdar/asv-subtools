//  Copyright xmuspeech (Author: Leo 2021-12-18)
#ifndef FRONTEND_FEATURE_FEATURE_ITF_H_
#define FRONTEND_FEATURE_FEATURE_ITF_H_ 1

#include "torch/script.h"

namespace subtools {
class Basefeature{
 public:
  virtual int Dim() const = 0; /// returns the feature dimension.

  virtual torch::Tensor ComputeFeatures(const std::vector<float>& wav, float vtln_warp) = 0;
  virtual int ComputeFeatures(const std::vector<float>& wav,std::vector<torch::Tensor> *feats, float vtln_warp)=0;
  virtual ~Basefeature(){};
};
}
#endif 