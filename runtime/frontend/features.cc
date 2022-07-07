/**
 * Copyright      xmuspeech (Leo 2021-12-23)
 */

#include "frontend/features.h"


namespace subtools {

template <class C>
GenericBaseFeature<C>::GenericBaseFeature(const typename C::Options &opts)
:feat_c(opts){};

template <class C>
torch::Tensor GenericBaseFeature<C>::ComputeFeatures(const std::vector<float>& wav, float vtln_warp){
  torch::NoGradGuard no_grad;
  torch::Tensor w = torch::tensor(wav).to(torch::kFloat32);

  return feat_c.ComputeFeatures(w,vtln_warp);
}

template <class C>
int GenericBaseFeature<C>::ComputeFeatures(const std::vector<float>& wav, std::vector<torch::Tensor> *feats,float vtln_warp){
  torch::NoGradGuard no_grad;
  torch::Tensor w = torch::tensor(wav).to(torch::kFloat32);

  w = feat_c.ComputeFeatures(w,vtln_warp);
  int num_frames = w.size(0);
  feats->resize(num_frames);
  for (int i = 0; i < num_frames; ++i){
    (*feats)[i]=std::move(w[i]);
  }

  return num_frames;
}

// instantiate the templates defined here for MFCC and filterbank classes.
template class GenericBaseFeature<kaldifeat::Mfcc>;
template class GenericBaseFeature<kaldifeat::Fbank>;


}  // namespace subtools
