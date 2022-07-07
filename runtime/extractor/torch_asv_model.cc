//  Copyright xmuspeech (Author: Leo 2021-12-18)

#include "extractor/torch_asv_model.h"
#include <utility>

namespace subtools{

  void TorchAsvModel::Read(const std::string& model_path, const int num_threads){
    TorchModule model = torch::jit::load(model_path);
    module_ = std::make_shared<TorchModule>(std::move(model));
    // For multi-thread performance
    at::set_num_threads(num_threads);
    torch::NoGradGuard no_grad;
    module_->eval();

    //LOG(INFO) << "torch model info num threads " << num_threads;
  }
}

