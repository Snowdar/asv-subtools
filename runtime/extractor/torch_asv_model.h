//  Copyright xmuspeech (Author: Leo 2021-12-18)
#ifndef EXTRACTOR_TORCH_ASV_MODEL_H_
#define EXTRACTOR_TORCH_ASV_MODEL_H_

#include <memory>
#include <string>

#include "torch/script.h"
#include "torch/torch.h"
#include "utils/utils.h"

namespace subtools{

using TorchModule = torch::jit::script::Module;

class TorchAsvModel {
 public:
  TorchAsvModel() = default;

  void Read(const std::string& model_path, const int num_threads = 1);

  std::shared_ptr<TorchModule> torch_model() const { return module_; }

 private:
  std::shared_ptr<TorchModule> module_ = nullptr;

 public:
  SUBTOOLS_DISALLOW_COPY_AND_ASSIGN(TorchAsvModel);
};

}

#endif
