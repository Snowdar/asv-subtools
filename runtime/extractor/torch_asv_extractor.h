//  Copyright xmuspeech (Author: Leo 2021-12-18)

#ifndef EXTRACTOR_ASV_EXTRACTOR_H_
#define EXTRACTOR_ASV_EXTRACTOR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"
#include "utils/utils.h"
#include "frontend/feature_pipeline.h"
#include "extractor/torch_asv_model.h"

namespace subtools{

using TorchModule = torch::jit::script::Module;

// This struct is copied/modified from kaldi/src/ivector/voice-activity-detection.h
struct VadEnergyOptions
{
    float vad_energy_threshold;
    float vad_energy_mean_scale;
    int   vad_frames_context;
    float vad_proportion_threshold;

    VadEnergyOptions(): vad_energy_threshold(5.5),
        vad_energy_mean_scale(0.5),
        vad_frames_context(2),
        vad_proportion_threshold(0.12) { }

    void Register(Options *opts)
    {
        opts->Register("vad-energy-threshold", &vad_energy_threshold);
        opts->Register("vad-energy-mean-scale", &vad_energy_mean_scale);
        opts->Register("vad-frames-context", &vad_frames_context);
        opts->Register("vad-proportion-threshold", &vad_proportion_threshold);
    }
};

struct ExtractOptions{
    std::string position = "near";
    int chunk_size = 15000;
    bool do_vad = true;
    bool feat_submean = true;
    VadEnergyOptions vad_opts;
};

class TorchAsvExtractor{
    public:
        TorchAsvExtractor(std::shared_ptr<FeaturePipeline> feature_pipeline,
        std::shared_ptr<TorchAsvModel> model,
        const ExtractOptions& opts);

        void ComputeVadEnergy(const VadEnergyOptions &opts,
                      torch::Tensor &feats, torch::Tensor &output_voiced);

        int Extract();

        torch::Tensor result() const { return xvector_; };

    private:
        std::shared_ptr<FeaturePipeline> feature_pipeline_;
        std::shared_ptr<TorchAsvModel> model_;
        const ExtractOptions& opts_;
        int num_frames_ = 0;
        torch::Tensor xvector_;
    public:
        SUBTOOLS_DISALLOW_COPY_AND_ASSIGN(TorchAsvExtractor);
    };
}

#endif
