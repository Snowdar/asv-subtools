//  Copyright xmuspeech (Author: Leo 2021-12-18)

#include <memory>
#include <string>

#include <algorithm>
#include <limits>
#include "extractor/torch_asv_extractor.h"
#include "utils/timer.h"

namespace subtools
{

void TorchAsvExtractor::ComputeVadEnergy(const VadEnergyOptions &opts,
        torch::Tensor &feats, torch::Tensor &output_voiced)
{
    int T = feats.size(0); //number of frames
    //LOG(INFO) << T;

    output_voiced = torch::ones({T}, torch::kFloat32);
    if (T == 0)
    {
        LOG(FATAL) << "Empty features";
        return;
    }

    //log_energy.CopyColFromMat(feats, 0); // column zero is log-energy.
    torch::Tensor log_energy = feats.select(1,0).clone();

    float log_energy_sum = (log_energy.sum()).item().toFloat();

    float energy_threshold = opts.vad_energy_threshold;
    if (opts.vad_energy_mean_scale != 0.0)
    {
        CHECK(opts.vad_energy_mean_scale > 0.0);
        energy_threshold += opts.vad_energy_mean_scale * log_energy_sum / T;
    }
    //LOG(INFO) << energy_threshold;

    CHECK(opts.vad_frames_context >= 0);
    CHECK(opts.vad_proportion_threshold > 0.0 &&
          opts.vad_proportion_threshold < 1.0);

    const float *log_energy_data = log_energy.data_ptr<float>();
    for (int t = 0; t < T; t++)
    {
        int num_count = 0, den_count = 0, context = opts.vad_frames_context;
        for (int t2 = t - context; t2 <= t + context; t2++)
        {
            if (t2 >= 0 && t2 < T)
            {
                den_count++;
                if (log_energy_data[t2] > energy_threshold)
                    num_count++;
            }
        }
        if (num_count >= den_count * opts.vad_proportion_threshold)
            output_voiced[t] = 1.0;
        else
            output_voiced[t] = 0.0;
    }
};

TorchAsvExtractor::TorchAsvExtractor(std::shared_ptr<FeaturePipeline> feature_pipeline,
                                     std::shared_ptr<TorchAsvModel> model,
                                     const ExtractOptions& opts)
    : feature_pipeline_(std::move(feature_pipeline)),
      model_(std::move(model)),
      opts_(opts) {};

int TorchAsvExtractor::Extract()
{
    const int feature_dim = feature_pipeline_->feature_dim();
    int num_requried_frames = 0;
    num_requried_frames = std::numeric_limits<int>::max();

    std::vector<torch::Tensor> feats;
    feature_pipeline_->Read(num_requried_frames, &feats);
    num_frames_ += feats.size();
    //LOG(INFO) << "Required " << num_requried_frames << " get "
    //          << feats.size();

    torch::NoGradGuard no_grad;

    torch::Tensor input_feats = torch::cat(feats, 0);

    torch::Tensor vad_result;
    if (opts_.do_vad){
        vad_result = torch::ones({num_frames_}, torch::kFloat32);
        ComputeVadEnergy(opts_.vad_opts, input_feats, vad_result);

        if((vad_result.sum()).item().toFloat()==0.0)
        {
            LOG(INFO) << "No frames were judged voiced for utterance!";
            return -1;
        }
    }

    //substract feature mean, hongqy 2022-01-06
    if(opts_.feat_submean)
        input_feats = input_feats - input_feats.mean(0);

    if (opts_.do_vad){
        torch::Tensor voice_index = torch::nonzero(vad_result).squeeze(1);

        input_feats = input_feats.index_select(0,voice_index);
    }

    Timer timer;
    int xvector_dim = model_->torch_model()->run_method("embedding_dim").toInt();
    std::vector<torch::jit::IValue> inputs = {input_feats,opts_.position,opts_.chunk_size};

    torch::Tensor output = model_->torch_model()
                           ->get_method("extract_embedding_whole")(inputs).toTensor();

    int forward_time = timer.Elapsed();
    CHECK_EQ(xvector_dim,output.size(0));
    xvector_ = output;
    //VLOG(3) << "forward takes " << forward_time << " ms.";

    return 0;
};

}

