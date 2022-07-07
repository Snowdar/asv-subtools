//  Copyright xmuspeech (Author: Leo 2021-12-18)

#include "frontend/feature_pipeline.h"
#include <algorithm>
#include <utility>

namespace subtools
{

FeaturePipelineConfig::FeaturePipelineConfig(
    const FeatureConfig &config)
{
    if (config.feature_type == "mfcc" || config.feature_type == "fbank")
    {
        feature_type = config.feature_type;
        if (feature_type == "mfcc")
        {
            config.GetMfccOptions(&mfcc_opts);
        }
        if (feature_type == "fbank")
        {
            config.GetFbankOptions(&fbank_opts);
        }
    }
    else
    {
        LOG(FATAL) << "Invalid feature type: " << config.feature_type << ". "
                   << "Supported feature types: mfcc, fbank.";
    }

    sample_rate = config.sample_frequency;
    window_size = static_cast<int>(sample_rate * 0.001f * config.frame_length);
    window_shift = static_cast<int>(sample_rate * 0.001f * config.frame_shift);
    vtln_warp = config.GetVtlnWarp();
};

FeaturePipeline::FeaturePipeline(
    const FeaturePipelineConfig &config):
    config_(config)
{
    Init();
};

void FeaturePipeline::Init()
{
    //LOG(INFO) << "FeaturePipeline::Init()";

    if (config_.feature_type == "mfcc")
    {
        kaldifeats_ = new Mfcc(config_.mfcc_opts);
    }
    else if (config_.feature_type == "fbank")
    {
        kaldifeats_ = new Fbank(config_.fbank_opts);
    }
    else
    {
        LOG(FATAL) << "Code error: invalid feature type " << config_.feature_type;
    }
    feature_dim_ = kaldifeats_->Dim();
    vtln_warp = config_.vtln_warp;
    num_frames_ = 0;
    input_finished_ = false;
};

void FeaturePipeline::AcceptWaveform(const std::vector<float>& wav)
{
    std::vector<float> waves;
    waves.insert(waves.end(), remained_wav_.begin(), remained_wav_.end());
    waves.insert(waves.end(), wav.begin(), wav.end());

    // std::vector<torch::Tensor> feats;
    // int num_frames = kaldifeats_->ComputeFeatures(waves, &feats,vtln_warp);
    torch::Tensor feats =  kaldifeats_->ComputeFeatures(waves,vtln_warp);

    int num_frames = feats.size(0);

    for (size_t i = 0; i < num_frames; ++i)
    {
        feature_queue_.Push(std::move((feats[i]).unsqueeze(0)));
    }

    num_frames_ += num_frames;

    int left_samples = waves.size() - config_.window_shift * num_frames;
    remained_wav_.resize(left_samples);
    std::copy(waves.begin() + config_.window_shift * num_frames, waves.end(),
              remained_wav_.begin());

    // We are still adding wave, notify input is not finished
    finish_condition_.notify_one();
}

void FeaturePipeline::set_input_finished()
{
    CHECK(!input_finished_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        input_finished_ = true;
    }

    finish_condition_.notify_one();
}

bool FeaturePipeline::ReadOne(torch::Tensor* feat)
{
    if (!feature_queue_.Empty())
    {
        *feat = std::move(feature_queue_.Pop());
        return true;
    }
    else
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!input_finished_)
        {
            // This will release the lock and wait for notify_one()
            // from AcceptWaveform() or set_input_finished()
            finish_condition_.wait(lock);
            if (!feature_queue_.Empty())
            {
                *feat = std::move(feature_queue_.Pop());
                return true;
            }
        }
        CHECK(input_finished_);
        CHECK(feature_queue_.Empty());

        return false;
    }
}

bool FeaturePipeline::Read(int num_frames,
                           std::vector<torch::Tensor>* feats)
{
    feats->clear();
    torch::Tensor feat;
    while (feats->size() < num_frames)
    {
        if (ReadOne(&feat))
        {
            feats->push_back(std::move(feat));
        }
        else
        {
            return false;
        }
    }

    return true;
}

void FeaturePipeline::Reset()
{
    input_finished_ = false;
    num_frames_ = 0;

    remained_wav_.clear();

    feature_queue_.Clear();
}

}  // namespace subtools
