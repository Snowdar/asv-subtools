//  Copyright xmuspeech (Author: Leo 2021-12-18)

#ifndef EXTRACTOR_PARAMS_H_
#define EXTRACTOR_PARAMS_H_

#include <memory>
#include <utility>
#include <vector>
#include "utils/utils.h"
#include "frontend/feature_pipeline.h"
#include "extractor/torch_asv_extractor.h"
#include "extractor/torch_asv_model.h"
#include "utils/options.h"
#include "yaml-cpp/yaml.h"

// // FeatureConfig
// extern std::string feature_conf;
// extern int asv_sample_rate;
// extern bool feat_submean;

// // ASV Model flags
// extern int asv_num_threads;
// extern std::string asv_model_path;

// // ExtractOptions
// extern std::string asv_position;
// extern int asv_chunk_size;
// extern bool asv_do_vad;

// // VAD
// extern float asv_vad_energy_threshold;
// extern float asv_vad_energy_mean_scale;
// extern int asv_vad_frames_context;
// extern float vad_proportion_threshol;


// FeatureConfig flags
DEFINE_string(feature_conf, "", "yaml file path for feature config");
// DEFINE_int32(asv_sample_rate, 16000, "sample rate for audio");
DEFINE_bool(feat_submean, true, "sentence level feature submean");

// ASV Model flags
DEFINE_int32(num_threads, 1, "num threads for GEMM");
DEFINE_string(model_path, "", "pytorch exported model path");

// ExtractOptions
DEFINE_string(position, "near", "position for xvector extracting");
DEFINE_int32(chunk_size, 15000, "maxChunk of inputs to split feat");
DEFINE_bool(do_vad, false, "energy based vad");

// VAD flags
DEFINE_double(vad_energy_threshold, 5.5, "Constant term in energy threshold");
DEFINE_double(vad_energy_mean_scale, 0.5, "If this is set to s, to get the actual threshold we "
                "let m be the mean log-energy of the file, and use "
                "s*m + vad-energy-threshold");
DEFINE_int32(vad_frames_context, 2, "Number of frames of context on each side of central frame, "
                "in window for which energy is monitored");
DEFINE_double(vad_proportion_threshold, 0.12, "Parameter controlling the proportion of frames within "
                    "the window that need to have more energy than the "
                    "threshold");


namespace subtools
{

std::shared_ptr<FeaturePipelineConfig> InitFeaturePiplineConfigFromFlags()
{
    YAML::Node config = YAML::LoadFile(FLAGS_feature_conf);
    config["kaldi_featset"]["feature_type"]=config["feature_type"];
    //config["kaldi_featset"]["sample_frequency"]=config["sample_frequency"];

    Options op;
    auto feature_config = std::make_shared<FeatureConfig>();
    feature_config->Register(&op);

    op.ReadYamlMap(config["kaldi_featset"]);

    // feature_config->sample_frequency = asv_sample_rate;

    auto feature_pipline_config = std::make_shared<FeaturePipelineConfig>(*feature_config);

    return feature_pipline_config;
};

std::shared_ptr<ExtractOptions> InitExtractOptionsFromFlags()
{
    auto extract_config = std::make_shared<ExtractOptions>();
    extract_config->chunk_size = FLAGS_chunk_size;
    extract_config->position = FLAGS_position;

    extract_config->do_vad = FLAGS_do_vad;

    extract_config->vad_opts.vad_energy_threshold = FLAGS_vad_energy_threshold;
    extract_config->vad_opts.vad_energy_mean_scale = FLAGS_vad_energy_mean_scale;
    extract_config->vad_opts.vad_frames_context = FLAGS_vad_frames_context;
    extract_config->vad_opts.vad_proportion_threshold = FLAGS_vad_proportion_threshold;

    extract_config->feat_submean = FLAGS_feat_submean;

    return extract_config;
};

std::shared_ptr<TorchAsvModel> InitTorchAsvModel()
{
    auto asv_model = std::make_shared<TorchAsvModel>();
    LOG(INFO) << "Reading model " << FLAGS_model_path;
    asv_model->Read(FLAGS_model_path, FLAGS_num_threads);

    return asv_model;
};

}

#endif  // EXTRACTOR_PARAMS_H_
