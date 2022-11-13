#include <iomanip>
#include <utility>
#include <iostream>
// #include "torch/script.h"
#include "extractor/params.h"
#include "frontend/wav.h"
#include "utils/timer.h"
#include "utils/utils.h"
#include "utils/string.h"

DEFINE_string(wav_path, "", "single wave path");
DEFINE_string(wav_scp, "", "input wav scp");
DEFINE_int32(warmup, 20, "num of warmup decode, 0 means no warmup");

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  google::SetLogDestination(google::GLOG_INFO,"./test.1.log");

  auto feature_config = subtools::InitFeaturePiplineConfigFromFlags();
  auto extractor_config = subtools::InitExtractOptionsFromFlags();
  auto asv_model = subtools::InitTorchAsvModel();
  if (FLAGS_wav_path.empty() && FLAGS_wav_scp.empty()) {
    LOG(FATAL) << "Please provide the wave path or the wav scp.";
  }
  std::vector<std::pair<std::string, std::string>> waves;
  if (!FLAGS_wav_path.empty()) {
    waves.emplace_back(make_pair("test", FLAGS_wav_path));
  } else {
    std::ifstream wav_scp(FLAGS_wav_scp);
    std::string line;
    while (getline(wav_scp, line)) {
      std::vector<std::string> strs;
      subtools::SplitString(line, &strs);
      CHECK_GE(strs.size(), 2);
      waves.emplace_back(make_pair(strs[0], strs[1]));
    }
  }

  // Warmup
  if (FLAGS_warmup > 0) {
    LOG(INFO) << "Warming up...";
    
      auto wav = waves[0];
      for (int i = 0; i < FLAGS_warmup; i++) { 
        subtools::WavReader wav_reader(wav.second);
        // CHECK_EQ(wav_reader.sample_rate(), FLAGS_sample_rate);
        auto feature_pipeline =
            std::make_shared<subtools::FeaturePipeline>(*feature_config);
        subtools::Timer timer;
        feature_pipeline->AcceptWaveform(std::vector<float>(
            wav_reader.data(), wav_reader.data() + wav_reader.num_sample()));
        feature_pipeline->set_input_finished();
        if(i==0){
          LOG(INFO) << "make features for "<< feature_pipeline->num_frames()
          << " frames takes " <<timer.Elapsed() << "ms.";
        }

        subtools::TorchAsvExtractor extractor(feature_pipeline, asv_model,
                                      *extractor_config);

        int wave_dur =
            static_cast<int>(static_cast<float>(wav_reader.num_sample()) /
                            wav_reader.sample_rate() * 1000);

        timer.Reset();
        extractor.Extract();
        int extract_time = timer.Elapsed();
        LOG(INFO) << " Warmup RTF " << static_cast<float>(extract_time) / wave_dur
                  << "ms.";
        LOG(INFO) << " Warmup num " << i+1 << " Done! " << std::endl;        
      }
      
    
    LOG(INFO) << "Warmup done.";
  }

  int total_waves_dur = 0;
  int total_extract_time = 0;
  for (auto &wav : waves) {
    LOG(INFO) << wav.first << " Start! " << std::endl;
    subtools::WavReader wav_reader(wav.second);
    // CHECK_EQ(wav_reader.sample_rate(), FLAGS_sample_rate);
    auto feature_pipeline =
        std::make_shared<subtools::FeaturePipeline>(*feature_config);
    subtools::Timer timer;
    feature_pipeline->AcceptWaveform(std::vector<float>(
        wav_reader.data(), wav_reader.data() + wav_reader.num_sample()));
    feature_pipeline->set_input_finished();
    
    LOG(INFO) << "num frames " << feature_pipeline->num_frames();
    LOG(INFO) << "make features for "<< feature_pipeline->num_frames()
              << " frames takes " <<timer.Elapsed() << "ms.";

    subtools::TorchAsvExtractor extractor(feature_pipeline, asv_model,
                                   *extractor_config);

    int wave_dur =
        static_cast<int>(static_cast<float>(wav_reader.num_sample()) /
                         wav_reader.sample_rate() * 1000);
 
    timer.Reset();
    extractor.Extract();
    int extract_time = timer.Elapsed();
    
    torch::Tensor result = extractor.result();
    std::cout<<result;
    LOG(INFO) << "extracted xvector of " << wave_dur << "ms audio taken " << extract_time
              << "ms.";
    LOG(INFO) <<"RTF: "<< static_cast<float>(extract_time) / wave_dur;
    LOG(INFO) << wav.first << " Done! " << std::endl;
    total_waves_dur += wave_dur;
    total_extract_time += extract_time;
  }
  LOG(INFO) << "Total: processed " << total_waves_dur << "ms audio taken "
            << total_extract_time << "ms.";
  LOG(INFO) << "RTF: " << std::setprecision(4)
            << static_cast<float>(total_extract_time) / total_waves_dur;
  google::ShutdownGoogleLogging();
  return 0;

}