//  Copyright xmuspeech (Author: Leo 2021-12-18)
//                               Qingyang Hong 2022-01-03)

#ifndef FRONTEND_FEATURE_PIPELINE_H_
#define FRONTEND_FEATURE_PIPELINE_H_

#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "frontend/feature-itf.h"
#include "frontend/features.h"
#include "utils/blocking_queue.h"
#include "utils/options.h"
#include "utils/utils.h"

namespace subtools
{

struct FeatureConfig
{
    // FeatureType
    std::string feature_type = "mfcc";

    // FrameExtraction
    float sample_frequency = 16000.0f;
    float frame_length = 25.0f;
    float frame_shift = 10.0f;
    float dither = 0.0f;
    float preemphasis_coefficient = 0.97f;
    bool remove_dc_offset = true;
    std::string window_type = "povey";
    bool round_to_power_of_two = true;
    float blackman_coeff = 0.42f;

    bool snip_edges = true;  // only support true now

    // MelBanks
    int num_mel_bins = 23;
    float high_freq = 0.0f;
    float low_freq = 20.0f;
    float vtln_high = -500.0f;
    float vtln_low = 100.0f;

    // Features
    bool use_energy = true;
    float energy_floor = 1.0f;
    bool raw_energy = true;
    bool htk_compat = false;
    float vtln_warp = 1.0f;

    // for mfcc
    int num_ceps = 13;
    float cepstral_lifter = 22.0f;

    // for fbank
    bool use_log_fbank = true;
    bool use_power = true;

    void Register(Options *opts)
    {
        opts->Register("feature-type",&feature_type);

        opts->Register("sample-frequency",&sample_frequency);
        opts->Register("frame-length",&frame_length);
        opts->Register("frame-shift",&frame_shift);
        opts->Register("dither",&dither);
        opts->Register("preemphasis-coefficient",&preemphasis_coefficient);
        opts->Register("remove-dc-offset",&remove_dc_offset);
        opts->Register("window-type",&window_type);
        opts->Register("round-to-power-of-two",&round_to_power_of_two);
        opts->Register("blackman-coeff",&blackman_coeff);
        opts->Register("snip-edges",&snip_edges);

        opts->Register("num-mel-bins",&num_mel_bins);
        opts->Register("high-freq",&high_freq);
        opts->Register("low-freq",&low_freq);
        opts->Register("vtln-high",&vtln_high);
        opts->Register("vtln-low",&vtln_low);

        opts->Register("use-energy",&use_energy);
        opts->Register("energy-floor",&energy_floor);
        opts->Register("raw-energy",&raw_energy);
        opts->Register("htk-compat",&htk_compat);
        opts->Register("vtln-warp",&vtln_warp);

        opts->Register("num-ceps",&num_ceps);
        opts->Register("cepstral-lifter",&cepstral_lifter);

        opts->Register("use-log-fbank",&use_log_fbank);
        opts->Register("use-power",&use_power);
    };

    void GetWindowOptions(kaldifeat::FrameExtractionOptions *fopts) const
    {
        fopts->samp_freq = sample_frequency;
        fopts->frame_shift_ms = frame_shift;
        fopts->frame_length_ms = frame_length;
        fopts->dither = dither;
        fopts->remove_dc_offset = remove_dc_offset;
        fopts->window_type = window_type;
        fopts->round_to_power_of_two = round_to_power_of_two;
        fopts->blackman_coeff = blackman_coeff;
        fopts->snip_edges = snip_edges;
    };

    void GetMelOptions(kaldifeat::MelBanksOptions *mopts) const
    {
        mopts->num_bins = num_mel_bins;
        mopts->low_freq = low_freq;
        mopts->high_freq = high_freq;
        mopts->vtln_low = vtln_low;
        mopts->vtln_high = vtln_high;
    };

    void GetFbankOptions(kaldifeat::FbankOptions *fbopts) const
    {
        this->GetWindowOptions(&(fbopts->frame_opts));
        this->GetMelOptions(&(fbopts->mel_opts));
        fbopts->use_energy=use_energy;
        fbopts->energy_floor=energy_floor;
        fbopts->raw_energy=raw_energy;
        fbopts->htk_compat=htk_compat;
        fbopts->use_log_fbank=use_log_fbank;
        fbopts->use_power=use_power;
    };

    void GetMfccOptions(kaldifeat::MfccOptions *mfopts) const
    {
        this->GetWindowOptions(&(mfopts->frame_opts));
        this->GetMelOptions(&(mfopts->mel_opts));
        mfopts->num_ceps=num_ceps;
        mfopts->use_energy=use_energy;
        mfopts->energy_floor=energy_floor;
        mfopts->raw_energy=raw_energy;
        mfopts->htk_compat=htk_compat;
        mfopts->cepstral_lifter=cepstral_lifter;
    };

    float GetVtlnWarp() const
    {
        return vtln_warp;
    };
};

struct FeaturePipelineConfig
{
    std::string feature_type;  // "mfcc" or "fbank"
    float sample_rate;
    int window_size;
    int window_shift;
    float vtln_warp;
    kaldifeat::MfccOptions mfcc_opts;  // options for MFCC computation,
    // if feature_type == "mfcc"
    kaldifeat::FbankOptions fbank_opts;  // Options for filterbank computation, if
    // feature_type == "fbank"

    FeaturePipelineConfig():
        feature_type("fbank"), sample_rate(16000.0f)
    {
        window_size = static_cast<int>(sample_rate * 0.001f * 25.0f);
        window_shift = static_cast<int>(sample_rate * 0.001f * 10.0f);
        vtln_warp = 1.0f;
    }

    FeaturePipelineConfig(const FeatureConfig &feature_config);
};


// Typically, FeaturePipeline is used in two threads: one thread A calls
// AcceptWaveform() to add raw wav data and set_input_finished() to notice
// the end of input wav, another thread B (extacter thread) calls Read() to
// consume features.So a BlockingQueue is used to make this class thread safe.

// The Read() is designed as a blocking method when there is no feature
// in feature_queue_ and the input is not finished.

class FeaturePipeline
{
public:
    explicit FeaturePipeline(const FeaturePipelineConfig& opts);

    // The feature extraction is done in AcceptWaveform().
    void AcceptWaveform(const std::vector<float>& wav);

    // Current extracted frames number.
    int num_frames() const
    {
        return num_frames_;
    }
    int feature_dim() const
    {
        return feature_dim_;
    }
    const FeaturePipelineConfig& config() const
    {
        return config_;
    }

    // The caller should call this method when speech input is end.
    // Never call AcceptWaveform() after calling set_input_finished() !
    void set_input_finished();

    // Return False if input is finished and no feature could be read.
    // Return True if a feature is read.
    // This function is a blocking method. It will block the thread when
    // there is no feature in feature_queue_ and the input is not finished.
    bool ReadOne(torch::Tensor* feat);

    // Read #num_frames frame features.
    // Return False if less then #num_frames features are read and the
    // input is finished.
    // Return True if #num_frames features are read.
    // This function is a blocking method when there is no feature
    // in feature_queue_ and the input is not finished.
    bool Read(int num_frames, std::vector<torch::Tensor>* feats);

    void Reset();
    bool IsLastFrame(int frame) const
    {
        return input_finished_ && (frame == num_frames_ - 1);
    };

    ~FeaturePipeline()
    {
        delete kaldifeats_;
    };

private:
    void Init();
    const FeaturePipelineConfig& config_;
    int feature_dim_;
    Basefeature *kaldifeats_;
    float vtln_warp;

    BlockingQueue<torch::Tensor> feature_queue_;
    int num_frames_;
    bool input_finished_;

    // The feature extraction is done in AcceptWaveform().
    // This wavefrom sample points are consumed by frame size.
    // The residual wavefrom sample points after framing are
    // kept to be used in next AcceptWaveform() calling.
    std::vector<float> remained_wav_;

    // Used to block the Read when there is no feature in feature_queue_
    // and the input is not finished.
    mutable std::mutex mutex_;
    std::condition_variable finish_condition_;
};

}  // namespace subtools

#endif  // FRONTEND_FEATURE_PIPELINE_H_
