// nnet3bin/nnet3-xvector-compute.cc

// Copyright 2017   Johns Hopkins University (author: Daniel Povey)
//           2017   Johns Hopkins University (author: Daniel Garcia-Romero)
//           2017   David Snyder
//           2019   xmuspeech (author: Snowdar)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Compile x-vector nnet and output kinds of chunk-compiled files\n"
        "Usage: nnet3-compile-xvector-net [options] <raw-nnet-in> <utt2chunk-rspecifier> <output-dir>\n"
        "e.g.:\n"
        "nnet3-compile-xvector-net --min-chun-size=25 --max-chunk-size=10000 final.tdnn6.raw ark:data/train/utt2num_frames.nosil exp/xv_compile\n"
        "Note:\n"
        "utt2chunk-rspecifier format: index    chunk-size\n"
        "                             utt-1    200\n"
        "                             utt-2    362\n"
        "                             ...         \n"
        "                             utt-724  927\n"
        "\n"
        "Make sure the output-dir is exist.\n"
        "And the output files will be ${output-dir}/${chunk-siez}.xv.compile.\n";


    ParseOptions po(usage);
    Timer timer;

    NnetSimpleComputationOptions opts;
    CachingOptimizingCompilerOptions compiler_config;

    opts.acoustic_scale = 1.0; // by default do no scaling in this recipe.

    std::string use_gpu = "no";
    bool binary = true;
    bool pad_input = true;

    int32 max_chunk_size = -1,
      min_chunk_size = 25;

    opts.Register(&po);
    compiler_config.Register(&po);

    po.Register("use-gpu", &use_gpu,
      "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("binary", &binary,
        "If false, write the compiled files with text format.");
    po.Register("pad-input", &pad_input, "If true, duplicate the first and "
        "last frames of the input features as required to equal min-chunk-size.");
    po.Register("max-chunk-size", &max_chunk_size,
      "If set, extracts xectors from specified chunk-size, and averages.  "
      "If not set, extracts an xvector from all available features.");
    po.Register("min-chunk-size", &min_chunk_size,
      "Minimum chunk-size allowed when extracting xvectors.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    if (min_chunk_size < 1) {
        KALDI_ERR << "min chunk size "<<min_chunk_size<<" is not >= 1";
    }
    if (max_chunk_size != -1 && max_chunk_size < min_chunk_size) {
        KALDI_ERR << "max chunk size  " << max_chunk_size << " is not -1 or not greater than min chunk size " << min_chunk_size;
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
        chunk_rspecifier = po.GetArg(2),
        output_dir = po.GetArg(3);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);
    SetBatchnormTestMode(true, &nnet);
    SetDropoutTestMode(true, &nnet);
    CollapseModel(CollapseModelConfig(), &nnet);

    CachingOptimizingCompiler compiler(nnet, opts.optimize_config, compiler_config);

    SequentialInt32Reader chunk_reader(chunk_rspecifier);

    typedef unordered_map<int32, int32> HashType;
    HashType compile_done;

    std::ofstream compile_file;

    int32 num_chunk = 0, num_duplicate = 0, num_fail = 0;
    char out_path[200];

    ComputationRequest request;
    request.need_model_derivative = false;
    request.store_component_stats = false;
    IoSpecification output_spec;
    output_spec.name = "output";
    output_spec.has_deriv = false;
    output_spec.indexes.resize(1);
    request.outputs.resize(1);
    request.outputs[0].Swap(&output_spec);

    Timer io_time;
    double tot_io_time = 0.0;

    for (; !chunk_reader.Done(); chunk_reader.Next()) {
        int32 this_chunk_size = chunk_reader.Value();

        if (this_chunk_size > max_chunk_size && max_chunk_size != -1) {
            KALDI_WARN << "The chunk "<< this_chunk_size <<" is greater than max chunk size " << max_chunk_size;
            this_chunk_size = this_chunk_size % max_chunk_size;
        }

        if ( !pad_input && this_chunk_size < min_chunk_size) {
            KALDI_WARN << "The chunk "<< this_chunk_size <<" is not greater than min chunk size " << min_chunk_size << " without pad.";
            num_fail++;
            continue;
        }
        else if (pad_input && this_chunk_size < min_chunk_size) {
            this_chunk_size = min_chunk_size;
        }

        if (compile_done.count(this_chunk_size) == 0) {
            request.inputs.clear();
            request.inputs.push_back(IoSpecification("input", 0, this_chunk_size));
            
            std::shared_ptr<const NnetComputation> computation(std::move(compiler.Compile(request)));

            if (output_dir.length() + 10 > 200) {
                KALDI_ERR << "Too long path " << output_dir;
            }
            sprintf(out_path, "%s/%d.xv.compile", output_dir.c_str(), this_chunk_size);

            io_time.Reset();
            compile_file.open(out_path);
            computation->Write(compile_file, binary);
            compile_file.close();
            tot_io_time += io_time.Elapsed();

            compile_done[this_chunk_size] = 1;
            num_chunk++;
        }
        else {
            num_duplicate++;
        }
    }


#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed << "s ( Including "<< tot_io_time << "s I/O ).";
    KALDI_LOG << "Done " << num_chunk << " chunk-nets, " << num_duplicate << " duplicated, "<< num_fail << " failed.";

    if (num_chunk != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
