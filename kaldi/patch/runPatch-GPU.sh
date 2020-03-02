#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-12-27)

# Change the kaldidir by yourself and run this patch script before compiling src (independent to tools)

kaldidir=/work/kaldi

# There is an another good way now [2019-07-27]

default_proportion=0.05 # If GPU has 12Gb memeory, 12Gb * 0.01 = 120M, it means that allocate 120M memeory every times
low_proportion=0.01 # Should keep default_proportion > low_proportion all the time, or there is a possible error 
defualt_num_subregions=40 

file=$kaldidir/src/cudamatrix/cu-allocator.h
cp $file $file.bk # backup

cd $kaldidir # Change workdir
git checkout -- $file # Recover to kaldi version. You can use this command anytime in kaldidir if you want to recover changes

# Could also change this file by hand. Just edit three places.
# Focus on "memory_proportion(0.05)". "memory_proportion >= 0.05" and "num_subregions(20)"
# in follows:
##CuAllocatorOptions():
##     cache_memory(true), memory_proportion(0.01), num_subregions(20) { }
##...
##KALDI_ASSERT(memory_proportion >= 0.001 && memory_proportion < 0.99);

sed -i 's/memory_proportion(.\+),/memory_proportion('"$default_proportion"'),/g' $file
sed -i 's/num_subregions(.\+) { }/num_subregions('"$defualt_num_subregions"') { }/g' $file
sed -i 's/\(memory_proportion \+>= \+\).\+\( \+&&\)/\1'"$low_proportion"'\2/g' $file

echo "All done. And $kaldidir/src is ready to be compiled now."

#####################################################################################################
## Deprecated ##

# # You can use 'git' command to recover these file or copy form .bk directory.
# mkdir -p $kaldidir/src/cudamatrix/.bk/
# mv $kaldidir/src/cudamatrix/{cu-allocator*,cu-device*} $kaldidir/src/cudamatrix/.bk/
# cp -f src/cudamatrix/* $kaldidir/src/cudamatrix/

# # This command could only deal with the Kaldi src with 2019-05-28 version
# sed -i 's/HAVE_CUDA == 1/HAVE_CUDA == 2/g' $kaldidir/src/nnet3/nnet-utils.cc
# sed -i 's/if (did_output_to_gpu)/\/\/if (did_output_to_gpu)/g' $kaldidir/src/nnet3/nnet-batch-compute.cc
# sed -i 's/SynchronizeGpu/\/\/SynchronizeGpu/g' $kaldidir/src/nnet3/nnet-batch-compute.cc
# sed -i 's/RegisterCuAllocatorOptions/\/\/RegisterCuAllocatorOptions/g' $kaldidir/src/nnet3bin/nnet3-train.cc
# sed -i 's/RegisterCuAllocatorOptions/\/\/RegisterCuAllocatorOptions/g' $kaldidir/src/chainbin/nnet3-chain-train.cc
# sed -i 's/RegisterCuAllocatorOptions/\/\/RegisterCuAllocatorOptions/g' $kaldidir/src/rnnlmbin/rnnlm-train.cc
# sed -i 's/cudadecoderbin//' $kaldidir/src/Makefile # Do not compile the cudadecoderbin

# for x in nnet3-latgen-faster-batch.cc nnet3-compute.cc nnet3-compute-batch.cc nnet3-xvector-compute.cc nnet3-latgen-faster-batch.cc;do
# sed -i 's/CuDevice::RegisterDeviceOptions/\/\/CuDevice::RegisterDeviceOptions/g' $kaldidir/src/nnet3bin/$x
# done

# echo "All done and you can start to recompile the src/cudamatrix, src/nnet3, src/nnet3bin and src/chainbin."
