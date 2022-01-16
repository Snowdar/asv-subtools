#!/bin/bash

# Copyright xmuspeech (Author: Snowdar 2020-05-03)

### A record of Snowdar's experiments about voxceleb1-O/E/H tasks. Suggest to execute every script one by one.

### Reference
##  Paper: Chung, Joon Son, Arsha Nagrani, and Andrew Zisserman. 2018. “Voxceleb2: Deep Speaker Recognition.” 
##         ArXiv Preprint ArXiv:1806.05622.
##  Data downloading: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/

### Dataset info (The voxceleb2.test is not used here and the avg length of utterances in both trainset and testset is about 8s.)
##  Only trainset: voxceleb2_dev = voxceleb2.dev (num_utts: 1092009, num_speakers: 5994, duration: >2300h)
##
##  Only testset: voxceleb1 = voxceleb1.dev + voxceleb1.test (num_utts: 153516, num_speakers: 1251)

### Task info
##  Original task of testset: voxceleb1-O 
##       num_utts of enroll: 4715, 
##       num_utts of test: 4713, 
##       total num_utts: just use 4715 from 4874 testset, 
##       num_speakers: 40, 
##       num_trials: 37720 (clean:37611)
##
##  Extended task of testset: voxceleb1-E 
##       num_utts of enroll: 145375, 
##       num_utts of test: 142764, 
##       total num_utts: just use 145375 from 153516 testset, 
##       num_speakers: 1251, 
##       num_trials: 581480 (clean:401308)
##
##  Hard task of testset: voxceleb1-H 
##       num_utts of enroll: 138137, 
##       num_utts of test: 135637, 
##       total num_utts: just use 138137 from 153516 testset, 
##       num_speakers: 1190, 
##       num_trials: 552536 (clean:550894)



### Start 
# [1] Prepare the data/voxceleb1_dev, data/voxceleb1_test and data/voxceleb2_dev.
# ==> Make sure the audio datasets (voxceleb1, voxceleb2, RIRS and Musan) have been downloaded by yourself.
voxceleb1_path=/data/voxceleb1
voxceleb2_path=/data/voxceleb2
rirs_path=/data/RIRS_NOISES/
musan_path=/data/musan

subtools/recipe/voxceleb/prepare/make_voxceleb1_v2.pl $voxceleb1_path dev data/voxceleb1_dev
subtools/recipe/voxceleb/prepare/make_voxceleb1_v2.pl $voxceleb1_path test data/voxceleb1_test
subtools/recipe/voxceleb/prepare/make_voxceleb2.pl $voxceleb2_path dev data/voxceleb2_dev

# [2] Combine testset voxceleb1 = voxceleb1_dev + voxceleb1_test
subtools/kaldi/utils/combine_data.sh data/voxceleb1 data/voxceleb1_dev data/voxceleb1_test

# [3] Get trials
# ==> Make sure all trials are in data/voxceleb1.
subtools/recipe/voxceleb/prepare/get_trials.sh --dir data/voxceleb1 --tasks "voxceleb1-O voxceleb1-E voxceleb1-H \
                                                          voxceleb1-O-clean voxceleb1-E-clean voxceleb1-H-clean"

# [4] Get the copies of dataset which is labeled by a prefix like fbank_81 or mfcc_23_pitch etc.
prefix=fbank_81
subtools/newCopyData.sh $prefix "voxceleb2_dev voxceleb1"

# [5] Augment trainset by clean:aug=1:4 with Kaldi augmentation (total 5 copies).
subtools/augmentDataByNoise.sh --rirs-noises $rirs_path --musan $musan_path --factor 4 \
                                data/$prefix/voxceleb2_dev/ data/$prefix/voxceleb2_dev_plus_augx4
# [6] Make features for trainset
feat_type=fbank
feat_conf=subtools/conf/sre-fbank-81.conf
subtools/makeFeatures.sh --nj 30 data/$prefix/voxceleb2_dev_plus_augx4/ $feat_type $feat_conf

# [7] Compute VAD for augmented trainset
vad_conf=subtools/conf/vad-5.5.conf
subtools/computeAugmentedVad.sh data/$prefix/voxceleb2_dev_plus_augx4 data/$prefix/voxceleb2_dev/utt2spk \
                                $vad_conf

# [8] Get a clean subset of voxceleb2_dev from voxceleb2_dev_plus_augx4 to prepare a cohort set for asnorm
subtools/filterDataDir.sh --split-aug false data/$prefix/voxceleb2_dev_plus_augx4/ data/voxceleb2_dev/utt2spk \
                                            data/$prefix/voxceleb2_dev

# [9] Make features for testset
subtools/makeFeatures.sh data/$prefix/voxceleb1/ $feat_type $feat_conf

# [10] Compute VAD for testset which is clean
subtools/computeVad.sh data/$prefix/voxceleb1/ $vad_conf

### Training (preprocess -> get_egs -> training -> extract_xvectors)
# Note that, the launcher is a python script which is the main pipeline for it is independent with the 
# data preparing and back-end scoring. Here, we run every step one by one to show how it works.

# [11] Sample egs. It will do cmn and vad firstly and then remove invalid utts. Finally, 
#                  it samples egs to fixed chunk-size with instance sampling.
subtools/runPytorchLauncher.sh subtools/recipe/voxcelebSRC/run-resnet34-fbank-81-benchmark.py --stage=0 --endstage=2

# [12] Train a thin Resnet34 model with AM-Softmax loss and 4 GPUs will be used to accelerate training
subtools/runPytorchLauncher.sh subtools/recipe/voxcelebSRC/run-resnet34-fbank-81-benchmark.py --stage=3 --endstage=3 --gpu-id=0,1,2,3

# [13] Extract near xvectors in epoch 6 for voxceleb1 and voxceleb2_dev
subtools/runPytorchLauncher.sh subtools/recipe/voxcelebSRC/run-resnet34-fbank-81-benchmark.py --stage=4

### Back-end scoring
# [14] Score with submean + Cosine + AS-Norm processes
tasks="vox1-O vox1-O-clean vox1-E vox1-E-clean vox1-H vox1-H-clean"
score_norm=false
for task in $tasks;do
    [ "$task" == "vox1-O" ] && score_norm=true
    [ "$task" == "vox1-O-clean" ] && score_norm=true
    subtools/recipe/voxcelebSRC/gather_results_from_epochs.sh --prefix $prefix --score cosine  --submean true \
         --vectordir "exp/resnet34_fbank_81_benchmark" --task $task --epochs "6" --postions "near" \
         --score-norm $score_norm --score-norm-method "asnorm" --top-n 100 --cohort-set voxceleb2_dev
done

#### Report ####
# Egs = Voxceleb2_dev(+augx4) + sequential sampling
# Optimization = [SGD (lr = 0.01) + ReduceLROnPlateau] x 4 GPUs (total batch-size=512)
# Resnet34 (channels = 32, 64, 128, 256) + Stats-Pooling + FC-ReLU-BN-FC-BN + AM-Softmax (margin = 0.2)
#
# Back-end = near + Cosine
#
#  EER%       vox1-O   vox1-O-clean   vox1-E   vox1-E-clean   vox1-H   vox1-H-clean
#  Baseline   1.304    1.159          1.35     1.223          2.357    2.238       
#  Submean    1.262    1.096          1.338    1.206          2.355    2.223       
#  AS-Norm    1.161    1.026          -        -              -        -           
#
###### Just A Baseline Here #######
