#!/bin/bash

# Copyright xmuspeech (Author: Snowdar 2020-05-03)

### A record of Snowdar's experiments about voxceleb1-O/E/H tasks. Suggest to execute every script one by one.

### Reference
##  Paper: Chung, Joon Son, Arsha Nagrani, and Andrew Zisserman. 2018. “Voxceleb2: Deep Speaker Recognition.” 
##         ArXiv Preprint ArXiv:1806.05622.
##  Data downloading: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/

### Dataset info (The voxceleb2.test is not used here and the avg length of utterances in both trainset and testset is about 8s.)
##  Only trainset: voxceleb2_train = voxceleb2.dev (num_utts: 1092009, num_speakers: 5994, duration: >2300h)
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
# Prepare the data/voxceleb1_train, data/voxceleb1_test and data/voxceleb2_train.
# ==> Make sure the audio datasets (voxceleb1, voxceleb2, RIRS and Musan) have been downloaded by yourself.
voxceleb1_path=/data/voxceleb1
voxceleb2_path=/data/voxceleb2
rirs_path=/data/RIRS_NOISES/
musan_path=/data/musan

subtools/recipe/voxceleb/prepare/make_voxceleb1_v2.pl $voxceleb1_path dev data/voxceleb1_train
subtools/recipe/voxceleb/prepare/make_voxceleb1_v2.pl $voxceleb1_path test data/voxceleb1_test
subtools/recipe/voxceleb/prepare/make_voxceleb2.pl $voxceleb2_path dev data/voxceleb2_train

# Combine testset voxceleb1 = voxceleb1_train + voxceleb1_test
subtools/kaldi/utils/combine_data.sh data/voxceleb1 data/voxceleb1_train data/voxceleb1_test

# Get trials
# ==> Make sure all trials are in data/voxceleb1.
subtools/recipe/voxceleb/prepare/get_trials.sh --dir data/voxceleb1 --tasks "voxceleb1-O voxceleb1-E voxceleb1-H \
                                                                             voxceleb1-O-clean voxceleb1-E-clean voxceleb1-H-clean"

# Get the copies of dataset which is labeled by a prefix like mfcc_23_pitch or fbank_40_pitch etc.
subtools/newCopyData.sh mfcc_23_pitch "voxceleb2_train voxceleb1"

# Augment trainset by clean:aug=1:4 with Kaldi augmentation (total 5 copies).
subtools/augmentDataByNoise.sh --rirs-noises $rirs_path --musan $musan_path --factor 4 \
                                data/mfcc_23_pitch/voxceleb2_train/ data/mfcc_23_pitch/voxceleb2_train_augx5
# Make features for trainset
subtools/makeFeatures.sh --pitch true --pitch-config subtools/conf/pitch.conf data/mfcc_23_pitch/voxceleb2_train_augx5/ mfcc \
                                subtools/conf/sre-mfcc-23.conf

# Compute VAD for augmented trainset
subtools/computeAugmentedVad.sh data/mfcc_23_pitch/voxceleb2_train_augx5 data/mfcc_23_pitch/voxceleb2_train/utt2spk \
                                subtools/conf/vad-5.5.conf

# Get a clean copy of voxceleb2_train by spliting from voxceleb2_train_augx5
subtools/filterDataDir.sh --split-aug false data/mfcc_23_pitch/voxceleb2_train_augx5/ data/voxceleb2_train/utt2spk \
                                data/mfcc_23_pitch/voxceleb2_train

# Make features for testset
subtools/makeFeatures.sh --pitch true --pitch-config subtools/conf/pitch.conf data/mfcc_23_pitch/voxceleb1/ mfcc \
                                subtools/conf/sre-mfcc-23.conf

# Compute VAD for testset which is clean
subtools/computeVad.sh data/mfcc_23_pitch/voxceleb1/ subtools/conf/vad-5.5.conf


### Training (preprocess -> get_egs -> training -> extract_xvectors)
# The launcher is a python script which is the main pipeline for it is independent with the data preparing and the scoring.
# The launcher just train an extended x-vector baseline system and other methods like multi-gpu training,
# AM-softmax loss etc. could be set by yourself. 
subtools/runPytorchLauncher.sh runExtendedXvector-voxceleb2-mfcc.py --stage=0

### Back-end scoring
# Scoring with only voxceleb1_train_aug trainig.
# extended_voxceleb2x5_mfcc is the model dir which is set in runExtendedXvector-voxceleb2-mfcc.py.

for task in voxceleb1_O voxceleb1_E voxceleb1_H voxceleb1_O_clean voxceleb1_E_clean voxceleb1_H_clean;do
    # Cosine: lda128 -> norm -> cosine -> AS-norm (Near emdedding is better)
    # If use AM-softmax, replace lda with submean (--lda false --submean true) could have better performace based on cosine scoring.
    subtools/recipe/voxceleb/gather_results_from_epochs.sh --vectordir exp/extended_voxceleb2x5_mfcc  \
                                                        --epochs "15" --score cosine --enrollset ${task}_enroll --testset ${task}_test \
                                                        --trainset voxceleb2_train --score-norm true

    # PLDA: lda256 -> norm -> PLDA
    subtools/recipe/voxceleb/gather_results_from_epochs.sh --vectordir exp/extended_voxceleb2x5_mfcc  \
                                                        --epochs "15" --score plda --enrollset ${task}_enroll --testset ${task}_test \
                                                        --trainset voxceleb2_train --score-norm false
done