#!/bin/bash

# Copyright xmuspeech (Author: Snowdar 2020-02-23)

### A record of Snowdar's experiments about voxceleb original task. Suggest to execute every script one by one.

### Reference
##  Paper: Chung, Joon Son, Arsha Nagrani, and Andrew Zisserman. 2018. “Voxceleb2: Deep Speaker Recognition.” 
##         ArXiv Preprint ArXiv:1806.05622.
##  Data downloading: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/

### Dataset info (The voxceleb2.test is not used here and the avg length of utterances in both trainset and testset is about 8s.)
##  Small trainset: voxceleb1_train = voxceleb1.dev (num_utts: 148642, num_speakers: 1211, duration: >330h)
##  Big trainset: voxceleb1o2_train = voxceleb1.dev + voxceleb2.dev (num_utts: 1240651 <= 148642 + 1092009, 
##                                                   num_speakers: 7205 <= 1211 + 5994, duration: >2630h)
##
##  Only testset: voxceleb1_test = voxceleb1.test (num_utts: 4874, num_speakers: 40)

### Task info
##  Only task of testset: voxceleb1-O
##       num_utts of enroll: 4715, 
##       num_utts of test: 4713, 
##       total num_utts: just use 4715 from 4874 testset, 
##       num_speakers: 40, 
##       num_trials: 37720


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

# Combine trainset voxceleb1o2_train = voxceleb1_train + voxceleb2_train (without test part both)
subtools/kaldi/utils/combine_data.sh data/voxceleb1o2_train data/voxceleb1_train data/voxceleb2_train

# Get trials
# ==> Make sure the original trials is in data/voxceleb1_test.
subtools/recipe/voxceleb/prepare/get_trials.sh --dir data/voxceleb1_test --tasks voxceleb1-O

# Get the copies of dataset which is labeled by a prefix like mfcc_23_pitch or fbank_40_pitch etc.
subtools/newCopyData.sh mfcc_23_pitch "voxceleb1o2_train voxceleb1_test"

# Augment trainset by clean:aug=1:1 with Kaldi augmentation (randomly select the same utts of clean dataset
# from reverb, noise, music and babble copies.)
subtools/augmentDataByNoise.sh --rirs-noises $rirs_path --musan $musan_path --factor 1 \
                                data/mfcc_23_pitch/voxceleb1o2_train/ data/mfcc_23_pitch/voxceleb1o2_train_aug
# Make features for trainset
subtools/makeFeatures.sh --pitch true --pitch-config subtools/conf/pitch.conf data/mfcc_23_pitch/voxceleb1o2_train_aug/ mfcc \
                                subtools/conf/sre-mfcc-23.conf

# Compute VAD for augmented trainset
subtools/computeAugmentedVad.sh data/mfcc_23_pitch/voxceleb1o2_train_aug data/mfcc_23_pitch/voxceleb1o2_train/utt2spk \
                                subtools/conf/vad-5.5.conf

# Get a copy of voxceleb1_train_aug by spliting from voxceleb1o2_train_aug
subtools/filterDataDir.sh --split-aug true data/mfcc_23_pitch/voxceleb1o2_train_aug/ data/voxceleb1_train/utt2spk \
                                data/mfcc_23_pitch/voxceleb1_train_aug

# Get a copy of clean dataset by spliting from voxceleb1o2_train_aug. Set --check=false to overwrite an exist dir.
# These datasets could be used by yourself in back-end scoring.
subtools/filterDataDir.sh --check false --split-aug false data/mfcc_23_pitch/voxceleb1o2_train_aug/ data/voxceleb1o2_train/utt2spk \
                                data/mfcc_23_pitch/voxceleb1o2_train

subtools/filterDataDir.sh --split-aug false data/mfcc_23_pitch/voxceleb1o2_train_aug/ data/voxceleb1_train/utt2spk \
                                data/mfcc_23_pitch/voxceleb1_train

subtools/filterDataDir.sh --split-aug false data/mfcc_23_pitch/voxceleb1o2_train_aug/ data/voxceleb2_train/utt2spk \
                                data/mfcc_23_pitch/voxceleb2_train

# Make features for testset
subtools/makeFeatures.sh --pitch true --pitch-config subtools/conf/pitch.conf data/mfcc_23_pitch/voxceleb1_test/ mfcc \
                                subtools/conf/sre-mfcc-23.conf

# Compute VAD for testset which is clean
subtools/computeVad.sh data/mfcc_23_pitch/voxceleb1_test/ subtools/conf/vad-5.5.conf


### Training (preprocess -> get_egs -> training -> extract_xvectors)
# The launcher is a python script which is the main pipeline for it is independent with the data preparing and the scoring.

subtools/runPytorchLauncher.sh runStandardXvector-voxceleb1.py --stage=0
#subtools/runPytorchLauncher.sh runStandardXvector-voxceleb1-InSpecAug-AM.py --stage=3
#subtools/runPytorchLauncher.sh runExtendedXvector-voxceleb1.py --stage=3
#subtools/runPytorchLauncher.sh runExtendedXvector-voxceleb1-InSpecAug-AM.py --stage=3

subtools/runPytorchLauncher.sh runStandardXvector-voxceleb1o2.py --stage=0
#subtools/runPytorchLauncher.sh runStandardXvector-voxceleb1o2-InSpecAug-AM.py --stage=3
#subtools/runPytorchLauncher.sh runExtendedXvector-voxceleb1o2.py --stage=3
#subtools/runPytorchLauncher.sh runExtendedXvector-voxceleb1o2-InSpecAug-AM.py --stage=3


### Back-end scoring
# Scoring the baseline with only voxceleb1_train_aug trainig.
# standard_voxceleb1 is the model dir which is set in runSnowdarXvector-voxceleb1.py.
# Cosine: lda128 -> norm -> cosine -> AS-norm (Near emdedding is better)
# If use AM-softmax, replace lda with submean (--lda false --submean true) could have better performace based on cosine scoring.
subtools/recipe/voxceleb/gather_results_from_epochs.sh --vectordir exp/standard_voxceleb1  \
                                                       --epochs "21" --score cosine --score-norm true
# PLDA: lda256 -> norm -> PLDA (Far emdedding is better and PLDA is better than Cosine here (w/o AM-softmax and just a small model))
subtools/recipe/voxceleb/gather_results_from_epochs.sh --vectordir exp/standard_voxceleb1  \
                                                       --epochs "21" --score plda --score-norm false

# Scoreing the baseline with voxceleb1o2_train_aug training.
subtools/recipe/voxceleb/gather_results_from_epochs.sh --vectordir exp/standard_voxceleb1o2 \
                                                       --epochs "15" --score cosine --score-norm true
subtools/recipe/voxceleb/gather_results_from_epochs.sh --vectordir exp/standard_voxceleb1o2 \
                                                       --epochs "15" --score plda --score-norm false

# Scoring for other models could be done like above.

### All Done ###