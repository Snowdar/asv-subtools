#!/bin/bash

# Copyright xmuspeech (Author: Snowdar 2020-02-23)

### A record of my experiments about voxceleb1 and voxceleb2 and suggest to execute every script one by one.


### Start 
# Please prepare the data/voxceleb1_train, data/voxceleb2_train and data/voxceleb1_test by yourself (or official make*.pl).
# voxceleb_train = voxceleb1_train + voxceleb2_train (without test both)
subtools/kaldi/utils/combine_data.sh data/voxceleb_train data/voxceleb1_train data/voxceleb2_train

# Get the copies of dataset which is labeled by a prefix like mfcc_23_pitch or fbank_40 etc.
subtools/newCopyData.sh mfcc_23_pitch "voxceleb_train voxceleb1_test"
cp data/voxceleb1_test/trials data/mfcc_23_pitch/voxceleb1_test

# Augment trainset by 1:1 (clean/aug) with Kaldi augmentation (reverb noise music babble)
subtools/augmentDataByNoise.sh --rirs-noises /data/ASV/SRE19/RIRS_NOISES/ --musan /data/ASV/SRE19/musan --factor 1 \
                                data/mfcc_23_pitch/voxceleb_train/ data/mfcc_23_pitch/voxceleb_train_aug
# Make features for trainset
subtools/makeFeatures.sh --pitch true --pitch-config subtools/conf/pitch.conf data/mfcc_23_pitch/voxceleb_train_aug/ mfcc \
                                subtools/conf/sre-mfcc-23.conf
# Compute VAD for augmented trainset
subtools/computeAugmentedVad.sh --vad-conf subtools/conf/vad-5.5.conf data/mfcc_23_pitch/voxceleb_train_aug/ \
                                data/mfcc_23_pitch/voxceleb_train/utt2spk
# Get a copy of voxceleb1_train_aug by spliting from voxceleb_train_aug
subtools/filterDataDir.sh --split-aug true data/mfcc_23_pitch/voxceleb_train_aug/ data/voxceleb1_train/utt2spk \
                                data/mfcc_23_pitch/voxceleb1_train_aug

# Get a copy of clean dataset for scoring by spliting from voxceleb_train_aug
subtools/filterDataDir.sh --check false --split-aug false data/mfcc_23_pitch/voxceleb_train_aug/ data/voxceleb_train/utt2spk \
                                data/mfcc_23_pitch/voxceleb_train

subtools/filterDataDir.sh --split-aug false data/mfcc_23_pitch/voxceleb_train_aug/ data/voxceleb1_train/utt2spk \
                                data/mfcc_23_pitch/voxceleb1_train

subtools/filterDataDir.sh --split-aug false data/mfcc_23_pitch/voxceleb_train_aug/ data/voxceleb2_train/utt2spk \
                                data/mfcc_23_pitch/voxceleb2_train

# Make features for testset
subtools/makeFeatures.sh --pitch true --pitch-config subtools/conf/pitch.conf data/mfcc_23_pitch/voxceleb1_test/ mfcc \
                                subtools/conf/sre-mfcc-23.conf

# Compute VAD for testset which is clean
subtools/computeVad.sh data/mfcc_23_pitch/voxceleb1_test/ subtools/conf/vad-5.5.conf


### Training (preprocess + get_egs + training + extract_xvectors)
# The launcher is a python script which is the main pipeline for it is independent with the data preparing and the scoring.
python3 subtools/pytorch/launcher/runSnowdarXvector-voxceleb1.py --stage=0
python3 subtools/pytorch/launcher/runSnowdarXvector-voxceleb2.py --stage=0


### Back-end scoring
# Training with only voxceleb1_train_aug
subtools/recipe/voxceleb/gather_results_from_epochs.sh --vectordir exp/standard_xv_baseline_warmR_voxceleb1_adam  \
                                                       --epochs "7 14 21" --score cosine --score-norm true
subtools/recipe/voxceleb/gather_results_from_epochs.sh --vectordir exp/standard_xv_baseline_warmR_voxceleb1_adam  \
                                                       --epochs "7 14 21" --score plda --score-norm true

# Training with voxceleb_train_aug
subtools/recipe/voxceleb/gather_results_from_epochs.sh --vectordir exp/standard_xv_baseline_warmR_voxceleb2_adam  \
                                                       --epochs "1 3 7 15" --score cosine --score-norm true
subtools/recipe/voxceleb/gather_results_from_epochs.sh --vectordir exp/ standard_xv_baseline_warmR_voxceleb2_adam \
                                                       --epochs "1 3 7 15" --score plda --score-norm true

### Done
#
#### Report EER% ####
