#!/bin/bash

# Copyright xmuspeech (Author: Snowdar 2020-02-23)

### A record of Snowdar's experiments about voxceleb dataset. Suggest to execute every script one by one.
### Testset is voxceleb1-O and trainset is voxceleb1.dev or voxceleb1.dev + voxceleb2.dev.

### Start 
# Please prepare the data/voxceleb1_train, data/voxceleb2_train and data/voxceleb1_test by yourself (or official make*.pl).
# voxceleb1o2_train = voxceleb1_train + voxceleb2_train (without test part both)
subtools/kaldi/utils/combine_data.sh data/voxceleb1o2_train data/voxceleb1_train data/voxceleb2_train

# Get the copies of dataset which is labeled by a prefix like mfcc_23_pitch or fbank_40_pitch etc.
# Make sure trials is in data/voxceleb1_test.
subtools/newCopyData.sh mfcc_23_pitch "voxceleb1o2_train voxceleb1_test"

# Augment trainset by 1:1 (clean/aug) with Kaldi augmentation (reverb noise music babble)
subtools/augmentDataByNoise.sh --rirs-noises /data/RIRS_NOISES/ --musan /data/musan --factor 1 \
                                data/mfcc_23_pitch/voxceleb1o2_train/ data/mfcc_23_pitch/voxceleb1o2_train_aug
# Make features for trainset
subtools/makeFeatures.sh --pitch true --pitch-config subtools/conf/pitch.conf data/mfcc_23_pitch/voxceleb1o2_train_aug/ mfcc \
                                subtools/conf/sre-mfcc-23.conf
# Compute VAD for augmented trainset
subtools/computeAugmentedVad.sh  data/mfcc_23_pitch/voxceleb1o2_train_aug data/mfcc_23_pitch/voxceleb1o2_train/utt2spk \
                                 subtools/conf/vad-5.5.conf

# Get a copy of voxceleb1_train_aug by spliting from voxceleb1o2_train_aug
subtools/filterDataDir.sh --split-aug true data/mfcc_23_pitch/voxceleb1o2_train_aug/ data/voxceleb1_train/utt2spk \
                                data/mfcc_23_pitch/voxceleb1_train_aug

# Get a copy of clean dataset for scoring by spliting from voxceleb1o2_train_aug
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
# Both two launchers just train a standard x-vector baseline system and other methods like multi-gpu training, extended xvector, 
# AM-softmax loss etc. could be set by yourself. 
subtools/runPytorchLauncher.sh subtools/recipe/voxceleb/runSnowdarXvector-voxceleb1.py --stage=0
subtools/runPytorchLauncher.sh subtools/recipe/voxceleb/runSnowdarXvector-voxceleb1o2.py --stage=0


### Back-end scoring
# Scoring with only voxceleb1_train_aug trainig.
# standard_voxceleb1 is the model dir which is set in runSnowdarXvector-voxceleb1.py.
# Cosine: lda128 -> norm -> cosine -> AS-norm (Near emdedding is better)
# If use AM-softmax, replace lda with submean (--lda false --submean true) could have better performace based on cosine scoring.
subtools/recipe/voxceleb/gather_results_from_epochs.sh --vectordir exp/standard_voxceleb1  \
                                                       --epochs "21" --score cosine --score-norm true
# PLDA: lda256 -> norm -> PLDA (Far emdedding is better and PLDA is better than Cosine here (w/o AM-softmax and just a small model))
subtools/recipe/voxceleb/gather_results_from_epochs.sh --vectordir exp/standard_voxceleb1  \
                                                       --epochs "21" --score plda --score-norm false

# Scoreing with voxceleb1o2_train_aug training.
subtools/recipe/voxceleb/gather_results_from_epochs.sh --vectordir exp/standard_voxceleb1o2 \
                                                       --epochs "15" --score cosine --score-norm true
subtools/recipe/voxceleb/gather_results_from_epochs.sh --vectordir exp/standard_voxceleb1o2 \
                                                       --epochs "15" --score plda --score-norm false

### All Done ###
##
#### Report####
