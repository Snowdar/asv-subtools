#!/bin/bash
# Copyright xmuspeech (Author:Binling Wang 2021-07-25)
train=data/train

# Get the copies of dataset which is labeled by a prefix like mfcc_23_pitch or fbank_40_pitch etc.
subtools/newCopyData.sh mfcc_20_5.0 "train"

# Augment trainset by clean:aug=1:2 with speed (0.9,1.0,1.1) and volume perturbation; Make features for trainset;  
# and compute VAD for augmented trainset
subtools/concatSpFeats.sh --stage 0 --endstage 1 --volume false --datasets "train" --prefix  mfcc_20_5.0 --feat_type mfcc \
                          --feat_conf subtools/conf/sre-mfcc-20.conf --vad_conf conf/vad-5.0.conf --pitch true



## Pytorch x-vector model training
# Training (preprocess -> get_egs -> training -> extract_xvectors)
# The launcher is a python script which is the main pipeline for it is independent with the data preparing and the scoring.
# This launcher just train a extended xvector baseline system, and other methods like multi-gpu training, 
# AM-softmax loss etc. could be set by yourself.
python3 subtools/recipe/olr2021-baseline/run_pytorch_xvector_train.py --stage=0


bash subtools/recipe/olr2021-baseline/scoreSets_olr.sh.sh


