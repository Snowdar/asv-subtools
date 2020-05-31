#!/bin/bash

# Copyright xmuspeech (Author: Snowdar, Zheng Li 2020-05-30)
# Corresponding email: ap_olr@163.com
# Please refer to https://speech.xmu.edu.cn/ or http://olr.cslt.org for more info

### A record of baselines of AP20-OLR. Suggest to execute every script one by one.

### Start 
# Prepare data sets, all of them contain at least wav.scp, utt2lang, spk2utt and utt2spk;
# spk2utt/utt2spk could be fake, e.g. the utt-id is just the spk-id, in the test set.
#Training set
train=data/train
#AP20-OLR-test
task1_test=data/task1_test
task2_test=data/task2_test
task3_test=data/task3_test
#AP20-OLR-ref-enroll
task1_enroll=data/task1_enroll
task2_enroll=data/task2_enroll
task3_enroll=data/task3_enroll
#AP20-OLR-ref-dev
task1_dev=data/task1_dev
task2_dev=data/task2_dev

# Get the copies of dataset which is labeled by a prefix like mfcc_23_pitch or fbank_40_pitch etc.
subtools/newCopyData.sh mfcc_20_5.0 "train task1_test task2_test task3_test task1_enroll task2_enroll task3_enroll task1_dev task2_dev"

# Augment trainset by clean:aug=1:2 with speed (0.9,1.0,1.1) and volume perturbation; Make features for trainset;  Compute VAD for augmented trainset
subtools/concatSpFeats.sh --stage 0 --endstage 1 --volume true --datasets "train" --prefix mfcc_20_5.0 --feat_type mfcc \
                          --feat_conf subtools/conf/sre-mfcc-20.conf --vad_conf conf/vad-5.0.conf --pitch true --suffix aug

# Make features for testsets, enrollsets and development sets
for data in $task1_test $task2_test $task3_test $task1_enroll $task2_enroll $task3_enroll $task1_dev $task2_dev; do
  subtools/makeFeatures.sh --pitch true --pitch-config subtools/conf/pitch.conf $data mfcc \
                                subtools/conf/sre-mfcc-20.conf
done

# Compute VAD for testsets, enrollsets and development sets
for data in $task1_test $task2_test $task3_test $task1_enroll $task2_enroll $task3_enroll $task1_dev $task2_dev; do
  subtools/computeVad.sh $data subtools/conf/vad-5.0.conf
done

## Pytorch x-vector model training
# Training (preprocess -> get_egs -> training -> extract_xvectors)
# The launcher is a python script which is the main pipeline for it is independent with the data preparing and the scoring.
# Both this launcher just train a extended xvector baseline system, and other methods like multi-gpu training, 
# AM-softmax loss etc. could be set by yourself. 
python3 run_pytorch_xvector.py --stage=0

## Kaldi x-vector model training
# Training (preprocess -> get_egs -> training -> extract_xvectors)
sh run_kaldi_xvector.sh

## Kaldi i-vector model training
# Training (preprocess -> get_egs -> training -> extract_ivectors)
sh run_kaldi_ivector.sh

### Back-end scoring: lda100 -> submean -> norm -> LR 

# For AP20-OLR-ref-dev, the referenced development sets are used to help estimate
# the system performance when participants repeat the baseline systems or prepare their own systems.
# Task 1: Cross-channel LID; Task2 : dialect identification; Task3: no ref-development set provided
for exp in exp/pytorch_xvector/far_epoch21 exp/pytorch_xvector/far_epoch21 exp/kaldi_xvector/embedding1 exp/kaldi_ivector;do
  for task in 1 2;do
    subtools/scoreSets.sh --eval false --vectordir $exp --prefix mfcc_20_5.0  --enrollset=task${task}_enroll --testset=task${task}_test \
                          --lda true --clda 100 --submean true --score "lr" --metric "Cavg"
  done
done

# You can compare your results on AP20-OLR-ref-dev with results.txt to check your systems.

# For AP20-OLR-test, note that in this stage, only scores will be computed, but no metric will be given, by setting --eval true
# Task 1: Cross-channel LID; Task2 : dialect identification; Task3: noisy LID
for exp in exp/pytorch_xvector/far_epoch21 exp/pytorch_xvector/far_epoch21 exp/kaldi_xvector/embedding1 exp/kaldi_ivector;do
  for task in 1 2 3;do
    subtools/scoreSets.sh --eval true --vectordir $exp --prefix mfcc_20_5.0  --enrollset=task${task}_enroll --testset=task${task}_test \
                          --lda true --clda 100 --submean true --score "lr" --metric "Cavg"
    # Transfer the format of score file to requred format.
    subtools/score2table.sh $exp/task${task}_test/lr_task${task}_enroll_task${task}_test_lda100_submean_norm.score $exp/task${task}_test/lr_task${task}_enroll_task${task}_test_lda100_submean_norm.score.requred
  done
done

### All done ###

