#!/bin/bash

# Copyright xmuspeech (Author: Snowdar, Tao Jiang 2022-01-29)
#
# Please refer to http://cnceleb.org/ for more info

### A record of baselines of CNCSRC2022-Task1-baseline. Suggest to execute every script one by one.

#Training set
train=train_2022
#Evaluation
task1_enroll=eval_enroll
task1_test=eval_test

if [[ $stage -le 1 && 1 -le $endstage ]];then
	### Start


	# Fixed dir and make sure that the various files in a data directory are correctly sorted and filtered
	for data in $train $task1_enroll $task1_test;do
		subtools/kaldi/utils/fix_data_dir.sh data/$data
	done
	
	# Make features for trainset, enrollset and testset
	for data in $train $task1_enroll $task1_test;do
		subtools/makeFeatures.sh data/$data fbank subtools/conf/sre-fbank-81.conf
	done
	
	# Compute VAD for trainset, enrollset and testset
	for data in $train $task1_enroll $task1_test;do
		subtools/computeVad.sh data/$data subtools/conf/vad-5.0.conf
	done
		
	# Get the copies of dataset which is labeled by a prefix like fbank_81 or mfcc_23_pitch etc.
	for data in $train $task1_enroll $task1_test;do
		subtools/newCopyData.sh fbank_81 $data 
	done
fi

if [[ $stage -le 2 && 2 -le $endstage ]];then
	## Pytorch x-vector model training
	# Training (preprocess -> get_egs -> training -> extract_xvectors)
	# This launcher is a python script which is the main pipeline for x-vector model training, it is independent with the data preparing and the scoring.
	# This launcher just train an SEResNet baseline system, and other methods like multi-gpu training etc. could be set by yourself. 
	python3 run_cnsrc_sv.py --stage=0

fi

if [[ $stage -le 3 && 3 -le $endstage ]];then
	## ### Back-end scoring
	exp=exp/SEResnet34_am_train_fbank81/near_epoch_6
	subtools/recipe/cnsrc/sv/scoreSets_sv.sh --eval false --vectordir $exp --prefix fbank_81  --enrollset $task1_enroll --testset $task1_test --trials data/fbank_81/eval_test/trials/trials.lst

fi
