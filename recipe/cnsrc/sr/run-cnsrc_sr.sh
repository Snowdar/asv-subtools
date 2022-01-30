#!/bin/bash

# Copyright xmuspeech (Author: Snowdar, Tao Jiang 2022-01-29)
#
# Please refer to http://cnceleb.org/ for more info

### A record of baselines of CNCSRC2022-Task2-baseline. Suggest to execute every script one by one.



#Training set
train=train_2022
#Evaluation
task2_dev_enroll=task2_dev_target
task2_dev_test=task2_dev_pool

stage=1
endstage=4

if [[ $stage -le 1 && 1 -le $endstage ]];then
	### Start
	# A example of getting Track2_dev's wav.scp and utt2spk etc. 
	echo "start"
	#dev_enroll
	data_path=/tsdata1/tsdata/cn-celeb/Task2_dev
	data=/tsdata1/VPR/cnceleb/data/fbank_81/$task2_dev_enroll
	mkdir -p $data
	awk -v data_path=$data_path '{print $2,data_path"/"$1}' $data_path/metadata/enroll.meta > $data/wav.scp
    awk '{split($2,a,"-");print $2,a[1]}' $data_path/metadata/enroll.meta > $data/utt2spk
	
	#dev_test
	data_path=/tsdata1/tsdata/cn-celeb/Task2_dev/pool #The original data path
	data=data/fbank_81/$task2_dev_test
	mkdir -p $data
    find $data_path -name *.wav  > $data/temp.list
    awk '{split($1,a,"/");{split(a[7],b,".")};
	      print b[1],b[1]}' $data/temp.list > $data/utt2spk
    awk '{split($1,a,"/");{split(a[7],b,".")};
	      print b[1],$1}' $data/temp.list > $data/wav.scp
	
	# Fixed dir and make sure that the various files in a data directory are correctly sorted and filtered
	for x in $train $task2_dev_enroll $task2_dev_test;do
		subtools/kaldi/utils/fix_data_dir.sh data/fbank_81/$x
	done
fi

if [[ $stage -le 2 && 2 -le $endstage ]];then

	# Get the copies of dataset which is labeled by a prefix like fbank_81 or mfcc_23_pitch etc.
	for data in $train $task2_dev_enroll $task2_dev_test;do
		subtools/newCopyData.sh fbank_81 $data 
	done
	
	# Make features for trainset, enrollset and testset
	for data in $train $task2_dev_enroll ;do
		subtools/makeFeatures.sh data/fbank_81/$data fbank subtools/conf/sre-fbank-81.conf
	done
	
	# Compute VAD for trainset, enrollset and testset
	for data in $train $task2_dev_enroll ;do
		subtools/computeVad.sh data/fbank_81/$data subtools/conf/vad-5.0.conf
	done
		

fi

if [[ $stage -le 3 && 3 -le $endstage ]];then
	## Pytorch x-vector model training
	# Training (preprocess -> get_egs -> training -> extract_xvectors)
	# This launcher is a python script which is the main pipeline for x-vector model training, it is independent with the data preparing and the scoring.
	# This launcher just train an SEResNet baseline system, and other methods like multi-gpu training etc. could be set by yourself. 
	python3 run_cnsrc_sr.py --stage=0 --endstage=3
	#extract_xvectors and calulate time
	startTime=`date +"%Y-%m-%d %H:%M:%S"`
	python3 recipe/cnsrc/sr/run_cnsrc_sr.py --stage=4 --endstage=4
	exp=exp/SEResnet34_am_train_fbank81/near_epoch_6
	endTime=`date +"%Y-%m-%d %H:%M:%S"`
	st=`date -d  "$startTime" +%s`
	et=`date -d  "$endTime" +%s`
	RunTime=$(($et-$st))
	echo "Modeling time is $RunTime s" >> $exp/time.txt
fi

if [[ $stage -le 4 && 4 -le $endstage ]];then
	# Generate Trials
	subtools/getTrials.sh 1 data/fbank_81/$task2_dev_enroll/spk2utt data/fbank_81/$task2_dev_test/utt2spk data/fbank_81/$task2_dev_test/trials
	## ### Back-end scoring
	exp=exp/SEResnet34_am_train_fbank81/near_epoch_6
	recipe/cnsrc/sr/scoreSets_sr.sh --eval true --vectordir $exp --prefix fbank_81  --enrollset $task2_dev_enroll --testset $task2_dev_test --trials data/fbank_81/$task2_dev_test/trials
	
	# Transfer the format of score file to requred format.
	startTime=`date +"%Y-%m-%d %H:%M:%S"`
	python recipe/cnsrc/sr/trans_score_format.py $exp/$task2_dev_test/score/cosine_${task2_dev_enroll}_${task2_dev_test}_norm.score $exp/$task2_dev_test/score/cosine_${task2_dev_enroll}_${task2_dev_test}_norm.score.top10
	endTime=`date +"%Y-%m-%d %H:%M:%S"`
	st=`date -d  "$startTime" +%s`
	et=`date -d  "$endTime" +%s`
	RunTime=$(($et-$st))
	echo "Retrieval time is $RunTime s" >> $exp/time.txt\
	
	# Compute the final mAP
	python recipe/cnsrc/sr/cal_mAP.py $exp/$task2_dev_test/score/cosine_${task2_dev_enroll}_${task2_dev_test}_norm.score.top10 /tsdata1/tsdata/cn-celeb/Task2_dev/metadata/test.meta
fi

# ### All done ###
