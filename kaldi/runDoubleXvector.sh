#!/bin/bash

# Copyright     Tsinghua  (Author:YiLiu earlier than 2018-10-01)
#               xmuspeech (Author:Snowdar 2018-10-01 2019-01-27)


# This script is to train a multitask xvector network which contains two xvector branches with different tasks.

set -e

stage=0
endstage=6

train_stage=-10

use_gpu=true
clean=true
remove_egs=true

sleep_time=3
model_limit=8

xv_min_chunk=60
xv_max_chunk=80 # equal to xv_min_len:utt-length should be always >= xv_max_chunk

other_min_chunk=60
other_max_chunk=80

num_archives=150

xvTrainData=data/plp_20_5.0/baseTrain
otherTrainData=data/plp_20_5.0/other_baseTrain  # the feat-type and dim of two traindatas should be consistent

outputname=base_multiTask_xv_plp_20_5.0_cmn # just a output name and the real output-path is exp/$outputname

. subtools/path.sh
. subtools/kaldi/utils/parse_options.sh

########## auto variables ################
nnet_dir=exp/$outputname
other_egs_dir=exp/$outputname/other_egs
xv_egs_dir=exp/$outputname/xvector_egs

mkdir -p $nnet_dir
echo -e "SleepTime=$sleep_time\nLimit=$model_limit" > $nnet_dir/control.conf

xv_feat_dim=$(feat-to-dim scp:$xvTrainData/feats.scp -) || exit 1
other_feat_dim=$(feat-to-dim scp:$otherTrainData/feats.scp -) || exit 1
[ $xv_feat_dim != $other_feat_dim ] && echo "[exit] Dim of $xvTrainData is not equal to $otherTrainData" && exit 1

feat_dim=$xv_feat_dim


#### stage --> go #####
if [[ $stage -le 0 && 0 -le $endstage ]];then
	echo "[stage 0] Prepare xvTrainData dir with no silence frames for xvector egs"
	rm -rf ${xvTrainData}_nosil
	rm -rf exp/features/${xvTrainData}_nosil
	subtools/kaldi/sid/nnet3/xvector/prepare_feats_for_egs.sh --nj 20 --cmd "run.pl" \
			$xvTrainData ${xvTrainData}_nosil exp/features/${xvTrainData}_nosil
			
	rm -rf ${otherTrainData}_nosil
	rm -rf exp/features/${otherTrainData}_nosil
	subtools/kaldi/sid/nnet3/xvector/prepare_feats_for_egs.sh --nj 20 --cmd "run.pl" \
			$otherTrainData ${otherTrainData}_nosil exp/features/${otherTrainData}_nosil
fi

if [[ $stage -le 1 && 1 -le $endstage ]];then
	echo "[stage 1] Remove utts whose length is less than the lower limit value"
	subtools/removeUtt.sh ${otherTrainData}_nosil  $other_min_chunk
	subtools/removeUtt.sh ${xvTrainData}_nosil $xv_max_chunk
fi


other_output="other_output"

if [[ $stage -le 2 && 2 -le $endstage ]];then 
	echo "[stage 2] Prepare multitask network config" 
	other_num_targets=$(awk '{print $1}' $otherTrainData/spk2utt | sort | wc -l | awk '{print $1}') || exit 1
	xv_num_targets=$(awk '{print $1}' $xvTrainData/spk2utt | sort | wc -l | awk '{print $1}') || exit 1
	max_chunk_size=10000
	min_chunk_size=25
	
	mkdir -p $nnet_dir/configs/other

	cat <<EOF > $nnet_dir/configs/network.xconfig
	  # please note that it is important to have input layer with the name=input

	  # The frame-level layers
	  input dim=${feat_dim} name=input
	  
	  # shared layers
	  relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=512
	  relu-batchnorm-layer name=tdnn2 input=Append(-2,0,2) dim=512
	  relu-batchnorm-layer name=tdnn3 input=Append(-3,0,3) dim=512
	  relu-batchnorm-layer name=tdnn4 dim=512

	  # other branch
	  relu-batchnorm-layer name=other_tdnn5 dim=1500 input=tdnn4
	  stats-layer name=other_stats config=mean+stddev(0:1:1:${max_chunk_size})
	  relu-batchnorm-layer name=other_tdnn6 dim=512 input=other_stats
	  relu-batchnorm-layer name=other_tdnn7 dim=512 
	  output-layer name=$other_output include-log-softmax=true dim=$other_num_targets
	  
	  # xvector branch
	  relu-batchnorm-layer name=tdnn5 dim=1500 input=tdnn4
	  
	  # The stats pooling layer. Layers after this are segment-level.
	  # In the config below, the first and last argument (0, and ${max_chunk_size})
	  # means that we pool over an input segment starting at frame 0
	  # and ending at frame ${max_chunk_size} or earlier.  The other arguments (1:1)
	  # mean that no subsampling is performed.
	  stats-layer name=stats config=mean+stddev(0:1:1:${max_chunk_size})

	  # This is where we usually extract the embedding (aka xvector) from.
	  relu-batchnorm-layer name=tdnn6 dim=512 input=stats

	  # This is where another layer the embedding could be extracted
	  # from, but usually the previous one works better.
	  relu-batchnorm-layer name=tdnn7 dim=512
	  output-layer name=output include-log-softmax=true dim=${xv_num_targets}
EOF
	
	# parse nnet config with other as main branch,but we just need the "vars" file here
	sed 's/name=output/name=xvector_output/g' $nnet_dir/configs/network.xconfig | \
	sed ''s/name=$other_output/name=output/g'' > $nnet_dir/configs/other/network.xconfig
	subtools/kaldi/steps/nnet3/xconfig_to_configs.py \
		--xconfig-file $nnet_dir/configs/other/network.xconfig \
		--config-dir $nnet_dir/configs/other
		
	# parse nnet config with xvector as main branch	and use it to init raw model
	subtools/kaldi/steps/nnet3/xconfig_to_configs.py \
		--xconfig-file $nnet_dir/configs/network.xconfig \
		--config-dir $nnet_dir/configs
		
	cp $nnet_dir/configs/vars $nnet_dir/configs/vars_xvec
	cp $nnet_dir/configs/other/vars $nnet_dir/configs/vars_am
	
	# some configs for extracting xvector
	echo "output-node name=output input=tdnn6.affine" > $nnet_dir/extract_tdnn6.config
	cp -f $nnet_dir/extract_tdnn6.config $nnet_dir/extract.config
	echo "output-node name=output input=tdnn7.affine" > $nnet_dir/extract_tdnn7.config
	echo "$max_chunk_size" > $nnet_dir/max_chunk_size
	echo "$min_chunk_size" > $nnet_dir/min_chunk_size
fi	

# note:
# for train_cvector_dnn.py script (by YiLiu), the num of egs of xvector and other should be equal and 
# *egs.*.scp is required to exist in both xvector and other egs dir.Next, the "archive_chunk_lengths"
# file should be exist in $xv_egs_dir rather than $xv_egs_dir/temp where this file is generated initially. 
#
# the reason why don't combine the two type egs before training is that the author provide a c++ paragram 
# "nnet3-copy-cvector-egs" which can combine multitask egs temporarily when training,but it is not must
# because "nnet3-copy-egs" (kaldi provide) can also achive this purpose by option --outputs,which could be
# tedious than "nnet3-copy-cvector-egs".Ok,the training c++ paragrams like nnet3-train,nnet3-compute-prob
# are how to recognize multi-egs to update params of different branch of a shared network is very interesting,
# which is refered to the format of egs,a string like "<NnetIo> output <I1V>",which means this egs will be 
# used for a output-node (see parsed config ) whose name is "output" and ignore other branch.Yeh，the output-node 
# named "output" is a main branch and others,such as "other_output", will be as a secondary branch,which 
# refering to "nnet3-compute",but by "nnet3-[am-]copy",you can still change the master-slave relationship always 
# when you just have a final.raw/final.mdl.

## xvector egs ##
##############################################
if [[ $stage -le 3 && 3 -le $endstage ]];then
	echo "[stage 3] get xvector egs"
	subtools/kaldi/sid/nnet3/xvector/get_egs.sh --cmd "run.pl" \
		--nj 20 \
		--stage 0 \
		--num-train-archives $num_archives \
		--frames-per-iter-diagnostic 100000 \
		--min-frames-per-chunk $xv_min_chunk \
		--max-frames-per-chunk $xv_max_chunk \
		--num-diagnostic-archives 3 \
		--num-repeats 6000 \
		"${xvTrainData}_nosil" $xv_egs_dir
	
	# training script needs this file
	cp -f $xv_egs_dir/temp/archive_chunk_lengths $xv_egs_dir
fi

## other egs ##
##############################################	
if [[ $stage -le 4 && 4 -le $endstage ]];then
	echo "[stage 4] get other egs"
	subtools/kaldi/sid/nnet3/xvector/get_egs.sh --cmd "run.pl" \
		--nj 20 \
		--stage 0 \
		--num-train-archives $num_archives \
		--frames-per-iter-diagnostic 100000 \
		--min-frames-per-chunk $other_min_chunk \
		--max-frames-per-chunk $other_max_chunk \
		--num-diagnostic-archives 3 \
		--num-repeats 6000 \
		"${otherTrainData}_nosil" $other_egs_dir
fi


if [[ $stage -le 5 && 5 -le $endstage ]]; then
	echo "[stage 5] train multitask nnet3 raw model"
	dropout_schedule='0,0@0.20,0.1@0.50,0'
	srand=123
	
	subtools/kaldi/steps_multitask/nnet3/train_cvector_dnn.py --stage=$train_stage \
	  --cmd="run.pl" \
	  --trainer.optimization.proportional-shrink 10 \
	  --trainer.optimization.momentum=0.5 \
	  --trainer.optimization.num-jobs-initial=2 \
	  --trainer.optimization.num-jobs-final=8 \
	  --trainer.optimization.initial-effective-lrate=0.001 \
	  --trainer.optimization.final-effective-lrate=0.0001 \
	  --trainer.optimization.minibatch-size="256;64" \
	  --trainer.srand=$srand \
	  --trainer.max-param-change=2 \
	  --trainer.num-epochs=3 \
	  --trainer.dropout-schedule="$dropout_schedule" \
	  --trainer.shuffle-buffer-size=1000 \
	  --cleanup.remove-egs=$remove_egs \
	  --cleanup.preserve-model-interval=500 \
	  --use-gpu=true \
	  --am-output-name=$other_output \
	  --am-weight=1.0 \
	  --am-egs-dir=$other_egs_dir \
	  --xvec-output-name="output" \
	  --xvec-weight=1.0 \
	  --xvec-egs-dir=$xv_egs_dir \
	  --dir=$nnet_dir  || exit 1;
fi

if [[ -f $nnet_dir/final.raw && "$clean" == "true" ]];then
        rm -f $xv_egs_dir/egs*
        rm -f $other_egs_dir/egs*
        rm -rf ${xvTrainData}_nosil
        rm -rf exp/features/${xvTrainData}_nosil
		rm -rf ${otherTrainData}_nosil
        rm -rf exp/features/${otherTrainData}_nosil
fi

if [[ $stage -le 6 && 6 -le $endstage ]]; then
	echo "[stage 8] extract multitask-xvectors of several datasets"
	prefix=plp_20_5.0
	toEXdata="baseTrain test_1s test_1s_concat_sp"
	layer="tdnn6"
	nj=20
	gpu=false
	cache=1000
	
	for x in $toEXdata ;do
		for y in $layer ;do
			num=0
			[ -f $nnet_dir/$y/$x/xvector.scp ] && num=$(grep ERROR $nnet_dir/$y/$x/log/extract.*.log | wc -l)
			[[ "$force" == "true" || ! -f $nnet_dir/$y/$x/xvector.scp || $num -gt 0 ]] && \
			subtools/kaldi/sid/nnet3/xvector/extract_xvectors.sh --cache-capacity $cache --extract-config extract_${y}.config \
				--use-gpu $gpu --nj $nj $nnet_dir data/${prefix}/$x $nnet_dir/$y/$x
			> $nnet_dir/$y/$x/$prefix
			echo "$y layer embeddings of data/$prefix/$x extracted done."
		done
	done
fi
