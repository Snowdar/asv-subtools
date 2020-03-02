#!/bin/bash

# Copyright     Tsinghua  (Author:YiLiu earlier than 2018-10-01)
#               xmuspeech (Author:Snowdar 2018-10-01)

set -e

stage=0
endstage=8

phonetic_train_stage=-10
xv_train_stage=0 # should be always >=0 in this case

use_gpu=true
clean=true
remove_egs=true
cmn=true # do sliding cmn when getting egs

sleep_time=3
model_limit=8

phonetic_min_len=20
phonetic_lr_factor=0.2 # if phonetic_lr_factor=0 ,the params of layers coming from phonetic raw are not changed when iterating.

xv_min_chunk=60
xv_max_chunk=80 # equal to xv_min_len:utt-length should be always >= xv_max_chunk

num_archives=150

xvTrainData=data/plp_20_5.0/baseTrain_concat_volume_sp
phoneticTrainData=data/plp_20_5.0/thchs30_train  # the feat-type and dim of two traindatas should be consistent
phoneticAliDir=exp/thchs30_train_dnn_ali # get ali from a am model by yourself

outputname=base_phonetic_xv_plp_20_5.0_cmn # just a output name and the real output-path is exp/$outputname

. subtools/path.sh
. subtools/kaldi/utils/parse_options.sh

########## auto variables ################
phonetic_nnet_dir=exp/$outputname/phonetic
phonetic_egs_dir=exp/$outputname/phonetic/egs

xv_nnet_dir=exp/$outputname/xvector
xv_egs_dir=exp/$outputname/xvector/egs

mkdir -p $phonetic_nnet_dir
mkdir -p $xv_nnet_dir
echo -e "SleepTime=$sleep_time\nLimit=$model_limit" > $xv_nnet_dir/control.conf

xv_feat_dim=$(feat-to-dim scp:$xvTrainData/feats.scp -) || exit 1
phonetic_feat_dim=$(feat-to-dim scp:$phoneticTrainData/feats.scp -) || exit 1
[ $xv_feat_dim != $phonetic_feat_dim ] && echo "[exit] Dim of $xvTrainData is not equal to $phoneticTrainData" && exit 1

feat_dim=$xv_feat_dim

mkdir -p $phonetic_nnet_dir
mkdir -p $xv_nnet_dir

#### stage --> go #####
if [[ $stage -le 0 && 0 -le $endstage ]];then
	echo "[stage 0] Prepare xvTrainData dir with no nonspeech frames"
	rm -rf ${xvTrainData}_nosil
	rm -rf exp/features/${xvTrainData}_nosil
	subtools/kaldi/sid/nnet3/xvector/prepare_feats_for_egs.sh --nj 20 --cmd "run.pl" \
			$xvTrainData ${xvTrainData}_nosil exp/features/${xvTrainData}_nosil
fi

if [[ $stage -le 1 && 1 -le $endstage ]];then
	echo "[stage 1] Remove utts whose length is less than the lower limit value"
	subtools/removeUtt.sh ${phoneticTrainData} $phonetic_min_len
	subtools/removeUtt.sh ${xvTrainData}_nosil $xv_max_chunk
fi

## phonetic ##
##############################################

phonetic_end_node=phonetic_tdnn5
if [[ $stage -le 2 && 2 -le $endstage ]];then
	echo "[stage 2] Prepare phonetic network config" 
	phonetic_num_targets=$(tree-info $phoneticAliDir/tree | grep num-pdfs | awk '{print $2}') || exit 1
  
	mkdir -p $phonetic_nnet_dir/configs

	cat <<EOF > $phonetic_nnet_dir/configs/network.xconfig
    input dim=$feat_dim name=input
    relu-batchnorm-layer name=phonetic_tdnn1 dim=650 input=Append(-2,-1,0,1,2)
    relu-batchnorm-layer name=phonetic_tdnn2 dim=650 input=Append(-1,0,1)
    relu-batchnorm-layer name=phonetic_tdnn3 dim=650 input=Append(-1,0,1)
    relu-batchnorm-layer name=phonetic_tdnn4 dim=650 input=Append(-3,0,3)
    relu-batchnorm-layer name=$phonetic_end_node dim=128 input=Append(-6,-3,0)
    output-layer name=output dim=$phonetic_num_targets max-change=1.5
EOF
	subtools/kaldi/steps/nnet3/xconfig_to_configs.py \
      --xconfig-file $phonetic_nnet_dir/configs/network.xconfig \
      --config-dir $phonetic_nnet_dir/configs
	cp $phonetic_nnet_dir/configs/final.config $phonetic_nnet_dir/nnet.config
fi	
	
if [[ $stage -le 3 && 3 -le $endstage ]];then
	echo "[stage 3] get egs for training phonetic nnet3 model"
	left_context=$(grep 'model_left_context' $phonetic_nnet_dir/configs/vars | cut -d '=' -f 2)
	right_context=$(grep 'model_right_context' $phonetic_nnet_dir/configs/vars | cut -d '=' -f 2)
	frame_subsampling_factor=1
	[ -f $phoneticAliDir/frame_subsampling_factor ] && frame_subsampling_factor=$(awk '{print $1}' $phoneticAliDir/frame_subsampling_factor)

	subtools/kaldi/sid/nnet3/get_egs.sh --cmd "run.pl" \
		--nj 10 \
		--stage 0 \
		--cmn $cmn \
		--frame-subsampling-factor $frame_subsampling_factor \
		--vad true \
		--frames-per-eg 1 \
		--left-context $left_context \
		--right-context $right_context \
		${phoneticTrainData} $phoneticAliDir $phonetic_egs_dir
	
	# [To indicate training without multitask, delete valid_diagnostic.scp .] why?
	rm -f $phonetic_egs_dir/valid_diagnostic.scp
fi

if [[ $stage -le 4 && 4 -le $endstage ]];then
	echo "[stage 4] train phonetic nnet3 raw model"
	subtools/kaldi/steps/nnet3/train_raw_dnn.py --stage=$phonetic_train_stage \
		--cmd="run.pl" \
		--trainer.optimization.num-jobs-initial=2 \
		--trainer.optimization.num-jobs-final=8 \
		--trainer.optimization.initial-effective-lrate=0.0015 \
		--trainer.optimization.final-effective-lrate=0.00015 \
		--trainer.optimization.minibatch-size=256,128 \
		--trainer.srand=123 \
		--trainer.max-param-change=2 \
		--trainer.num-epochs=3 \
		--egs.frames-per-eg=1 \
		--egs.dir="$phonetic_egs_dir" \
		--cleanup=true \
		--cleanup.remove-egs=$remove_egs \
		--cleanup.preserve-model-interval=10 \
		--use-gpu=$use_gpu \
		--dir=$phonetic_nnet_dir  || exit 1;
fi

## xvector ##
##############################################
if [[ $stage -le 5 && 5 -le $endstage ]];then
	echo "[stage 5] get egs for training xvector nnet3 model"
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
fi

if [[ $stage -le 6 && 6 -le $endstage ]];then 
	echo "[stage 6] prepare xvector network config based phonetic network config and init the joining nnet3 raw model"
	xv_num_targets=$(wc -w $xv_egs_dir/pdf2num | awk '{print $1}')
	max_chunk_size=10000
	min_chunk_size=25

	mkdir -p $xv_nnet_dir/configs

	cat <<EOF > $xv_nnet_dir/configs/network.xconfig
	  # please note that it is important to have input layer with the name=input

	  # The frame-level layers
	  input dim=${feat_dim} name=input
	  relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=512
	  relu-batchnorm-layer name=tdnn2 input=Append(-2,0,2) dim=512
	  relu-batchnorm-layer name=tdnn3 input=Append(-3,0,3) dim=512
	  relu-batchnorm-layer name=tdnn4 dim=512
	  relu-batchnorm-layer name=tdnn5 dim=1500 input=Append(tdnn4,${phonetic_end_node}.batchnorm)

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

	# --existing-model option could make sure that the config can be parsed successfully because 
	# ${phonetic_end_node}.batchnorm is not exist in this nnet.config after all.
	subtools/kaldi/steps/nnet3/xconfig_to_configs.py \
		--existing-model $phonetic_nnet_dir/final.raw \
		--xconfig-file $xv_nnet_dir/configs/network.xconfig \
		--config-dir $xv_nnet_dir/configs

	# if phonetic_lr_factor=0 ,the params of layers coming from phonetic raw are not changed when iterating.
	# And by nnet3-init(nnet3-copy can also complete it ),we can get a new nnet3 network which joins both 
	# phonetic and xvector network.
	run.pl $xv_nnet_dir/log/generate_input_mdl.log \
            nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=$phonetic_lr_factor" \
	    $phonetic_nnet_dir/final.raw - \| \
		nnet3-init --srand=1 - $xv_nnet_dir/configs/final.config $xv_nnet_dir/input.raw  || exit 1;
	
	# some configs for extracting xvector
	echo "output-node name=output input=tdnn6.affine" > $xv_nnet_dir/extract_tdnn6.config
	cp -f $xv_nnet_dir/extract_tdnn6.config $xv_nnet_dir/extract.config
	echo "output-node name=output input=tdnn7.affine" > $xv_nnet_dir/extract_tdnn7.config
	echo "$max_chunk_size" > $xv_nnet_dir/max_chunk_size
	echo "$min_chunk_size" > $xv_nnet_dir/min_chunk_size
fi


if [[ $stage -le 7 && 7 -le $endstage ]]; then
	echo "[stage 7] train xvector nnet3 model with some phonetic hidden layers"
	dropout_schedule='0,0@0.20,0.1@0.50,0'
	srand=123
	
	# iteration starts from 0.raw which is a copy file of input.raw and make sure xv_train_stage from 0 rather than -10
	cp -f $xv_nnet_dir/input.raw $xv_nnet_dir/0.raw
	
	subtools/kaldi/steps/nnet3/train_raw_dnn.py --stage=$xv_train_stage \
		--cmd="run.pl" \
		--trainer.optimization.proportional-shrink 10 \
		--trainer.optimization.momentum=0.5 \
		--trainer.optimization.num-jobs-initial=2 \
		--trainer.optimization.num-jobs-final=8 \
		--trainer.optimization.initial-effective-lrate=0.001 \
		--trainer.optimization.final-effective-lrate=0.0001 \
		--trainer.optimization.minibatch-size=128 \
		--trainer.srand=$srand \
		--trainer.max-param-change=2 \
		--trainer.num-epochs=3 \
		--trainer.dropout-schedule="$dropout_schedule" \
		--trainer.shuffle-buffer-size=1000 \
		--egs.frames-per-eg=1 \
		--egs.dir=$xv_egs_dir \
		--cleanup=true \
		--cleanup.remove-egs=$remove_egs \
		--cleanup.preserve-model-interval=500 \
		--use-gpu=$use_gpu \
		--dir=$xv_nnet_dir  || exit 1;
fi

if [[ -f $xv_nnet_dir/final.raw && "$clean" == "true" ]];then
	rm -f $xv_egs_dir/egs*
	rm -f $phonetic_egs_dir/egs*
	rm -rf ${xvTrainData}_nosil
	rm -rf exp/features/${xvTrainData}_nosil
fi

if [[ $stage -le 8 && 8 -le $endstage ]]; then
	echo "[stage 8] extract phonetic-xvectors of several datasets"
	prefix=plp_20_5.0
	toEXdata="baseTrain_volume_sp test_1s_concat_sp"
	layer="tdnn6"
	nj=20
	gpu=false
	cache=1000
	
	for x in $toEXdata ;do
		for y in $layer ;do
			num=0
			[ -f $xv_nnet_dir/$y/$x/xvector.scp ] && num=$(grep ERROR $xv_nnet_dir/$y/$x/log/extract.*.log | wc -l)
			[[ "$force" == "true" || ! -f $xv_nnet_dir/$y/$x/xvector.scp || $num -gt 0 ]] && \
			subtools/kaldi/sid/nnet3/xvector/extract_xvectors.sh --cache-capacity $cache --extract-config extract_${y}.config \
				--use-gpu $gpu --nj $nj $xv_nnet_dir data/${prefix}/$x $xv_nnet_dir/$y/$x
			> $xv_nnet_dir/$y/$x/$prefix
			echo "$y layer embeddings of data/$prefix/$x extracted done."
		done
	done
fi
