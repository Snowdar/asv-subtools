#!/bin/bash

# Copyright xmuspeech (Author: Snowdar, Zheng Li 2020-05-30)

. ./cmd.sh
. ./path.sh

set -e

stage=0
endstage=5
train_stage=-11
use_gpu=true
remove_egs=false

traindata=data/mfcc_20_5.0/train_aug
outputname=kaldi_xvector

. ./path.sh
. ./utils/parse_options.sh

nnet_dir=exp/$outputname
egs_dir=exp/$outputname/egs

sleep_time=3
model_limit=8
mkdir -p $nnet_dir
echo -e "SleepTime=$sleep_time\nLimit=$model_limit" > $nnet_dir/control.conf
if [[ $stage -le 0 && 0 -le $endstage ]];then
rm -rf ${traindata}_nosil
rm -rf exp/features/${traindata}_nosil
subtools/kaldi/sid/nnet3/xvector/prepare_feats_for_egs.sh --nj 20 --cmd "run.pl" \
		$traindata ${traindata}_nosil exp/features/${traindata}_nosil
fi

mkdir -p $nnet_dir
num_pdfs=$(awk '{print $2}' $traindata/utt2spk | sort | uniq -c | wc -l)

min_chunk=60
max_chunk=80

if [[ $stage -le 1 && 1 -le $endstage ]];then
subtools/removeUtt.sh ${traindata}_nosil $max_chunk
fi

#generate the nnet examples
if [[ $stage -le 2 && 2 -le $endstage ]];then

subtools/kaldi/sid/nnet3/xvector/get_egs.sh --cmd "run.pl" \
    --nj 20 \
    --stage 0 \
    --frames-per-iter 5500000000 \
    --frames-per-iter-diagnostic 100000 \
    --min-frames-per-chunk $min_chunk \
    --max-frames-per-chunk $max_chunk \
    --num-diagnostic-archives 3 \
    --num-repeats 6000 \
    "${traindata}_nosil" $egs_dir
fi

#generate the nnet config
if [[ $stage -le 3 && 3 -le $endstage ]];then
	num_targets=$(wc -w $egs_dir/pdf2num | awk '{print $1}')
	feat_dim=$(cat $egs_dir/info/feat_dim)

	max_chunk_size=10000
	min_chunk_size=25

	mkdir -p $nnet_dir/configs
	cat <<EOF > $nnet_dir/configs/network.xconfig
	# please note that it is important to have input layer with the name=input

	# The frame-level layers
	input dim=${feat_dim} name=input
	relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=512
	relu-batchnorm-layer name=tdnn2 dim=512
	relu-batchnorm-layer name=tdnn3 input=Append(-2,0,2) dim=512
	relu-batchnorm-layer name=tdnn4 dim=512
	relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=512
	relu-batchnorm-layer name=tdnn6 dim=512
	relu-batchnorm-layer name=tdnn7 input=Append(-4,0,4) dim=512
	relu-batchnorm-layer name=tdnn8 dim=512
	relu-batchnorm-layer name=tdnn9 dim=512
	relu-batchnorm-layer name=tdnn10 dim=1500

	# The stats pooling layer. Layers after this are segment-level.
	# In the config below, the first and last argument (0, and ${max_chunk_size})
	# means that we pool over an input segment starting at frame 0
	# and ending at frame ${max_chunk_size} or earlier.  The other arguments (1:1)
	# mean that no subsampling is performed.
	stats-layer name=stats config=mean+stddev(0:1:1:${max_chunk_size})

	# This is where we usually extract the embedding (aka xvector) from.
	relu-batchnorm-layer name=embedding1 dim=512 input=stats

	# This is where another layer the embedding could be extracted
	# from, but usually the previous one works better.
	relu-batchnorm-layer name=embedding2 dim=512
	output-layer name=output include-log-softmax=true dim=${num_targets}
EOF

	steps/nnet3/xconfig_to_configs.py \
      --xconfig-file $nnet_dir/configs/network.xconfig \
      --config-dir $nnet_dir/configs
	cp $nnet_dir/configs/final.config $nnet_dir/nnet.config

	# These three files will be used by sid/nnet3/xvector/extract_xvectors.sh
	echo "output-node name=output input=embedding1.affine" > $nnet_dir/extract.config
	echo "$max_chunk_size" > $nnet_dir/max_chunk_size
	echo "$min_chunk_size" > $nnet_dir/min_chunk_size

fi

if [[ $stage -le 4 && 4 -le $endstage ]]; then
	dropout_schedule='0,0@0.20,0.1@0.50,0'
	srand=123
	
  subtools/kaldi/steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --trainer.optimization.proportional-shrink 10 \
    --trainer.optimization.momentum=0.5 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=8 \
    --trainer.optimization.initial-effective-lrate=0.005 \
    --trainer.optimization.final-effective-lrate=0.0005 \
    --trainer.optimization.minibatch-size=128 \
    --trainer.srand=$srand \
    --trainer.max-param-change=2 \
    --trainer.num-epochs=5 \
    --trainer.dropout-schedule="$dropout_schedule" \
    --trainer.shuffle-buffer-size=1000 \
    --egs.frames-per-eg=1 \
    --egs.dir="$egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=true \
    --dir=$nnet_dir  || exit 1;
fi

if [[ $stage -le 5 && 5 -le $endstage ]]; then
	echo "[stage 5] extract xvectors of several datasets"
	prefix=mfcc_20_5.0
	toEXdata="train_aug task1_test task2_test task3_test task1_enroll task2_enroll task3_enroll task1_dev task2_dev"
	layer="embedding1"
	nj=5
	gpu=true
	cache=3000
	
	for x in $toEXdata ;do
		for y in $layer ;do
			num=0
			[ -f $nnet_dir/$y/$x/xvector.scp ] && num=$(grep ERROR $nnet_dir/$y/$x/log/extract.*.log | wc -l)
			[[ "$force" == "true" || ! -f $nnet_dir/$y/$x/xvector.scp || $num > 0 ]] && \
			subtools/kaldi/sid/nnet3/xvector/extract_xvectors.sh --cache-capacity $cache --extract-config extract.config \
				--use-gpu $gpu --nj $nj $nnet_dir data/${prefix}/$x $nnet_dir/$y/$x
			> $nnet_dir/$y/$x/$prefix
			echo "$y layer embeddings of data/$prefix/$x extracted done."
		done
	done
fi


