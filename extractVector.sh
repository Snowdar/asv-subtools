#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-7-27)

use_gpu=false #just for x-vector
cache=3000 #just for x-vector
extract_config=extract.config #just for x-vector
nj=30
split_type=order #just for x-vector
model=final.raw #just for x-vector

. subtools/parse_options.sh

if [[ $# != 4 ]];then
echo "[exit] Num of parameters is not equal to 4"
echo "usage:$0 [--use-gpu false|true] [--nj 20|int] <vector-type> <model-dir> <data-dir> <output-dir>"
echo "[note] <vector-type> can be ivector or xvector only and --use-gpu just for x-vector"
exit 1
fi

vector_type=$1
model_dir=$2
data=$3
output=$4

case $vector_type in
	ivector) subtools/kaldi/sid/extract_ivectors.sh --cmd "run.pl" --nj $nj $model_dir $data $output || exit 1 ;;
	xvector) subtools/kaldi/sid/nnet3/xvector/extract_xvectors.sh --model $model --split-type $split_type --cache-capacity $cache --extract-config $extract_config --use-gpu $use_gpu --nj $nj $model_dir $data $output || exit 1;;
	*) echo "Not a valid vector type,just supporting ivector and xvector" && exit 1;;
esac

