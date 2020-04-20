#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-7-25)

pitch=false
pitch_config=subtools/conf/pitch.conf
cmvn=false
use_gpu=false
nj=20 #num-jobs
exp=exp/features

. subtools/parse_options.sh

if [[ $# != 3 ]];then
echo "[exit] Num of parameters is not equal to 3"
echo "usage:$0 [--pitch false|true] [--pitch-config subtools/conf/pitch.conf] [--nj 20|int] <data-dir> <feature-type> <feature-config>"
echo "[note] Base <feature-type> could be fbank/mfcc/plp/spectrogram and the option --pitch defaults false"
exit 1
fi

data=$1
feat_type=$2
config=$3

suffix=
cuda=

[ "$use_gpu" == "true" ] && cuda=cuda

pitch_string=
if [ $pitch == "true" ];then
suffix=pitch
pitch_string="--pitch-config $pitch_config"
fi

case $feat_type in 
	mfcc) ;;
	fbank) ;;
	plp) ;;
	spectrogram) ;;
	*) echo "[exit] Invalid base feature type $feat_type ,just fbank/mfcc/plp" && exit 1;;
esac

name=`echo "$data" | sed 's/\// /g' | awk '{for(i=1;i<=NF-1;i++){printf $i"_";}printf $NF}'`
subtools/kaldi/steps/make_${feat_type}${suffix:+_$suffix}${cuda:+_$cuda}.sh $pitch_string --${feat_type}-config $config --nj $nj --cmd "run.pl" $data $exp/${feat_type}/$name/log $exp/${feat_type}/$name || exit 1

echo "Make features done."

if [ $cmvn == "true" ];then
subtools/kaldi/steps/compute_cmvn_stats.sh $data $exp/cmvn/log $exp/cmvn || exit 1
echo "Compute cmvn stats done."
fi
