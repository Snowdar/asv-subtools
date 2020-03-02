#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-7-25)

nj=20
exp=exp/features

. subtools/parse_options.sh

if [[ $# != 2 ]];then
echo "[exit] Num of parameters is not equal to 2"
echo "usage:$0 <data-dir> <vad-config>"
exit 1
fi

data=$1
config=$2



name=`echo "$data" | sed 's/\// /g' | awk '{for(i=1;i<=NF-1;i++){printf $i"_";}printf $NF}'`
subtools/kaldi/sid/compute_vad_decision.sh --nj $nj --cmd "run.pl" --vad-config $config $data $exp/vad/$name/log $exp/vad/$name || exit 1

echo "Compute VAD done."
