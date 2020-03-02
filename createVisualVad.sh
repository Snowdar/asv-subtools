#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2019-01-11)

# This script is used to generate a visual vad.scp to be compatible some script which needs vad.scp when
# we don't need vad process actually. Therefore, it could avoid to rewrite some codes always in this case.

nj=20
exp=exp/features

. subtools/parse_options.sh
. subtools/path.sh

if [[ $# != 1 ]];then
echo "[exit] Num of parameters is not equal to 1"
echo "usage:$0 <data-dir>"
exit 1
fi

data=$1

[ ! -f "$data/utt2num_frames" ] && feat-to-len scp:$data/feats.scp ark,t:$data/utt2num_frames


name=`echo "$data" | sed 's/\// /g' | awk '{for(i=1;i<=NF-1;i++){printf $i"_";}printf $NF}'`
outdir=$exp/visual/vad/$name
mkdir -p $outdir

awk '{value="";for(i=0;i<$2;i++){value=value" 1"} print $1,"[",value,"]"}' $data/utt2num_frames | copy-vector \
ark:- ark,scp:$outdir/vad.ark,$data/vad.scp

echo "Create visual vad done."