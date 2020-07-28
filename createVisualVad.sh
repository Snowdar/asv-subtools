#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2019-01-11)

# This script is used to generate a visual vad.scp to be compatible some script which needs vad.scp when
# we don't need vad process actually. Therefore, it could avoid to rewrite some codes always in this case.

nj=20
exp=exp/features

. subtools/parse_options.sh
. subtools/linux/functions.sh
. subtools/path.sh

if [[ $# != 1 ]];then
echo "[exit] Num of parameters is not equal to 1"
echo "usage:$0 <data-dir>"
exit 1
fi

data=$1

[ ! -s "$data/utt2num_frames" ] && subtools/get_utt2num_frames_from_feats.sh --nj $nj $data

name=`echo "$data" | sed 's/\// /g' | awk '{for(i=1;i<=NF-1;i++){printf $i"_";}printf $NF}'`
outdir=$exp/visual/vad/$name
mkdir -p $outdir

function create_vad_label(){
    awk '{value="";for(i=0;i<$2;i++){value=value" 1"} print $1,"[",value,"]"}' $1 | copy-vector \
        ark:- ark,scp:$outdir/vad.$3.ark,$2
    # $3 is the num of nj
    return 0
}

echo "Create visual vad by $data/num2num_frames..."
do_lines_task_parallel --nj $nj create_vad_label $data/utt2num_frames $data/vad.scp
echo "Create visual vad done."
