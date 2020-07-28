#!/bin/bash

# Copyright xmuspeech (Author: Snowdar 2019-10-17)

nj=10

. subtools/parse_options.sh
. subtools/linux/functions.sh
. subtools/path.sh

set -e

if [[ $# != 1 ]];then
echo "[exit] Num of parameters is not equal to 1"
echo "usage:$0 <data-dir>"
exit 1
fi

data=$1

[ ! -d $data ] && echo "[exit] No such dir $data." && exit 1
[ ! -f $data/feats.scp ] && echo "[exit] Expected $data/feats.scp to exist" && exit 1

# Check
num=$(wc -l $data/feats.scp | awk '{print $1}')

rm -f $data/.error

if [ -f $data/utt2num_frames ];then
    awk -v num=$num -v data=$data '{if(NF!=2){print $1 > data"/.error"}}
            END{if(NR!=num){print $1 > data"/.error"}}' $data/utt2num_frames
    [ ! -f $data/.error ] && echo "[Note] The file $data/utt2num_frames is exist, so do not generate it again." && exit 0
    [ -f $data/.error ] && echo "[Warning] There is an error in $data/utt2num_frames, so generate it again."
fi

function feat_to_len(){
    feat-to-len scp:$1 ark,t:$2
    return 0
}

echo "Generate $data/utt2num_frames according to $data/feats.scp"
do_lines_task_parallel --nj $nj feat_to_len $data/feats.scp $data/utt2num_frames
echo "Get $data/utt2num_frames done"
