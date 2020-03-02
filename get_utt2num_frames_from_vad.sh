#!/bin/bash  

# Copyright xmuspeech (Author: Snowdar 2019-10-17)

nj=10
nosil=true

. subtools/parse_options.sh
. subtools/linux/functions.sh
. subtools/path.sh

set -e

if [[ $# != 1 ]];then
echo "[exit] Num of parameters is not equal to 1"
echo "usage:$0 [--nosil true|false] <data-dir>"
echo "note: it will generate utt2num_frames.nosil to data-dir if --nosil is true 
      else utt2num_frames which contains silent frames."
exit 1
fi

data=$1

[ ! -d $data ] && echo "[exit] No such dir $data." && exit 1
[ ! -f $data/vad.scp ] && echo "[exit] Expected $data/vad.scp to exist" && exit 1

# Check
num=$(wc -l $data/vad.scp | awk '{print $1}')

rm -f $data/.error
if [ "$nosil" == "true" ];then
    if [ -f $data/utt2num_frames.nosil ];then
        awk -v num=$num -v data=$data '{if(NF!=2){print $1 > data"/.error"}}
                                  END{if(NR!=num){print $1 > data"/.error"}}' $data/utt2num_frames.nosil
        [ ! -f $data/.error ] && echo "[Note] The file $data/utt2num_frames.nosil is exist, so do not generate it again." && exit 0
        [ -f $data/.error ] && echo "[Warning] There is sames a error in $data/utt2num_frames.nosil, so generate it again."
    fi
else
    if [ -f $data/utt2num_frames ];then
        awk -v num=$num -v data=$data '{if(NF!=2){print $1 > data"/.error"}}
                                  END{if(NR!=num){print $1 > data"/.error"}}' $data/utt2num_frames
        [ ! -f $data/.error ] && echo "[Note] The file $data/utt2num_frames is exist, so do not generate it again." && exit 0
        [ -f $data/.error ] && echo "[Warning] There is sames a error in $data/utt2num_frames, so generate it again."
    fi
fi

function copy_vector(){
    copy-vector scp:$1 ark,t:- | awk '{print $1,NF-1}' > $2
    return 0
}

function copy_vector_nosil(){
    copy-vector scp:$1 ark,t:- | awk '{m=0;for(i=2;i<=NF;i++){m=m+$i}print $1,m}' > $2
    return 0
}

if [ "$nosil" == "true" ];then
    echo "Generate $data/utt2num_frames.nosil according to $data/vad.scp"
    do_lines_task_parallel --nj $nj copy_vector_nosil $data/vad.scp $data/utt2num_frames.nosil
    echo "Get $data/utt2num_frames.nosil done"
else
    echo "Generate $data/utt2num_frames according to $data/vad.scp"
    do_lines_task_parallel --nj $nj copy_vector $data/vad.scp $data/utt2num_frames
    echo "Get $data/utt2num_frames done"
fi

