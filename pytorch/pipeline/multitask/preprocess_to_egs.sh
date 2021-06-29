#!/bin/bash

# Copyright xmuspeech

# Author: Zheng Li 2021-06-08 at xmuspeech.
# Update the multi-task learning preparation.

set -e

stage=0
endstage=3

subsampling=false
multi_task_learning=true

force_clear=true
fetures_exp=exp/features

# Do vad and traditional cmn process
nj=20
cmn=true 
compress=false # Should be false to make use of kaldi_io I/O

# Remove utts
min_chunk=200
limit_utts=8

# Get chunk egs
valid_sample=true
valid_num_utts=1024
valid_split_type="--total-spk"
sample_type="speaker_balance" # sequential | speaker_balance
chunk_num=-1
scale=1.5
overlap=0.1
valid_sample_type="every_utt" # With split type [--total-spk] and sample type [every_utt], we will get enough spkers as more
                              # as possible and finally we get valid_num_utts * valid_chunk_num = 1024 * 2 = 2048 valid chunks.
valid_chunk_num=2

. subtools/path.sh
. subtools/parse_options.sh

if [[ $# != 3 ]];then
echo "[exit] Num of parameters is not equal to 3"
echo "usage:$0 <data-dir> <egs-dir> <ali-dir>"
exit 1
fi

# Key params
traindata=$1
egsdir=$2
alidir=$3

[ ! -d "$traindata" ] && echo "The traindata [$traindata] is not exist." && exit 1

if [[ $stage -le 0 && 0 -le $endstage ]];then
    echo "$0: stage 0"
    if [ "$force_clear" == "true" ];then
        rm -rf ${traindata}_nosil
        rm -rf $fetures_exp/${traindata}_nosil

        [ ! -d "${traindata}_nosil" ] && \
        subtools/kaldi/sid/nnet3/xvector/prepare_feats_for_egs.sh --nj $nj --cmd "run.pl" --compress $compress --cmn $cmn \
                                                   $traindata ${traindata}_nosil $fetures_exp/${traindata}_nosil || exit 1
        utils/data/fix_data_dir.sh ${traindata}_nosil
    else
        echo "Note, the ${traindata}_nosil is exist but force_clear is not true, so do not prepare feats again."
    fi
fi

if [[ $stage -le 1 && 1 -le $endstage ]];then
    echo "$0: stage 1"
    subtools/removeUtt.sh --limit-utts $limit_utts ${traindata}_nosil $min_chunk || exit 1
	utils/data/fix_data_dir.sh ${traindata}_nosil
	cp -rf ${traindata}/ali.scp ${traindata}_nosil/ali.scp
fi

if [[ $stage -le 2 && 2 -le $endstage ]];then
    echo "$0: stage 2"
    if [ "$subsampling" == "false" ];then
      subtools/pytorch/pipeline/multitask/get_alignments.sh --stage 4 ${traindata} $alidir ${traindata}_nosil || exit 1
    fi
    if [ "$subsampling" == "true" ];then
      subtools/pytorch/pipeline/multitask/get_alignments_subsampling.sh ${traindata} $alidir ${traindata} || exit 1
      mv ${traindata}/ali.scp ${traindata}_nosil/
      mv ${traindata}/ali.ark ${traindata}_nosil/
      mv ${traindata}/tree ${traindata}_nosil/
      mv  ${traindata}/phones ${traindata}_nosil/
    fi
fi

if [[ $stage -le 3 && 3 -le $endstage ]];then
    echo "$0: stage 3"
    [ "$egsdir" == "" ] && echo "The egsdir is not specified." && exit 1
	mkdir -p $egsdir/info
	cp -rf ${traindata}/phones $egsdir/info/num_phones

    python3 subtools/pytorch/pipeline/onestep/get_chunk_egs.py \
		--multi-task-learning=$multi_task_learning \
        --chunk-size=$min_chunk \
        --valid-sample=$valid_sample \
        --valid-num-utts=$valid_num_utts \
        --valid-split-type=$valid_split_type \
        --sample-type=$sample_type \
        --chunk-num=$chunk_num \
        --scale=$scale \
        --overlap=$overlap \
        --valid-chunk-num=$valid_chunk_num \
        --valid-sample-type=$valid_sample_type \
        ${traindata}_nosil $egsdir || exit 1
fi

exit 0
