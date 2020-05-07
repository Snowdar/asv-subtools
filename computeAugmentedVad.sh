#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2020-01-22)

aug_suffixes="reverb noise music babble"

. subtools/parse_options.sh

if [[ $# != 2 && $# != 3 ]];then
echo "[exit] Num of parameters is not equal to 2 or 3"
echo "usage:$0 <aug-data-dir> <clean-list> <vad_conf>"
echo "usage:$0 <aug-data-dir> <clean-vad-scp>"
exit 1
fi

datadir=$1
clean_list=$2

vad_conf=""

if  [[ $# == 3 ]];then
    vad_conf=$3
fi

if [ "$vad_conf" != "" ];then
    echo "Compute vad for clean data firstly."

    [ ! -f "$vad_conf" ] && echo "Expected vad conf to exist." && exit 1
    [ ! -f "$datadir/feats.scp" ] && echo "Expected $datadir/feats.scp to exist." && exit 1

    subtools/filterDataDir.sh $datadir $clean_list $datadir/clean
    subtools/computeVad.sh $datadir/clean $vad_conf

    clean_vad=$datadir/clean/vad.scp
else
    clean_vad=$clean_list
fi

cat $clean_vad > $datadir/aug.vad
for aug_suffix in $aug_suffixes;do
    awk -v suffix=$aug_suffix '{print $1"-"suffix, $2}' $clean_vad >> $datadir/aug.vad
done

> $datadir/lost_clean.utts
awk -v data=$datadir 'NR==FNR{a[$1]=$2}NR>FNR{if(!a[$1]){print $1 >> data"/lost_clean.utts"}else{print $1,a[$1]}}' \
                     $datadir/aug.vad $datadir/utt2spk > $datadir/vad.scp

num=$(wc -l $datadir/lost_clean.utts | awk '{print $1}')

[ $num -gt 0 ] && echo "[exit] Could not find $num clean items for augmented utts which are in $datadir/lost_clean.utts." && \
                rm -rf $datadir/clean $datadir/aug.vad && exit 1

rm -rf $datadir/clean $datadir/aug.vad $datadir/lost.clean.utts

echo "Compute VAD for augmented data done."

