#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-12-16)

prefix="" # If NULL, add spk-id to correspondent utt-id (concatenated by -).
          # If -, just do backup or recover data dir from recovering.
as_suffix=false # if true, add a suffix rather than prefix

extra_files= # could be exp/model/tdnn6/train/xvector.scp etc.

. subtools/parse_options.sh

if [[ $# != 1 ]];then
echo "[exit] Num of parameters is not equal to 1"
echo "usage:$0 <data-dir>"
exit 1
fi

datas=$1

for data in $datas;do

[ ! -d $data ] && echo "[exit] No such dir $data" && exit 1


files=""
for x in wav.scp utt2spk text utt2dur utt2num_frames utt2len feats.scp vad.scp cmvn.scp;do
[ -f $data/$x.bk ] && cp -f $data/$x.bk $data/$x
[ ! -f $data/$x.bk ] && [ -f $data/$x ] && cp -f $data/$x $data/$x.bk
[ -f $data/$x ] && files="$files $data/$x"
done

for x in $extra_files;do
[ -f $x.bk ] && cp -f $x.bk $x
[ ! -f $x.bk ] && [ -f $x ] && cp -f $x $x.bk
[ -f $x ] && files="$files $x"
done

[ "$prefix" == "-" ] && echo "Prefix is - , then just do backup or recovering and exit now." && exit 1

if [ "$prefix" == "" ];then
echo "Prefix is NULL, so add spk-id to utt-id..."
[ ! -f $data/utt2spk ] && echo "[exit] $data/utt2spk is expected to exist."
for x in $files;do
awk -v suffix=$as_suffix 'NR==FNR{a[$1]=$2}NR>FNR{if(suffix=="false"){$1=a[$1]"-"$1;}else{$1=$1"-"a[$1];}print $0}' $data/utt2spk.bk $x > $x.tmp && mv -f $x.tmp $x
echo "$x done."
done
else
for x in $files;do
awk -v suffix=$as_suffix -v prefix=$prefix '{if(suffix=="false"){$1=prefix"-"$1;}else{$1=$1"-"prefix;}print $0}' $x > $x.tmp && mv -f $x.tmp $x
echo "$x done."
done
fi

subtools/kaldi/utils/fix_data_dir.sh $data
rm -rf $data/.backup
echo "$data done."
done
echo "All done."




