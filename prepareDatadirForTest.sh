#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-11-29)

if [ $# != 2 ];then
echo "usage: $0 <wav-dir> <out-dir>"
echo "e.g.: $0  test/wav test/outdir"
exit 1
fi

wav_path=$1
outdir=$2

echo "Create $outdir/wav.scp firstly..."
mkdir -p $outdir
> $outdir/wav.scp
> $outdir/utt2spk
for x in $(ls $wav_path | sort );do
if [[ -f $wav_path/$x && $x == *".wav" ]] ;then
utt=${x%.*}
echo "$utt $wav_path/$x" >> $outdir/wav.scp
echo "$utt $utt" >> $outdir/utt2spk
fi
done
subtools/utils/fix_data_dir.sh $outdir
echo "done."
