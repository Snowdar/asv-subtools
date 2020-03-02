#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-7-25)

nj=20 #num-jobs
exp=exp/features/select
force=false
compress=true

. subtools/path.sh
. subtools/parse_options.sh

if [[ $# != 3 ]];then
echo "[exit] Num of parameters is not equal to 3"
echo "usage:$0 <data-dir> <selection> <out-data-dir>"
echo "<selection> is such as using 0-22 to get mfcc features from 23mfcc+3pitch raw features. It is 0-based."
exit 1
fi

data=$1
selection=$2
out_data=$3

for x in feats.scp;do
	[ ! -s $data/$x ] && echo "[exit] expect $data/$x to exist." && exit 1
done

[ "$data" == "$out_data" ] && echo "data-dir $data is same to out-data-dir" && exit 1
[ -d $out_data ] && [ "$force" == "false" ] && echo "[exit] out-data-dir $out_data is exist, please delete it carefully by yourself" && exit 1

subtools/kaldi/utils/copy_data_dir.sh $data $out_data

raw_name=`echo "$data" | sed 's/\// /g' | awk '{for(i=1;i<=NF-1;i++){printf $i"_";}printf $NF}'`
name=`echo "$out_data" | sed 's/\// /g' | awk '{for(i=1;i<=NF-1;i++){printf $i"_";}printf $NF}'`

outdir=$exp/$name
mkdir -p $outdir/log

split_scps=""
for n in $(seq $nj); do
    split_scps="$split_scps $outdir/log/raw_feats.$n.scp"
done

subtools/kaldi/utils/split_scp.pl $data/feats.scp $split_scps || exit 1;

run.pl JOB=1:$nj $outdir/log/select_feats.JOB.log \
	select-feats $selection scp:$outdir/log/raw_feats.JOB.scp ark:- \| \
	copy-feats --compress=$compress ark:- ark,scp:$outdir/from_$raw_name.JOB.ark,$outdir/from_$raw_name.JOB.scp

for n in $(seq $nj); do
  cat $outdir/from_$raw_name.$n.scp || exit 1;
done > $out_data/feats.scp || exit 1

rm -f $outdir/log/raw_feats.*.scp

echo "Succeeded selecting features from ${data} to ${out_data} with selection ${selection}"

