#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-10-08)
# paste feats for iv-system

set -e

prefixes=plp_20_5.0
datasets="task_1_concat_sp task_2_concat_sp task_3_concat_sp"
phonetic_model=exp/concat_volume_sp_phonetic_xv_${prefixes}_cmn/xvector/final.raw
out_layer="phonetic_tdnn5.affine"
left_context=13
right_context=7
gpu=yes
nj=20

. subtools/parse_options.sh
. subtools/path.sh

srcdirs=""
for x in $datasets;do
srcdirs="$srcdirs data/$prefixes/$x"
done

for srcdir in $srcdirs;do
echo "paste phonetic embbedings from $srcdir"

datadir=${srcdir}_paste_phonetic
[ ! -d $datadir ] && cp -rf $srcdir $datadir

[ ! -f $phonetic_model.$out_layer ] && \
nnet3-copy --nnet-config="echo output-node name=output input=$out_layer |" --edits="remove-orphans" \
$phonetic_model $phonetic_model.$out_layer

prefixes=delta_feats_paste_phonetic
name=`basename $datadir`
featsname=`echo "${datadir}" | sed 's/\// /g' | awk '{for(i=1;i<=NF-1;i++){printf $i"_";}printf $NF}'`
featsdir=exp/features/paste/${featsname}
rm -rf $featsdir
mkdir -p $featsdir/log

scp=$srcdir/feats.scp
split_scps=""
for n in $(seq $nj); do
    split_scps="$split_scps $featsdir/log/feats_${name}.$n.scp"
done

subtools/kaldi/utils/split_scp.pl $scp $split_scps || exit 1;

run.pl JOB=1:$nj $featsdir/log/${prefixes}_$name.JOB.log \
	paste-feats "ark:add-deltas --delta-window=3 --delta-order=2 scp:$featsdir/log/feats_${name}.JOB.scp ark:- |" \
	"ark:nnet3-compute --use-gpu=$gpu --extra-left-context=$left_context --extra-right-context=$right_context \
	--frames-per-chunk=50 $phonetic_model.$out_layer scp:$featsdir/log/feats_${name}.JOB.scp ark:- |" \
	ark,scp:$featsdir/${prefixes}_$name.JOB.ark,$featsdir/${prefixes}_$name.JOB.scp

for n in $(seq $nj); do
  cat $featsdir/${prefixes}_$name.$n.scp || exit 1;
done > ${datadir}/feats.scp || exit 1
echo "Succeeded pasting phoneticfeatures for ${name}"
done
