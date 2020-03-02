#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2019-03-08)

set -e

min_len=100 # 10ms * 100 = 1s
max_len=1000
position=random # could be 'start' 'end' and 'random'
fixed=false #if true, cut utts with a fixed duration, max_len
remove_sil=false # if true, accumulate the length without silent frames
exp=exp/features

. subtools/parse_options.sh
. subtools/path.sh

if [[ $# != 2 ]];then
echo "[exit] Num of parameters is not equal to 2"
echo "$0 <src-data-dir> <out-data-dir>"
exit 1
fi

srcdata=$1
outdata=$2

[[ "$srcdata" == "$outdata" ]] && echo "[exit] the srcdata is euqal to outdata" && exit 1

for x in feats.scp vad.scp ;do
[ ! -f $srcdata/$x ] && echo "[exit] expect $data/$x to exist." && exit 1
done

if [ ! -d "$outdata" ];then
mkdir -p $outdata
cp $srcdata/{utt2spk,wav.scp,spk2utt} $outdata
fi

feats=$srcdata/feats.scp
vad=$srcdata/vad.scp
if [ "$remove_sil" == "true" ];then
	select-voiced-frames scp:$srcdata/feats.scp scp:$srcdata/vad.scp ark,scp:$srcdata/feats.nosil.ark,$srcdata/feats.nosil.scp
	copy-vector scp:$srcdata/vad.scp ark,t:- | \
	awk '{
	for(i=3;i<NF;i++){
	if($i==0){$i="";}
	}
	print $0;
	}' | copy-vector ark:- ark,scp:$srcdata/vad.nosil.ark,$srcdata/vad.nosil.scp
	feats=$srcdata/feats.nosil.scp
	vad=$srcdata/vad.nosil.scp
fi

vadpath=$outdata/vad.ark
> $vadpath

copy-vector scp:$vad ark,t:- | \
awk -v fixed=$fixed -v min_len=$min_len -v max_len=$max_len -v position=$position -v vadpath=$vadpath 'BEGIN{srand();}{
raw_len=NF-3;

if(fixed=="false"){
offset=sprintf("%d",rand()*1000000/(max_len-min_len+1));
len=min_len+offset;
}else{
len=max_len;
}

if(raw_len-len<=0){
len=raw_len;
}

if(position=="start"){
start=0;
}else if(position="end"){
start=raw_len-len;
}else{
start=sprintf("%d",rand()*1000000/(raw_len-len+1));
}

vad=$1" "$2
for(i=3;i<NF;i++){
tmp=i-3;
if(tmp>=start&&tmp<start+len){
vad=vad" "$i;
$i=1;
}else{
$i=0;
}
}
vad=vad" ] ";
print vad >> vadpath;
print $0;
}' > $outdata/vad.tmp.ark

name=`echo "$outdata" | sed 's/\// /g' | awk '{for(i=1;i<=NF-1;i++){printf $i"_";}printf $NF}'`
outdir=$exp/cut/$name
mkdir -p $outdir

select-voiced-frames scp:$feats ark:$outdata/vad.tmp.ark ark,scp:$outdir/$name.ark,$outdata/feats.scp
copy-vector ark:$outdata/vad.ark ark,scp:$outdir/vad.ark,$outdata/vad.scp

subtools/kaldi/utils/fix_data_dir.sh $outdata

rm -f $outdata/vad.tmp.ark $outdata/vad.ark $srcdata/vad.nosil.ark $srcdata/vad.nosil.scp $srcdata/feats.nosil.ark $srcdata/feats.nosil.scp
echo "Cut done."