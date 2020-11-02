#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2019-04-25)

vad=true
outputdir= # If NULL, default $datadir/split*
force=false # If true, split again whatever

. subtools/parse_options.sh
. subtools/path.sh

set -e

if [[ $# != 2 ]];then
echo "[exit] Num of parameters is not equal to 2"
echo "usage:$0 [--outputdir ""] [--force false|true] <data-dir> <num-nj>"
exit 1
fi

data=$1
nj=$2

[ ! -d $data ] && echo "[exit] No such dir $data." && exit 1

for x in feats.scp;do
    [ ! -f $data/$x ] && echo "[exit] expect $data/$x to exist." && exit 1
done

[ "$outputdir" == "" ] && outputdir=$data/split${nj}order
[ "$force" != "true" ] && [ -f "$outputdir/.done" ] && echo "[Note] Do not split $data again..." && exit 0

echo "Split $data with $nj nj according to length-order ..."

if [ "$vad" == "true" ];then
    utt2num_frames=utt2num_frames.nosil
    subtools/get_utt2num_frames_from_vad.sh --nosil true --nj $nj $data 
else
    utt2num_frames=utt2num_frames
    subtools/get_utt2num_frames_from_feats.sh $data
fi

mkdir -p $outputdir

sort -r -n -k 2 $data/$utt2num_frames > $outputdir/$utt2num_frames.order

tot_num=$(wc -l $outputdir/$utt2num_frames.order | awk '{print $1}')

[[ "$tot_num" -lt "$nj" ]] && echo "nj $nj is too large for $tot_num utterances." && exit 1

num_frames=$(awk '{a=a+$2}END{print a}' $outputdir/$utt2num_frames.order ) 

average_frames=$num_frames
[ "$nj" != 1 ] && average_frames=$[$num_frames/$nj + 1]

echo -e "num_frames:$num_frames\naverage_frames:$average_frames"

for i in $(seq $nj);do
mkdir -p $outputdir/$i
> $outputdir/$i/$utt2num_frames.order
done

# split 
awk -v nj=$nj -v mean=$average_frames -v dir=$outputdir  -v name=$utt2num_frames '{a[NR]=$1;b[NR]=$2;}END{
num=1;
max=b[num]+1;
avoid_dead_cycle=1;
while(num<=NR){
    out=0;
    for(i=1;i<=nj;i++){
        while(c[i]<max && c[i]+b[num]<mean){
        c[i]=c[i]+b[num];
        print a[num],b[num] >> dir"/"i"/"name".order"
        out=out+1;
        num=num+1;
        if(num>NR){break;}
        }
        if(num>NR){break;}
        if(c[i]>=max){max=c[i];}
    }

    for(i=nj;i>=1;i--){
        while(c[i]<=max && c[i]<mean && NR-num+avoid_dead_cycle>=i){
        c[i]=c[i]+b[num];
        print a[num],b[num] >> dir"/"i"/"name".order"
        out=out+1;
        num=num+1;
        if(num>NR){break;}
        }
        if(num>NR){break;}
        if(c[i]>=max){max=c[i];}    
    }
    if(out==0){
    avoid_dead_cycle=avoid_dead_cycle+1;
    }else if(avoid_dead_cycle>1){
    avoid_dead_cycle=avoid_dead_cycle-1;
    }
}
}' $outputdir/$utt2num_frames.order

# filter
for i in $(seq $nj);do
subtools/filterDataDir.sh --check false $data $outputdir/$i/$utt2num_frames.order $outputdir/$i/ >/dev/null
done

> $outputdir/.done # a mark file

rm -rf $outputdir/*/.backup

echo "Split done."
