#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-12-26)

vector_type= # xvector or ivector or any other. If NULL, find Automatically.

. subtools/parse_options.sh
. subtools/path.sh

set -e

if [[ $# != 3 ]];then
echo "[exit] Num of parameters is not equal to 2"
echo "usage:$0 <origin-data-dir> <sp-data-dir> <sp-vector-dir>"
echo "e.g.: $0 data/test_1s data/test_1s_sp exp/base_xv/tdnn6/test_1s_sp"
exit 1
fi

data=$1
spdata=$2
vecdir=$3

for x in $data $spdata $vector;do
[ ! -d $x ] && echo "[exit] No such dir $x." && exit 1
done

for x in utt2spk vad.scp ;do
[ ! -f $data/$x ] && echo "[exit] expect $data/$x to exist." && exit 1
[ ! -f $spdata/$x ] && echo "[exit] expect $data/$x to exist." && exit 1
done

if [ "$vector_type" == "" ];then
[ -f "$vecdir/ivector.scp" ] && vector_type=ivector && echo "[Auto find] Your vector is i-vector"
[ -f "$vecdir/xvector.scp" ] && vector_type=xvector && echo "[Auto find] Your vector is x-vector"
fi

num1=`wc -l $data/utt2spk | awk '{print $1}'`
num2=`wc -l $spdata/utt2spk | awk '{print $1}'`

[[ $(echo "$num1 * 3" | bc) != "$num2" ]] && echo "num of utts in $data * 3 != num of utts in $spdata, which means that it's not in sp case." && exit 1

vectorfile=$vecdir/$vector_type.scp
[ ! -f "$vectorfile" ] && echo "[exit] No such file $vectorfile." && exit 1

newdatadir=`dirname ${spdata}`/`basename ${spdata}`_mean
outdir=`dirname $vecdir`/`basename $spdata`_mean

if [ ! -d "$newdatadir" ];then
mkdir -p $newdatadir
cp -f $data/{utt2spk,wav.scp,spk2utt} $newdatadir
fi

[ ! -f "$spdata/utt2num_frames.nosil" ] && copy-vector scp:$spdata/vad.scp ark,t:- | awk '{m=0;for(i=2;i<=NF;i++){m=m+$i}print $1,m}' \
> $spdata/utt2num_frames.nosil

mkdir -p $outdir

copy-vector scp:$vectorfile ark,t:- | 
awk 'NR==FNR{a[$1]=$2}NR>FNR{
    if(FILENAME!=ARGV[ARGC-1]){
	   dim=NF-3
       for(i=0;i<dim;i++){
           b[$1i]=$(i+3);
       }
    }else if(a[$1]){
            printf $1" [ ";
            for(i=0;i<dim;i++){
                printf (a[$1]*b[$1i]+a[$1"-sp0.9"]*b[$1"-sp0.9"i]+a[$1"-sp1.1"]*b[$1"-sp1.1"i])/(a[$1]+a[$1"-sp0.9"]+a[$1"-sp1.1"])" ";
            }
            print "]";
       }
}' $spdata/utt2num_frames.nosil - $newdatadir/utt2spk | \
copy-vector ark:- ark,scp:$outdir/$vector_type.ark,$outdir/$vector_type.scp
