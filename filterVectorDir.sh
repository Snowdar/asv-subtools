#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2019-02-17)

f=1 # field of utt-id in id-file
exclude=false
share=true # if false, generate a copy of ark for out-vector-dir as a single dir but it will need some space.

. subtools/parse_options.sh
. subtools/path.sh

if [[ $# != 3 ]];then
echo "[exit] Num of parameters is not equal to 3"
echo "$0 [--exclude false|true] [--f 1] <in-vector-scp> <id-list> <out-vector-dir>"
exit 1
fi

inscp=$1
idlist=$2
outdir=$3

[ ! -f "$inscp" ] && echo "[exit] No such file $inscp" && exit 1
[ ! -f "$idlist" ] && echo "[exit] No such file $idlist" && exit 1
[ -d "$outdir" ] && echo "[exit] $outdir is exist." && exit 1

mkdir -p $outdir/log

exclude_string=""
[[ "$exclude" == "true" ]] && exclude_string="--exclude"

name=`basename ${inscp%.*}`
if [ "$share" == "true" ];then
run.pl $outdir/log/filter.log \
  awk -v f=$f '{print $f}' $idlist \| subtools/kaldi/utils/filter_scp.pl $exclude_string - $inscp \> $outdir/$name.scp
else
run.pl $outdir/log/filter.log \
  awk -v f=$f '{print $f}' $idlist \| subtools/kaldi/utils/filter_scp.pl $exclude_string - $inscp \| copy-vector scp:- ark,scp:$outdir/$name.ark,$outdir/$name.scp
fi
echo "Filter $outdir done."
