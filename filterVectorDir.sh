#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2019-02-17)

f=1 # field of utt-id in id-file
exclude=false
force=false
share=true # if false, generate a copy of ark for out-vector-dir as a single dir but it will need some space.
scp_type=xvector.scp

. subtools/parse_options.sh
. subtools/path.sh

if [[ $# != 3 ]];then
echo "[exit] Num of parameters is not equal to 3"
echo "$0 [--exclude false|true] [--f 1] <in-vector-dir> <id-list> <out-vector-dir>"
exit 1
fi

indir=$1
idlist=$2
outdir=$3

[ ! -f "$indir/$scp_type" ] && echo "[exit] No such file $indir/$scp_type" && exit 1
[ ! -f "$idlist" ] && echo "[exit] No such file $idlist" && exit 1
[ "$force" == "true" ] && rm -rf $outdir && exit 1
[ -d "$outdir" ] && echo "[exit] $outdir is exist." && exit 1

mkdir -p $outdir/log

exclude_string=""
[[ "$exclude" == "true" ]] && exclude_string="--exclude"

if [ "$share" == "true" ];then
run.pl $outdir/log/filter.log \
  awk -v f=$f '{print $f}' $idlist \| subtools/kaldi/utils/filter_scp.pl $exclude_string - $indir/$scp_type \> $outdir/$scp_type
else
run.pl $outdir/log/filter.log \
  awk -v f=$f '{print $f}' $idlist \| subtools/kaldi/utils/filter_scp.pl $exclude_string - $indir/$scp_type \| copy-vector scp:- ark,scp:$outdir/$name.ark,$outdir/$scp_type
fi
echo "Filter $outdir done."
