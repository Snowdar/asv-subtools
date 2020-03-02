#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2019-02-22)

share=true # if false, generate a copy of ark for out-vector-dir as a single dir but it will need some space.

. subtools/parse_options.sh
. subtools/path.sh

if [[ $# != 3 ]];then
echo "[exit] Num of parameters is not equal to 3"
echo "$0 <in-vector-scp1> <in-vector-scp2> <out-vector-dir>"
exit 1
fi

inscp1=$1
inscp2=$2
outdir=$3

[ ! -f "$inscp1" ] && echo "[exit] No such file $inscp1" && exit 1
[ ! -f "$inscp2" ] && echo "[exit] No such file $inscp2" && exit 1
[ -d "$outdir" ] && echo "[exit] $outdir is exist." && exit 1

name1=`basename ${inscp1%.*}`
name2=`basename ${inscp2%.*}`

[ "$name1" != "$name2" ] && echo "[exit] the vector type of $inscp1 is not equal to $inscp2" && exit 1

mkdir -p $outdir/log

if [ "$share" == "true" ];then
cat $inscp1 $inscp2 > $outdir/$name1.scp
else
run.pl $outdir/log/combine.log \
	cat $inscp1 $inscp2 \| copy-vector scp:- ark,scp:$outdir/$name1.ark,$outdir/$name1.scp
fi
echo "Combine done."