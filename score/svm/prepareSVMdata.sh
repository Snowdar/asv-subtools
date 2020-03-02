#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-8-17)


normalize=true

. subtools/parse_options.sh

. subtools/path.sh || exit 1

if [[ $# != 3 ]];then
echo "$0 $@"
echo "[exit] Num of parameters is not equal to 3"
echo "usage:$0 <data-dir> <vector-specifier ark|scp> <output-file>"
exit 1
fi

datadir=$1
specifier=$2
outfile=$3

awk 'BEGIN{i=0}NR==FNR{a[$1]=i;i++}NR>FNR{print $1,a[$2]}' $datadir/spk2utt $datadir/utt2spk > $datadir/utt2lable
ivector-normalize-length --normalize=$normalize $specifier ark,t:- | awk 'NR==FNR{a[$1]=$2}NR>FNR{$1=a[$1];print $0}' $datadir/utt2lable - | \
sed 's/[]\[]//g' > $outfile
echo "Prepare done."

