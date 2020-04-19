#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2020-04-17)


force=false

. subtools/parse_options.sh

if [[ $# != 1 ]];then
echo "[exit] Num of parameters is not equal to 1"
echo "usage:$0 <datadir>"
exit 1
fi

datadir=$1

for x in $datadir/utt2spk $datadir/spk2gender;do
    [ ! -f "$x" ] && echo "Expected $x to exist." && exit 1
done

[ "$force" == "true" ] && rm -f $datadir/utt2gender

[ -f "$datadir/utt2gender" ] && echo "$datadir/utt2gender is exist, please delete it by yourself or use '--force true' option." && exit 1

awk 'NR==FNR{spk2gender[$1]=$2}NR>FNR{print $1,spk2gender[$2]}' $datadir/spk2gender $datadir/utt2spk > $datadir/utt2gender || exit 1

num1=$(wc -l $datadir/utt2spk | awk '{print $1}')
num2=$(wc -l $datadir/utt2gender | awk '{print $1}')
[ $num1 -ne $num2 ] && echo "[Error] The gender information of spk2gender is not complete." && exit 1

echo "Generate done."