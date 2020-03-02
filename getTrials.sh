#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-06-29)

only_pair=false # if true,generate a unknow(don't know target/nontarget) trials for some testset and the trials only contains pairs.

. subtools/parse_options.sh

if [ $# != 4 ];then
echo "usage: $0 <mod 1(get raw trl)> <registerset-spk2utt/spk.list-path> <testset-utt2spk-path> <output-trials>"
echo "usage: $0 <mod 2(remove invalid utt)> <in-trials-path> <etracted-vector-dir|invalid-utt-list-file> <output-trials>"
echo "usage: $0 <mod 3(get full trials)> <list-1> <list-2> <output-trials>"
echo "note:<extracted-vector-dir> needs contain log-dir which includes speaker_mean.log"
echo "note:if you want to execute both mod-1 and mod-2,you should execute this script twice by different parametes."
exit 1
fi

mod=$1
path1=$2 
path2=$3 
output=$4

if [ "$mod" == "1" ];then
if [ "$only_pair" == "false" ];then
echo "Get raw trials..."
registerspk=`awk '{print $1}' $path1 | sort -u`
testspk=`awk '{print $2}' $path2 | sort -u `
echo "[Note] You should check it by yourself to avoid spks-lable-discrepancy in two set."
echo "test spks are [" $testspk "]"
echo "register spks are [" $registerspk "]" # e.g.,spks in olr2017 train and dev are not consistent.
else
echo "Get trials with only pairs for testset..."
fi
awk '{print $1}' $path1 | sort -u | \
awk -v pair=$only_pair 'NR==FNR{a[NR]=$1}NR>FNR{for(i=1;i<=NR-FNR;i++){if(pair=="false"){if($2==a[i]){print a[i]" "$1" target"}else{print a[i]" "$1" nontarget"}}else{print a[i]" "$1}}}' - $path2 > $output
echo "Write to $output done."
elif [ "$mod" == "2" ];then
echo "Remove invalid trials..."
[ -d $path2 ] && utt=$(awk '{if($3=="No" && $4=="iVector") print $NF}' $path2/log/speaker_mean.log)
[ -f $path2 ] && utt=`cat $path2`
echo "$utt" | sed 's/ \t/\n/g' | sed '/^$/d' > $output.invalid.utt.list
[ -s $output.invalid.utt.list ] && awk 'NR==FNR{a[$1]=1}NR>FNR{if(!a[$2]){print $0}}' \
$output.invalid.utt.list $path1 > $output.tmp && mv -f $output.tmp $output # replace "for and sed" to speed up the process
[ ! -s $output.invalid.utt.list ] && [ $path1 != $output ] && cp -f $path1 $output
num=`wc -l $output.invalid.utt.list | awk '{print $1}'`
echo "$num utt's trials have been removed and the invalid utts is in $output.invalid.utt.list"
echo "Write to $output done."
elif [ "$mod" == "3" ];then
awk '{print $1}' $path1 | sort -u | \
awk -v pair=$only_pair 'NR==FNR{a[NR]=$1}NR>FNR{for(i=1;i<=NR-FNR;i++){if(pair=="false"){if($1==a[i]){print a[i]" "$1" target"}else{print a[i]" "$1" nontarget"}}else{print a[i]" "$1}}}' - $path2 > $output
echo "Write to $output done."
else
echo "[exit] Your mod is not 1, 2 or 3, exit."
exit 1
fi
