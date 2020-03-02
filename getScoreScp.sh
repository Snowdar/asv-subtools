#!/bin/bash

#Copyright xmuspeech (Author:Snowdar 2018-09-24)

keys=""
excepts=""

. subtools/parse_options.sh

if [[ $# != 2 ]];then
echo "[exit] Num of parameters is not equal to 2"
echo "usage:$0 [--keys ''] [--except ''] <score-dir> <out-scp-file>"
exit 1
fi

dir=$1
scp=$2

scpdir=`dirname $scp`

[ ! -d $dir ] && echo "[exit] No such dir $dir" && exit 1
[ ! -d $scpdir ] && echo "[exit] No such dir $scpdir" && exit 1

scorepath=$(find $dir -name "*.score")

for key in $keys;do
tmp=$(find $dir -maxdepth 1 -name "*${key}*.score" )
scorepath=$(echo -e "${scorepath}\n${tmp}" | sort | uniq -d )
done

for except in $excepts;do
tmp=$(find $dir ! -maxdepth 1 -name "*${except}*.score")
scorepath=$(echo -e "${scorepath}\n${tmp}" | sort | uniq -d )
done

[[ $scorepath == "" ]] && echo "[exit] Find nothing." && exit 1
scorename=$(echo "$scorepath" | sed 's/[\/.]/ /g' | awk '{print $(NF-1)}')

> $scp
i=1
for score in $scorename;do
index="#$i"
[ -f $dir/$score.eer ] && eer=$(cat $dir/$score.eer)
[ "$eer" != "" ] && index="${eer}#$i"
echo "$index" | paste - <(echo "$dir/$score.score") >> $scp
i=`expr $i + 1`
done

echo "Get $scp done."


