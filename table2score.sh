#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-10-05)

trials= # if provided,score will be sorted as this trials' order
remove_inf=true 

. subtools/parse_options.sh

if [[ $# != 2 ]];then
echo "[exit] Num of parameters is not equal to 2"
echo "usage:$0 [--remove-inf true | false] [--trials trials-path] <in-table> <out-score>"
exit 1
fi

table=$1
score=$2

>tmp.count
awk 'NR==1{
for(i=1;i<=NF;i++){
a[i]=$i
}
}
NR>1{
for(i=2;i<=NF;i++){
if($i=="-inf"){print $i >>"tmp.count"}
print a[i-1],$1,$i
}
}' $table > $score
num=`wc -l tmp.count | awk '{print $1}'`
echo "Num of -inf in $table is :$num"


if [ -s "$trials" ];then
>tmp.count
awk 'NR==FNR{a[$1$2]=$3}NR>FNR{if(!a[$1$2]||a[$1$2]=="-inf"){a[$1$2]="-inf";print a[$1$2] >>"tmp.count"} print $1,$2,a[$1$2]}' $score $trials > $score.tmp && \
mv $score.tmp $score
num=`wc -l tmp.count | awk '{print $1}'`
echo "Num of -inf with $trials is :$num (if != 0,then the num of valid trials in $table is less than target num of $trials)"
fi

if [ "$remove_inf" == "true" ];then
echo "Remove -inf ..." && sed -i '/-inf/d' $score
fi
rm -f tmp.count
echo "Transforming table to score done."
