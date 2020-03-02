#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-10-05)

raw_trials= # if provided,generate a table including all pairs in this trials

. subtools/parse_options.sh

if [[ $# != 2 ]];then
echo "[exit] Num of parameters is not equal to 2"
echo "usage:$0 [--raw-trials trials-path] <in-score> <out-table>"
echo "[note] where is no score will be valued '-inf'"
exit 1
fi

score=$1
table=$2

if [ ! -s "$raw_trials" ];then
awk 'BEGIN{i=1;j=1}
{
if(!a[$1]){speak[i]=$1;a[$1]=1;i=i+1}
if(!b[$2]){utt[j]=$2;b[$2]=1;j=j+1}
if(!score[$1$2]){if(NF<3){$3="-inf";}score[$1$2]=$3}
}
END{
printf("\t")
for(k=1;k<i;k++){
printf("\t"speak[k])
}
printf("\n")
for(k=1;k<j;k++){
printf(utt[k])
for(h=1;h<i;h++){
if(!score[speak[h]utt[k]]){score[speak[h]utt[k]]="-inf"}
printf("  "score[speak[h]utt[k]])
}
printf("\n")
}
}' $score > $table
else
awk 'BEGIN{i=1;j=1}
NR==FNR{
if(!a[$1]){speak[i]=$1;a[$1]=1;i=i+1}
if(!b[$2]){utt[j]=$2;b[$2]=1;j=j+1}
}
NR>FNR{
if(!score[$1$2]){if(NF<3){$3="-inf";}score[$1$2]=$3}
}
END{
printf("\t")
for(k=1;k<i;k++){
printf("\t"speak[k])
}
printf("\n")
for(k=1;k<j;k++){
printf(utt[k])
for(h=1;h<i;h++){
if(!score[speak[h]utt[k]]){score[speak[h]utt[k]]="-inf"}
printf("  "score[speak[h]utt[k]])
}
printf("\n")
}
}' $raw_trials $score > $table
fi

echo "Tansforming score to table done."
