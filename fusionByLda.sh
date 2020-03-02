#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018/09/28)

train_normalize=true
form_normalize=false # value of normalize is tuning

keep_mat=true # if false,rm *.mat in the end
computeCavg=false
printlog=true # if you don't want to print c++ program log,you can use 2>/dev/null when calling this script with this in the end of your calling. 
equal_fusion=true # If true,print fusion eer with equal weight as the same time.
eval=false # if true ,use a exist mat to do eval fusion
mat= 

. subtools/parse_options.sh
. subtools/path.sh

set -e 
if [[ $# != 3 ]];then
echo "[exit] Num of parameters is not equal to 3"
echo "usage:$0  <train/test-trials> <train/test-score-scp|train/test-score-list> <out-fusion-score>"
exit 1
fi

trials=$1
scorescp=$2
outscore=$3


for x in $trials $scorescp;do
[[ ! -f $x || ! -s $x ]] && echo "No such file $x or the file $x is empty" && exit 1
done

[[ "$eval" == "true" && "$mat" == "" ]] && echo "[exit] --test-mod is true but you have not provided a lda-mat" && exit 1

[ "$eval" == "true" ] && equal_fusion=false && echo "Now,it's test mod..."

dir=$(dirname $scorescp)
name=$(basename ${scorescp%.*})

list=$(sed '/^#/d' $scorescp | awk '{print $NF}')

duplicate=$(echo "$list" | sort | uniq -d)


[ "$duplicate" != "" ] && echo "[exit] Exist duplicated scores "$duplicate && exit 1

for x in $list;do
[[ ! -f $x || ! -s $x ]] && echo "No such file $x or the file $x is empty" && exit 1
done

num=$(echo "$list" | wc -l)

[ "$printlog" == "true" ] && echo "Num of valid scores is $num"
[ "$printlog" == "true" ] && echo "Get score vector to be computed by lda matrix... "

trialsnum=$(wc -l $trials | awk '{print $1}')

awk 'NR==FNR{a[$1"-"$2]=$1"-"$2" ["}
    NR>FNR{if(FILENAME!=ARGV[ARGC-1]){
a[$1"-"$2]=a[$1"-"$2]" "$3;}else{print a[$1"-"$2]" ]"}}' \
$trials $list $trials | awk -v num=$num '{if(NF==num+3)print $0}' >$dir/$name.ark

trialsnum=$(wc -l $trials | awk '{print $1}')
arknum=$(wc -l $dir/$name.ark | awk '{print $1}')

[ $trialsnum != $arknum ] && echo "[exit] Num of trials is not equal to num of score-pairs. Maybe you miss some scores in foo score file and you should provide the intersection-trials with enough score-pairs which should appear in every score file." && exit 1

if [ "$eval" == "false" ];then
[ "$printlog" == "true" ] && echo "Get vector lable for lda matrix training..."
awk '{print $1"-"$2,$3}' $trials > $dir/$name.lable

[ "$printlog" == "true" ] && echo "Train lda matrix..."
ivector-compute-lda --binary=false --dim=1 --total-covariance-factor=0.1 \
    "ark:ivector-normalize-length --normalize=$train_normalize ark:$dir/$name.ark  ark:- |" \
    ark:$dir/$name.lable \
    $dir/$name.mat 

[ "$printlog" == "true" ] && echo "Check LDA matrix to make sure that mean of target sets is always greater than nontarget sets..."
awk '{if(NR==2){$(NF-1)=0;k=0;
for(i=1;i<NF-1;i++){
if($i<=0){
k++;
}
}
for(i=1;i<NF;i++){
if(k>=(NF-2)/2.0 && $i<0){
$i=-$i}
printf " "$i}print " ]"}else{printf "["}}' $dir/$name.mat > $dir/$name.mat.tmp

mv -f $dir/$name.mat.tmp $dir/$name.mat

mat=$dir/$name.mat
fi

[ "$printlog" == "true" ] && echo "Do fusion..."
ivector-transform $mat \
    "ark:ivector-normalize-length --normalize=$form_normalize ark:$dir/$name.ark ark:- |"  \
    ark,t:$outscore.tmp 

[ "$printlog" == "true" ] && echo "Write fusion score to $outscore..."
awk '{print $1,$2}' $trials | paste - <(awk '{print $3}' $outscore.tmp) > $outscore

if [ "$equal_fusion" == "true" ];then
[ "$printlog" == "true" ] && echo "Do extra equal fusion..."
awk '{printf "[";for(i=1;i<NF-2;i++){printf " 1"}print " 0 ]"}' $mat > $mat.equal

ivector-transform $mat.equal \
    "ark:ivector-normalize-length --normalize=$form_normalize ark:$dir/$name.ark ark:- |"  \
    ark,t:$outscore.tmp

awk '{print $1,$2}' $trials | paste - <(awk '{print $3}' $outscore.tmp) > $outscore.equal
fi

rm -f $dir/$name.ark
rm -f $dir/$name.lable
rm -f $outscore.tmp

if [ "$eval" == "false" ];then
echo "Fusion weights are as follows:"
cat $mat
[ "$keep_mat" == "false" ] && [ "$eval" == "false" ] && rm -f $mat
echo "Get fusion eer:"
subtools/computeEER.sh $outscore 3 $trials 3 2>/dev/null
[ "$equal_fusion" == "true" ] && echo "( fuision with equal weight :" `subtools/computeEER.sh $outscore.equal 3 $trials 3 2>/dev/null` ")" && rm -f $mat.equal
[ "$computeCavg" == "true" ] && echo "Get fusion Cavg:"
[ "$computeCavg" == "true" ] && subtools/computeCavg.py -pairs $trials $outscore
else
echo -e  "\n[Fusion done] Your fusion score is $outscore."
fi
[ "$printlog" == "true" ] && echo "All done."
