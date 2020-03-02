#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018/02/20)

mod=lda # lda | svm | equal
metric_function="subtools/score/metric/getMintDCF.sh" # this function should return a float or a string-float metric only
# There are some functions in subtools/score/metric/*.sh

stop_early=false

. subtools/parse_options.sh

set -e 

if [[ $# != 3 ]];then
echo "[exit] Num of parameters is not equal to 3"
echo "usage:$0 <trials> <score-list> <out-fusion-dir>"
exit 1
fi

trials=$1
scp=$2
outdir=$3

mkdir -p $outdir/tmp

scorelist=$(sed '/^#/d' $scp |awk '{print $NF}')

echo "Compute metric for initialization..."
> $outdir/init.sort.scp
for x in $scorelist;do
metric=$($metric_function $trials $x)
echo "$metric $x" >> $outdir/init.sort.scp
done

sort -n -k 1 $outdir/init.sort.scp -o $outdir/init.sort.scp

primary=$(head -n 1 $outdir/init.sort.scp)
remaining=$(tail -n +2 $outdir/init.sort.scp)
echo $primary > $outdir/final.scp
> $outdir/record.mat
echo "iter 0"
echo "$primary" | awk '{print "Current",$1}'

index=0
for iter in $(seq $(echo "$remaining" | wc -l ));do
echo "iter $iter"
> $outdir/tmp/$index.scp
count=1
	for x in $(echo "$remaining" | awk '{print $2}');do
		echo -e "$primary\n2 $x" > $outdir/tmp/$index.$count.scp
		case $mod in
			lda) subtools/fusionByLda.sh $trials $outdir/tmp/$index.$count.scp $outdir/tmp/$index.$count.score 2>/dev/null 1>/dev/null ;;
			svm) subtools/fusionBySvm.py --write-weight="$outdir/tmp/$index.$count.mat" $trials $outdir/tmp/$index.$count.scp $outdir/tmp/$index.$count.score 2>/dev/null 1>/dev/null;;
			equal)score1=$(echo "$primary" | awk '{print $2}')
			echo "[ 1 1 0 ]" > $outdir/tmp/$index.$count.mat
			subtools/weightScore.sh --weight1 1 --weight2 1 $score1 $x $outdir/tmp/$index.$count.score 2>/dev/null 1>/dev/null;;
			*) echo "Do not support this mod $mod" && exit 1;;
		esac
		metric=$($metric_function $trials $outdir/tmp/$index.$count.score)
		echo "$metric $outdir/tmp/$index.$count.score" >> $outdir/tmp/$index.scp
		count=$[ $count + 1 ]
	done
sort -n -k 1 $outdir/tmp/$index.scp -o $outdir/tmp/$index.scp

oldmetric=$(echo "$primary" | awk '{print $1}')
primary=$(head -n 1 $outdir/tmp/$index.scp)
newmetric=$(echo "$primary" | awk '{print $1}')
[[ $(echo "$newmetric > $oldmetric" | bc) > 0 ]] && echo "Notice..." && [ "$stop_early" == "true" ] && echo "Get the top and stop early."  && break
cp -f $(echo "$primary" | awk '{print $2}') $outdir/final.score.tmp
echo "$primary" | awk '{print "Current",$1}'
select=$(basename $(echo $primary | awk '{print $2}') | sed 's/\./ /g' | awk '{print $2}')
cat $outdir/tmp/$index.$select.mat >> $outdir/record.mat
remaining=$(echo "$remaining" | awk -v select=$select -v outdir=$outdir '{if(NR!=select){print $0}else{print $0 >> outdir"/final.scp"}}')
index=$[ $index + 1 ]
done

awk 'BEGIN{n=1;w[n]=1;b=0;}{
b=b*$2+$4;
for(i=1;i<=n;i++){
w[i]=w[i]*$2;
}
n=n+1;
w[n]=$3;
}END{
printf "[";
for(i=1;i<=n;i++){
printf " "w[i];
}
print " "b" ]\n";
}' $outdir/record.mat >$outdir/final.greedy.mat

mv -f $outdir/final.score.tmp $outdir/final.score
rm -rf $outdir/tmp 
echo "Done."







