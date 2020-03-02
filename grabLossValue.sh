#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2019-03-16)

objects="diagnostic valid train" # default all.
outputname="output"

. subtools/parse_options.sh

if [[ $# != 2 ]];then
echo "[exit] Num of parameters is not equal to 2"
echo "usage:$0 <model-logdir> <outdir>"
exit 1
fi

logdir=$1
outdir=$2

mkdir -p $outdir

for object in $objects;do
case $object in
	diagnostic)
	log=$(ls $logdir/compute_prob_train.*.log)
	[ "$log" == "" ] && echo "[Warning] No $object logs in $logdir, so skip it." 
	> $outdir/$object.loss
	max=0
	for x in $log;do
		iter=$(echo "$x" | sed 's/\./ /g' | awk '{print $(NF-1)}')
		[ "$iter" != "final" ] && [ "$iter" -gt "$max" ] && max=$iter
		value=$(awk -v name=$outputname '{if($3=="Overall" && $4=="log-likelihood" && $6=="'\''"name"'\''") {value=$8;}}END{print value}' $x)
		[ "$value" != "" ] && echo "$iter $value" >> $outdir/$object.loss
	done
	max=`expr $max + 2`
	sed -i 's/final/'"$max"'/g' $outdir/$object.loss
	sort -n -k 1 $outdir/$object.loss -o $outdir/$object.loss
	;;
	
	valid)
	log=$(ls $logdir/compute_prob_valid.*.log)
	[ "$log" == "" ] && echo "[Warning] No $object logs in $logdir, so skip it." 
	> $outdir/$object.loss
	max=0
	for x in $log;do
		iter=$(echo "$x" | sed 's/\./ /g' | awk '{print $(NF-1)}')
		[ "$iter" != "final" ] && [ "$iter" -gt "$max" ] && max=$iter
		value=$(awk -v name=$outputname '{if($3=="Overall" && $4=="log-likelihood" && $6=="'\''"name"'\''") {value=$8;}}END{print value}' $x)
		[ "$value" != "" ] && echo "$iter $value" >> $outdir/$object.loss
	done
	max=`expr $max + 2`
	sed -i 's/final/'"$max"'/g' $outdir/$object.loss
	sort -n -k 1 $outdir/$object.loss -o $outdir/$object.loss
	;;
	
	train)
	log=$(ls $logdir/train.*.*.log)
	[ "$log" == "" ] && echo "[Warning] No $object logs in $logdir, so skip it." 
	> $outdir/$object.parallel.loss
	for x in $log;do
		iter=$(echo "$x" | sed 's/\./ /g' | awk '{print $(NF-2),$(NF-1)}')
		iter=$(echo $iter | awk '{$1=$1+1;print $0}')
		value=$(awk -v name=$outputname '{if($3=="Overall" && $5=="objective" && $8=="'\''"name"'\''") {value=$10;}}END{print value}' $x)
		[ "$value" != "" ] && echo "$iter $value" >> $outdir/$object.parallel.loss
	done
	sort -n -k 1 -n -k 2 $outdir/$object.parallel.loss -o $outdir/$object.parallel.loss
	
	awk 'BEGIN{current=-1;}{a[$1]=a[$1]+$3;b[$1]=b[$1]+1;if($1!=current){if(current!=-1){print current,a[current]/b[current];}current=$1;}}
	END{print current,a[current]/b[current];}' \
	$outdir/$object.parallel.loss > $outdir/$object.loss
	;;
	*) echo "[exit] Invalid object $object" && exit 1 ;;
esac
echo "Grabing loss of $object ..."
done

echo "All done."
