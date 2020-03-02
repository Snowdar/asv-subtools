#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-7-25)

set -e

cnum=64 # num of Gaussions
gmm_dir_name=gmm # a relative path to preserve gmm models
cleanup=false # if true , clean gmm models in the end
num_iters=4 # for every GMM
num_gselect=30
num_iters_init=20
num_frames=500000
num_frames_den=500000
min_gaussian_weight=0.0001
adapt=false
mmi=true
init_mmi=true
tau=400
weight_tau=10
smooth_tau=100
E=2
nj=20

. subtools/parse_options.sh
. subtools/path.sh

if [[ $# != 5 ]];then
echo "[exit] Num of parameters is not equal to 5"
echo "usage:$0 <train-vector-rspecifier> <train-utt2spk> <test-vector-rspecifier> <trials|pairs> <out-score>"
echo "[note] The tials or pairs should have 1-field with class-id and 2-field with utt-id."
exit 1
fi

traindata=$1
utt2spk=$2
testdata=$3
trials=$4
score=$5

path=$(echo "$traindata" | sed 's/ark\://g' | sed 's/scp\://g')
dir=$(dirname $path)/$gmm_dir_name
name=$(basename ${path%.*})

mkdir -p $dir

spks=$(awk '{print $2}' $utt2spk | sort -u )

if [ "$traindata" == "scp:"* ];then
cp -f $traindata $dir/$name.scp
else
copy-vector $traindata ark,scp:$dir/$name.ark,$dir/$name.scp
fi

for x in $spks;do
> $dir/$x.scp
done

awk -v dir=$dir 'NR==FNR{a[$1]=$2}NR>FNR{if(a[$1]){print $0 >>dir"/"a[$1]".scp"}
else{print "[warning] utt-id "$1" is not in utt2spk"}}' $utt2spk $dir/$name.scp

iters=$(echo "$num_iters" | sed 's/-/ /g')

init_string=
if [ "$adapt" == "true" ];then
iter=$(echo "$iters" | awk '{print $1}')
subtools/score/gmm/train_diag_gmm_with_vector.sh --num-iters-init $num_iters_init --num-frames $num_frames --min-gaussian-weight $min_gaussian_weight --num-gselect $num_gselect --nj $nj --num-iters $iter scp:$dir/$name.scp $cnum $dir/$name && \
mv $dir/$name/final.dubm $dir/$name.gmm
init_string="--init-model $dir/$name.gmm"
fi

i=1
for x in $spks;do
echo "->Train gmm for $x class"
den=
if [ "$mmi" == "true" ];then
> $dir/$x.den.scp
for y in $spks;do
[ "$y" != "$x" ] && cat $dir/$y.scp >> $dir/$x.den.scp
done
den="--den-rspecifier scp:$dir/$x.den.scp"
init_string=
fi
iter=$(echo "$iters" | awk -v i=$i '{if(i<=NF){print $(i)}else{print $NF}}')
i=`expr $i + 1`
subtools/score/gmm/train_diag_gmm_with_vector.sh $den --E $E --num-iters-init $num_iters_init \
  --num-frames $num_frames --num-frames-den $num_frames_den --min-gaussian-weight $min_gaussian_weight \
  --num-gselect $num_gselect --nj $nj --smooth-tau $smooth_tau --init-mmi $init_mmi \
  --num-iters $iter --tau $tau --weight-tau $weight_tau $init_string scp:$dir/$x.scp $cnum $dir/$x && \
mv $dir/$x/final.dubm $dir/$x.gmm
done

##############
testpath=$(echo "$testdata" | sed 's/ark\://g' | sed 's/scp\://g')
testdir=$(dirname $testpath)/tmp
testname=$(basename ${testpath%.*})
mkdir -p $testdir
copy-vector $testdata ark,t:$testdir/$testname.ark.txt

if [ "$adapt" == "true" ];then
gmm-global-get-frame-likes $dir/$name.gmm ark:$testdir/$testname.ark.txt "ark,t:| awk '{print \$1,\$3}' > $testdir/$name.score"
fi

> $testdir/$testname.score
for x in $spks;do
gmm-global-get-frame-likes $dir/$x.gmm ark:$testdir/$testname.ark.txt "ark,t:| awk '{print \"$x\",\$1,\$3}' > $testdir/$x.score"
cat $testdir/$x.score >> $testdir/$testname.score
done
awk 'NR==FNR{a[$1$2]=$3}NR>FNR{print $1,$2,a[$1$2]}' $testdir/$testname.score $trials > $score

# other two generation ways of score 
#awk 'NR==FNR{a[$1]=$2}NR>FNR{print $1,$2,$3-a[$2]}' $testdir/$name.score $testdir/$testname.score > $testdir/$testname.score.tmp
#awk 'NR==FNR{a[$1$2]=$3}NR>FNR{print $1,$2,a[$1$2]}' $testdir/$testname.score.tmp $trials > $score
#
#awk 'NR==FNR{if($3<-745){$3=0}a[$1$2]=exp($3);b[$2]=b[$2]+exp($3)}NR>FNR{print $1,$2,a[$1$2]/b[$2]}' $testdir/$testname.score.tmp $trials > $score

rm -rf $testdir
