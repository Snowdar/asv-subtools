#!/bin/bash

# Copyright xmuspeech (Author: Snowdar 2019-06-03)
#                     (Author: Leo  2021-10-13 "Online extractor")

nj=10
cmd="run.pl"
stage=1

model=final.params

data_type="raw"
de_silence=false
amp_th=100
use_gpu=false
gpu_id=""
force=false
sleep_time=3
feat_config=config/feat_conf.yaml
nnet_config=config/nnet.config


echo "$0 $@"

set -e 

if [ -f subtools/path.sh ]; then . subtools/path.sh; fi
. ./subtools/parse_options.sh || exit 1;

if [[ $# != 3 ]];then
echo "[exit] Num of parameters is not equal to 3"
echo "usage:$0 <model-dir> <data-dir> <output-dir>"
exit 1
fi

srcdir=$1
data=$2
dir=$3

# Check
mkdir -p $dir/log

num=0

[ -s $dir/xvector.scp ] && num=$(grep -E "ERROR|Error" $dir/log/extract.*.log | wc -l)

[[ "$force" != "true" && -s $dir/xvector.scp && $num == 0 ]] && echo "Do not extract xvectors of [ $data ] to [ $dir ] again with force=$force." && exit 0

rm -rf $dir/log/* # It is important for the checking.

# Start



for f in $srcdir/$model $srcdir/$feat_config $srcdir/$nnet_config; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

[ ! -f $data/wav.csv ] && [ ! -f $data/wav.scp ] && echo "no wav list in $data" && exit 1;

sdata=$data/split${nj}utt/JOB
if [ "$data_type" = "raw" ];then
  subtools/kaldi/utils/split_data.sh --per-utt $data $nj || exit 1
  wavs=${sdata}/wav.scp
else
    if [ -f $data/wav.scp ];then
      lines1=$(cat $data/wav.scp | wc -l)
      lines2=$(awk '{if(NR>1) sum+= $3};END{print sum}' $data/wav.csv)
      [[ $lines1 != $lines2 ]] && echo "num_utt not equal between wav.csv and wav.scp in $data; check it." && exit 1;

    fi

  python3 subtools/pytorch/pipeline/onestep/split_csv.py --nj=$nj $data || exit 1
  wavs=${sdata}/wav.csv

fi

echo "$0: extracting xvectors for $data"


output="ark:| copy-vector ark:- ark,scp:$dir/xvector.JOB.ark,$dir/xvector.JOB.scp"

if [ $stage -le 1 ]; then
      echo "$0: extracting xvectors from pytorch nnet"
      trap "subtools/linux/kill_pid_tree.sh --show true $$ && echo -e '\nAll killed\n' && exit 1" INT
      if $use_gpu; then
        pids=""
        for g in $(seq $nj); do
          $cmd --gpu 1 ${dir}/log/extract.$g.log \
            python3 subtools/pytorch/pipeline/onestep/extract_embeddings_new.py --use-gpu=$use_gpu --gpu-id="$gpu_id" \
                    --data-type=$data_type --de-silence=$de_silence --amp-th=$amp_th \
                    --feat-config=$srcdir/$feat_config --nnet-config=$srcdir/$nnet_config \
                    "$srcdir/$model" "`echo $wavs | sed s/JOB/$g/g`" "`echo $output | sed s/JOB/$g/g`" || exit 1 &
          sleep $sleep_time
        pids="$pids $!"
        done
      trap "subtools/linux/kill_pid_tree.sh --show true $pids && echo -e '\nAll killed' && exit 1" INT
      wait
      else
      $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
          python3 subtools/pytorch/pipeline/onestep/extract_embeddings_new.py --use-gpu="false" \
                  --data-type=$data_type --de-silence=$de_silence --amp-th=$amp_th \
                  --feat-config=$srcdir/$feat_config --nnet-config=$srcdir/$nnet_config \
                  "$srcdir/$model" "$wavs" "$output" || exit 1;
      fi

      num=$(grep -E "ERROR|Error" $dir/log/extract.*.log | wc -l)
      [ $num -gt 0 ] && echo "There are some ERRORS in $dir/log/extract.*.log." && exit 1
fi

if [ $stage -le 2 ]; then
      echo "$0: combining xvectors across jobs"
      for j in $(seq $nj); do cat $dir/xvector.$j.scp; done >$dir/xvector.scp || exit 1;
fi

echo "Embeddings of [ $data ] has been extracted to [ $dir ] done."

exit 0
