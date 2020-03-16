#!/bin/bash

# Copyright     2017  David Snyder
#               2017  Johns Hopkins University (Author: Daniel Povey)
#               2017  Johns Hopkins University (Author: Daniel Garcia Romero)
#               2019  xmuspeech (Author: Snowdar)
#               2020  xmuspeech (Author: Hao Lu "Add diarisation")
# Apache 2.0.

# This script extracts embeddings (called "xvectors" here) from a set of
# utterances, given features and a trained DNN.  The purpose of this script
# is analogous to subtools/sid/extract_ivectors.sh: it creates archives of
# vectors that are used in speaker recognition.  Like ivectors, xvectors can
# be used in PLDA or a similar backend for scoring.

# Begin configuration section.
nj=30
cmd="run.pl"

cache_capacity=64 # Cache capacity for x-vector extractor
chunk_size=-1     # The chunk size over which the embedding is extracted.
                  # If left unspecified, it uses the max_chunk_size in the nnet
                  # directory.
extract_config=extract.config
use_gpu=false
stage=0
cmn=true
cmn_window=300

clean=false
model=final.raw # or any other model, such 180.raw 
offline=true # If true, use offline mod to speed up
split_type=order # default | order . The order way is used to speed up with offline mod.

# Diarisation
sliding=false
window=1.5
period=0.75
min_segment=0.5
hard_min=false

echo "$0 $@"  # Print the command line for logging

if [ -f subtools/path.sh ]; then . ./subtools/path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <nnet-dir> <data> <xvector-dir>"
  echo " e.g.: $0 exp/xvector_nnet data/train exp/xvectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (subtools/kaldi/utils/run.pl|subtools/kaldi/utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --cache-capacity <n|64>                          # To speed-up xvector extraction"
  echo "  --chunk-size <n|-1>                              # If provided, extracts embeddings with specified"
  echo "                                                   # chunk size, and averages to produce final embedding"
fi

srcdir=$1
data=$2
dir=$3

for f in $srcdir/$model $srcdir/min_chunk_size $srcdir/max_chunk_size $data/feats.scp $data/vad.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

min_chunk_size=`cat $srcdir/min_chunk_size 2>/dev/null`
max_chunk_size=`cat $srcdir/max_chunk_size 2>/dev/null`

nnet=$srcdir/$model
if [ -f $srcdir/$extract_config ] ; then
  echo "$0: using $srcdir/$extract_config to extract xvectors"
  nnet="nnet3-copy --nnet-config=$srcdir/$extract_config $srcdir/$model - |"
else
echo "[exit] No such config $srcdir/$extract_config"
exit 1
fi

if [ $chunk_size -le 0 ]; then
  chunk_size=$max_chunk_size
fi

if [ $max_chunk_size -lt $chunk_size ]; then
  echo "$0: specified chunk size of $chunk_size is larger than the maximum chunk size, $max_chunk_size" && exit 1;
fi


mkdir -p $dir/log

# Diarisation
if [ "$sliding" == "true" ]; then
sub_data=$dir/subsegments_data
mkdir -p $sub_data
# Set up sliding-window subsegments
  if $hard_min; then
    awk -v min=$min_segment '{if($4-$3 >= min){print $0}}' $data/segments \
        > $dir/pruned_segments
    segments=$dir/pruned_segments
  else
    segments=$data/segments
  fi

  [ ! -f $segments ] && echo "Expected $segments to exist." && exit 1

  subtools/kaldi/utils/data/get_uniform_subsegments.py \
      --max-segment-duration=$window \
      --overlap-duration=$(perl -e "print ($window-$period);") \
      --max-remaining-duration=$min_segment \
      --constant-duration=True \
      $segments > $dir/subsegments
  subtools/kaldi/utils/data/subsegment_data_dir.sh $data \
      $dir/subsegments $sub_data

# Creat visual vad
subtools/createVisualVad.sh $sub_data
data=$sub_data
fi

case $split_type in
    default)
    subtools/kaldi/utils/split_data.sh --per-utt $data $nj
    sdata=$data/split${nj}utt/JOB
    ;;
    order)
    subtools/splitDataByLength.sh $data $nj
    sdata=$data/split${nj}order/JOB
    ;;
    *) echo "[exit] Do not support $split_type split-type" && exit 1;;
esac

echo "$0: extracting xvectors for $data"


# Set up the featuresa
if [ "$cmn" == "true" ];then
feat="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=$cmn_window scp:${sdata}/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:${sdata}/vad.scp ark:- |"
else
feat="ark:select-voiced-frames scp:${sdata}/feats.scp scp,s,cs:${sdata}/vad.scp ark:- |"
fi


if [ "$offline" == "false" ];then
    if [ $stage -le 1 ]; then
      echo "$0: extracting xvectors from nnet"

      trap "subtools/linux/kill_pid_tree.sh --show true $$ && echo -e '\nAll killed'" INT
      if $use_gpu; then
        pids=""
        for g in $(seq $nj); do
          $cmd --gpu 1 ${dir}/log/extract.$g.log \
            nnet3-xvector-compute --use-gpu=yes --min-chunk-size=$min_chunk_size --chunk-size=$chunk_size --cache-capacity=${cache_capacity} \
            "$nnet" "`echo $feat | sed s/JOB/$g/g`" ark,scp:${dir}/xvector.$g.ark,${dir}/xvector.$g.scp || exit 1 &
        pids="$pids $!"
        done
        trap "subtools/linux/kill_pid_tree.sh --show true $pids && echo -e '\nAll killed'" INT
        wait
      else
        $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
          nnet3-xvector-compute --use-gpu=no --min-chunk-size=$min_chunk_size --chunk-size=$chunk_size --cache-capacity=${cache_capacity} \
          "$nnet" "$feat" ark,scp:${dir}/xvector.JOB.ark,${dir}/xvector.JOB.scp || exit 1;
      fi
    fi
else
    echo "$0: Extract x-vectors offline..."

    if [ $stage -le 0 ]; then
        echo "$0: Compile xvector nnet"
        [ ! -f "$data/utt2num_frames.nosil" ] && copy-vector scp:$data/vad.scp ark,t:- | awk '{m=0;for(i=2;i<=NF;i++){m=m+$i}print $1,m}' \
        > $data/utt2num_frames.nosil

        order_len=$(awk '{print $2}' $data/utt2num_frames.nosil | sort -n)
        
        echo "$order_len" | uniq -u | awk '{print "u",$1}' > $dir/log/chunk.all
        echo "$order_len" | uniq -d | awk '{print "d",$1}' >> $dir/log/chunk.all
        
        num=$(wc -l $dir/log/chunk.all | awk '{print $1}')

        if [[ "num" -lt "$nj" ]];then
        this_nj=1
        else
        this_nj=$nj
        fi
        
        chunks=
        for i in $(seq $this_nj);do
        chunks="$chunks $dir/log/chunk.$i"
        done
        
        subtools/kaldi/utils/split_scp.pl $dir/log/chunk.all $chunks
        
        echo "min $min_chunk_size" >> $dir/log/chunk.1
        echo "max $chunk_size" >> $dir/log/chunk.1
        
        mkdir -p $dir/log/compile

        $cmd JOB=1:$this_nj $dir/log/compile_xv_nnet.JOB.log \
          nnet3-compile-xvector-net --binary=true --min-chunk-size=$min_chunk_size \
            --max-chunk-size=$chunk_size "$nnet" ark:$dir/log/chunk.JOB $dir/log/compile
        
        
    fi
    
    if [ $stage -le 1 ]; then
    echo "$0: extracting xvectors from compiled nnet"
       $cmd JOB=1:$nj $dir/log/generate_compiled_table.JOB.log \
          subtools/kaldi/utils/filter_scp.pl -f 1 $sdata/utt2spk $data/utt2num_frames.nosil \| \
          awk -v dir=$dir/log/compile '{print $2,dir"/"$2".xv.compile"}' \> $dir/log/compiled.JOB.table  
        
       for i in $(seq $nj);do
       echo "$min_chunk_size $dir/log/compile/$min_chunk_size.xv.compile" >> $dir/log/compiled.$i.table 
       echo "$chunk_size $dir/log/compile/$chunk_size.xv.compile" >> $dir/log/compiled.$i.table 
       done
    
        trap "subtools/linux/kill_pid_tree.sh --show true $$ && echo -e '\nAll killed'" INT
        if $use_gpu; then
        pids=""
        for g in $(seq $nj); do
          $cmd --gpu 1 ${dir}/log/extract.$g.log \
            nnet3-offline-xvector-compute --use-gpu=yes --compiled-path-table=$dir/log/compiled.$g.table --binary=true --min-chunk-size=$min_chunk_size \
            --max-chunk-size=$chunk_size --cache-capacity=${cache_capacity} \
            "$nnet" "`echo $feat | sed s/JOB/$g/g`" ark,scp:${dir}/xvector.$g.ark,${dir}/xvector.$g.scp || exit 1 &
        pids="$pids $!"
        done
        trap "subtools/linux/kill_pid_tree.sh --show true $pids && echo -e '\nAll killed'" INT
        wait
        
      else
        $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
          nnet3-offline-xvector-compute --use-gpu=no --compiled-path-table=$dir/log/compiled.JOB.table --binary=true --min-chunk-size=$min_chunk_size \
          --max-chunk-size=$chunk_size --cache-capacity=${cache_capacity} \
          "$nnet" "$feat" ark,scp:${dir}/xvector.JOB.ark,${dir}/xvector.JOB.scp || exit 1;
      fi
        
    fi
    
    if [ "$clean" == "true" ];then
      rm -rf $dir/log/compile
      rm -f $dir/log/chunk.* $dir/log/compiled.*.table
    fi
fi

if [ $stage -le 2 ]; then
      echo "$0: combining xvectors across jobs"
      for j in $(seq $nj); do cat $dir/xvector.$j.scp; done | awk 'NR==FNR{a[$1]=$2}NR>FNR{if(a[$1]){print $1,a[$1]}}' - $data/feats.scp >$dir/xvector.scp || exit 1;
fi

if [ $stage -le 3 ]; then
      # Average the utterance-level xvectors to get speaker-level xvectors.
      echo "$0: computing mean of xvectors for each speaker"
      $cmd $dir/log/speaker_mean.log \
        ivector-mean ark:$data/spk2utt scp:$dir/xvector.scp \
        ark,scp:$dir/spk_xvector.ark,$dir/spk_xvector.scp ark,t:$dir/num_utts.ark || exit 1;
fi


