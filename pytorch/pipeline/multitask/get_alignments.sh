#!/bin/bash

# Copyright xmuspeech (Author:Zheng Li 2021-06-08)

cmd=run.pl

nj=6
stage=3

if [ -f subtools/path.sh ]; then . ./subtools/path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [opts] <data> <ali-dir> <egs-dir>"
  echo " e.g.: $0 data/train exp/tri3_ali exp/tri4_nnet/egs"
  
  exit 1;
fi

data=$1
alidir=$2
dir=$3

if [ $stage -le 0 ]; then
# Check some files.
for f in $alidir/ali.1.gz $alidir/final.mdl $alidir/tree ; do
   [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

if [ ! -d $dir ]; then
  mkdir -p $dir
fi

sdata=$data/split${nj}utt
subtools/kaldi/utils/split_data.sh --per-utt $data $nj

#mkdir -p $dir/log $dir/info
#cp $alidir/tree $dir

num_ali_jobs=$(cat $alidir/num_jobs) || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: copying data alignments"
  cmd_string="copy-int-vector ark:-"
  for id in $(seq $num_ali_jobs); do gunzip -c $alidir/ali.$id.gz; done | \
  $cmd_string ark,scp:$dir/ali.ark,$dir/ali.scp || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: transforing data alignments"
  ali-to-pdf $alidir/final.mdl scp:$dir/ali.scp ark,scp:$dir/ali.pdf.ark,$dir/ali.pdf.scp
  mv $dir/ali.pdf.scp $dir/ali.scp
  rm -rf $dir/ali.ark  
  phonetic_num_targets=$(tree-info $alidir/tree | grep num-pdfs | awk '{print $2}') || exit 1
  echo "$phonetic_num_targets" > $data/phones
fi

if [ $stage -le 3 ]; then
  echo "$0: check and fix $data, to keep num_feats < num_ali"
  if [ ! -f $data/ali.scp ]; then
    echo "copy ali.scp from $alidir/ali.scp"
    cp -rf $alidir/ali.scp $data/
  fi
  feats_lines=`wc -l $data/feats.scp` 
  ali_lines=`wc -l $data/ali.scp` 
  if [ "$feats_lines" != "$ali_lines"  ]; then
    echo "The number of feats: $feats_lines is not equal to alignments: $ali_lines"
    echo "Fix this by reducing the number of feats that of alignments: $ali_lines"
    mv $data/feats.scp $data/feats.scp.temp
    mv $data/ali.scp $data/feats.scp
    utils/data/fix_data_dir.sh $data
    mv $data/feats.scp $data/ali.scp
    mv $data/feats.scp.temp $data/feats.scp
    utils/data/fix_data_dir.sh $data
  fi
fi

if [ $stage -le 4 ]; then
  if [ ! -f $data/ali.scp ]; then
    echo "copy ali.scp from $alidir/ali.scp"
    cp -rf $alidir/ali.scp $data/
  fi
  select-voiced-ali scp:$data/ali.scp scp:$data/vad.scp ark,scp:$dir/ali.ark,$dir/ali.scp
  cp $data/phones $dir/
fi

