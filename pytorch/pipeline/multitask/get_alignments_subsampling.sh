#!/bin/bash

# Copyright xmuspeech (Author:Zheng Li 2021-06-08)

cmd=run.pl

nj=6
stage=0

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

# Check some files.
for f in $alidir/ali.1.gz $alidir/final.mdl $alidir/tree ; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

if [ ! -d $dir ]; then
  mkdir -p $dir
fi

sdata=$data/split${nj}utt
subtools/kaldi/utils/split_data.sh --per-utt $data $nj

mkdir -p $dir/log $dir/info
cp $alidir/tree $dir

num_ali_jobs=$(cat $alidir/num_jobs) || exit 1;


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
  echo "$0: now recover the subsampling"
   copy-int-vector scp:$dir/ali.scp ark,t:$dir/ali.sb.txt
   awk '{print "changelinelabel" $1;for(i=2;i<=NF;i++){print $i,$i,$i }}' $dir/ali.sb.txt > $dir/ali.sb.new.txt
   sed -i ':a;N;$!ba;s/\n/ /g' $dir/ali.sb.new.txt
   sed -i 's/changelinelabel/\n/g' $dir/ali.sb.new.txt
   rm -rf $dir/ali.scp
   rm -rf $dir/ali.pdf.ark
   copy-int-vector ark:$dir/ali.sb.new.txt ark,scp:$dir/ali.ark,$dir/ali.scp
fi

if [ $stage -le 3 ]; then
  echo "$0: check and fix $data, to keep num_feats < num_ali"
  feats_lines=`wc -l $dir/feats.scp` 
  ali_lines=`wc -l $dir/ali.scp` 
  if [ "$feats_lines" != "$ali_lines"  ]; then
    echo "The number of feats: $feats_lines is not equal to alignments: $ali_lines"
    echo "Fix this by reducing the number of feats that of alignments: $ali_lines"
    mv $dir/feats.scp $dir/feats.scp.temp
    mv $dir/ali.scp $dir/feats.scp
    utils/data/fix_data_dir.sh $dir
    mv $dir/feats.scp $dir/ali.scp
    mv $dir/feats.scp.temp $dir/feats.scp
    utils/data/fix_data_dir.sh $dir
  fi
fi

 if [ $stage -le 4 ]; then
  copy-int-vector scp:$data/ali.scp ark,t:$data/ali.rd.txt
  paste $data/utt2num_frames $data/ali.rd.txt > $dir/utt2num_frame.ali.rd.txt
  awk '{num=$2 +3 ;i=3;print "changelinelabel" ;for(i=3;i<=num;i++)  {print $i}}' $dir/utt2num_frame.ali.rd.txt > $dir/ali.rd.txt
  sed -i ':a;N;$!ba;s/\n/ /g' $dir/ali.rd.txt
  sed -i 's/changelinelabel/\n/g' $dir/ali.rd.txt
  copy-int-vector ark:$dir/ali.rd.txt ark,scp:$dir/ali.rd.ark,$dir/ali.rd.scp
  select-voiced-ali scp:$dir/ali.rd.scp scp:$data/vad.scp ark,scp:$dir/ali.ark,$dir/ali.scp
  rm -rf $dir/ali.rd.scp
fi

