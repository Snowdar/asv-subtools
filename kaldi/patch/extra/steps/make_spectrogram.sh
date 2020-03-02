#!/bin/bash

# Copyright 2012-2016  Karel Vesely  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
spectrogram_config=conf/spectrogram.conf
compress=true
write_utt2num_frames=false  # if true writes utt2num_frames
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f subtools/path.sh ]; then . ./subtools/path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
   echo "Usage: $0 [options] <data-dir> [<log-dir> [<spectrogram-dir>] ]";
   echo "e.g.: $0 data/train exp/make_spectrogram/train spectrogram"
   echo "Note: <log-dir> defaults to <data-dir>/log, and <spectrogram-dir> defaults to <data-dir>/data"
   echo "Options: "
   echo "  --spectrogram-config <config-file>                     # config passed to compute-spectrogram-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (subtools/kaldi/utils/run.pl|subtools/kaldi/utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --write-utt2num-frames <true|false>     # If true, write utt2num_frames file."
   exit 1;
fi

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=$data/log
fi
if [ $# -ge 3 ]; then
  spectrogramdir=$3
else
  spectrogramdir=$data/data
fi


# make $spectrogramdir an absolute pathname.
spectrogramdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $spectrogramdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $spectrogramdir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/wav.scp

required="$scp $spectrogram_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_spectrogram.sh: no such file $f"
    exit 1;
  fi
done

subtools/kaldi/utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

for n in $(seq $nj); do
  # the next command does nothing unless $spectrogramdir/storage/ exists, see
  # subtools/kaldi/utils/create_data_link.pl for more info.
  subtools/kaldi/utils/create_data_link.pl $spectrogramdir/raw_spectrogram_$name.$n.ark
done

if $write_utt2num_frames; then
  write_num_frames_opt="--write-num-frames=ark,t:$logdir/utt2num_frames.JOB"
else
  write_num_frames_opt=
fi

if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."
  split_segments=""
  for n in $(seq $nj); do
    split_segments="$split_segments $logdir/segments.$n"
  done

  subtools/kaldi/utils/split_scp.pl $data/segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null

  $cmd JOB=1:$nj $logdir/make_spectrogram_${name}.JOB.log \
    extract-segments scp,p:$scp $logdir/segments.JOB ark:- \| \
    compute-spectrogram-feats --verbose=2 --config=$spectrogram_config ark:- ark:- \| \
    copy-feats --compress=$compress $write_num_frames_opt ark:- \
     ark,scp:$spectrogramdir/raw_spectrogram_$name.JOB.ark,$spectrogramdir/raw_spectrogram_$name.JOB.scp \
     || exit 1;

else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for n in $(seq $nj); do
    split_scps="$split_scps $logdir/wav.$n.scp"
  done

  subtools/kaldi/utils/split_scp.pl $scp $split_scps || exit 1;

  $cmd JOB=1:$nj $logdir/make_spectrogram_${name}.JOB.log \
    compute-spectrogram-feats --verbose=2 --config=$spectrogram_config scp,p:$logdir/wav.JOB.scp ark:- \| \
    copy-feats --compress=$compress $write_num_frames_opt ark:- \
     ark,scp:$spectrogramdir/raw_spectrogram_$name.JOB.ark,$spectrogramdir/raw_spectrogram_$name.JOB.scp \
     || exit 1;

fi


if [ -f $logdir/.error.$name ]; then
  echo "Error producing spectrogram features for $name:"
  tail $logdir/make_spectrogram_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $spectrogramdir/raw_spectrogram_$name.$n.scp || exit 1;
done > $data/feats.scp

if $write_utt2num_frames; then
  for n in $(seq $nj); do
    cat $logdir/utt2num_frames.$n || exit 1;
  done > $data/utt2num_frames || exit 1
  rm $logdir/utt2num_frames.*
fi

rm $logdir/wav.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/feats.scp | wc -l`
nu=`cat $data/utt2spk | wc -l`
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully ($nf != $nu);"
  echo "consider using subtools/kaldi/utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating filterbank features for $name"
