#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

if [ $# -ne 1 ]; then
  echo "This script outputs a mapping from recording to a list of utterances "
  echo "corresponding to the recording. It is analogous to the content of "
  echo "a spk2utt file, but is indexed by recording instead of speaker."
  echo "Usage: get_reco2utt.sh <data>"
  echo " e.g.: get_reco2utt.sh data/train"
  exit 1
fi

data=$1

if [ ! -s $data/segments ]; then
  subtools/kaldi/utils/data/get_segments_for_data.sh $data > $data/segments
fi

cut -d ' ' -f 1,2 $data/segments | subtools/kaldi/utils/utt2spk_to_spk2utt.pl
