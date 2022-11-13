#!/bin/bash
# Copyright xmuspeech (Author: Leo 2021-08-26)
# prepare csv egs.
echo "$0 $@"
set -e
nj=16
cmd="run.pl"
stage=0
endstage=2

force_clear=true

# generate chunk
expected_files='utt2spk:wav.scp:spk2utt'
whole_utt=false
random_segment=false
seg_dur=2.015
de_silence=true              
vad_wav_savdir=""           # path to save wavs after trimming silence when de_silence=true
vad_win_len=0.1
amp_th=50

# Remove utts
min_len=2.0
max_len=1000.0
limit_utts=8

# split tran valid
valid_sample=true
valid_num_utts=2048
valid_split_type="--default" #"--total-spk"
valid_chunk_num=2            # if 0, means use all the chunk of valid utt.
valid_fix_chunk_num=false    # With split type [--total-spk] and fix_chun_num=true
# finally we get valid_num_utts * valid_chunk_num = 1024 * 2 = 2048 valid chunks.

# Prepare speech augmention csv files.
pre_rirmusan=false
openrir_folder="export/path" # where has openslr rir. (eg. ./data/RIRS_NOISES, then, openrir_folder="./data") 
musan_folder="export/path"   # where has openslr musan.
csv_aug_folder="exp/aug_csv" # noise clips id indexes.
savewav_folder=""            # noise clips for online speechaug, set it in SSD. /export/speech_aug_2_0
max_noise_len=$seg_dur       # match with seg_dur

# share or raw
data_type=raw
num_utts_per_shard=2000
shard_dir=""                 # path to save training shards when data_type=shard

if [ -f subtools/path.sh ]; then . subtools/path.sh; fi
. subtools/parse_options.sh || exit 1

if [[ $# != 3 ]]; then
  echo "[exit] Num of parameters is not equal to 3"
  echo "usage:$0 <data-dir> <data-dir-egs> <egs-dir>"
  exit 1
fi
# Key params
traindata=$1
traindata_for_egs=$2
egsdir=$3

[ ! -d "$traindata" ] && echo "The traindata [$traindata] is not exist." && exit 1

vad_suffix=
if [ "$de_silence" == "true" ]; then
  vad_suffix="_vad_amp_th$amp_th"

fi
mkdir -p ${egsdir}/info

if [[ $stage -le 0 && 0 -le $endstage ]]; then
  echo -e "$0: stage 0: Generate raw wav kaldidir which contains utt2chunk utt2sr and utt2dur.\n "
  logdir=${traindata}/log
  [ -d "$logdir" ] && rm -rf $logdir
  mkdir -p $logdir
  echo "You can check the progress bar in $logdir."
  subtools/kaldi/utils/split_data.sh --per-utt $traindata $nj
  sdata=$traindata/split${nj}utt
  name=`basename $traindata`
  vad_train_data=$vad_wav_savdir/${name}${vad_suffix}/split${nj}
  trap "subtools/linux/kill_pid_tree.sh --show true $$ && echo -e '\nAll killed\n' && exit 1" INT

  $cmd JOB=1:$nj $logdir/generate_raw_wav_kaldidir.JOB.log \
    python3 subtools/pytorch/pipeline/onestep/generate_raw_wav_kaldidir.py \
    --expected-files=$expected_files --whole-utt=$whole_utt \
    --random-segment=$random_segment --seg-dur=$seg_dur \
    --de-silence=$de_silence --vad-save-dir=$vad_train_data/JOB \
    --vad-win-len=$vad_win_len --amp-th=$amp_th \
    $sdata/JOB $sdata/JOB/temp || exit 1

  num=$(grep -E "ERROR|Error" $logdir/generate_raw_wav_kaldidir.*.log | wc -l)
  [ $num -gt 0 ] && echo "There are some ERRORS in $logdir/generate_raw_wav_kaldidir.*.log" && exit 1
  if [ "$force_clear" == "true" ]; then
    rm -rf ${traindata}/rawwav${vad_suffix}
  fi

  mkdir -p ${traindata}/rawwav${vad_suffix}/info
  dst_files="wav.scp utt2spk utt2dur utt2chunk utt2sr ori_dur desil_dur"
  for file in $dst_files; do
    for j in $(seq $nj); do cat $sdata/$j/temp/$file; done >${traindata}/rawwav${vad_suffix}/$file || exit 1
  done
  subtools/kaldi/utils/utt2spk_to_spk2utt.pl <${traindata}/rawwav${vad_suffix}/utt2spk >${traindata}/rawwav${vad_suffix}/spk2utt
  old_num_utt=$(cat ${traindata}/wav.scp | wc -l)
  new_num_utt=$(cat ${traindata}/rawwav${vad_suffix}/wav.scp | wc -l)
  for f in ori_dur desil_dur;do
    mv -f ${traindata}/rawwav${vad_suffix}/$f ${traindata}/rawwav${vad_suffix}/${f}_bk
    awk 'BEGIN{sum=0}{sum=sum+$1}END{print sum}' ${traindata}/rawwav${vad_suffix}/${f}_bk >  ${traindata}/rawwav${vad_suffix}/${f}
    rm -rf ${traindata}/rawwav${vad_suffix}/${f}_bk
  done
  old_dur=`cat ${traindata}/rawwav${vad_suffix}/ori_dur`
  new_dur=`cat ${traindata}/rawwav${vad_suffix}/desil_dur`

  utt_reduced=$(awk 'BEGIN{printf "%.1f%%\n",(('$old_num_utt'-'$new_num_utt')/'$old_num_utt')*100}')
  dur_reduced=$(awk 'BEGIN{printf "%.1f%%\n",(('$old_dur'-'$new_dur')/'$old_dur')*100}')

  echo -e "Generate utt2chunk utt2sr and utt2dur from ${traindata} to ${traindata}/rawwav${vad_suffix} done\n old_utt_num:$old_num_utt  \n new_utt_num:$new_num_utt \n reduced utts num:$utt_reduced\n old_dur:$old_dur(h) \n new_dur:$new_dur(h) \n reduced dur:$dur_reduced"

  echo -e "Generate utt2chunk utt2sr and utt2dur from ${traindata} to ${traindata}/rawwav${vad_suffix} done\n old_utt_num:$old_num_utt  \n new_utt_num:$new_num_utt \n reduced utts num:$utt_reduced\n old_dur:$old_dur(h) \n new_dur:$new_dur(h) \n reduced dur:$dur_reduced" >${traindata}/rawwav${vad_suffix}/info/chunk_info

  mkdir -p $traindata_for_egs && cp -r ${traindata}/rawwav${vad_suffix}/* $traindata_for_egs
fi

if [[ $stage -le 1 && 1 -le $endstage ]]; then
  echo "$0: stage 1.1: filter dataset"

  subtools/removeUtt_raw_wav.sh --limit-utts $limit_utts --min-len $min_len --max-len $max_len \
    $traindata_for_egs || exit 1
  num_utt=$(cat $traindata_for_egs/wav.scp | wc -l)
  num_spk=$(cat $traindata_for_egs/spk2utt | wc -l)
  new_dur=$(awk 'BEGIN{sum=0}{sum=sum+$2}END{printf "%.2f\n",sum/3600}' ${traindata_for_egs}/utt2dur)


  if [ -f ${traindata}/wav.scp ] && [ -f ${traindata}/spk2utt ]; then 
    old_num_utt=$(cat ${traindata}/wav.scp | wc -l)
    old_num_spk=$(cat ${traindata}/spk2utt | wc -l)
    reduced_utt=$(awk 'BEGIN{printf "%.1f%%\n",(('$old_num_utt'-'$num_utt')/'$old_num_utt')*100}')
    reduced_spk=$(awk 'BEGIN{printf "%.1f%%\n",(('$old_num_spk'-'$num_spk')/'$old_num_spk')*100}')

    echo -e "From ${traindata} to ${traindata_for_egs}.\n utt_num reduced from $old_num_utt to $num_utt,reduced:$reduced_utt,dur remain: $new_dur.\n spk_num reduced from $old_num_spk to $num_spk.\n reduced:$reduced_spk.\n"
    echo -e "From ${traindata} to ${traindata_for_egs}.\n utt_num reduced from $old_num_utt to $num_utt,reduced:$reduced_utt,dur remain: $new_dur.\n spk_num reduced from $old_num_spk to $num_spk.\n reduced:$reduced_spk.\n" >$traindata_for_egs/info/remove_info
  else
    echo "${traindata}/rawwav${vad_suffix}: utt_num:$num_utt   spk_num:$num_spk  dur:$new_dur"
    echo "${traindata}/rawwav${vad_suffix}: utt_num:$num_utt   spk_num:$num_spk  dur:$new_dur" >$traindata_for_egs/info/remove_info
  fi

  echo "$0: stage 1.2: prepare speech augmention csv files. "
  if [ "$pre_rirmusan" == "true" ]; then
    [ "$savewav_folder" == "" ] && echo "The savewav_folder for aug wav is not specified, suggest set into SSD" && exit 1
    python3 subtools/pytorch/pipeline/onestep/prepare_speechaug_csv.py \
      --openrir-folder=$openrir_folder \
      --musan-folder=$musan_folder \
      --savewav-folder=$savewav_folder \
      --force-clear=$force_clear \
      --max-noise-len=$max_noise_len \
      $csv_aug_folder || exit 1
  fi
fi

if [[ $stage -le 2 && 2 -le $endstage ]]; then
  echo "$0: stage 2"
  [ "$egsdir" == "" ] && echo "The egsdir is not specified." && exit 1
  if [ "$data_type" == "raw" ]; then
    python3 subtools/pytorch/pipeline/onestep/get_raw_wav_chunk.py \
      --valid-sample=$valid_sample \
      --valid-split-type=$valid_split_type \
      --valid-num-utts=$valid_num_utts \
      --valid-chunk-num=$valid_chunk_num \
      --valid-fix-chunk-num=$valid_fix_chunk_num \
      $traindata_for_egs $egsdir || exit 1
  fi
  if [ "$data_type" == "shard" ]; then
    [ "$shard_dir" == "" ] && echo "The shard_dir is not specified." && exit 1
    suffix=
    if [ "$whole_utt" == "true" ]; then
      suffix="whole"
    else
      suffix=${seg_dur/./_}
    fi
    raw_eg_dir=${traindata}/shard/raw_eg${vad_suffix}_${suffix}
    shard_eg_dir=${traindata}/shard/shard_eg${vad_suffix}_${suffix}
    mkdir -p $raw_eg_dir $shard_eg_dir
    python3 subtools/pytorch/pipeline/onestep/get_raw_wav_chunk.py \
      --valid-sample=$valid_sample \
      --valid-split-type=$valid_split_type \
      --valid-num-utts=$valid_num_utts \
      --valid-chunk-num=$valid_chunk_num \
      --valid-fix-chunk-num=$valid_fix_chunk_num \
      $traindata_for_egs $raw_eg_dir || exit 1
    mkdir -p $shard_dir/log
    for x in train valid; do
      mkdir -p $shard_dir/${x}
      [ -f ${raw_eg_dir}/${x}.egs.csv ] && python3 subtools/pytorch/pipeline/onestep/make_shard_list.py \
        --nj $nj \
        --num-utts-per-shard=$num_utts_per_shard \
        $raw_eg_dir/${x}.egs.csv $shard_dir/${x} \
        $shard_eg_dir/${x}.egs.csv 1>$shard_dir/log/mkshrd_error_${x}_eg.log || exit 1
    done

    mkdir -p ${egsdir}
    cp -rf $raw_eg_dir/info ${shard_eg_dir}
    cp -rf $shard_eg_dir/* ${egsdir}
  fi

fi

exit 0
