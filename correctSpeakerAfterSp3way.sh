#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-09-14)

# To avoid adding some spkear-id with sp prefix in lre or sre task,you can use this script to correct your datadir
extra_files= # you can add some you want to fix but needs utt-id in 1th field
extra_factor= # could be 0.8 or 1.2 or other factor which have 3 chars.

. subtools/parse_options.sh
. subtools/path.sh

if [[ $# != 1 ]];then
echo "[exit] Num of parameters is not equal to 1"
echo "$0 <data-dir>"
exit 1
fi

data=$1

files=
for x in wav.scp utt2spk feats.scp vad.scp utt2dur utt2uniq reco2dur reco2utt;do
[ -f $data/$x ] && files="$files $data/$x"
done

for x in $extra_files;do
[ -f $x ] && files="$files $x"
done

for file in $files; do
[ -f $file ] && \
awk -v factor=$extra_factor '{if(match($1,"^sp1.1-")||match($1,"^sp0.9-")||match($1,"^sp"factor"-")){$1=substr($1,7)"-"substr($1,1,5)} print $0}' $file > $file.tmp && mv -f $file.tmp $file
# you also can use this script to fix xvector.scp or ivector.scp or any others scp with utt-id
echo "$data/$file done"
done

[[ "$extra_factor" != "" ]] && sed -i 's/sp'"$extra_factor"'-//g' $data/utt2spk
sed -i 's/sp0.9-//g' $data/utt2spk
sed -i 's/sp1.1-//g' $data/utt2spk
echo "recover spearker-id done"

subtools/kaldi/utils/fix_data_dir.sh $data
echo "All done."
