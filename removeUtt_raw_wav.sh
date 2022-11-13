#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-7-25)

limit_utts=0 # The spker whose utts < limit_utts will be removed. <=0 means no removing.
min_len=0.0
max_len=1000.0


. subtools/path.sh
. subtools/parse_options.sh

if [[ $# != 1 ]];then
echo "[exit] Num of parameters is not equal to 1"
echo "usage:$0 <data-dir>"
echo "[note] The utts whose length is not satisfied will be removed but keep the back-up so that you can recover while running this script with a small enough length."
exit 1
fi

data=$1



required="wav.scp utt2chunk utt2dur spk2utt utt2spk"

for f in $required; do
  if [ ! -f $data/$f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done
[ -f $data/utt2dur.backup ] && cp -f $data/utt2dur.backup $data/utt2dur
[ -f $data/spk2utt.backup ] && cp -f $data/spk2utt.backup $data/spk2utt


utt=`awk -v min=$min_len -v max=$max_len  '{if($2<min || $2>max) printf $1" "}' ${data}/utt2dur`

[ "$limit_utts" -gt 0 ] && utt=$utt`echo $utt | awk -v limit=$limit_utts 'NR==FNR{for(i=1;i<=NF;i++){a[$i]=1;}} NR>FNR{tot=NF-1;
for(i=2;i<=NF;i++){
   if(a[$i]==1){
      tot=tot-1;
   }
}
if(tot<limit){$1="";print $0;}
}' - $data/spk2utt`

list=`echo "$utt" | sed 's/ /\n/g' | sed '/^$/d' | sort -u`
num=0
if [ "$list" != "" ];then
num=`echo "$list" | wc -l | awk '{print $1}'`
else
echo "Need to remove nothing. It means that your datadir will be recovered form bakeup if you used this script before."
fi

echo -e "[`echo $list`] $num utts here will be removed."

# Backup and Recover
for x in wav.scp utt2spk spk2utt feats.scp vad.scp utt2num_frames utt2gender spk2gender utt2chunk utt2dur text utt2sr;do
[ -f $data/$x.backup ] && cp -f $data/$x.backup $data/$x
[ ! -f $data/$x.backup ] && [ -f $data/$x ] &&  cp $data/$x $data/$x.backup
done

# Remove
for x in wav.scp utt2spk feats.scp vad.scp utt2num_frames utt2gender utt2chunk utt2dur text utt2sr;do
[ -f $data/$x ] && [ "$list" != "" ] && echo "$list" | awk 'NR==FNR{a[$1]=1}NR>FNR{if(!a[$1]){print $0}}' - \
   $data/$x > $data/$x.tmp && \
   mv -f $data/$x.tmp $data/$x
echo "$data/$x done"
done

subtools/kaldi/utils/utt2spk_to_spk2utt.pl <$data/utt2spk >$data/spk2utt
echo "$data/spk2utt done"

if [ -f $data/spk2gender ];then
subtools/kaldi/utils/filter_scp.pl $data/spk2utt $data/spk2gender > $data/spk2gender.tmp && mv -f $data/spk2gender.tmp $data/spk2gender
echo "$data/spk2gender done"
fi

echo 'Remove invalid utts done.'
