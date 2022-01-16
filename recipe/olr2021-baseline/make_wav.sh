#!bin/bash

data_path=$1
data=./data/pre


mkdir -p $data
mkdir -p $data/wavflist
mkdir -p $data/wavscp
mkdir -p $data/utt2lan



#list all utt2lang & wav.scp
find $data_path  -name wav.scp > $data/wavscp.list
# find $data_path  -name utt2lang > $data/utt2lang.list

#conbine wav.scp files & conbine utt2lang files
find $data_path -name wav.scp | xargs cat | sort -k 1 -u  > $data/wav_raw.scp
awk '!a[$1]++{print}' $data/wav_raw.scp > $data/wav.scp 
wc -l $data/wav_raw.scp
wc -l $data/wav.scp



# find $data_path -name utt2lang | xargs cat | sort -k 1 -u > $data/utt2lang





#separate by the language code 
name="Kazak Minnan Shanghai Sichuan Tibet Uyghu ct-cn id-id ja-jp ko-kr ru-ru vi-vn zh-cn UNKNOWN ms-my th-th hi-di te-in"

for x in $name;do
cat $data/wav.scp |awk -F " " '{if(substr($1,1,5)==substr("'$x'",1,5)) print $2}'  | sort -u >$data/wavflist/$x'.flist'
cat $data/wav.scp |awk -F " " '{if(substr($1,1,5)==substr("'$x'",1,5)) print $1,$2}'  | sort -u >$data/wavscp/$x'.scp'
cat $data/wav.scp |awk -F " " '{if(substr($1,1,5)==substr("'$x'",1,5)) print $1,"'$x'"}'| sort -u >  $data/utt2lan/$x'.utt2lang'
done




#conbine the 13 target languages
 mkdir -p ./data/train
 target="Kazak Minnan Shanghai Sichuan Tibet Uyghu ct-cn id-id ja-jp ko-kr ru-ru vi-vn zh-cn"

for x in $target;do
echo  $data/utt2lan/$x'.utt2lang'; done  |  xargs  -i  cat  {}  >> ./data/train/utt2lang

for x in $target;do
echo  $data/wavscp/$x'.scp'; done  |  xargs  -i  cat  {}  >> ./data/train/wav.scp