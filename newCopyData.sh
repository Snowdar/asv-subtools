#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2018-7-25)

topdir=data
force=false # for overwrite

. subtools/parse_options.sh

if [[ $# != 2 ]];then
echo "[exit] Num of parameters is not equal to 2"
echo "usage:$0 --force [false|true] <prefixes> <src-dirs>"
exit 1
fi

pre=$1
srcs=$2

for src in $srcs;do
[ ! -d "$src" ] && tmp=$src && src=data/$src && [ ! -d "$src" ] && echo "[exit] No such dir $tmp or $src" && exit 1
name=`basename $src`
target=$topdir/${pre}/$name

[ "$src" == "$target" ] && echo "[Warning] data-dir $src is same to target-data-dir, so skip it" && continue
[ -d $target ] && [ "$force" == "false" ] && echo "[exit] $target is exist, please delete it carefully by yourself" && exit 1

rm -rf $target
mkdir -p $target

echo "Copy $src to $target..."

for x in wav.scp utt2spk spk2utt;do
[ ! -f $src/$x ] && echo "[exit] Expected $src/$x to exist at least." && exit 1
done

trials=""
for path in $(find $src -name "*trials");do
trials="$trials $(basename $path)"
done

for x in wav.scp utt2spk spk2utt feats.scp vad.scp utt2num_frames utt2dur reco2dur text utt2gender spk2gender $trials;do
[ -f $src/$x ] && cp $src/$x $target/ && echo "[ $x ] copy done"
done
echo "Note, your new datadir is $target\n"
done
echo "Copy done."
