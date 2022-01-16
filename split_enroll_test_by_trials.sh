#!/bin/bash

# Snowdar 

vectordir=
outname=
force=false
clear=false
scp_type=xvector.scp

. subtools/parse_options.sh
. subtools/linux/functions.sh
. subtools/path.sh

#   #
  # 
check_params 2 "[--vectordir ""] <data-dir> <trials>"
  #
#   #

datadir=$1
trials=$2

[ ! -d $datadir ] && echo "Expected $datadir to exist" && exit 1
[ ! -f $trials ] && echo "Expected $trials to exist" && exit 1

datadir=$(dirname $datadir)/$(basename $datadir)


if [ "$outname" == "" ];then
    outname=$datadir
else
    outname=$(dirname $datadir)/$outname
fi

this_name=$(basename $outname)
enroll_list=${this_name:+${this_name}_}enroll.list
test_list=${this_name:+${this_name}_}test.list

# Get list
echo "[1] Generate $datadir/$enroll_list..."
# Note that, the meaning of $ in awk should be transferred.
force_for "awk '{print \$1}' $trials | sort -u > $datadir/$enroll_list" f:$datadir/$enroll_list

echo "[2] Generate $datadir/$test_list..."
force_for "awk '{print \$2}' $trials | sort -u > $datadir/$test_list" f:$datadir/$test_list

# Subset
echo "[3] Generate ${outname}_enroll..."
force_for "subtools/filterDataDir.sh $datadir $datadir/$enroll_list ${outname}_enroll" d:${outname}_enroll

echo "[4] Generate ${outname}_test..."
force_for "subtools/filterDataDir.sh $datadir $datadir/$test_list ${outname}_test" d:${outname}_test
echo "[5] Copy $trials to ${outname}_test/trials..."
force_for "cp -f $trials ${outname}_test/trials" f:${outname}_test/trials

if [ "$vectordir" != "" ];then
    vectordir_dir=$(dirname $vectordir)
    dataset_name=$(basename $outname)

    echo "[6] Generate $vectordir_dir/${dataset_name}_enroll ..."
    force_for "subtools/filterVectorDir.sh --scp-type $scp_type $vectordir $datadir/$enroll_list $vectordir_dir/${dataset_name}_enroll" \
              d:$vectordir_dir/${dataset_name}_enroll

    echo "[7] Generate $vectordir_dir/${dataset_name}_test ..."
    force_for "subtools/filterVectorDir.sh --scp-type $scp_type $vectordir $datadir/$test_list $vectordir_dir/${dataset_name}_test" \
              d:$vectordir_dir/${dataset_name}_test
fi

if [ "$clear" == "true" ];then
    rm -f $datadir/$enroll_list $datadir/$test_list
fi

echo "## Split done ##"
