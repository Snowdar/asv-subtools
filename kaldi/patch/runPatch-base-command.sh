#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2019-01-14)

# Make sure your kaldi has been compiled.

current_dir=$(subtools/linux/decode_symbolic_link.sh $PWD)
kaldi_root_dir=$(dirname $(dirname $(dirname $current_dir)))

# There is an another good way now [2019-07-27]

# compile these files
cfile="
src/nnet3bin/nnet3-compile-xvector-net.cc 
src/nnet3bin/nnet3-offline-xvector-compute.cc
src/gmmbin/gmm-global-init-from-feats-mmi.cc
src/gmmbin/gmm-global-est-gaussians-ebw.cc
src/gmmbin/gmm-global-est-map.cc
src/gmmbin/gmm-global-est-weights-ebw.cc
"
. subtools/kaldi/utils/parse_options.sh

cd subtools/kaldi/patch

for x in $cfile;do
	name=`basename ${x%.*}`
	dir=`dirname $x`

	[ ! -f "$kaldi_root_dir/$dir/Makefile" ] && echo "$kaldi_root_dir/$dir/Makefile is not exist." && exit 1;

	sed -e ':a' -e 'N' -e '$!ba' -e 's/\( *\)\\\n \+/ /g;s/\\\n//g' "$kaldi_root_dir/$dir/Makefile" | \
		awk -v name=$name '{if($1=="BINFILES"){$0="BINFILES = "name;} print $0}' > $kaldi_root_dir/$dir/Makefile.tmp

	# Copy *.cc
	[ -f $kaldi_root_dir/$x ] && mv $kaldi_root_dir/$x $kaldi_root_dir/$x.bk
	rm -f $kaldi_root_dir/$dir/$name.o $kaldi_root_dir/$dir/$name
	cp -f $x $kaldi_root_dir/$dir

	# Make
	echo "[Enter $kaldi_root_dir/$dir...]"
	cd  $kaldi_root_dir/$dir
	make -f $kaldi_root_dir/$dir/Makefile.tmp
	rm -f $kaldi_root_dir/$dir/Makefile.tmp
	echo "[Leave $kaldi_root_dir/$dir...]"
	cd - >/dev/null
done

cd -
echo "Compile done."

##########################################################################
## Deprecated ##  Reasons: it is dependent to atlas or mkl version and not enough free
# if paths of *.so are not right, you could fix them by yourself.
# atlasInclude=$(grep  "ATLASINC.*=" $kaldi_root_dir/src/kaldi.mk | cut -d "=" -f 2 )
# atlasSo=$(grep  "ATLASLIBS.*=" $kaldi_root_dir/src/kaldi.mk | cut -d "=" -f 2 )
# openfstInclude=$(grep  "OPENFSTINC.*=" $kaldi_root_dir/src/kaldi.mk | cut -d "=" -f 2 )
# openfstSo=$(grep  "OPENFSTLIBS.*=" $kaldi_root_dir/src/kaldi.mk | cut -d "=" -f 2 )
# openfstLib=$(grep  "OPENFSTLDFLAGS.*=" $kaldi_root_dir/src/kaldi.mk | cut -d " " -f 3)
# cuda=$(grep  "CUDATKDIR.*=" $kaldi_root_dir/src/kaldi.mk | cut -d "=" -f 2 | sed 's/^ \+//g')

# for x in $cfile;do
# name=`basename ${x%.*}`
# dir=`dirname $x`
# type=`basename $dir`
# [ -f $kaldi_root_dir/$dir/$x ] && mv $kaldi_root_dir/$dir/$x.bk
# cp -f $x $kaldi_root_dir/$dir
# cd  $kaldi_root_dir/$dir
# echo "compile $kaldi_root_dir/$x"
# case $type in
	# ivectorbin)
	# g++ -std=c++11 -I.. -I$openfstInclude -Wno-sign-compare -Wall -Wno-sign-compare -Wno-unused-local-typedefs -Wno-deprecated-declarations -Winit-self -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -I$atlasInclude -msse -msse2 -pthread -g  -DHAVE_CUDA -I$cuda/include -c -o $name.o $name.cc
	
	# g++  $openfstLib  -rdynamic  $name.o ../ivector/kaldi-ivector.a ../hmm/kaldi-hmm.a ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a   $openfstSo $atlasSo -lm -lpthread -ldl -o $name
	# cd -
	# ;;
	# nnet3bin)
	# g++ -std=c++11 -I.. -I$openfstInclude -Wno-sign-compare -Wall -Wno-sign-compare -Wno-unused-local-typedefs -Wno-deprecated-declarations -Winit-self -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -I$atlasInclude -msse -msse2 -pthread -g  -DHAVE_CUDA -I$cuda/include -c -o $name.o $name.cc
	
	# g++  $openfstLib  -rdynamic -L$cuda/lib64 -Wl,-rpath,$cuda/lib64  $name.o ../nnet3/kaldi-nnet3.a ../chain/kaldi-chain.a ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a   $openfstSo $atlasSo -lm -lpthread -ldl -lcublas -lcusparse -lcudart -lcurand  -o $name
	# cd -
	# ;;
	# gmmbin)
	# g++ -std=c++11 -I.. -isystem $openfstInclude -O1 -Wno-sign-compare -Wall -Wno-sign-compare -Wno-unused-local-typedefs -Wno-deprecated-declarations -Winit-self -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -I$atlasInclude -msse -msse2 -pthread -g  -DHAVE_CUDA -I$cuda/include   -c -o $name.o $name.cc
	
	# g++  $openfstLib  -rdynamic  $name.o ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a ../feat/kaldi-feat.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a  $openfstSo $atlasSo -lm -lpthread -ldl -o $name
	# cd -
	# ;;
	# *) echo "don't support this type [$type]" && exit 1;;
# esac

# echo "Compile $name done."
# echo "Start to test this command and if you can see usages of this command later,then it means compiling successfully."
# $kaldi_root_dir/$dir/$name
# done
# echo "All done."
