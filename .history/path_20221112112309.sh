# Auto-config environment by Snowdar (2020-04-16)

# Note: please make sure your project path is just like kaldi/egs/xmuspeech/yourproject, where
# the project should be in sub-sub-sub-dir of kaldi root. If not, modify KALDI_ROOT by yourself.

# Use decode_symbolic_link.sh rather than ../../../ to get the KALDI_ROOT so that it could
# support the case that the project is linked by a symbolic and $PWD contains the symbolic.

current_dir=$(subtools/linux/decode_symbolic_link.sh $PWD)
<<<<<<< HEAD
# kaldi_root_dir=$(dirname $(dirname $(dirname $current_dir)))
kaldi_root_dir=/work/kaldi_test
=======
kaldi_root_dir=$(dirname $(dirname $(dirname $current_dir)))
# kaldi_root_dir=/work/kaldi
>>>>>>> master
[ ! -d $kaldi_root_dir/tools ] && echo >&2 "[KALDI_ROOT ERROR] Got an invalid path $kaldi_root_dir when source environment (See the 'Note' in subtools/path.sh to correct it by yourself)." && exit 1

export KALDI_ROOT=$kaldi_root_dir
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/subtools/kaldi/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
