# Auto-config environment by Snowdar (2020-04-16)

current_dir=$(subtools/linux/decode_symbolic_link.sh $PWD)
kaldi_root_dir=$(dirname $(dirname $(dirname $current_dir)))

export KALDI_ROOT=$kaldi_root_dir
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/subtools/kaldi/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
