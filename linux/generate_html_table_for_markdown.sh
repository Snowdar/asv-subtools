#!/bin/bash  

# Copyright xmuspeech (Author: Snowdar 2020-06-04)

head=true

. subtools/parse_options.sh

if [[ $# != 2 ]];then
echo "[exit] Num of parameters is not equal to 2"
echo "usage:$0 <table-data-file> <out-html-file>"
exit 1
fi

data=$1
html=$2

awk -F '|' -v head=$head '
BEGIN{print "<table>";}
{
    if(NR==1&&head=="true"){
        print "<tr style=\"white-space: nowrap;text-align:left;\">";
        for(i=1;i<=NF;i++){
            if($i=="="){$i="";}
            print "<th>"$i"</th>";
        }
        print "</tr>";
    }
    else{
        print "<tr style=\"white-space: nowrap;text-align:left;\">";
        for(i=1;i<=NF;i++){
            if($i=="="){$i="";}
            print "<td>"$i"</td>";
        }
        print "</tr>";
    }
}
END{print "</table>";}' $1 > $2

echo "Generate done."

# "|": sep
# = : NULL
# Example:
#
# Index|Features|Model|InSpecAug|AM-Softmax (m=0.2)|Back-End|voxceleb1-O*|voxceleb1-O|voxceleb1-E|voxceleb1-H
# 1|mfcc23&pitch|extended x-vector|no|no|PLDA|1.622|2.089|2.221|3.842
# 2|fbank40&pitch|resnet34-2d|no|no|PLDA|1.909|3.065|2.392|3.912
# =|=|=|=|=|Cosine->+AS-Norm|2.158->-|2.423->2.344|2.215->2.01|4.873->3.734
# 3|fbank40&pitch|resnet34-2d|no|yes|PLDA|1.622|1.893|1.962|3.546
# =|=|=|=|=|Cosine->+AS-Norm|1.612->1.543|1.713->1.591|1.817->1.747|3.269->3.119
# 4|fbank40&pitch|resnet34-2d|yes|yes|PLDA|1.495|1.813|1.920|3.465
# =|=|=|=|=|Cosine->+AS-Norm|1.601->1.559|1.676->1.601|1.817->1.742|3.233->3.097
# 5|fbank80|resnet34-2d|no|yes|PLDA|1.511|1.808|1.847|3.251
# =|=|=|=|=|Cosine->+AS-Norm|1.538->-|1.628->1.538|1.767->1.705|3.111->2.985