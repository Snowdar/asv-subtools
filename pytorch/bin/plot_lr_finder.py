# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-10-17)

import sys
import argparse
import pandas as pd

sys.path.insert(0, 'subtools/pytorch')

import libs.support.figure as figure
import libs.support.utils as utils


csv = sys.argv[1]
savepng = sys.argv[2]

df1 = pd.read_csv("exp/standard_xv_baseline_warmR_voxceleb1_v2/log/lr_finder_wd1e-1.csv")
df2 = pd.read_csv("exp/standard_xv_baseline_warmR_voxceleb1_v2/log/lr_finder_wd1.0.csv")

log_lrs1 = df1["log_lr"]
train_loss1 = df1["valid_loss"]
#valid_loss = df["valid_loss"]
log_lrs2 = df2["log_lr"]
train_loss2 = df2["valid_loss"]

data = [(log_lrs1[5:-400], train_loss1[5:-400], {"label":"wd=0.1"}),
        (log_lrs2[5:-400], train_loss2[5:-400], {"label":"wd=1.0"})]

figure.plot_lr_finder(data, "exp/standard_xv_baseline_warmR_voxceleb1_v2/log/wd.png")



