# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-05-21)

import torch

from thop.vision.basic_hooks import *
from .components import *
from .loss import *

# This rules are not right now.
custom_ops = {
    TdnnAffine: TdnnAffine.thop_count,
    ReluBatchNormTdnnLayer: zero_ops,
    StatisticsPooling: StatisticsPooling.thop_count,
    SoftmaxLoss: zero_ops,
    torch.nn.modules.loss.CrossEntropyLoss: zero_ops
}
