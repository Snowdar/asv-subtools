# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-05-29)

import numpy as np

import torch
import torch.nn.functional as F

from libs.support.utils import to_device
from .components import *

## TopVirtualLoss ✿
class TopVirtualLoss(torch.nn.Module):
    """ This is a virtual loss class to be suitable for pipline scripts, such as train.py. And it requires
    to implement the function get_posterior to compute accuracy. But just using self.posterior to record the outputs
    before computing loss in forward is more convenient.
    For example,
        def forward(self, inputs, targets):
            outputs = softmax(inputs)
            self.posterior = outputs
            loss = CrossEntropy(outputs, targets)
        return loss
    It means that get_posterior should be called after forward.
    """

    def __init__(self, *args, **kwargs):
        super(TopVirtualLoss, self).__init__()
        self.posterior = None
        self.init(*args, **kwargs)

    def init(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *inputs):
        raise NotImplementedError

    def get_posterior(self):
        assert self.posterior is not None
        return self.posterior

#############################################

## Loss ✿
"""
Note, there are some principles about loss implements:
    In process: torch.nn.CrossEntropyLoss = softmax + log + torch.nn.NLLLoss()
    In function: torch.nn.NLLLoss() <-> - (sum(torch.tensor.gather())
so, in order to keep codes simple and efficient, do not using 'for' or any other complex grammar to implement what could be replaced by above.
"""

class SoftmaxLoss(TopVirtualLoss):
    """ An usual log-softmax loss with affine component.
    """
    def init(self, input_dim, num_targets, t=1, reduction='mean', special_init=False):
        self.affine = TdnnAffine(input_dim, num_targets)
        self.t = t # temperature
        # CrossEntropyLoss() has included the LogSoftmax, so do not add this function extra.
        self.loss_function = torch.nn.CrossEntropyLoss(reduction=reduction)

        # The special_init is not recommended in this loss component
        if special_init :
            torch.nn.init.xavier_uniform_(self.affine.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

    def forward(self, inputs, targets):
        """Final outputs should be a (N, C) matrix and targets is a (1,N) matrix where there are 
        N targets-indexes (index value belongs to 0~9 when target-class C = 10) for N examples rather than 
        using one-hot format directly.
        One example, one target.
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[2] == 1

        posterior = self.affine(inputs)
        self.posterior = posterior.detach()

        # The frames-index is 1 now 
        outputs = torch.squeeze(posterior, dim=2)
        return self.loss_function(outputs/self.t, targets)


class FocalLoss(TopVirtualLoss):
    """Implement focal loss according to [Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. 
    "Focal loss for dense object detection", IEEE international conference on computer vision, 2017.]
    """
    def init(self, input_dim, num_targets, gamma=2, reduction='sum', eps=1.0e-10):

        self.softmax_affine = SoftmaxAffineLayer(input_dim, num_targets, dim=1, log=False, bias=True)
        self.loss_function = torch.nn.NLLLoss(reduction=reduction)

        self.gamma = gamma
        # self.alpha = alpha
        self.eps = eps

    def forward(self, inputs, targets):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[2] == 1

        posterior = self.softmax_affine(inputs)
        self.posterior = posterior.detach()

        focal_posterior = (1 - posterior)**self.gamma * torch.log(posterior.clamp(min=self.eps))
        outputs = torch.squeeze(focal_posterior, dim=2)
        return self.loss_function(outputs, targets)


class MarginSoftmaxLoss(TopVirtualLoss):
    """MarginSoftmaxLoss
    """
    def init(self, input_dim, num_targets,
             m=0.2, s=30., t=1.,
             feature_normalize=True,
             method="am",
             mhe_loss=False, mhe_w=0.01,
             reduction='mean', eps=1.0e-10, init=True):

        self.input_dim = input_dim
        self.num_targets = num_targets
        self.weight = torch.nn.Parameter(torch.randn(num_targets, input_dim, 1))
        self.s = s # scale factor with feature normalization
        self.m = m # margin
        self.t = t # temperature
        self.feature_normalize = feature_normalize
        self.method = method # am | aam | sm1 | sm2 | sm3
        self.feature_normalize = feature_normalize
        self.mhe_loss = mhe_loss
        self.mhe_w = mhe_w

        self.eps = eps

        if feature_normalize :
            p_target = [0.9, 0.95, 0.99]
            suggested_s = [ (num_targets-1)/num_targets*np.log((num_targets-1)*x/(1-x)) for x in p_target ]

            if self.s < suggested_s[0]:
                print("Warning : using feature noamlization with small scalar s={s} could result in bad convergence. \
                There are some suggested s : {suggested_s} w.r.t p_target {p_target}.".format(
                    s=self.s, suggested_s=suggested_s, p_target=p_target))

        self.loss_function = torch.nn.CrossEntropyLoss(reduction=reduction)

        # Init
        if init:
             # torch.nn.init.xavier_normal_(self.weight, gain=1.0)
            torch.nn.init.normal_(self.weight, 0., 0.01) # it seems well

    def forward(self, inputs, targets):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[2] == 1

        batch_size = inputs.shape[0]

        normalized_x = F.normalize(inputs.squeeze(dim=2), dim=1)
        normalized_weight = F.normalize(self.weight.squeeze(dim=2), dim=1)
        cosine_theta = F.linear(normalized_x, normalized_weight) # Y = W*X

        if not self.feature_normalize :
            self.s = inputs.norm(2, dim=1) # [batch-size, l2-norm]

        self.posterior = (self.s * cosine_theta).detach().unsqueeze(2) # accuracy must be reported before margin penalty added

        ## Margin

        # cosine_theta [batch_size, num_class]
        # targets.unsqueeze(1) [batch_size, 1]
        cosine_theta_target = cosine_theta.gather(1, targets.unsqueeze(1))

        if self.method == "am":
            penalty_cosine_theta = cosine_theta_target - self.m
        elif self.method == "aam":
            # Another implementation w.r.t cosine(theta+m) = cosine_theta * cos_m - sin_theta * sin_m
            # penalty_cosine_theta = self.cos_m * cosine_theta_target - self.sin_m * torch.sqrt((1-cosine_theta_target**2).clamp(min=0.))
            penalty_cosine_theta = torch.cos(torch.acos(cosine_theta_target) + self.m)
        elif self.method == "sm1":
            # penalty_cosine_theta = cosine_theta_target - (1 - cosine_theta_target) * self.m
            penalty_cosine_theta = (1 + self.m) * cosine_theta_target - self.m
        elif self.method == "sm2":
            penalty_cosine_theta = cosine_theta_target - (1 - cosine_theta_target**2) * self.m
        elif self.method == "sm3":
            penalty_cosine_theta = cosine_theta_target - (1 - cosine_theta_target)**2 * self.m
        else:
            raise ValueError("Do not support this {0} margin w.r.t [ am | aam | sm1 | sm2 | sm3 ]".format(self.method))

        outputs = self.s * cosine_theta.scatter(1, targets.unsqueeze(1), penalty_cosine_theta)

        # Here reported loss will be always higher than softmax loss for the absolute margin penalty and 
        # it is a lie about why we can not decrease the loss to a mininum value. We 
        # should not report the loss after margin penalty but we really do that to avoid computing the 
        # training loss twice.
        if self.mhe_loss:
            sub_weight = normalized_weight - torch.index_select(normalized_weight, 0, targets).unsqueeze(dim=1)
            # [N, C]
            normed_sub_weight = sub_weight.norm(2, dim=2)
            mask = torch.full_like(normed_sub_weight, True, dtype=torch.bool).scatter_(1, targets.unsqueeze(dim=1), False)
            # [N, C-1]
            normed_sub_weight_clean = torch.masked_select(normed_sub_weight, mask).reshape(targets.size()[0], -1)
            # torch.mean means 1/(N*(C-1))
            the_mhe_loss = self.mhe_w * torch.mean((normed_sub_weight_clean**2).clamp(min=self.eps)**-1)

            return self.loss_function(outputs/self.t, targets) + the_mhe_loss
        else:
            return self.loss_function(outputs/self.t, targets)

    def extra_repr(self):
        return '(~affine): ~(input_dim={input_dim}, num_targets={num_targets}, method={method}, margin={m}, s={s}, t={t}, ' \
               'feature_normalize={feature_normalize}, mhe_loss={mhe_loss}, mhe_w={mhe_w}, eps={eps})'.format(**self.__dict__)


class LogisticAffinityLoss(TopVirtualLoss):
    """LogisticAffinityLoss.
    Reference: Peng, J., Gu, R., & Zou, Y. (2019). 
               LOGISTIC SIMILARITY METRIC LEARNING VIA AFFINITY MATRIX FOR TEXT-INDEPENDENT SPEAKER VERIFICATION. 
    """
    def init(self, init_w=5., init_b=-1., reduction='mean'):
        self.reduction = reduction

        self.w = torch.nn.Parameter(torch.tensor(init_w))
        self.b = torch.nn.Parameter(torch.tensor(init_b))

    def forward(self, inputs, targets):
        # This loss has no way to compute accuracy
        S = F.normalize(inputs.squeeze(dim=2), dim=1)
        A = torch.sigmoid(self.w * torch.mm(S, S.t()) + self.b) # This can not keep the diag-value equal to 1 and it maybe a question.

        targets_matrix = targets + torch.zeros_like(A)
        condition = targets_matrix - targets_matrix.t()
        outputs = -torch.log(torch.where(condition==0, A, 1-A))

        if self.reduction == 'sum':
            return outputs.sum()
        elif self.reduction == 'mean':
            return outputs.sum() / targets.shape[0]
        else:
            raise ValueError("Do not support this reduction {0}".format(self.reduction))