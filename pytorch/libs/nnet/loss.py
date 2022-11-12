# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-05-29)

import numpy as np
import math
import torch
import torch.nn.functional as F

import pandas as pd

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

    @utils.for_device_free
    def get_accuracy(self, targets):
        """
        @return: return accuracy
        """
        return self.compute_accuracy(self.get_posterior(), targets)

    @utils.for_device_free
    def predict(self, outputs):
        """
        @outputs: the outputs tensor with [batch-size,n,1] shape comes from affine before computing softmax or 
                  just softmax for n classes
        @return: an 1-dimensional vector including class-id (0-based) for prediction
        """
        with torch.no_grad():
            prediction = torch.squeeze(torch.argmax(outputs, dim=1))

        return prediction

    @utils.for_device_free
    def compute_accuracy(self, outputs, targets):
        """
        @outputs: the outputs tensor with [batch-size,n,1] shape comes from affine before computing softmax or 
                 just softmax for n classes
        @return: the float accuracy
        """
        assert outputs.shape[0] == len(targets)

        with torch.no_grad():
            prediction = self.predict(outputs)
            num_correct = (targets==prediction).sum()

        return num_correct.item()/len(targets)

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
    def init(self, input_dim, num_targets, t=1, reduction='mean', special_init=False,label_smoothing = 0.0):
        self.affine = TdnnAffine(input_dim, num_targets)
        self.t = t # temperature
        # CrossEntropyLoss() has included the LogSoftmax, so do not add this function extra.
        self.loss_function = torch.nn.CrossEntropyLoss(reduction=reduction,label_smoothing=label_smoothing)

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

        # The frames-index is 1 now.
        outputs = torch.squeeze(posterior, dim=2)
        return self.loss_function(outputs/self.t, targets)


class SoftmaxLoss_frame_phone_fix(TopVirtualLoss):
    #Zheng Li 2021-06-08
    """ An usual log-softmax loss with affine component.
    """
    def init(self, input_dim, num_targets, t=1, reduction='mean', special_init=False,label_smoothing = 0.0):
        self.affine = TdnnAffine(input_dim, num_targets)
        self.t = t # temperature
        self.num_phones = num_targets - 1
        # CrossEntropyLoss() has included the LogSoftmax, so do not add this function extra.
        self.loss_function = torch.nn.CrossEntropyLoss(reduction=reduction,label_smoothing=label_smoothing)

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
        posterior = self.affine(inputs)
        self.posterior = posterior.detach()
        outputs = torch.squeeze(posterior, dim=2)
        targets = targets.long()
        t = torch.squeeze(targets, dim=1)
        o = torch.squeeze(outputs, dim=2)

        t_max = torch.max(t).item()
        t_min = torch.min(t).item()
        num_phones = self.num_phones
        if t_max > num_phones or t_min < 0:
            zero = torch.zeros_like(t)
            t = torch.where(t > num_phones, zero, t)
            t = torch.where(t < 0, zero, t)
            loss = self.loss_function(o/self.t, t)
        else:
            loss = self.loss_function(o/self.t, t)

        return loss


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
    """Margin softmax loss.
    There are AM, AAM, Double-AM, SM1 (Snowdar Margin softmax loss), SM2 and SM3. 
    Do not provide A-softmax loss again for its complex implementation and margin limitation.
    Reference:
            [1] Liu, W., Wen, Y., Yu, Z., & Yang, M. (2016). Large-margin softmax loss for convolutional neural networks. 
                Paper presented at the ICML.

            [2] Liu, W., Wen, Y., Yu, Z., Li, M., Raj, B., & Song, L. (2017). Sphereface: Deep hypersphere embedding for 
                face recognition. Paper presented at the Proceedings of the IEEE conference on computer vision and pattern 
                recognition.

            [3] Wang, F., Xiang, X., Cheng, J., & Yuille, A. L. (2017). Normface: l2 hypersphere embedding for face 
                verification. Paper presented at the Proceedings of the 25th ACM international conference on Multimedia.

            [4] Wang, F., Cheng, J., Liu, W., & Liu, H. (2018). Additive margin softmax for face verification. IEEE Signal 
                Processing Letters, 25(7), 926-930.

            [5] Wang, H., Wang, Y., Zhou, Z., Ji, X., Gong, D., Zhou, J., . . . Liu, W. (2018). Cosface: Large margin cosine 
                loss for deep face recognition. Paper presented at the Proceedings of the IEEE Conference on Computer Vision 
                and Pattern Recognition.

            [6] Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). Arcface: Additive angular margin loss for deep face 
                recognition. Paper presented at the Proceedings of the IEEE Conference on Computer Vision and Pattern 
                Recognition.

            [7] Zhou, S., Chen, C., Han, G., & Hou, X. (2020). Double Additive Margin Softmax Loss for Face Recognition. 
                Applied Sciences, 10(1), 60. 
    """
    def init(self, input_dim, num_targets, 
             m=0.2, s=30., t=1.,
             feature_normalize=True,
             method="am",
             double=False,
             mhe_loss=False, mhe_w=0.01,
             inter_loss=0.,
             ring_loss=0.,
             curricular=False,
             reduction='mean', eps=1.0e-10, scale_init=False,label_smoothing = 0.0):

        self.input_dim = input_dim
        self.num_targets = num_targets
        self.weight = torch.nn.Parameter(torch.randn(num_targets, input_dim, 1))
        self.s = s # scale factor with feature normalization
        self.init_m = m
        self.m = m # margin
        self.t = t # temperature
        self.feature_normalize = feature_normalize
        self.method = method # am | aam | sm1 | sm2 | sm3 
        self.double = double
        self.feature_normalize = feature_normalize
        self.mhe_loss = mhe_loss
        self.mhe_w = mhe_w
        self.inter_loss = inter_loss
        self.ring_loss = ring_loss
        self.lambda_m = 1
        self.curricular = CurricularMarginComponent() if curricular else None

        if self.ring_loss > 0:
            self.r = torch.nn.Parameter(torch.tensor(20.))
            self.feature_normalize = False

        self.eps = eps

        if feature_normalize :
            p_target = [0.9, 0.95, 0.99]
            suggested_s = [ (num_targets-1)/num_targets*np.log((num_targets-1)*x/(1-x)) for x in p_target ]

            if self.s < suggested_s[0]:
                print("Warning : using feature noamlization with small scalar s={s} could result in bad convergence. \
                There are some suggested s : {suggested_s} w.r.t p_target {p_target}.".format(
                s=self.s, suggested_s=suggested_s, p_target=p_target))

        self.loss_function = torch.nn.CrossEntropyLoss(reduction=reduction,label_smoothing = label_smoothing)
        self.weight_scale =None
        self.scale_init=scale_init
        # Init weight.
        if scale_init:
            initial_scale = torch.tensor(1.0).log()
            self.weight_scale = torch.nn.Parameter(initial_scale.clone().detach())
            std = 0.1
            a = (3 ** 0.5) * std
            torch.nn.init.uniform_(self.weight, -a, a)
            fan_in = self.weight.shape[1] *self.weight[0][0].numel()
            scale = fan_in ** -0.5  # 1/sqrt(fan_in)
            with torch.no_grad():
                self.weight_scale += torch.tensor(scale / std).log()

        else:
             # torch.nn.init.xavier_normal_(self.weight, gain=1.0)
            torch.nn.init.normal_(self.weight, 0., 0.01) # It seems better.

    def get_weight(self):
        if self.scale_init:
            return self.weight * self.weight_scale.exp()
        else:
            return self.weight
    def forward(self, inputs, targets):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[2] == 1
        # with torch.cuda.amp.autocast(enabled=False):
        ## Normalize

        normalized_x = F.normalize(inputs.squeeze(dim=2), dim=1) # [512, 512] [batch_size, output_dim]
        normalized_weight = F.normalize(self.get_weight().squeeze(dim=2), dim=1) # [1000, 512] [target_num, output_dim]

        with torch.cuda.amp.autocast(enabled=False):
            cosine_theta = F.linear(normalized_x, normalized_weight) # Y = W*X

            if not self.feature_normalize :
                self.s = inputs.norm(2, dim=1) # [batch-size, l2-norm]
                # The accuracy must be reported before margin penalty added
                self.posterior = (self.s.detach() * cosine_theta.detach()).unsqueeze(2) 
            else:
                self.posterior = (self.s * cosine_theta.detach()).unsqueeze(2)

            if not self.training:
                # For valid set.
                outputs = self.s * cosine_theta
                return self.loss_function(outputs, targets)

            ## Margin Penalty
            # cosine_theta [batch_size, num_class]
            # targets.unsqueeze(1) [batch_size, 1]
            cosine_theta_target = cosine_theta.gather(1, targets.unsqueeze(1))

            if self.inter_loss > 0:
                inter_cosine_theta = torch.softmax(self.s * cosine_theta, dim=1)
                inter_cosine_theta_target = inter_cosine_theta.gather(1, targets.unsqueeze(1))
                inter_loss = torch.log((inter_cosine_theta.sum(dim=1) - inter_cosine_theta_target)/(self.num_targets - 1) + self.eps).mean()

            if self.method == "am":
                penalty_cosine_theta = cosine_theta_target - self.m
                if self.double:
                    double_cosine_theta = cosine_theta + self.m
            elif self.method == "aam":
                # Another implementation w.r.t cosine(theta+m) = cosine_theta * cos_m - sin_theta * sin_m
                # penalty_cosine_theta = self.cos_m * cosine_theta_target - self.sin_m * torch.sqrt((1-cosine_theta_target**2).clamp(min=0.))
                penalty_cosine_theta = torch.cos(torch.acos(cosine_theta_target) + self.m)
                if self.double:
                    double_cosine_theta = torch.cos(torch.acos(cosine_theta).add(-self.m))
            elif self.method == "sm1":
                # penalty_cosine_theta = cosine_theta_target - (1 - cosine_theta_target) * self.m
                penalty_cosine_theta = (1 + self.m) * cosine_theta_target - self.m
            elif self.method == "sm2":
                penalty_cosine_theta = cosine_theta_target - (1 - cosine_theta_target**2) * self.m
            elif self.method == "sm3":
                penalty_cosine_theta = cosine_theta_target - (1 - cosine_theta_target)**2 * self.m
            else:
                raise ValueError("Do not support this {0} margin w.r.t [ am | aam | sm1 | sm2 | sm3 ]".format(self.method))

            penalty_cosine_theta = self.lambda_m * penalty_cosine_theta + \
                                (1 - self.lambda_m) * cosine_theta_target

            if self.double:
                cosine_theta = self.lambda_m * double_cosine_theta + (1 - self.lambda_m) * cosine_theta

            if self.curricular is not None:
                cosine_theta = self.curricular(cosine_theta, cosine_theta_target, penalty_cosine_theta)

            # cosine_theta = cosine_theta.float()
            # penalty_cosine_theta = penalty_cosine_theta.float()

            outputs = self.s * cosine_theta.scatter(1, targets.unsqueeze(1), penalty_cosine_theta)

            ## Other extra loss
            # Final reported loss will be always higher than softmax loss for the absolute margin penalty and 
            # it is a lie about why we can not decrease the loss to a mininum value. We should not report the 
            # loss after margin penalty did but we really report this invalid loss to avoid computing the 
            # training loss twice.

            if self.ring_loss > 0:
                ring_loss = torch.mean((self.s - self.r)**2)/2
            else:
                ring_loss = 0.

            if self.mhe_loss:
                sub_weight = normalized_weight - torch.index_select(normalized_weight, 0, targets).unsqueeze(dim=1)
                # [N, C]
                normed_sub_weight = sub_weight.norm(2, dim=2)
                mask = torch.full_like(normed_sub_weight, True, dtype=torch.bool).scatter_(1, targets.unsqueeze(dim=1), False)
                # [N, C-1]
                normed_sub_weight_clean = torch.masked_select(normed_sub_weight, mask).reshape(targets.size()[0], -1)
                # torch.mean means 1/(N*(C-1))
                the_mhe_loss = self.mhe_w * torch.mean((normed_sub_weight_clean**2).clamp(min=self.eps)**-1)

                return self.loss_function(outputs/self.t, targets) + the_mhe_loss + self.ring_loss * ring_loss
            elif self.inter_loss > 0:
                return self.loss_function(outputs/self.t, targets) + self.inter_loss * inter_loss + self.ring_loss * ring_loss
            else:
                return self.loss_function(outputs/self.t, targets) + self.ring_loss * ring_loss
    
    def step(self, lambda_m, add_margin=None):
        self.lambda_m = lambda_m 
        if add_margin is not None:
            self.m = max(0,self.init_m+add_margin)


    def extra_repr(self):
        return '(~affine): (input_dim={input_dim}, num_targets={num_targets}, method={method}, double={double}, ' \
               'margin={m}, s={s}, t={t}, feature_normalize={feature_normalize}, mhe_loss={mhe_loss}, mhe_w={mhe_w}, ' \
               'eps={eps}, scale_init={scale_init})'.format(**self.__dict__)


#  Leo 2022-11-08
class MarginWarm(torch.nn.Module):
    def __init__(self,                 
                 start_epoch,
                 end_epoch,                 
                 offset_margin = 0.0,
                 init_lambda  = 1.0,
                 epoch_iter=None):
        '''
        between start_epoch and end_epoch, the offset_margin is
        exponentially increasing from offset_margin (usually negative) to 0.
        And the lambda_t is linearly increasing from init_lambda to 1.
        It is designed to control the margin_softmaxloss through `margin + offset_margin` and 
        `penalty_cosine_theta = lambda * penalty_cosine_theta + (1 - lambda) * cosine_theta_target`
        '''
        super().__init__()
        if end_epoch < start_epoch:
            raise  ValueError("End_epoch should not smaller then start_epoch, but got end_epoch: {}, start_epoch:{}"
                             .format(end_epoch,start_epoch))
        assert abs(init_lambda - 0.5)<=0.5,"init_lambda should be in [0, 1]"              
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.offset_margin = offset_margin
        self.init_lambda = init_lambda
        self.epoch_iter = epoch_iter
        if epoch_iter:
            self.update_step_range(epoch_iter)


    def update_step_range(self,epoch_iter,overwrite=False):
        if not overwrite and self.epoch_iter:
            raise ValueError("epoch_iter has been set as {}, but overwrite = {} now".format(self.epoch_iter,overwrite))
        else:
            self.epoch_iter = epoch_iter
        self.increase_start_iter = (self.start_epoch - 1) * epoch_iter
        self.fix_start_iter = (self.end_epoch - 1) * epoch_iter
        self.step_range = self.fix_start_iter - self.increase_start_iter

    def get_increase_margin(self, cur_step):
        initial_val = 1.0
        final_val = 1e-3

        cur_pos = cur_step - self.increase_start_iter

        ratio = math.exp(
            (cur_pos / self.step_range) *
            math.log(final_val / (initial_val + 1e-6))) * initial_val
        offset_margin = self.offset_margin * ratio
        lambda_t =  self.init_lambda + (cur_pos / self.step_range) * (1-self.init_lambda)

        return offset_margin, lambda_t

    def step(self, cur_step):
        if self.epoch_iter<0 or not isinstance(self.epoch_iter, int):
            raise  ValueError("MarginWarm expected positive integer epoch_iter, but got {}"
                             .format(self.epoch_iter))
        if cur_step >= self.fix_start_iter:
            return 0,1
        elif cur_step > self.increase_start_iter:
            return self.get_increase_margin(cur_step)
        else:
            return self.offset_margin,self.init_lambda

    def extra_repr(self):
        return 'start_epoch={start_epoch}, end_epoch={end_epoch}, epoch_iter={epoch_iter}, ' \
               'offset_margin={offset_margin} init_lambda={init_lambda})'.format(**self.__dict__)



class CurricularMarginComponent(torch.nn.Module):
    """CurricularFace is implemented as a called component for MarginSoftmaxLoss.
    Reference: Huang, Yuge, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, 
               and Feiyue Huang. 2020. “CurricularFace: Adaptive Curriculum Learning Loss for Deep Face 
               Recognition.” ArXiv E-Prints arXiv:2004.00288.
    Github: https://github.com/HuangYG123/CurricularFace. Note, the momentum of this github is a wrong value w.r.t
            the above paper. The momentum 't' should not increase so fast and I have corrected it as follow.

    By the way, it does not work in my experiments.
    """
    def __init__(self, momentum=0.01):
        super(CurricularMarginComponent, self).__init__()
        self.momentum = momentum
        self.register_buffer('t', torch.zeros(1))

    def forward(self, cosine_theta, cosine_theta_target, penalty_cosine_theta):
        with torch.no_grad():
            self.t = (1 - self.momentum) * cosine_theta_target.mean() + self.momentum * self.t

        mask = cosine_theta > penalty_cosine_theta
        hard_example = cosine_theta[mask]
        # Use clone to avoid problem "RuntimeError: one of the variables needed for gradient computation 
        # has been modified by an inplace operation"
        cosine_theta_clone = cosine_theta.clone()
        cosine_theta_clone[mask] = hard_example * (self.t + hard_example)

        return cosine_theta_clone


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


class MixupLoss(TopVirtualLoss):
    """Implement a mixup component to augment data and increase the generalization of model training.
    Reference: 
        [1] Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. n.d. Mixup: BEYOND EMPIRICAL RISK MINIMIZATION.
        [2] Zhu, Yingke, Tom Ko, and Brian Mak. 2019. “Mixup Learning Strategies for Text-Independent Speaker Verification.”

    Github: https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
    """
    def init(self, base_loss, mixup_component):
        
        self.base_loss = base_loss
        self.mixup_component = mixup_component

    def forward(self, inputs, targets):
        if self.training:
            lam = self.mixup_component.lam
            index = self.mixup_component.index

            loss = lam * self.base_loss(inputs, targets) + \
                (1 - lam) * self.base_loss(inputs, targets[index])
        else:
            loss = self.base_loss(inputs, targets)

        return loss

    def get_accuracy(self, targets):
        if self.training:
            # It is not very clear to compute accuracy for mixed data.
            lam = self.mixup_component.lam
            index = self.mixup_component.index
            return lam * self.compute_accuracy(self.base_loss.get_posterior(), targets) + \
                   (1 - lam) * self.compute_accuracy(self.base_loss.get_posterior(), targets[index])
        else:
            return self.compute_accuracy(self.base_loss.get_posterior(), targets)


