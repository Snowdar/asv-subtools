# -*- coding:utf-8 -*-

# Reference: https://github.com/espnet/espnet.

import torch
import torch.nn.functional as F

class Trans_Bat(torch.nn.BatchNorm1d):
    """BatchNorm1d  module

    :param int nout: output dim size
    :param int dim: dimension to be normalized
    :param bool transpose: transpose T and F.
    :param float eps: invalid, compatible with LayerNorm.
    """

    def __init__(self, nout, transpose=False,eps=1e-12,learnabel_affine: bool = True):
        super(Trans_Bat,self).__init__(nout)
        self.norm = torch.nn.BatchNorm1d(nout,affine=learnabel_affine)
        self.transpose = transpose
    def forward(self, x):
        """Apply BatchNorm1d normalization

        :param torch.Tensor x: input tensor
        :return: batch normalized tensor
        :rtype torch.Tensor
        """

        if self.transpose:
            return self.norm(x.transpose(1, -1)).transpose(1, -1)
        else:
            return self.norm(x)

# class LayerNorm(torch.nn.Module):
#     """Layer normalization module

#     :param int nout: output dim size
#     :param int dim: dimension to be normalized
#     """
#     def __init__(
#         self,
#         nout: int,
#         dim: int = -1,  # CAUTION: see documentation.
#         eps: float = 1e-5,
#         learnabel_affine: bool = True,
#     ) -> None: 
#         super(LayerNorm, self).__init__()  
#         self.dim = dim
         
#         self.norm = torch.nn.LayerNorm(nout,eps=eps,elementwise_affine=learnabel_affine)

#     def forward(self,x):
#         if self.dim == -1:
#             return self.norm(x)

#         return self.norm(x.transpose(1, -1)).transpose(1, -1)
class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module

    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1, eps=1e-5,learnabel_affine: bool = True):
        super(LayerNorm, self).__init__(nout, eps=eps,elementwise_affine=learnabel_affine)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization

        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps)
        return F.layer_norm(
            x.transpose(1, -1), self.normalized_shape, self.weight, self.bias, self.eps).transpose(1, -1)        
        #     return super().forward(x)
        # return super().forward(x.transpose(1, -1)).transpose(1, -1)
class BasicNorm(torch.nn.Module):
    """
    This is intended to be a simpler, and hopefully cheaper, replacement for
    LayerNorm.  The observation this is based on, is that Transformer-type
    networks, especially with pre-norm, sometimes seem to set one of the
    feature dimensions to a large constant value (e.g. 50), which "defeats"
    the LayerNorm because the output magnitude is then not strongly dependent
    on the other (useful) features.  Presumably the weight and bias of the
    LayerNorm are required to allow it to do this.
    So the idea is to introduce this large constant value as an explicit
    parameter, that takes the role of the "eps" in LayerNorm, so the network
    doesn't have to do this trick.  We make the "eps" learnable.
    Args:
       num_channels: the number of channels, e.g. 512.
      channel_dim: the axis/dimension corresponding to the channel,
        interprted as an offset from the input's ndim if negative.
        shis is NOT the num_channels; it should typically be one of
        {-2, -1, 0, 1, 2, 3}.
       eps: the initial "epsilon" that we add as ballast in:
             scale = ((input_vec**2).mean() + epsilon)**-0.5
          Note: our epsilon is actually large, but we keep the name
          to indicate the connection with conventional LayerNorm.
       learn_eps: if true, we learn epsilon; if false, we keep it
         at the initial value.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
        eps: float = 0.25,
        learn_eps: bool = True,
    ) -> None:
        super(BasicNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.tensor(eps).log().detach())
        else:
            self.register_buffer("eps", torch.tensor(eps).log().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scales = (
            torch.mean(x ** 2, dim=self.channel_dim, keepdim=True)
            + self.eps.exp()
        ) ** -0.5
        return x * scales