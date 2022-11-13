# -*- coding:utf-8 -*-

# Reference: https://github.com/espnet/espnet.

from turtle import xcor
import torch
from typing import Tuple
import torch.nn as nn
from .scaling import ScaledLinear,ScaledConv2d,ActivationBalancer
from libs.nnet.activation import DoubleSwish
from .layer_norm import BasicNorm
class TooShortUttError(Exception):
    """Raised when the utt is too short for subsampling.
    Args:
        message (str): Message for error catch
        actual_size (int): the short size that cannot pass the subsampling
        limit (int): the limit size for subsampling
    """

    def __init__(self, message, actual_size, limit):
        """Construct a TooShortUttError for error handler."""
        super().__init__(message)
        self.actual_size = actual_size
        self.limit = limit


def check_short_utt(ins, size):
    """Check if the utterance is too short for subsampling."""
    if isinstance(ins, Conv2dSubsampling2) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling4) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling6) and size < 11:
        return True, 11
    if isinstance(ins, Conv2dSubsampling8) and size < 15:
        return True, 15
    return False, -1

class LinearNoSubsampling(torch.nn.Module):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc_class (torch.nn.Module): Custom position encoding layer.

    """
    def __init__(self, idim: int, odim: int, dropout_rate: float,mlp_head: bool,
                 pos_enc_class: torch.nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            # torch.nn.LayerNorm(odim, eps=1e-5),
            # torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 1
        self.mlp_head = mlp_head
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, odim)) if self.mlp_head else torch.empty(0)
    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        if self.mlp_head:
            b,_,_ = x.shape
            cls_tokens = self.cls_token.repeat(b,1,1)

            x = torch.cat((cls_tokens,x),dim=1)

        x, pos_emb = self.pos_enc(x)
        return x, pos_emb, x_mask


class Conv2dSubsampling4(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc_class (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, 
                 mlp_head: bool, pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling4, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        )
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 4
        self.mlp_head = mlp_head
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, odim)) if self.mlp_head else torch.empty(0)
    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if self.mlp_head:
            b,_,_ = x.shape
            cls_tokens = self.cls_token.repeat(b,1,1)
            x = torch.cat((cls_tokens,x),dim=1)
        x, pos_emb = self.pos_enc(x)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.
        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.
        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]

class ReConv2dSubsampling4(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc_class (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, 
                 idim: int, 
                 odim: int, 
                 dropout_rate: float, 
                 mlp_head: bool, 
                 pos_enc_class: torch.nn.Module,
                 layer1_channels: int = 8,
                 layer2_channels: int = 32,
                 layer3_channels: int = 128,
    ):
        """Construct an Conv2dSubsampling object."""
        assert idim >= 7
        super(ReConv2dSubsampling4, self).__init__()
        self.conv = nn.Sequential(
            ScaledConv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=1,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
            ScaledConv2d(
                in_channels=layer1_channels,
                out_channels=layer2_channels,
                kernel_size=3,
                stride=2,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
            ScaledConv2d(
                in_channels=layer2_channels,
                out_channels=layer3_channels,
                kernel_size=3,
                stride=2,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
        )
        self.out = ScaledLinear(
            layer3_channels * (((idim - 1) // 2 - 1) // 2), odim
        )
        # set learn_eps=False because out_norm is preceded by `out`, and `out`
        # itself has learned scale, so the extra degree of freedom is not
        # needed.
        self.out_norm = BasicNorm(odim, learn_eps=False)
        # constrain median of output to be close to zero.
        self.out_balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55
        )        
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 4
        self.mlp_head = mlp_head
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, odim)) if self.mlp_head else torch.empty(0)
    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x = self.out_norm(x)
        x = self.out_balancer(x)
        if self.mlp_head:
            b,_,_ = x.shape
            cls_tokens = self.cls_token.repeat(b,1,1)
            x = torch.cat((cls_tokens,x),dim=1)

        x, pos_emb = self.pos_enc(x)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]

class SVConv2dSubsampling4(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc_class (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, 
                 mlp_head: bool, pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling object."""
        super(SVConv2dSubsampling4, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, (2,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, (2,1)),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (idim-4) , odim)
        )
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 4
        self.mlp_head = mlp_head
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, odim)) if self.mlp_head else torch.empty(0)
    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if self.mlp_head:
            b,_,_ = x.shape
            cls_tokens = self.cls_token.repeat(b,1,1)
            x = torch.cat((cls_tokens,x),dim=1)
        x, pos_emb = self.pos_enc(x)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.
        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.
        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc_class (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, mlp_head: bool, pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2)), odim))
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 2
        self.mlp_head = mlp_head
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, odim)) if self.mlp_head else torch.empty(0)
    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if self.mlp_head:
            b,_,_ = x.shape
            cls_tokens = self.cls_token.repeat(b,1,1)
            x = torch.cat((cls_tokens,x),dim=1)
        x, pos_emb = self.pos_enc(x)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:1]

    def __getitem__(self, key):
        """Get item.
        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.
        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]

class SVConv2dSubsampling2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc_class (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, mlp_head: bool, pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling2 object."""
        super(SVConv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, (2,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (idim-4), odim))
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 2
        self.mlp_head = mlp_head
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, odim)) if self.mlp_head else torch.empty(0)
    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if self.mlp_head:
            b,_,_ = x.shape
            cls_tokens = self.cls_token.repeat(b,1,1)
            x = torch.cat((cls_tokens,x),dim=1)
        x, pos_emb = self.pos_enc(x)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:1]
        # return x, pos_emb, x_mask[:, :, :-2:2]
    def __getitem__(self, key):
        """Get item.
        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.
        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]

class Conv2dSubsampling6(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, mlp_head: bool, pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling6 object."""
        super(Conv2dSubsampling6, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim),
        )
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 6
        self.mlp_head = mlp_head
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, odim)) if self.mlp_head else torch.empty(0)
    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if self.mlp_head:
            b,_,_ = x.shape
            cls_tokens = self.cls_token.repeat(b,1,1)
            x = torch.cat((cls_tokens,x),dim=1)

        x, pos_emb = self.pos_enc(x)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/8 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, mlp_head: bool, pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling8 object."""
        super(Conv2dSubsampling8, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        self.mlp_head = mlp_head
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, odim)) if self.mlp_head else torch.empty(0)
    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if self.mlp_head:
            b,_,_ = x.shape
            cls_tokens = self.cls_token.repeat(b,1,1)
            x = torch.cat((cls_tokens,x),dim=1)
        x, pos_emb = self.pos_enc(x)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]


class FrameMerger(nn.Module):
    def __init__(self, dim: int, 
                 normalize_before: bool = True,
                 norm_type: str = "layer_norm",):
        super().__init__()
        # self.conv = torch.nn.Sequential(
        #     torch.nn.Conv1d(dim, dim*2, 3,2),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv1d(dim*2, dim*2, 3, 2),
        #     torch.nn.ReLU(),
        # )
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(dim, 2*dim, 3,2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(2*dim, 2*dim, 3, 2),
            torch.nn.ReLU(),
        ) 
        # self.conv = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, dim, 3, (2,1)),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(dim, dim, 3, (2,1)),
        #     torch.nn.ReLU(),
        # )       
        
        self.normalize_before = normalize_before
        if self.normalize_before:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.LayerNorm(dim*2)

    def forward(self, x,pos_emb:torch.Tensor,mask:torch.Tensor,offset: int = 0):
        if self.normalize_before:
            x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        if not self.normalize_before:
            x = self.norm(x)

        return x,pos_emb[:, offset:offset + x.size(1)],mask[:, :, :-2:2][:, :, :-2:2]