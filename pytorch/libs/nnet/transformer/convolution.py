# -*- coding:utf-8 -*-

# Reference: https://github.com/espnet/espnet.

"""ConvolutionModule definition."""


from typing import Optional, Tuple
import torch
from torch import nn
from .scaling import ActivationBalancer,ScaledConv1d

class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        causal (int): Whether use causal convolution or not
    """

    def __init__(
        self, 
        channels: int, 
        kernel_size: int = 15, 
        activation:nn.Module = nn.ReLU(), 
        norm: str = 'batch_norm',
        causal: bool=False,
        activation_balancer: bool=False,
        bias: bool=True
    ):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()


        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # self.lorder is used to distinguish if it's a causal convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0            
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )
        assert norm in ['batch_norm', 'layer_norm', "basic_norm"]
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation
        self.balancer1,self.balancer2 = None,None
        self.activation_balancer = activation_balancer
        if activation_balancer:
            self.balancer1=ActivationBalancer(
            channel_dim=1, max_abs=10.0, min_positive=0.05, max_positive=1.0
            )
            self.balancer2=ActivationBalancer(
            channel_dim=1,  min_positive=0.05, max_positive=1.0
            )            

    def forward(self,
        x: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = None,
    )-> torch.Tensor:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time)
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # mask batch padding
        if mask_pad is not None:
            x.masked_fill_(~mask_pad, 0.0)
        if self.lorder>0:
            x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        if self.activation_balancer and self.balancer1 is not None:
            x = self.balancer1(x)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)

            
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.norm(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)

        if self.activation_balancer and self.balancer2 is not None:
            x = self.balancer2(x)

        x = self.activation(x) 
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad is not None:
            x.masked_fill_(~mask_pad, 0.0)
        return x.transpose(1, 2)

class ReConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        causal (int): Whether use causal convolution or not
    """

    def __init__(
        self, 
        channels: int, 
        kernel_size: int = 15, 
        activation:nn.Module = nn.ReLU(), 
        causal: bool=False,
        bias: bool=True
    ):
        """Construct an ConvolutionModule object."""
        super(ReConvolutionModule, self).__init__()


        self.pointwise_conv1 = ScaledConv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # self.lorder is used to distinguish if it's a causal convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0            
        self.depthwise_conv = ScaledConv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        self.pointwise_conv2 = ScaledConv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            initial_scale=0.25,
        )
        self.activation = activation

        self.balancer1=ActivationBalancer(
        channel_dim=1, max_abs=10.0, min_positive=0.05, max_positive=1.0
        )
        self.balancer2=ActivationBalancer(
        channel_dim=1,  min_positive=0.05, max_positive=1.0
        )            

    def forward(self,
        x: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = None,
    )-> torch.Tensor:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time)
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)
        # mask batch padding
        if mask_pad is not None:
            x.masked_fill_(~mask_pad, 0.0)
        if self.lorder>0:
            x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)

        x = self.balancer1(x)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)

        x = self.balancer2(x)

        x = self.activation(x) 
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad is not None:
            x.masked_fill_(~mask_pad, 0.0)
        return x.transpose(1, 2)