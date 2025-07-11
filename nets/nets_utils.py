#
# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Helper functions for creating architectures.

Author: Ken C. L. Wong
"""

import torch
from torch import nn
import numpy as np
from typing import List

from .fourier_operator import FourierOperator
from .hartley_operator import HartleyOperator

__author__ = 'Ken C. L. Wong'


def spatial_padcrop(x: torch.Tensor, target_shape: List[int]):
    """Performs spatial cropping and/or padding.
    Nothing is done if the shapes are already matched.

    Args:
        x: The tensor to be reshaped.
        target_shape: Target shape.

    Returns:
        A reshaped tensor.
    """
    ndim = x.ndim
    assert ndim in (3, 4, 5) and ndim == len(target_shape) + 2
    padding, cropping = get_spatial_padcrop(x, target_shape)

    if sum(padding) != 0:
        x = torch.nn.functional.pad(x, padding)

    if sum(cropping) != 0:
        if ndim == 3:
            l, u = cropping
            u = x.shape[-1] - u  # To handle 0
            x = x[..., l:u]
        elif ndim == 4:
            wl, wu, hl, hu = cropping
            wu = x.shape[-1] - wu
            hu = x.shape[-2] - hu
            x = x[..., hl:hu, wl:wu]
        else:
            wl, wu, hl, hu, dl, du = cropping
            wu = x.shape[-1] - wu
            hu = x.shape[-2] - hu
            du = x.shape[-3] - du
            x = x[..., dl:du, hl:hu, wl:wu]

    return x


def get_spatial_padcrop(x: torch.Tensor, target_shape: List[int]):
    """Computes the amount needed to be padded and cropped.

    Args:
        x: The tensor to be reshaped.
        target_shape: Target shape.

    Returns:
        The padding and cropping tuples.
    """
    shape = x.shape[2:]

    ndim = len(shape)

    if target_shape == shape:
        return (0, 0) * ndim, (0, 0) * ndim

    diff = [t - s for t, s in zip(target_shape, shape)]

    # Regardless of dimension, at most one padding and one cropping is enough
    padding = [0, 0] * ndim
    cropping = [0, 0] * ndim
    for i, d in enumerate(diff[::-1]):  # PyTorch padding input is in reversed order
        if d >= 0:
            q = d // 2
            if d % 2 == 0:
                padding[i * 2] = padding[i * 2 + 1] = q
            else:
                padding[i * 2] = q
                padding[i * 2 + 1] = q + 1
        else:
            d = -d
            q = d // 2
            if d % 2 == 0:
                cropping[i * 2] = cropping[i * 2 + 1] = q
            else:
                cropping[i * 2] = q
                cropping[i * 2 + 1] = q + 1

    return padding, cropping


def init_weights_for_snn(module):
    """Initializes weights of a layer for a self-normalizing neural network (SNN).

    Args:
        module: A layer module.
    """
    target_cls = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, HartleyOperator)
    if isinstance(module, target_cls):
        nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
        if module.bias is not None:
            nn.init.uniform_(module.bias, -0.001, 0.001)
    elif isinstance(module, FourierOperator):
        nn.init.kaiming_normal_(module.weight_real, nonlinearity='linear')
        nn.init.kaiming_normal_(module.weight_imag, nonlinearity='linear')
        if module.bias is not None:
            nn.init.uniform_(module.bias, -0.001, 0.001)


class _OpNormAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = None
        self.normalization = None
        self.activation = None

    def forward(self, x):
        x = self.op(x)
        if self.normalization is not None:
            x = self.normalization(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvNormAct(_OpNormAct):
    """A module that performs convolution, layer normalization (optional), and activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size (default: 1).
        stride: Stride (default: 1).
        use_bias: Use bias or not (default: True).
        activation: Activation (default: 'selu').
        use_snn: If True, self-normalizing neural network (SNN) is used, thus no normalization (default: True).
        ndim: Dimension of input tensor, 4 for 2D and 5 for 3D (default: 5).
        device: Device index to select, e.g., 'cuda', 'cpu' (default: None).
    """
    def __init__(self, in_channels, out_channels, *, kernel_size=1, stride=1, use_bias=True, activation='selu',
                 use_snn=True, ndim=5, device=None):
        super().__init__()

        assert ndim in (4, 5)

        if np.all(np.array(stride) == 1):
            padding = 'same'
        elif np.isscalar(kernel_size):
            padding = kernel_size // 2
        else:
            padding = np.array(kernel_size) // 2
        op = nn.Conv2d if ndim == 4 else nn.Conv3d
        self.op = op(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias, device=device)

        if use_snn:  # No normalization for self-normalizing neural network
            # SNN needs SELU to function properly
            if activation != 'selu' and activation != nn.functional.selu:
                raise RuntimeError('Self-normalizing neural network (SNN) must be used with SELU.')
        else:
            self.normalization = nn.GroupNorm(1, out_channels, device=device)

        self.activation = activation
        if isinstance(self.activation, str):
            self.activation = getattr(nn.functional, self.activation)


class ConvTransposeNormAct(_OpNormAct):
    """A module that performs transposed convolution, layer normalization (optional), and activation.
    Only works for stride=2.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size (default: 2).
        use_bias: Use bias or not (default: True).
        activation: Activation (default: 'selu').
        ndim: Dimension of input tensor, 4 for 2D and 5 for 3D (default: 5).
        device: Device index to select, e.g., 'cuda', 'cpu' (default: None).
    """
    def __init__(self, in_channels, out_channels, *, kernel_size=2, use_bias=True, activation='selu',
                 ndim=5, device=None):
        super().__init__()

        assert ndim in (4, 5)
        if np.isscalar(kernel_size):
            padding = kernel_size // 2
        else:
            padding = np.array(kernel_size) // 2
        output_padding = 1
        stride = 2
        op = nn.ConvTranspose2d if ndim == 4 else nn.ConvTranspose3d
        self.op = op(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=use_bias,
                     device=device)

        # SELU is self-normalizing, thus normalization is not required
        if activation != 'selu' and activation != nn.functional.selu:
            self.normalization = nn.GroupNorm(1, out_channels, device=device)

        self.activation = activation
        if isinstance(self.activation, str):
            self.activation = getattr(nn.functional, self.activation)
