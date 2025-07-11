#
# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import numpy as np
import math

import torch
from torch.nn import Module, Parameter, init

from .dht import dht2, dht3

__author__ = 'Ken C. L. Wong'


class HartleyOperator(Module):
    """A layer that applies the convolution theorem through the Hartley transform.
    The input tensor is Hartley transformed, modified by the learnable weights in the
    frequency domain, and inverse Hartley transformed back to the spatial domain.

    Note that the "_notransform" functions are for HNOSeg-XS.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_modes: Number of frequency modes (k_max). Can be an int or a list of int (d, h, w) (default: None).
            Note that `num_modes` must be smaller than half of the input spatial size in each dimension.
        use_bias: If True, learned bias is added to the output tensor (default: False).
        weights_type: Type of weights in the frequency domain.
            Must be 'individual' or 'shared' (default: 'shared').
        use_transform: If True, the Hartley transform is used (default: True).
            Otherwise, the inputs are already in the frequency domain.
        ndim: Input tensor dimension, i.e., 5 for 3D problems, 4 for 2D problems (default: 5).
        device: Device index to select, e.g., 'cuda', 'cpu' (default: None).
        dtype: Data type (default: None).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_modes=None,
                 use_bias=False,
                 weights_type='shared',
                 use_transform=True,
                 ndim=5,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        valid_weights_type_str = {'individual', 'shared'}
        if weights_type not in valid_weights_type_str:
            raise ValueError(
                f'weights_type must be one of {valid_weights_type_str}')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_modes = num_modes
        self.use_bias = use_bias
        self.weights_type = weights_type
        self.use_transform = use_transform

        # Ensures self.num_modes is a tuple
        if self.num_modes is not None:
            if np.isscalar(self.num_modes):
                self.num_modes = (self.num_modes,) * (ndim - 2)
            else:
                assert len(self.num_modes) == ndim - 2
                self.num_modes = tuple(self.num_modes)

        if self.weights_type == 'shared':
            weight_shape = (self.out_channels, self.in_channels)
        else:
            assert self.num_modes is not None
            weight_shape = (self.out_channels, self.in_channels) + tuple(np.array(self.num_modes) * 2)
        self.weight = Parameter(torch.empty(weight_shape, **factory_kwargs))

        if self.use_bias:
            self.bias = Parameter(torch.empty((1, self.out_channels) + (1,) * (ndim - 2), **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, inputs):
        if self.use_transform:
            # Note that self.bias is added before inverse transform if not None
            if inputs.ndim == 4:
                x = self._call2d(inputs)
            else:
                x = self._call3d(inputs)
        else:
            if inputs.ndim == 4:
                x = self._call2d_notransform(inputs)
            else:
                x = self._call3d_notransform(inputs)

            if self.use_bias:
                x = x + self.bias

        return x

    def _call2d(self, inputs):
        s0, s1 = inputs.shape[2:]  # Spatial size
        modes_0, modes_1 = self.num_modes

        if self.weights_type == 'shared':
            if modes_0 * 2 > s0:
                modes_0 = s0 // 2
            if modes_1 * 2 > s1:
                modes_1 = s1 // 2
        else:
            assert s0 >= 2 * modes_0 and s1 >= 2 * modes_1

        x = inputs

        x = dht2(x)

        if self.weights_type == 'shared':
            weight = self.weight
            equation = 'oi,bihw->bohw'
            ll = torch.einsum(equation, weight, x[..., :modes_0, :modes_1])
            lh = torch.einsum(equation, weight, x[..., :modes_0, -modes_1:])
            hl = torch.einsum(equation, weight, x[..., -modes_0:, :modes_1])
            hh = torch.einsum(equation, weight, x[..., -modes_0:, -modes_1:])
        else:
            weight = self.weight
            weight_reverse = get_reverse(weight, [-2, -1])
            x_reverse = get_reverse(x, [-2, -1])
            equation = 'oihw,bihw->bohw'
            ll = hartley_conv(equation,
                              weight[..., :modes_0, :modes_1], weight_reverse[..., :modes_0, :modes_1],
                              x[..., :modes_0, :modes_1], x_reverse[..., :modes_0, :modes_1])
            lh = hartley_conv(equation,
                              weight[..., :modes_0, -modes_1:], weight_reverse[..., :modes_0, -modes_1:],
                              x[..., :modes_0, -modes_1:], x_reverse[..., :modes_0, -modes_1:])
            hl = hartley_conv(equation,
                              weight[..., -modes_0:, :modes_1], weight_reverse[..., -modes_0:, :modes_1],
                              x[..., -modes_0:, :modes_1], x_reverse[..., -modes_0:, :modes_1])
            hh = hartley_conv(equation,
                              weight[..., -modes_0:, -modes_1:], weight_reverse[..., -modes_0:, -modes_1:],
                              x[..., -modes_0:, -modes_1:], x_reverse[..., -modes_0:, -modes_1:])

        pad_shape = (x.shape[0], self.out_channels, modes_0, s1 - 2 * modes_1)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        low = torch.cat([ll, pad_zeros, lh], dim=-1)
        high = torch.cat([hl, pad_zeros, hh], dim=-1)

        pad_shape = (x.shape[0], self.out_channels, s0 - 2 * modes_0, s1)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([low, pad_zeros, high], dim=-2)

        if self.use_bias:
            x = x + self.bias

        # This improves the operator by providing nonlinearity
        x = torch.nn.functional.selu(x)

        x = dht2(x, is_inverse=True)

        return x

    def _call3d(self, inputs):
        s0, s1, s2 = inputs.shape[2:]  # Spatial size
        modes_0, modes_1, modes_2 = self.num_modes

        if self.weights_type == 'shared':
            if modes_0 * 2 > s0:
                modes_0 = s0 // 2
            if modes_1 * 2 > s1:
                modes_1 = s1 // 2
            if modes_2 * 2 > s2:
                modes_2 = s2 // 2
        else:
            assert s0 >= 2 * modes_0 and s1 >= 2 * modes_1 and s2 >= 2 * modes_2

        x = inputs

        x = dht3(x)

        if self.weights_type == 'shared':
            weight = self.weight
            equation = 'oi,bidhw->bodhw'
            lll = torch.einsum(equation, weight, x[..., :modes_0, :modes_1, :modes_2])
            lhl = torch.einsum(equation, weight, x[..., :modes_0, -modes_1:, :modes_2])
            hll = torch.einsum(equation, weight, x[..., -modes_0:, :modes_1, :modes_2])
            hhl = torch.einsum(equation, weight, x[..., -modes_0:, -modes_1:, :modes_2])
            llh = torch.einsum(equation, weight, x[..., :modes_0, :modes_1, -modes_2:])
            lhh = torch.einsum(equation, weight, x[..., :modes_0, -modes_1:, -modes_2:])
            hlh = torch.einsum(equation, weight, x[..., -modes_0:, :modes_1, -modes_2:])
            hhh = torch.einsum(equation, weight, x[..., -modes_0:, -modes_1:, -modes_2:])
        else:
            weight = self.weight
            weight_reverse = get_reverse(weight, [-3, -2, -1])
            x_reverse = get_reverse(x, [-3, -2, -1])
            equation = 'oidhw,bidhw->bodhw'
            lll = hartley_conv(
                equation,
                weight[..., :modes_0, :modes_1, :modes_2], weight_reverse[..., :modes_0, :modes_1, :modes_2],
                x[..., :modes_0, :modes_1, :modes_2], x_reverse[..., :modes_0, :modes_1, :modes_2]
            )
            lhl = hartley_conv(
                equation,
                weight[..., :modes_0, -modes_1:, :modes_2], weight_reverse[..., :modes_0, -modes_1:, :modes_2],
                x[..., :modes_0, -modes_1:, :modes_2], x_reverse[..., :modes_0, -modes_1:, :modes_2]
            )
            hll = hartley_conv(
                equation,
                weight[..., -modes_0:, :modes_1, :modes_2], weight_reverse[..., -modes_0:, :modes_1, :modes_2],
                x[..., -modes_0:, :modes_1, :modes_2], x_reverse[..., -modes_0:, :modes_1, :modes_2]
            )
            hhl = hartley_conv(
                equation,
                weight[..., -modes_0:, -modes_1:, :modes_2], weight_reverse[..., -modes_0:, -modes_1:, :modes_2],
                x[..., -modes_0:, -modes_1:, :modes_2], x_reverse[..., -modes_0:, -modes_1:, :modes_2]
            )
            llh = hartley_conv(
                equation,
                weight[..., :modes_0, :modes_1, -modes_2:], weight_reverse[..., :modes_0, :modes_1, -modes_2:],
                x[..., :modes_0, :modes_1, -modes_2:], x_reverse[..., :modes_0, :modes_1, -modes_2:]
            )
            lhh = hartley_conv(
                equation,
                weight[..., :modes_0, -modes_1:, -modes_2:], weight_reverse[..., :modes_0, -modes_1:, -modes_2:],
                x[..., :modes_0, -modes_1:, -modes_2:], x_reverse[..., :modes_0, -modes_1:, -modes_2:]
            )
            hlh = hartley_conv(
                equation,
                weight[..., -modes_0:, :modes_1, -modes_2:], weight_reverse[..., -modes_0:, :modes_1, -modes_2:],
                x[..., -modes_0:, :modes_1, -modes_2:], x_reverse[..., -modes_0:, :modes_1, -modes_2:]
            )
            hhh = hartley_conv(
                equation,
                weight[..., -modes_0:, -modes_1:, -modes_2:], weight_reverse[..., -modes_0:, -modes_1:, -modes_2:],
                x[..., -modes_0:, -modes_1:, -modes_2:], x_reverse[..., -modes_0:, -modes_1:, -modes_2:]
            )

        # Padding along spatial dim 2, shape = (b, c, modes_0, modes_1, s2)
        pad_shape = [x.shape[0], self.out_channels, modes_0, modes_1, s2 - 2 * modes_2]
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        ll = torch.cat([lll, pad_zeros, llh], dim=-1)
        lh = torch.cat([lhl, pad_zeros, lhh], dim=-1)
        hl = torch.cat([hll, pad_zeros, hlh], dim=-1)
        hh = torch.cat([hhl, pad_zeros, hhh], dim=-1)

        # Padding along spatial dim 1, shape = (b, c, modes_0, s1, s2)
        pad_shape = (x.shape[0], self.out_channels, modes_0, s1 - 2 * modes_1, s2)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        low = torch.cat([ll, pad_zeros, lh], dim=-2)
        high = torch.cat([hl, pad_zeros, hh], dim=-2)

        # Padding along spatial dim 0, shape = (b, c, s0, s1, s2)
        pad_shape = (x.shape[0], self.out_channels, s0 - 2 * modes_0, s1, s2)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([low, pad_zeros, high], dim=-3)

        if self.use_bias:
            x = x + self.bias

        # This activation is crucial for improving the accuracy of the operator
        # by providing nonlinearity in the frequency domain.
        x = torch.nn.functional.selu(x)

        x = dht3(x, is_inverse=True)

        return x

    def _call2d_notransform(self, inputs):
        x = inputs
        if self.weights_type == 'shared':
            weight = self.weight
            equation = 'oi,bihw->bohw'
            return torch.einsum(equation, weight, x)
        else:
            # NOTE: reverse after cropping has a single difference at the highest negative frequency in each dimension
            weight = self.weight
            weight_reverse = get_reverse(weight, [-2, -1])
            x_reverse = get_reverse(x, [-2, -1])
            equation = 'oihw,bihw->bohw'
            return hartley_conv(equation, weight, weight_reverse, x, x_reverse)

    def _call3d_notransform(self, inputs):
        x = inputs
        if self.weights_type == 'shared':
            weight = self.weight
            equation = 'oi,bidhw->bodhw'
            return torch.einsum(equation, weight, x)
        else:
            # NOTE: reverse after cropping has a single difference at the highest negative frequency in each dimension
            weight = self.weight
            weight_reverse = get_reverse(weight, [-3, -2, -1])
            x_reverse = get_reverse(x, [-3, -2, -1])
            equation = 'oidhw,bidhw->bodhw'
            return hartley_conv(equation, weight, weight_reverse, x, x_reverse)


def hartley_conv(equation, weight, weight_reverse, x, x_reverse):
    """Applies Hartley convolution theorem in the frequency domain.

    Args:
        equation: An equation that describes how kernel and data (x) interact.
        weight: A weight matrix.
        weight_reverse: `weight` with the frequency axes reversed.
        x: Data
        x_reverse: `x` with the frequency axes reversed.

    Returns:
        A tensor in the frequency domain.
    """
    h1 = torch.einsum(equation, weight, x + x_reverse)
    h2 = torch.einsum(equation, weight_reverse, x - x_reverse)
    return (h1 + h2) * 0.5


def get_reverse(x, dims):
    """Get x[N-k] by 'reverse then roll by 1'.
    'reverse' converts x[k] to x[N-1-k], then 'roll by 1' changes x[N-1-k] to x[N-k] as x[0] = x[N].

    Args:
        x: Input tensor.
        dims: Which dimensions to reverse and roll. Must be a list or tuple.

    Returns:
        x[N-k]
    """
    assert isinstance(dims, (list, tuple))
    shifts = [1] * len(dims)
    return torch.roll(torch.flip(x, dims), shifts, dims)
