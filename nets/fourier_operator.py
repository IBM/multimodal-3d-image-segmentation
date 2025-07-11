#
# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import numpy as np
import math

import torch
from torch.nn import Module, Parameter, init

__author__ = 'Ken C. L. Wong'


class FourierOperator(Module):
    """A layer that applies the convolution theorem through the Fourier transform.
    The input tensor is Fourier transformed, modified by the learnable weights in the
    frequency domain, and inverse Fourier transformed back to the spatial domain.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_modes: Number of frequency modes (k_max). Can be an int or a list of int (d, h, w) (default: None).
            Note that `num_modes` must be smaller than half of the input spatial size in each dimension.
        use_bias: If True, learned bias is added to the output tensor (default: False).
        weights_type: Type of weights in the frequency domain.
            Must be 'individual' or 'shared' (default: 'shared').
        use_transform: If True, the Fourier transform is used (default: True).
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
        else:  # rfft omits the negative frequencies in the last dimension
            assert self.num_modes is not None
            weight_shape = ((self.out_channels, self.in_channels) +
                            tuple(np.array(self.num_modes[:-1]) * 2) + self.num_modes[-1:])
        # PyTorch can handle complex parameters, but separating real and imag allows
        # counting parameters in terms of float numbers
        self.weight_real = Parameter(torch.empty(weight_shape, **factory_kwargs))
        self.weight_imag = Parameter(torch.empty(weight_shape, **factory_kwargs))

        if self.use_bias:
            self.bias = Parameter(torch.empty((1, self.out_channels) + (1,) * (ndim - 2), **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
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

        x = torch.fft.rfftn(x, dim=(-2, -1), norm='forward')

        weight = torch.complex(self.weight_real, self.weight_imag)

        if self.weights_type == 'shared':
            equation = 'oi,bihw->bohw'
            low = torch.einsum(equation, weight, x[..., :modes_0, :modes_1])
            high = torch.einsum(equation, weight, x[..., -modes_0:, :modes_1])
        else:
            equation = 'oihw,bihw->bohw'
            low = torch.einsum(equation, weight[..., :modes_0, :modes_1], x[..., :modes_0, :modes_1])
            high = torch.einsum(equation, weight[..., -modes_0:, :modes_1], x[..., -modes_0:, :modes_1])

        # Padding needs to be done manually as ifft only pads at the end
        pad_shape = (x.shape[0], self.out_channels, s0 - 2 * modes_0, modes_1)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([low, pad_zeros, high], dim=-2)

        if self.use_bias:
            x = x + self.bias

        x = torch.fft.irfftn(x, s=(-1, s1), dim=(-2, -1), norm='forward')

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

        x = torch.fft.rfftn(x, dim=(-3, -2, -1), norm='forward')

        weight = torch.complex(self.weight_real, self.weight_imag)

        if self.weights_type == 'shared':
            equation = 'oi,bidhw->bodhw'
            ll = torch.einsum(equation, weight, x[..., :modes_0, :modes_1, :modes_2])
            lh = torch.einsum(equation, weight, x[..., :modes_0, -modes_1:, :modes_2])
            hl = torch.einsum(equation, weight, x[..., -modes_0:, :modes_1, :modes_2])
            hh = torch.einsum(equation, weight, x[..., -modes_0:, -modes_1:, :modes_2])
        else:
            equation = 'oidhw,bidhw->bodhw'
            ll = torch.einsum(
                equation,
                weight[..., :modes_0, :modes_1, :modes_2], x[..., :modes_0, :modes_1, :modes_2]
            )
            lh = torch.einsum(
                equation,
                weight[..., :modes_0, -modes_1:, :modes_2], x[..., :modes_0, -modes_1:, :modes_2]
            )
            hl = torch.einsum(
                equation,
                weight[..., -modes_0:, :modes_1, :modes_2], x[..., -modes_0:, :modes_1, :modes_2]
            )
            hh = torch.einsum(
                equation,
                weight[..., -modes_0:, -modes_1:, :modes_2], x[..., -modes_0:, -modes_1:, :modes_2]
            )

        # Padding needs to be done manually as irfft only pads at the end

        # Padding along spatial dim 1, shape = (b, c, modes_0, s1, modes_2)
        pad_shape = (x.shape[0], self.out_channels, modes_0, s1 - 2 * modes_1, modes_2)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        low = torch.cat([ll, pad_zeros, lh], dim=-2)
        high = torch.cat([hl, pad_zeros, hh], dim=-2)

        # Padding along spatial dim 0, shape = (b, c, s0, s1, modes_2)
        pad_shape = (x.shape[0], self.out_channels, s0 - 2 * modes_0, s1, modes_2)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([low, pad_zeros, high], dim=-3)

        if self.use_bias:
            x = x + self.bias

        x = torch.fft.irfftn(x, s=(-1, -1, s2), dim=(-3, -2, -1), norm='forward')

        return x

    def _call2d_notransform(self, inputs):
        x = inputs
        weight = torch.complex(self.weight_real, self.weight_imag)
        equation = 'oi,bihw->bohw' if self.weights_type == 'shared' else 'oihw,bihw->bohw'
        return torch.einsum(equation, weight, x)

    def _call3d_notransform(self, inputs):
        x = inputs
        weight = torch.complex(self.weight_real, self.weight_imag)
        equation = 'oi,bidhw->bodhw' if self.weights_type == 'shared' else 'oidhw,bidhw->bodhw'
        return torch.einsum(equation, weight, x)
