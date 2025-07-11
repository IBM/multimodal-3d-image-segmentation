#
# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import numpy as np
from typing import Union
import math

import torch
from torch.nn import Module, Parameter, init

from .dht import dht2, dht3

__author__ = 'Ken C. L. Wong'


class HartleyMultiHeadAttention(Module):
    """A layer that applies the Hartley multi-head attention.
    The input tensor is Hartley transformed and multi-head self-attention is applied
    in the frequency domain. The inverse Hartley transform converts the results back
    to the spatial domain.

    The number of output channels is given by `value_dim`. If it is None, `key_dim` is used.

    Args:
        in_channels: Number of input channels for query. It also represents `key_in_channels`
            or `value_in_channels` if they are None.
        key_dim: Number of output channels of each attention head for query and key.
            It also represents `value_dim` if it is None.
        num_heads: Number of attention heads.
        num_modes: Number of frequency modes (k_max). Can be an int or a list of int (d, h, w).
            Note that `num_modes` must be smaller than half of the input spatial size in each dimension,
            and must be divisible by `patch_size`.
        patch_size: Patch size for grouping in the frequency domain (default: None).
        attention_activation: Activation applied on the attention matrix (default: 'selu').
            If a str is provided, the activation from torch.nn.functional is used.
        value_dim: Number of output channels of each attention head for value (default: None).
            If None, `key_dim` is used.
        key_in_channels: Number of input channels for key. If None, `in_channels` is used (default: None).
        value_in_channels: Number of input channels for value. If None, `in_channels` is used (default: None).
        use_bias: If True, biases are added to the query, value, key, and output tensors (default: False).
        use_transform: If True, the Hartley transform is used (default: True).
            Otherwise, the inputs are already in the frequency domain.
        ndim: Input tensor dimension, i.e., 5 for 3D problems, 4 for 2D problems (default: 5).
        device: Device index to select, e.g., 'cuda', 'cpu' (default: None).
        dtype: Data type (default: None).
    """
    def __init__(self,
                 in_channels,
                 key_dim,
                 num_heads,
                 num_modes,
                 patch_size=None,
                 attention_activation: Union[str, callable] = 'selu',
                 value_dim=None,
                 key_in_channels=None,
                 value_in_channels=None,
                 use_bias=False,
                 use_transform=True,
                 ndim=5,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_channels = in_channels
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.num_modes = num_modes
        self.patch_size = patch_size
        self.attention_activation = attention_activation
        self.value_dim = value_dim or key_dim
        self.key_in_channels = key_in_channels or in_channels
        self.value_in_channels = value_in_channels or self.key_in_channels
        self.use_bias = use_bias
        self.use_transform = use_transform

        # Ensures self.num_modes is a tuple
        if np.isscalar(self.num_modes):
            self.num_modes = (self.num_modes,) * (ndim - 2)
        else:
            assert len(self.num_modes) == ndim - 2
            self.num_modes = tuple(self.num_modes)

        if np.isscalar(self.patch_size):
            self.patch_size = (self.patch_size,) * (ndim - 2)

        if isinstance(self.attention_activation, str):
            self.attention_activation = getattr(torch.nn.functional, self.attention_activation)

        self.weight_query = Parameter(torch.empty(
            (self.num_heads, self.key_dim, self.in_channels), **factory_kwargs))
        self.weight_key = Parameter(torch.empty(
            (self.num_heads, self.key_dim, self.key_in_channels), **factory_kwargs))
        self.weight_value = Parameter(torch.empty(
            (self.num_heads, self.value_dim, self.value_in_channels), **factory_kwargs))
        self.weight_out = Parameter(torch.empty(
            (self.value_dim, self.value_dim * self.num_heads), **factory_kwargs))

        if self.use_bias:
            self.bias_query = Parameter(torch.empty(
                (1, self.num_heads, self.key_dim) + (1,) * (ndim - 2), **factory_kwargs))
            self.bias_key = Parameter(torch.empty(
                (1, self.num_heads, self.key_dim) + (1,) * (ndim - 2), **factory_kwargs))
            self.bias_value = Parameter(torch.empty(
                (1, self.num_heads, self.value_dim) + (1,) * (ndim - 2), **factory_kwargs))
            self.bias_out = Parameter(torch.empty(
                (1, self.value_dim) + (1,) * (ndim - 2), **factory_kwargs))
        else:
            self.register_parameter('bias_query', None)
            self.register_parameter('bias_key', None)
            self.register_parameter('bias_value', None)
            self.register_parameter('bias_out', None)

        self.reset_parameters()

    def reset_parameters(self):
        self._reset_parameters(self.weight_query, self.bias_query)
        self._reset_parameters(self.weight_key, self.bias_key)
        self._reset_parameters(self.weight_value, self.bias_value)
        self._reset_parameters(self.weight_out, self.bias_out)

    @staticmethod
    def _reset_parameters(weight, bias):
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            init.zeros_(bias)

    def forward(self, inputs):
        if self.use_transform:
            return self._call(inputs)
        else:
            return self._call_notransform(inputs)

    def _call(self, inputs):
        if not isinstance(inputs, (tuple, list)):  # Single input
            spatial_shape = inputs.shape[2:]
            query = key = value = self.dht(inputs)
        elif len(inputs) == 2:
            spatial_shape = inputs[0].shape[2:]
            query = self.dht(inputs[0])
            key = value = self.dht(inputs[1])
        elif len(inputs) == 3:
            spatial_shape = inputs[0].shape[2:]
            query = self.dht(inputs[0])
            key = self.dht(inputs[1])
            value = self.dht(inputs[2])
        else:
            raise ValueError('Invalid inputs.')

        ndim = query.ndim
        assert ndim in (4, 5)

        # Ensures proper modes range
        if ndim == 4:
            s0, s1 = spatial_shape
            modes_0, modes_1 = self.num_modes
            assert s0 >= 2 * modes_0 and s1 >= 2 * modes_1
        else:
            s0, s1, s2 = spatial_shape
            modes_0, modes_1, modes_2 = self.num_modes
            assert s0 >= 2 * modes_0 and s1 >= 2 * modes_1 and s2 >= 2 * modes_2

        if ndim == 4:
            query = self.freq_conv2d(self.weight_query, query)  # (B, HEADS, C, H, W)
            key = self.freq_conv2d(self.weight_key, key)
            value = self.freq_conv2d(self.weight_value, value)
        else:
            query = self.freq_conv3d(self.weight_query, query)  # (B, HEADS, C, D, H, W)
            key = self.freq_conv3d(self.weight_key, key)
            value = self.freq_conv3d(self.weight_value, value)

        if self.use_bias:
            query = query + self.bias_query
            key = key + self.bias_key
            value = value + self.bias_value

        # Dimension reduction by grouping, (B, HEADS, C * prod(patch_size), num_d, num_h, num_w)
        if self.patch_size is not None:
            if ndim == 4:
                query = grouping2d(query, self.patch_size)
                key = grouping2d(key, self.patch_size)
                value = grouping2d(value, self.patch_size)
            else:
                query = grouping3d(query, self.patch_size)
                key = grouping3d(key, self.patch_size)
                value = grouping3d(value, self.patch_size)

        spatial_shape_freq = query.shape[3:]  # Spatial shape before flattening

        query = self.spatial_flatten(query)  # (B, HEADS, C * prod(patch_size), num_d * num_h * num_w)
        key = self.spatial_flatten(key)
        value = self.spatial_flatten(value)

        att = torch.einsum('bzcq,bzck->bzqk', query, key)
        att = att / np.sqrt(key.shape[2])
        if self.attention_activation is not None:
            att = self.attention_activation(att)

        output = torch.einsum('bzqk,bzck->bzcq', att, value)
        output = torch.reshape(output, (-1,) + output.shape[1:3] + spatial_shape_freq)

        # Get back the original spatial shape of query, (B, HEADS, C, D, H, W)
        if self.patch_size is not None:
            if ndim == 4:
                output = ungrouping2d(output, self.value_dim, self.patch_size)
            else:
                output = ungrouping3d(output, self.value_dim, self.patch_size)

        # Group heads and channels
        shape = output.shape
        output = torch.reshape(output, (-1,) + (shape[1] * shape[2],) + shape[3:])
        # Get MHA output
        equation = 'oi,bihw->bohw' if ndim == 4 else 'oi,bidhw->bodhw'
        output = torch.einsum(equation, self.weight_out, output)
        if self.use_bias:
            output = output + self.bias_out

        output = self.inverse2d(output, spatial_shape) if ndim == 4 else self.inverse3d(output, spatial_shape)

        return output

    def _call_notransform(self, inputs):
        if not isinstance(inputs, (tuple, list)):  # Single input
            query = key = value = inputs
        elif len(inputs) == 2:
            query = inputs[0]
            key = value = inputs[1]
        elif len(inputs) == 3:
            query = inputs[0]
            key = inputs[1]
            value = inputs[2]
        else:
            raise ValueError('Invalid inputs.')

        ndim = query.ndim
        assert ndim in (4, 5)

        if ndim == 4:
            query = self.freq_conv2d_notransform(self.weight_query, query)  # (B, HEADS, C, H, W)
            key = self.freq_conv2d_notransform(self.weight_key, key)
            value = self.freq_conv2d_notransform(self.weight_value, value)
        else:
            query = self.freq_conv3d_notransform(self.weight_query, query)  # (B, HEADS, C, D, H, W)
            key = self.freq_conv3d_notransform(self.weight_key, key)
            value = self.freq_conv3d_notransform(self.weight_value, value)

        if self.use_bias:
            query = query + self.bias_query
            key = key + self.bias_key
            value = value + self.bias_value

        # Dimension reduction by grouping, (B, HEADS, C * prod(patch_size), num_d, num_h, num_w)
        if self.patch_size is not None:
            if ndim == 4:
                query = grouping2d(query, self.patch_size)
                key = grouping2d(key, self.patch_size)
                value = grouping2d(value, self.patch_size)
            else:
                query = grouping3d(query, self.patch_size)
                key = grouping3d(key, self.patch_size)
                value = grouping3d(value, self.patch_size)

        spatial_shape_freq = query.shape[3:]  # Spatial shape before flattening

        query = self.spatial_flatten(query)  # (B, HEADS, C * prod(patch_size), num_d * num_h * num_w)
        key = self.spatial_flatten(key)
        value = self.spatial_flatten(value)

        att = torch.einsum('bzcq,bzck->bzqk', query, key)
        att = att / np.sqrt(key.shape[2])
        if self.attention_activation is not None:
            att = self.attention_activation(att)

        output = torch.einsum('bzqk,bzck->bzcq', att, value)
        output = torch.reshape(output, (-1,) + output.shape[1:3] + spatial_shape_freq)

        # Get back the original spatial shape of query, (B, HEADS, C, D, H, W)
        if self.patch_size is not None:
            if ndim == 4:
                output = ungrouping2d(output, self.value_dim, self.patch_size)
            else:
                output = ungrouping3d(output, self.value_dim, self.patch_size)

        # Group heads and channels
        shape = output.shape
        output = torch.reshape(output, (-1,) + (shape[1] * shape[2],) + shape[3:])
        # Get MHA output
        equation = 'oi,bihw->bohw' if ndim == 4 else 'oi,bidhw->bodhw'
        output = torch.einsum(equation, self.weight_out, output)
        if self.use_bias:
            output = output + self.bias_out

        return output

    def freq_conv2d(self, weight, x):
        equation = 'zoi,bihw->bzohw'  # z: heads
        modes_0, modes_1 = self.num_modes

        ll = torch.einsum(equation, weight, x[..., :modes_0, :modes_1])
        lh = torch.einsum(equation, weight, x[..., :modes_0, -modes_1:])
        hl = torch.einsum(equation, weight, x[..., -modes_0:, :modes_1])
        hh = torch.einsum(equation, weight, x[..., -modes_0:, -modes_1:])

        low = torch.cat([ll, lh], dim=-1)
        high = torch.cat([hl, hh], dim=-1)
        return torch.cat([low, high], dim=-2)

    def freq_conv3d(self, kernel, x):
        equation = 'zoi,bidhw->bzodhw'  # z: heads
        modes_0, modes_1, modes_2 = self.num_modes

        lll = torch.einsum(equation, kernel, x[..., :modes_0, :modes_1, :modes_2])
        lhl = torch.einsum(equation, kernel, x[..., :modes_0, -modes_1:, :modes_2])
        hll = torch.einsum(equation, kernel, x[..., -modes_0:, :modes_1, :modes_2])
        hhl = torch.einsum(equation, kernel, x[..., -modes_0:, -modes_1:, :modes_2])
        llh = torch.einsum(equation, kernel, x[..., :modes_0, :modes_1, -modes_2:])
        lhh = torch.einsum(equation, kernel, x[..., :modes_0, -modes_1:, -modes_2:])
        hlh = torch.einsum(equation, kernel, x[..., -modes_0:, :modes_1, -modes_2:])
        hhh = torch.einsum(equation, kernel, x[..., -modes_0:, -modes_1:, -modes_2:])

        # Combine along spatial dim 2
        ll = torch.cat([lll, llh], dim=-1)
        lh = torch.cat([lhl, lhh], dim=-1)
        hl = torch.cat([hll, hlh], dim=-1)
        hh = torch.cat([hhl, hhh], dim=-1)

        # Combine along spatial dim 1
        low = torch.cat([ll, lh], dim=-2)
        high = torch.cat([hl, hh], dim=-2)

        # Combine along spatial dim 0
        return torch.cat([low, high], dim=-3)

    @staticmethod
    def freq_conv2d_notransform(weight, x):
        equation = 'zoi,bihw->bzohw'  # z: heads
        return torch.einsum(equation, weight, x)

    @staticmethod
    def freq_conv3d_notransform(kernel, x):
        equation = 'zoi,bidhw->bzodhw'  # z: heads
        return torch.einsum(equation, kernel, x)

    def inverse2d(self, x, spatial_shape):
        s0, s1 = spatial_shape
        modes_0, modes_1 = self.num_modes

        ll = x[..., :modes_0, :modes_1]
        lh = x[..., :modes_0, -modes_1:]
        hl = x[..., -modes_0:, :modes_1]
        hh = x[..., -modes_0:, -modes_1:]

        # Padding
        pad_shape = x.shape[:2] + (modes_0, s1 - 2 * modes_1)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        low = torch.cat([ll, pad_zeros, lh], dim=-1)
        high = torch.cat([hl, pad_zeros, hh], dim=-1)

        pad_shape = x.shape[:2] + (s0 - 2 * modes_0, s1)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([low, pad_zeros, high], dim=-2)

        x = dht2(x, is_inverse=True)

        return x

    def inverse3d(self, x, spatial_shape):
        s0, s1, s2 = spatial_shape
        modes_0, modes_1, modes_2 = self.num_modes

        lll = x[..., :modes_0, :modes_1, :modes_2]
        lhl = x[..., :modes_0, -modes_1:, :modes_2]
        hll = x[..., -modes_0:, :modes_1, :modes_2]
        hhl = x[..., -modes_0:, -modes_1:, :modes_2]
        llh = x[..., :modes_0, :modes_1, -modes_2:]
        lhh = x[..., :modes_0, -modes_1:, -modes_2:]
        hlh = x[..., -modes_0:, :modes_1, -modes_2:]
        hhh = x[..., -modes_0:, -modes_1:, -modes_2:]

        # Padding needs to be done manually as ifft only pads at the end

        # Padding along spatial dim 2, shape = (b, c, modes_0, modes_1, s2)
        pad_shape = x.shape[:2] + (modes_0, modes_1, s2 - 2 * modes_2)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        ll = torch.cat([lll, pad_zeros, llh], dim=-1)
        lh = torch.cat([lhl, pad_zeros, lhh], dim=-1)
        hl = torch.cat([hll, pad_zeros, hlh], dim=-1)
        hh = torch.cat([hhl, pad_zeros, hhh], dim=-1)

        # Padding along spatial dim 1, shape = (b, c, modes_0, s1, s2)
        pad_shape = x.shape[:2] + (modes_0, s1 - 2 * modes_1, s2)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        low = torch.cat([ll, pad_zeros, lh], dim=-2)
        high = torch.cat([hl, pad_zeros, hh], dim=-2)

        # Padding along spatial dim 0, shape = (b, c, s0, s1, s2)
        pad_shape = x.shape[:2] + (s0 - 2 * modes_0, s1, s2)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([low, pad_zeros, high], dim=-3)

        x = dht3(x, is_inverse=True)

        return x

    @staticmethod
    def spatial_flatten(x):
        shape = (-1,) + x.shape[1:3] + (np.prod(x.shape[3:]),)
        return torch.reshape(x, shape)

    @staticmethod
    def dht(x):
        ndim = x.ndim
        assert ndim in (4, 5)
        if ndim == 4:
            return dht2(x)
        return dht3(x)


def grouping2d(x, patch_size):
    """Groups pixels into patches.

    Args:
        x: Input tensor of shape (batch_size, num_heads, num_channels, height, width).
        patch_size: Patch size.

    Returns:
        A reshaped tensor with each new pixel as a concatenation of the original pixels,
        with shape (batch_size, num_heads, num_grouped_channels, num_patches_h, num_patches_w).
    """
    assert len(patch_size) == 2

    patch_h, patch_w = patch_size
    _, z, c, h, w = x.shape  # z is num_heads

    assert h % patch_h == 0 and w % patch_w == 0
    num_h = h // patch_h
    num_w = w // patch_w

    x = torch.reshape(x, (-1, z, c, num_h, patch_h, num_w, patch_w))
    x = torch.permute(x, (0, 1, 2, 4, 6, 3, 5))  # (b, z, c, patch_h, patch_w, num_h, num_w)
    x = torch.reshape(x, (-1, z, c * patch_h * patch_w, num_h, num_w))

    return x


def ungrouping2d(x, num_channels, patch_size):
    """Ungroups patches back to pixels.

    Args:
        x: Input tensor of shape (batch_size, num_heads, num_grouped_channels, num_patches_h, num_patches_w).
        num_channels: Number of original channels before grouping.
        patch_size: Patch size used for grouping.

    Returns:
        A reshaped tensor with the original shape before grouping,
        with shape (batch_size, num_heads, num_channels, height, width).
    """
    assert len(patch_size) == 2

    patch_h, patch_w = patch_size
    c = num_channels
    _, z, _, num_h, num_w = x.shape

    x = torch.reshape(x, (-1, z, c, patch_h, patch_w, num_h, num_w))
    x = torch.permute(x, (0, 1, 2, 5, 3, 6, 4))  # (b, z, c, num_h, patch_h, num_w, patch_w)
    x = torch.reshape(x, (-1, z, c, num_h * patch_h, num_w * patch_w))

    return x


def grouping3d(x, patch_size):
    """Groups pixels into patches.

    Args:
        x: Input tensor of shape (batch_size, num_heads, num_channels, depth, height, width).
        patch_size: Patch size.

    Returns:
        A reshaped tensor with each new pixel as a concatenation of the original pixels,
        with shape (batch_size, num_heads, num_grouped_channels, num_patches_d, num_patches_h, num_patches_w).
    """
    assert len(patch_size) == 3

    patch_d, patch_h, patch_w = patch_size
    _, z, c, d, h, w = x.shape  # z is num_heads

    assert d % patch_d == 0 and h % patch_h == 0 and w % patch_w == 0
    num_d = d // patch_d
    num_h = h // patch_h
    num_w = w // patch_w

    x = torch.reshape(x, (-1, z, c, num_d, patch_d, num_h, patch_h, num_w, patch_w))
    x = torch.permute(x, (0, 1, 2, 4, 6, 8, 3, 5, 7))  # (b, z, c, patch_d, patch_h, patch_w, num_d, num_h, num_w)
    x = torch.reshape(x, (-1, z, c * patch_d * patch_h * patch_w, num_d, num_h, num_w))

    return x


def ungrouping3d(x, num_channels, patch_size):
    """Ungroups patches back to pixels.

    Args:
        x: Input tensor of shape
            (batch_size, num_heads, num_grouped_channels, num_patches_d, num_patches_h, num_patches_w).
        num_channels: Number of original channels before grouping.
        patch_size: Patch size used for grouping.

    Returns:
        A reshaped tensor with the original shape before grouping,
        with shape (batch_size, num_heads, num_channels, depth, height, width).
    """
    assert len(patch_size) == 3

    patch_d, patch_h, patch_w = patch_size
    c = num_channels
    _, z, _, num_d, num_h, num_w = x.shape

    x = torch.reshape(x, (-1, z, c, patch_d, patch_h, patch_w, num_d, num_h, num_w))
    x = torch.permute(x, (0, 1, 2, 6, 3, 7, 4, 8, 5))  # (b, z, c, num_d, patch_d, num_h, patch_h, num_w, patch_w)
    x = torch.reshape(x, (-1, z, c, num_d * patch_d, num_h * patch_h, num_w * patch_w))

    return x
