#
# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import numpy as np
from functools import partial
from typing import Union

import torch
from torch import nn

from .nets_utils import spatial_padcrop, ConvNormAct, init_weights_for_snn
from .hartley_operator import HartleyOperator
from .dht import dhtn

__author__ = 'Ken C. L. Wong'


class HNOSegXS(nn.Module):
    """This class is used to create the HNOSeg-XS architecture.
    Please refer to our IEEE-TMI 2025 paper for more details.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels of the model.
        filters: Number of filters for all layers except the output layer.
        num_transform_blocks: A list of int, each int represents the number of frequency-domain convolutions.
            The length of the list corresponds to :math:`n_B` (number of blocks) in our paper,
            and each element corresponds to :math:`n_{XS}` (number of frequency-domain convolutions).
        num_modes: Number of frequency modes (k_max). Can be an int or a list of int (d, h, w).
            Note that num_modes must be smaller than half of the image size in each dimension.
        weights_type: Type of weights in the frequency domain. Must be 'individual' or 'shared' (default: 'shared').
        use_resize: If True, learnable input downsampling and output upsampling are used (default: True).
        use_deep_supervision: If True, deep supervision is used (default: False).
        use_unet_skip: If True, the first half of the blocks are treated as encoding blocks, and
            their tensors are concatenated with the inputs to the decoding blocks (second half) via
            U-Net style skip connections (default: True).
        use_block_concat: If True, concat is used with the skip connection in a block.
            Otherwise, add is used (default: True).
        activation: Activation (default: 'selu').
        output_activation: Activation for the output, usually 'softmax' for segmentation (default: 'softmax').
        ndim: Dimension of input tensor, 4 for 2D and 5 for 3D (default: 5).
        device: Device index to select, e.g., 'cuda', 'cpu' (default: None).
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            filters,
            num_transform_blocks,
            num_modes,
            weights_type='shared',
            use_resize=True,
            use_deep_supervision=False,
            use_unet_skip=True,
            use_block_concat=True,
            activation='selu',
            output_activation: Union[str, callable] = 'softmax',
            ndim=5,
            device=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.num_transform_blocks = num_transform_blocks
        self.num_modes = num_modes
        self.weights_type = weights_type
        self.use_resize = use_resize
        self.use_deep_supervision = use_deep_supervision
        self.use_unet_skip = use_unet_skip
        self.use_block_concat = use_block_concat
        self.activation = activation
        self.output_activation = output_activation
        self.ndim = ndim
        self.device = device

        self.conv_in = None
        self.conv1 = None
        self.layers = None
        self.conv_out = None

        assert self.ndim in (4, 5)

        if np.isscalar(self.num_transform_blocks):
            self.num_transform_blocks = [self.num_transform_blocks]

        self.block = partial(HNOXSBlock, num_modes=self.num_modes, weights_type=self.weights_type, ndim=self.ndim,
                             activation=self.activation, device=self.device, use_block_concat=self.use_block_concat)

        self.create_layers()

    def create_layers(self):
        ds_out_channels = []  # For deep supervision
        encode_out_channels = dict()  # For unet skip connection

        cur_in_channels = self.in_channels  # Remember the current in_channels
        filters = self.filters

        if self.use_resize:
            self.conv_in = ConvNormAct(cur_in_channels, filters, kernel_size=2, stride=2, use_bias=True,
                                       activation=self.activation, ndim=self.ndim, device=self.device)
            cur_in_channels = filters

        self.conv1 = ConvNormAct(cur_in_channels, filters, use_bias=True, activation=self.activation, ndim=self.ndim,
                                 device=self.device)
        cur_in_channels = filters
        if self.use_deep_supervision:
            ds_out_channels.append(cur_in_channels)

        assert isinstance(self.num_transform_blocks, (list, tuple))
        self.layers = nn.ModuleList()
        num_blocks = len(self.num_transform_blocks)
        for i, num_convs in enumerate(self.num_transform_blocks):
            # Decoding. Always exclude i == num_blocks // 2, as it is the median if num_blocks is odd,
            # and its concatenating tensor is its input tensor if num_blocks is even
            if self.use_unet_skip and i > num_blocks // 2:
                cur_in_channels += encode_out_channels[num_blocks - 1 - i]

            self.layers.append(self.block(num_convs, cur_in_channels, filters))
            cur_in_channels = filters

            if self.use_deep_supervision:
                ds_out_channels.append(cur_in_channels)
            if self.use_unet_skip and i < num_blocks // 2:  # Encoding
                encode_out_channels[i] = cur_in_channels

        if ds_out_channels:
            cur_in_channels = sum(ds_out_channels)

        op = nn.Conv2d if self.ndim == 4 else nn.Conv3d
        self.conv_out = op(cur_in_channels, self.out_channels, kernel_size=1, bias=False, device=self.device)

        if isinstance(self.output_activation, str):
            dim = 1 if self.output_activation == 'softmax' else None
            self.output_activation = getattr(nn.functional, self.output_activation)
            if dim is not None:
                self.output_activation = partial(self.output_activation, dim=dim)

        if self.activation == 'selu' or self.activation == nn.functional.selu:
            self.apply(init_weights_for_snn)

    def forward(self, x):
        image_size = x.shape[2:]
        ds_tensors = []
        encode_tensors = dict()

        if self.use_resize:
            x = self.conv_in(x)

        x = self.conv1(x)
        if self.use_deep_supervision:
            ds_tensors.append(x)

        num_blocks = len(self.num_transform_blocks)
        for i, layer in enumerate(self.layers):
            # Decoding. Always exclude i == num_blocks // 2, as it is the median if num_blocks is odd,
            # and its concatenating tensor is its input tensor if num_blocks is even
            if self.use_unet_skip and i > num_blocks // 2:
                x = torch.cat([x, encode_tensors[num_blocks - 1 - i]], dim=1)

            x = layer(x)

            if self.use_deep_supervision:
                ds_tensors.append(x)
            if self.use_unet_skip and i < num_blocks // 2:  # Encoding
                encode_tensors[i] = x

        if ds_tensors:
            x = torch.cat(ds_tensors, dim=1)

        if self.use_resize:
            mode = 'bilinear' if self.ndim == 4 else 'trilinear'
            x = nn.functional.interpolate(x, size=image_size, mode=mode)

        x = self.conv_out(x)
        x = spatial_padcrop(x, image_size)  # To handle the spatial size mismatch after down- and up-sampling
        x = self.output_activation(x)

        return x


class HNOXSBlock(nn.Module):
    """This is the HNO-XS block with the block skip connection.

    Args:
        num_convs: Number of frequency-domain convolutions.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_modes: Number of frequency modes (k_max). Can be an int or a list of int (d, h, w).
            Note that num_modes must be smaller than half of the image size in each dimension.
        weights_type: Type of weights in the frequency domain. Must be 'individual' or 'shared' (default: 'shared').
        ndim: Dimension of input tensor, 4 for 2D and 5 for 3D (default: 5).
        activation: Activation (default: 'selu').
        device: Device index to select, e.g., 'cuda', 'cpu' (default: None).
        use_conv_branch: If True, the spatial convolution branch is used (default: False).
        use_block_concat: If True, concat is used by the block skip connections instead of add (default: True).
    """
    def __init__(
            self,
            num_convs,
            in_channels,
            out_channels,
            num_modes,
            weights_type='shared',
            ndim=5,
            activation='selu',
            device=None,
            use_conv_branch=False,
            use_block_concat=True,
    ):
        super().__init__()

        cur_in_channels = in_channels

        # Change the number of channels as the blocks use identity mapping.
        # Must be used with use_unet_skip.
        self.mapping_conv = None
        if cur_in_channels != out_channels:
            self.mapping_conv = ConvNormAct(cur_in_channels, out_channels, use_bias=True, activation=activation,
                                            ndim=ndim, device=device)
            cur_in_channels = out_channels

        self.transform_crop = TransformCrop(num_modes, ndim)

        # The n_{XS} frequency-domain convolutions
        self.conv_blocks = nn.ModuleList()
        for _ in range(num_convs):
            self.conv_blocks.append(
                NeuralOperatorBlock(cur_in_channels, out_channels, num_modes, weights_type, ndim, activation, device,
                                    use_conv_branch))
            cur_in_channels = out_channels

        self.pad_inverse = PadInverse(ndim)

        # SELU is self-normalizing, thus normalization is not required
        self.normalization = None
        if activation != 'selu' and activation != nn.functional.selu:
            self.normalization = nn.GroupNorm(1, cur_in_channels, device=device)

        self.activation = activation
        if isinstance(self.activation, str):
            self.activation = getattr(nn.functional, self.activation)

        self.conv_concat = None
        if use_block_concat:
            cur_in_channels += out_channels
            self.conv_concat = ConvNormAct(cur_in_channels, out_channels, use_bias=True, activation=activation,
                                           ndim=ndim, device=device)

    def forward(self, x):
        if self.mapping_conv is not None:
            x = self.mapping_conv(x)  # For changing the number of channels if needed

        tmp = x

        spatial_shape = x.shape[2:]
        x = self.transform_crop(x)
        for block in self.conv_blocks:  # The n_{XS} frequency-domain convolutions
            x = block(x)
        x = self.pad_inverse(x, spatial_shape)

        if self.normalization is not None:
            x = self.normalization(x)
        if self.activation is not None:
            x = self.activation(x)

        # The block skip connection.
        # This must be placed after normalization and activation.
        # Probably because of the intensity range of pad_inverse output.
        if self.conv_concat is not None:
            x = torch.cat([x, tmp], dim=1)
            x = self.conv_concat(x)
        else:
            x = x + tmp

        return x


class NeuralOperatorBlock(nn.Module):
    """This is for a single frequency-domain convolution."""
    def __init__(self, in_channels, out_channels, num_modes, weights_type, ndim, activation, device,
                 use_conv_branch=False):
        super().__init__()

        # When weights_type == 'shared', this is equivalent to convolution with the kernel size of 1 (Conv1).
        self.op = HartleyOperator(in_channels, out_channels, num_modes, use_bias=False, weights_type=weights_type,
                                  use_transform=False, ndim=ndim, device=device)

        self.conv_branch = None
        if use_conv_branch:
            op = nn.Conv2d if ndim == 4 else nn.Conv3d
            self.conv_branch = op(in_channels, out_channels, kernel_size=1, bias=False, device=device)

        # SELU is self-normalizing, thus normalization is not required
        self.normalization = None
        if activation != 'selu' and activation != nn.functional.selu:
            self.normalization = nn.GroupNorm(1, out_channels, device=device)

        # This activation is crucial
        self.activation = activation
        if isinstance(self.activation, str):
            self.activation = getattr(nn.functional, self.activation)

    def forward(self, x):
        x1 = self.op(x) if self.op is not None else None
        x2 = self.conv_branch(x) if self.conv_branch is not None else None

        assert x1 is not None or x2 is not None

        tmp = x

        if x1 is not None and x2 is not None:
            x = x1 + x2
        elif x1 is not None:
            x = x1
        else:
            x = x2

        x = x + tmp

        if self.normalization is not None:
            x = self.normalization(x)
        if self.activation is not None:  # This activation is crucial
            x = self.activation(x)

        return x


class TransformCrop(nn.Module):
    def __init__(self, num_modes, ndim):
        super().__init__()

        self.num_modes = num_modes

        assert ndim in (4, 5)
        dim = (-2, -1) if ndim == 4 else (-3, -2, -1)  # Axes to be transformed
        self.transform = partial(dhtn, dim=dim)

        # Ensures self.num_modes is a tuple
        if np.isscalar(self.num_modes):
            self.num_modes = (self.num_modes,) * (ndim - 2)
        else:
            assert len(self.num_modes) == ndim - 2
            self.num_modes = tuple(self.num_modes)

    def forward(self, x):
        if x.ndim == 4:
            return self._call2d(x)
        else:
            return self._call3d(x)

    def _call2d(self, inputs):
        s0, s1 = inputs.shape[2:]  # Spatial size
        modes_0, modes_1 = self.num_modes

        if modes_0 * 2 > s0:
            modes_0 = s0 // 2
        if modes_1 * 2 > s1:
            modes_1 = s1 // 2

        x = inputs

        x = self.transform(x)

        ll = x[..., :modes_0, :modes_1]
        lh = x[..., :modes_0, -modes_1:]
        hl = x[..., -modes_0:, :modes_1]
        hh = x[..., -modes_0:, -modes_1:]

        low = torch.cat([ll, lh], dim=-1)
        high = torch.cat([hl, hh], dim=-1)

        return torch.cat([low, high], dim=-2)

    def _call3d(self, inputs):
        s0, s1, s2 = inputs.shape[2:]  # Spatial size
        modes_0, modes_1, modes_2 = self.num_modes

        if modes_0 * 2 > s0:
            modes_0 = s0 // 2
        if modes_1 * 2 > s1:
            modes_1 = s1 // 2
        if modes_2 * 2 > s2:
            modes_2 = s2 // 2

        x = inputs

        x = self.transform(x)

        lll = x[..., :modes_0, :modes_1, :modes_2]
        lhl = x[..., :modes_0, -modes_1:, :modes_2]
        hll = x[..., -modes_0:, :modes_1, :modes_2]
        hhl = x[..., -modes_0:, -modes_1:, :modes_2]
        llh = x[..., :modes_0, :modes_1, -modes_2:]
        lhh = x[..., :modes_0, -modes_1:, -modes_2:]
        hlh = x[..., -modes_0:, :modes_1, -modes_2:]
        hhh = x[..., -modes_0:, -modes_1:, -modes_2:]

        ll = torch.cat([lll, llh], dim=-1)
        lh = torch.cat([lhl, lhh], dim=-1)
        hl = torch.cat([hll, hlh], dim=-1)
        hh = torch.cat([hhl, hhh], dim=-1)

        low = torch.cat([ll, lh], dim=-2)
        high = torch.cat([hl, hh], dim=-2)

        return torch.cat([low, high], dim=-3)


class PadInverse(nn.Module):
    def __init__(self, ndim):
        super().__init__()

        assert ndim in (4, 5)
        dim = (-2, -1) if ndim == 4 else (-3, -2, -1)  # Axes to be transformed
        self.transform = partial(dhtn, dim=dim, is_inverse=True)

    def forward(self, x, spatial_shape):
        if x.ndim == 4:
            return self._call2d(x, spatial_shape)
        else:
            return self._call3d(x, spatial_shape)

    def _call2d(self, inputs, spatial_shape):
        s0, s1 = spatial_shape  # Spatial size

        x = inputs

        modes_0 = inputs.shape[2] // 2
        modes_1 = inputs.shape[3] // 2
        assert s0 >= 2 * modes_0 and s1 >= 2 * modes_1

        ll = x[..., :modes_0, :modes_1]
        lh = x[..., :modes_0, -modes_1:]
        hl = x[..., -modes_0:, :modes_1]
        hh = x[..., -modes_0:, -modes_1:]

        pad_shape = (x.shape[0], x.shape[1], modes_0, s1 - 2 * modes_1)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        low = torch.cat([ll, pad_zeros, lh], dim=-1)
        high = torch.cat([hl, pad_zeros, hh], dim=-1)

        pad_shape = (x.shape[0], x.shape[1], s0 - 2 * modes_0, s1)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([low, pad_zeros, high], dim=-2)

        x = self.transform(x)

        return x

    def _call3d(self, inputs, spatial_shape):
        s0, s1, s2 = spatial_shape  # Spatial size

        x = inputs

        modes_0 = inputs.shape[2] // 2
        modes_1 = inputs.shape[3] // 2
        modes_2 = inputs.shape[4] // 2
        assert s0 >= 2 * modes_0 and s1 >= 2 * modes_1 and s2 >= 2 * modes_2

        lll = x[..., :modes_0, :modes_1, :modes_2]
        lhl = x[..., :modes_0, -modes_1:, :modes_2]
        hll = x[..., -modes_0:, :modes_1, :modes_2]
        hhl = x[..., -modes_0:, -modes_1:, :modes_2]
        llh = x[..., :modes_0, :modes_1, -modes_2:]
        lhh = x[..., :modes_0, -modes_1:, -modes_2:]
        hlh = x[..., -modes_0:, :modes_1, -modes_2:]
        hhh = x[..., -modes_0:, -modes_1:, -modes_2:]

        # Padding along spatial dim 2, shape = (b, c, modes_0, modes_1, s2)
        pad_shape = [x.shape[0], x.shape[1], modes_0, modes_1, s2 - 2 * modes_2]
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        ll = torch.cat([lll, pad_zeros, llh], dim=-1)
        lh = torch.cat([lhl, pad_zeros, lhh], dim=-1)
        hl = torch.cat([hll, pad_zeros, hlh], dim=-1)
        hh = torch.cat([hhl, pad_zeros, hhh], dim=-1)

        # Padding along spatial dim 1, shape = (b, c, modes_0, s1, s2)
        pad_shape = (x.shape[0], x.shape[1], modes_0, s1 - 2 * modes_1, s2)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        low = torch.cat([ll, pad_zeros, lh], dim=-2)
        high = torch.cat([hl, pad_zeros, hh], dim=-2)

        # Padding along spatial dim 0, shape = (b, c, s0, s1, s2)
        pad_shape = (x.shape[0], x.shape[1], s0 - 2 * modes_0, s1, s2)
        pad_zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([low, pad_zeros, high], dim=-3)

        x = self.transform(x)

        return x
