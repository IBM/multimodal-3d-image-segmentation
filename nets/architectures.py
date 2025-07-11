#
# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Different network architectures for image segmentation.
All architectures work for both 2D and 3D.

Author: Ken C. L. Wong
"""

from functools import partial
from typing import Union

import torch
from torch import nn

from .nets_utils import spatial_padcrop, ConvNormAct, ConvTransposeNormAct, init_weights_for_snn
from .fourier_operator import FourierOperator
from .hartley_operator import HartleyOperator
from .hartley_mha import HartleyMultiHeadAttention

__author__ = 'Ken C. L. Wong'


class VNetDS(nn.Module):
    """A network architecture modified from V-Net with deep supervision.
    Learnable input downsampling and output upsampling can also be used.

    Please refer to our MICCAI 2018 paper for more details:

    Wong, K.C.L., Moradi, M., Tang, H., Syeda-Mahmood, T. (2018). 3D Segmentation
    with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes. In:
    Medical Image Computing and Computer Assisted Intervention – MICCAI 2018.
    https://doi.org/10.1007/978-3-030-00931-1_70

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels of the model.
        base_num_filters: The number of filters of each convolutional layer before the first downsampling.
        num_blocks: A list of int specifying the encoding path, e.g., [1, 2, 3] means
            Conv -> Downsample -> Conv -> Conv -> Downsample -> Conv -> Conv -> Conv.
            The decoding path is the reverse without the last int.
        use_resize: If True, learnable input downsampling and output upsampling are used (default: True).
        right_leg_indexes: A list of int specifying the right leg from the decoding path (deep supervision).
            E.g., [0, 1, 2, 3, 4]. Set to None if deep supervision is not used (default: None).
        kernel_size: Kernel size for convolutional layers (default: 3).
        activation: Activation (default: 'elu').
        use_snn: If True, self-normalizing neural network (SNN) is used (default: False).
        output_activation: Activation for the output, usually 'softmax' for segmentation (default: 'softmax').
        use_residual: If True, residual connections are used (default: True).
        ndim: Dimension of input tensor, 4 for 2D and 5 for 3D (default: 5).
        device: Device index to select, e.g., 'cuda', 'cpu' (default: None).
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            base_num_filters,
            num_blocks,
            use_resize=True,
            right_leg_indexes=None,
            kernel_size=3,
            activation='elu',
            use_snn=False,
            output_activation='softmax',
            use_residual=True,
            ndim=5,
            device=None,
    ):
        super().__init__()
        assert isinstance(num_blocks, (list, tuple))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.use_resize = use_resize
        self.right_leg_indexes = right_leg_indexes
        self.output_activation = output_activation
        self.use_residual = use_residual
        self.ndim = ndim

        if self.right_leg_indexes is None:
            self.right_leg_indexes = [0]

        assert self.ndim in (4, 5)
        num_sections = len(self.num_blocks)

        conv = partial(ConvNormAct, stride=1, use_bias=True, activation=activation, use_snn=use_snn, ndim=self.ndim,
                       device=device)

        cur_in_channels = in_channels  # Remember the current in_channels

        self.conv_in = None
        if self.use_resize:
            self.conv_in = ConvNormAct(cur_in_channels, base_num_filters, kernel_size=2, stride=2, use_bias=True,
                                       activation=activation, use_snn=use_snn, ndim=self.ndim, device=device)
            cur_in_channels = base_num_filters

        encode_out_channels = dict()
        right_leg_out_channels = dict()

        # Encoding layers
        self.encode_layers = nn.ModuleDict()  # key: section number (str); val: ModuleList
        for i in range(num_sections):
            layers = nn.ModuleList()
            filters = base_num_filters * (2 ** i)

            tmp_in_channels = cur_in_channels if self.use_residual else None

            for _ in range(self.num_blocks[i]):
                layers.append(conv(cur_in_channels, filters, kernel_size=kernel_size))
                cur_in_channels = filters

            if self.use_residual:
                layers.append(conv(tmp_in_channels, filters, kernel_size=1))
                cur_in_channels = filters

            if i != num_sections - 1:
                encode_out_channels[i] = filters
                layers.append(  # Downsampling
                    ConvNormAct(cur_in_channels, filters, kernel_size=kernel_size, stride=2, use_bias=True,
                                activation=activation, use_snn=use_snn, ndim=self.ndim, device=device))
                cur_in_channels = filters
            elif i in self.right_leg_indexes:
                right_leg_out_channels[i] = cur_in_channels

            self.encode_layers[str(i)] = layers

        self.decode_layers = nn.ModuleDict()  # key: section number (str); val: ModuleList
        for i in reversed(range(num_sections - 1)):
            layers = nn.ModuleList()
            filters = base_num_filters * (2 ** i)

            layers.append(  # Upsampling
                ConvTransposeNormAct(cur_in_channels, filters, kernel_size=kernel_size, use_bias=True,
                                     activation=activation, ndim=self.ndim, device=device))
            cur_in_channels = filters

            cur_in_channels = cur_in_channels + encode_out_channels[i]  # Concat with encoding tensor

            tmp_in_channels = cur_in_channels if self.use_residual else None

            for _ in range(self.num_blocks[i]):
                layers.append(conv(cur_in_channels, filters, kernel_size=kernel_size))
                cur_in_channels = filters

            if self.use_residual:
                layers.append(conv(tmp_in_channels, filters, kernel_size=1))
                cur_in_channels = filters

            if i in self.right_leg_indexes:
                right_leg_out_channels[i] = cur_in_channels

            self.decode_layers[str(i)] = layers

        # Right leg
        self.conv_ds = None
        if len(right_leg_out_channels) == 1:
            cur_in_channels = right_leg_out_channels[0]
        else:
            cur_in_channels = sum(right_leg_out_channels.values())
            # To avoid OOM
            self.conv_ds = ConvNormAct(cur_in_channels, self.out_channels, use_bias=True, activation=activation,
                                       use_snn=use_snn, ndim=self.ndim, device=device)
            cur_in_channels = self.out_channels

        op = nn.Conv2d if self.ndim == 4 else nn.Conv3d
        self.conv_out = op(cur_in_channels, self.out_channels, kernel_size=1, bias=False, device=device)

        if isinstance(self.output_activation, str):
            dim = 1 if self.output_activation == 'softmax' else None
            self.output_activation = getattr(nn.functional, self.output_activation)
            if dim is not None:
                self.output_activation = partial(self.output_activation, dim=dim)

        self.encode_tensors = None
        self.right_leg = None

        if use_snn and (activation == 'selu' or activation == nn.functional.selu):
            self.apply(init_weights_for_snn)

    def forward(self, x):
        image_size = x.shape[2:]

        self.encode_tensors = dict()
        self.right_leg = dict()

        if self.use_resize:
            x = self.conv_in(x)

        x = self.encode(x)
        x = self.decode(x)

        if self.use_resize:
            mode = 'bilinear' if self.ndim == 4 else 'trilinear'
            x = nn.functional.interpolate(x, size=image_size, mode=mode)

        x = self.conv_out(x)
        x = spatial_padcrop(x, image_size)  # To handle the spatial size mismatch after down- and up-sampling
        x = self.output_activation(x)

        return x

    def encode(self, x):
        num_sections = len(self.num_blocks)
        for i in range(num_sections):
            layers_iter = iter(self.encode_layers[str(i)])  # noqa

            tmp = x if self.use_residual else None

            for _ in range(self.num_blocks[i]):
                x = next(layers_iter)(x)

            if tmp is not None:
                x = x + next(layers_iter)(tmp)

            if i != num_sections - 1:
                self.encode_tensors[i] = x
                x = next(layers_iter)(x)  # Downsampling
            elif i in self.right_leg_indexes:
                self.right_leg[i] = x

        return x

    def decode(self, x):
        num_sections = len(self.num_blocks)
        for i in reversed(range(num_sections - 1)):
            layers_iter = iter(self.decode_layers[str(i)])  # noqa

            x = next(layers_iter)(x)  # Upsampling
            x = spatial_padcrop(x, self.encode_tensors[i].shape[2:])  # Handle spatial shape difference
            x = torch.cat([x, self.encode_tensors[i]], dim=1)

            tmp = x if self.use_residual else None

            for _ in range(self.num_blocks[i]):
                x = next(layers_iter)(x)

            if tmp is not None:
                x = x + next(layers_iter)(tmp)

            if i in self.right_leg_indexes:
                self.right_leg[i] = x

        if len(self.right_leg) == 1:
            x = self.right_leg[0]
        else:
            x = torch.cat(upsampling(self.right_leg), dim=1)
            x = self.conv_ds(x)

        return x


class _TransSeg(nn.Module):
    """The parent class of NeuralOperatorSeg and HartleyMHASeg
        as they have similar block architectures.
    """
    def __init__(self):
        super().__init__()

        # Arguments from child classes
        self.in_channels = None
        self.out_channels = None
        self.filters = None
        self.num_transform_blocks = None
        self.use_resize = None
        self.use_deep_supervision = None
        self.activation = None
        self.output_activation = None
        self.ndim = None
        self.device = None
        self.block = None

        # PyTorch operations
        self.conv_in = None
        self.conv1 = None
        self.layers = None
        self.conv_out = None
        self.conv_ds = None

    def create_layers(self):
        ds_out_channels = []  # For deep supervision

        cur_in_channels = self.in_channels  # Remember the current in_channels

        if self.use_resize:
            self.conv_in = ConvNormAct(cur_in_channels, self.filters, kernel_size=2, stride=2, use_bias=True,
                                       activation=self.activation, ndim=self.ndim, device=self.device)
            cur_in_channels = self.filters

        self.conv1 = ConvNormAct(cur_in_channels, self.filters, use_bias=True, activation=self.activation,
                                 ndim=self.ndim, device=self.device)
        cur_in_channels = self.filters
        if self.use_deep_supervision:
            ds_out_channels.append(cur_in_channels)

        self.layers = nn.ModuleList()
        filters = self.filters
        for _ in range(self.num_transform_blocks):
            self.layers.append(self.block(cur_in_channels, filters))
            cur_in_channels = filters
            if self.use_deep_supervision:
                ds_out_channels.append(cur_in_channels)

        if ds_out_channels:
            cur_in_channels = sum(ds_out_channels)
            # To avoid OOM
            self.conv_ds = ConvNormAct(cur_in_channels, self.out_channels, use_bias=True, activation=self.activation,
                                       ndim=self.ndim, device=self.device)
            cur_in_channels = self.out_channels

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
        tensors = []

        if self.use_resize:
            x = self.conv_in(x)

        x = self.conv1(x)
        if self.use_deep_supervision:
            tensors.append(x)

        for layer in self.layers:
            x = layer(x)
            if self.use_deep_supervision:
                tensors.append(x)

        if tensors:
            x = torch.cat(tensors, dim=1)
            x = self.conv_ds(x)

        if self.use_resize:
            mode = 'bilinear' if self.ndim == 4 else 'trilinear'
            x = nn.functional.interpolate(x, size=image_size, mode=mode)

        x = self.conv_out(x)
        x = spatial_padcrop(x, image_size)  # To handle the spatial size mismatch after down- and up-sampling
        x = self.output_activation(x)

        return x


class NeuralOperatorSeg(_TransSeg):
    """A family of architectures related to the Fourier neural operator (FNO).
    This class can be used to create the FNO, FNOSeg, and HNOSeg architectures
    by different combinations of input arguments.

    Please refer to our IEEE-TMI 2025 paper for more details.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels of the model.
        filters: Number of filters for all layers except the output layer.
        num_transform_blocks: Number of transform (FNO or HNO) blocks (int).
        num_modes: Number of frequency modes (k_max). Can be an int or a list of int (d, h, w).
            Note that num_modes must be smaller than half of the image size in each dimension.
        transform_type: Type of transformation. Can be 'Fourier' or 'Hartley'.
        weights_type: Type of weights in the frequency domain. Must be 'individual' or 'shared' (default: 'shared').
        use_resize: If True, learnable input downsampling and output upsampling are used (default: True).
        use_deep_supervision: If True, deep supervision is used (default: False).
        use_bias_conv_branch: If True, bias is used in the conv_branch in each block (default: False).
        use_block_skip: If True, skip connection is used in a block (default: True).
        use_block_concat: If True, concat is used by the block skip connections instead of add (default: True).
            Only used when `use_block_skip` is True.
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
            transform_type,
            weights_type='shared',
            use_resize=True,
            use_deep_supervision=False,
            use_bias_conv_branch=False,
            use_block_skip=True,
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
        self.transform_type = transform_type
        self.weights_type = weights_type
        self.use_resize = use_resize
        self.use_deep_supervision = use_deep_supervision
        self.use_bias_conv_branch = use_bias_conv_branch
        self.use_block_skip = use_block_skip
        self.use_block_concat = use_block_concat
        self.activation = activation
        self.output_activation = output_activation
        self.ndim = ndim
        self.device = device

        assert self.transform_type in ('Fourier', 'Hartley')
        assert self.ndim in (4, 5)

        self.block = partial(NeuralOperatorBlock, num_modes=self.num_modes, transform_type=self.transform_type,
                             weights_type=self.weights_type, ndim=self.ndim, activation=self.activation,
                             device=self.device, use_bias_conv_branch=self.use_bias_conv_branch,
                             use_block_skip=self.use_block_skip, use_block_concat=self.use_block_concat)

        self.create_layers()


class HartleyMHASeg(_TransSeg):
    """HartleyMHA architecture.
    This architecture comprises layers that perform self-attention in the frequency domain.

    Please refer to our MICCAI 2023 paper for more details.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels of the model.
        filters: Number of filters for all layers except the output layer.
        num_transform_blocks: Number of transform (Hartley MHA) blocks (int).
        num_heads: Number of attention heads.
        num_modes: Number of frequency modes (k_max). Can be an int or a list of int (d, h, w).
            Note that `num_modes` must be smaller than half of the input spatial size in each dimension,
            and must be divisible by `patch_size`.
        patch_size: Patch size for grouping in the frequency domain.
        attention_activation: Activation applied on the attention matrix (default: 'selu').
        use_resize: If True (default), learnable input downsampling and output upsampling are used.
        use_deep_supervision: If True (default), deep supervision is used.
        use_bias_conv_branch: If True, bias is used in the conv_branch in each block (default: False).
        use_block_skip: If True, skip connection is used in a block (default: True).
        use_block_concat: If True, concat is used by the block skip connections instead of add (default: True).
            Only used when use_block_skip is True.
        activation: Activation (default: 'selu').
        output_activation: Activation for the output, usually 'softmax' (default) for segmentation.
        ndim: Dimension of input tensor, 4 for 2D and 5 for 3D (default: 5).
        device: Device index to select, e.g., 'cuda', 'cpu' (default: None).
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            filters,
            num_transform_blocks,
            num_heads,
            num_modes,
            patch_size,
            attention_activation='selu',
            use_resize=True,
            use_deep_supervision=True,
            use_bias_conv_branch=False,
            use_block_skip=True,
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
        self.num_heads = num_heads
        self.num_modes = num_modes
        self.patch_size = patch_size
        self.attention_activation = attention_activation
        self.use_resize = use_resize
        self.use_deep_supervision = use_deep_supervision
        self.use_bias_conv_branch = use_bias_conv_branch
        self.use_block_skip = use_block_skip
        self.use_block_concat = use_block_concat
        self.activation = activation
        self.output_activation = output_activation
        self.ndim = ndim
        self.device = device

        assert self.ndim in (4, 5)

        self.block = partial(HartleyMHABlock, num_heads=self.num_heads, num_modes=self.num_modes,
                             patch_size=self.patch_size, attention_activation=self.attention_activation, ndim=self.ndim,
                             activation=self.activation, device=self.device,
                             use_bias_conv_branch=self.use_bias_conv_branch, use_block_skip=self.use_block_skip,
                             use_block_concat=self.use_block_concat)

        self.create_layers()


class _TransBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = None
        self.conv_branch = None
        self.normalization = None
        self.activation = None
        self.use_block_skip = None
        self.conv_concat = None

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

        if self.normalization is not None:
            x = self.normalization(x)
        if self.activation is not None:
            x = self.activation(x)

        if self.use_block_skip:
            if self.conv_concat is not None:
                x = torch.cat([x, tmp], dim=1)
                x = self.conv_concat(x)
            else:
                x = x + tmp

        return x


class NeuralOperatorBlock(_TransBlock):
    """This is the FNO/HNO block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_modes: Number of frequency modes (k_max). Can be an int or a list of int (d, h, w).
            Note that num_modes must be smaller than half of the image size in each dimension.
        transform_type: Type of transformation. Can be 'Fourier' or 'Hartley'.
        weights_type: Type of weights in the frequency domain. Must be 'individual' or 'shared' (default: 'shared').
        ndim: Dimension of input tensor, 4 for 2D and 5 for 3D (default: 5).
        activation: activation: Activation (default: 'selu').
        device: Device index to select, e.g., 'cuda', 'cpu' (default: None).
        use_conv_branch: If True, the spatial convolution branch is used (default: True).
        use_bias_conv_branch: If True, bias is used in the conv_branch in each block (default: False).
        use_block_skip: If True, skip connection is used in a block (default: True).
        use_block_concat: If True, concat is used by the block skip connections instead of add (default: True).
            Only used when `use_block_skip` is True.
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            num_modes,
            transform_type,
            weights_type='shared',
            ndim=5,
            activation='selu',
            device=None,
            use_conv_branch=True,
            use_bias_conv_branch=False,
            use_block_skip=True,
            use_block_concat=True,
    ):
        super().__init__()

        assert transform_type in ('Fourier', 'Hartley')
        self.use_block_skip = use_block_skip

        op = FourierOperator if transform_type == 'Fourier' else HartleyOperator
        self.op = op(in_channels, out_channels, num_modes, use_bias=False, weights_type=weights_type, ndim=ndim,
                     device=device)

        if use_conv_branch:
            op = nn.Conv2d if ndim == 4 else nn.Conv3d
            self.conv_branch = op(in_channels, out_channels, kernel_size=1, bias=use_bias_conv_branch, device=device)

        # SELU is self-normalizing, thus normalization is not required
        if activation != 'selu' and activation != nn.functional.selu:
            self.normalization = nn.GroupNorm(1, out_channels, device=device)

        self.activation = activation
        if isinstance(self.activation, str):
            self.activation = getattr(nn.functional, self.activation)

        if self.use_block_skip and use_block_concat:
            self.conv_concat = ConvNormAct(in_channels + out_channels, out_channels, use_bias=True,
                                           activation=activation, ndim=ndim, device=device)


class HartleyMHABlock(_TransBlock):
    def __init__(self, in_channels, key_dim, num_heads, num_modes, patch_size, attention_activation, ndim, activation,
                 device, use_conv_branch=True, use_bias_conv_branch=False, use_block_skip=True, use_block_concat=True):
        super().__init__()

        self.use_block_skip = use_block_skip

        self.op = HartleyMultiHeadAttention(
            in_channels, key_dim, num_heads, num_modes, patch_size, attention_activation, ndim=ndim, device=device)

        if use_conv_branch:
            op = nn.Conv2d if ndim == 4 else nn.Conv3d
            self.conv_branch = op(in_channels, key_dim, kernel_size=1, bias=use_bias_conv_branch, device=device)

        # SELU is self-normalizing, thus normalization is not required
        if activation != 'selu' and activation != nn.functional.selu:
            self.normalization = nn.GroupNorm(1, key_dim, device=device)

        self.activation = activation
        if isinstance(self.activation, str):
            self.activation = getattr(nn.functional, self.activation)

        if self.use_block_skip and use_block_concat:
            self.conv_concat = ConvNormAct(in_channels + key_dim, key_dim, use_bias=True,
                                           activation=activation, ndim=ndim, device=device)


def upsampling(tensors):
    """Upsamples a dict of tensors to the same spatial size.
    The tensor of key 0 is assumed to be the largest tensor and used as a reference.

    Args:
        tensors: a dict of tensors with keys as integers and values as tensors.

    Returns:
        A list of upsampled tensors.
    """
    ref_tensor = tensors[0]  # Must be the tensor with the largest size
    ref_size = ref_tensor.shape[2:]

    up = [nn.functional.interpolate(t, ref_size) for t in tensors.values()]

    return up
