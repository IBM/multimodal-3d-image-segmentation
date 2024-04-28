#
# Copyright 2023 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Different network architectures for image segmentation.
All architectures work for both 2D and 3D.

Author: Ken C. L. Wong
"""

import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose
from keras.layers import UpSampling2D, UpSampling3D, Concatenate, Add, Activation
from keras.layers import GroupNormalization

from nets.nets_utils import spatial_padcrop, get_loss
from nets.custom_objects import custom_objects
from nets.fourier_operator import FourierOperator
from nets.hartley_operator import HartleyOperator
from nets.hartley_mha import HartleyMultiHeadAttention

__author__ = 'Ken C. L. Wong'


class VNetDS:
    """A network architecture modified from V-Net with deep supervision.
    Learnable input downsampling and output upsampling can also be used.

    Please refer to our MICCAI 2018 paper for more details:

    Wong, K.C.L., Moradi, M., Tang, H., Syeda-Mahmood, T. (2018). 3D Segmentation
    with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes. In:
    Medical Image Computing and Computer Assisted Intervention – MICCAI 2018.
    https://doi.org/10.1007/978-3-030-00931-1_70

    Args:
        image_size: Spatial image size, used with `num_input_channels` to create the input layer.
        num_input_channels: Number of input channels, used with `image_size` to create the input layer.
        base_num_filters: The number of filters of each convolutional layer before the first downsampling.
        num_blocks: A list of int specifying the encoding path, e.g., [1, 2, 3] means
            Conv -> Downsample -> Conv -> Conv -> Downsample -> Conv -> Conv -> Conv.
            The decoding path is the reverse without the last int.
        num_output_channels: Number of output channels of the model.
        optimizer: Optimizer.
        use_resize: If True (default), learnable input downsampling and output upsampling are used.
        right_leg_indexes: A list of int specifying the right leg from the decoding path (deep supervision).
            E.g., [0, 1, 2]. None (default) if deep supervision is not used.
        kernel_size: Kernel size for convolutional layers (default: 3).
        activation: Activation (default: 'selu').
        output_activation: Activation for the output, usually 'softmax' (default) for segmentation.
        loss: Loss function(s). Can be a str, a callable, or a list of them (default: 'PCCLoss').
        loss_args: Optional loss function arguments.
            If it is a list, each element is only used if it is a dict (default: None).
        kernel_initializer: Kernel initializer of convolutional layers (default: 'glorot_uniform').
        use_residual: If True (default), residual connections are used.
    """
    def __init__(
            self,
            image_size,
            num_input_channels,
            base_num_filters,
            num_blocks,
            num_output_channels,
            optimizer,
            use_resize=True,
            right_leg_indexes=None,
            kernel_size=3,
            activation='selu',
            output_activation='softmax',
            loss='PCCLoss',
            loss_args=None,
            kernel_initializer='glorot_uniform',
            use_residual=True,
    ):
        assert isinstance(num_blocks, (list, tuple))

        self.image_size = image_size
        self.num_input_channels = num_input_channels
        self.base_num_filters = base_num_filters
        self.num_blocks = num_blocks
        self.num_output_channels = num_output_channels
        self.optimizer = optimizer
        self.use_resize = use_resize
        self.right_leg_indexes = right_leg_indexes
        self.kernel_size = kernel_size
        self.activation = activation
        self.output_activation = output_activation
        self.kernel_initializer = kernel_initializer
        self.use_residual = use_residual

        if self.right_leg_indexes is None:
            self.right_leg_indexes = [0]

        self.encode_tensors = dict()
        self.right_leg = dict()

        self.loss = get_loss(loss, loss_args, custom_objects)

        ndim = len(self.image_size)
        self.conv = Conv2D if ndim == 2 else Conv3D
        self.conv_transpose = Conv2DTranspose if ndim == 2 else Conv3DTranspose

    def __call__(self):
        inputs = Input(self.image_size + (self.num_input_channels,))

        x = inputs

        if self.use_resize:
            x = self.conv_block(self.base_num_filters, kernel_size=2, strides=2)(x)

        x = self.encode(x)
        x = self.decode(x)

        if self.use_resize:
            op = self.conv_transpose(filters=self.num_output_channels, kernel_size=2, strides=2, padding='same',
                                     kernel_initializer=self.kernel_initializer)
        else:
            op = self.conv(filters=self.num_output_channels, kernel_size=1, kernel_initializer=self.kernel_initializer)
        x = op(x)

        x = spatial_padcrop(x, self.image_size)  # To handle the spatial size mismatch after down- and up-sampling

        x = Activation(self.output_activation)(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(optimizer=self.optimizer, loss=self.loss)

        return model

    def encode(self, x):
        num_sections = len(self.num_blocks)
        for i in range(num_sections):
            filters = self.base_num_filters * (2 ** i)

            tmp = x if self.use_residual else None

            for _ in range(self.num_blocks[i]):
                x = self.conv_block(filters)(x)

            if tmp is not None:
                tmp = self.conv_block(filters, kernel_size=1)(tmp)
                x = Add()([tmp, x])

            if i != num_sections - 1:
                self.encode_tensors[i] = x
                x = self.conv_block(filters, strides=2)(x)  # Downsampling
            elif i in self.right_leg_indexes:
                self.right_leg[i] = x

        return x

    def decode(self, x):
        num_sections = len(self.num_blocks)
        for i in reversed(range(num_sections - 1)):
            filters = self.base_num_filters * (2 ** i)
            x = self.conv_transpose_block(filters)(x)  # Upsampling
            x = spatial_padcrop(x, tuple(self.encode_tensors[i].shape[1:-1]))  # Handle spatial shape difference
            x = Concatenate()([x, self.encode_tensors[i]])

            tmp = x if self.use_residual else None

            for _ in range(self.num_blocks[i]):
                x = self.conv_block(filters)(x)

            if tmp is not None:
                tmp = self.conv_block(filters, kernel_size=1)(tmp)
                x = Add()([tmp, x])

            if i in self.right_leg_indexes:
                self.right_leg[i] = x

        if len(self.right_leg) == 1:
            x = self.right_leg[0]
        else:
            x = Concatenate()(upsampling(self.right_leg))

        return x

    def conv_block(self, filters, kernel_size=None, strides=1):
        def inner(x):
            ks = kernel_size or self.kernel_size
            op = self.conv(filters, kernel_size=ks, strides=strides, padding='same',
                           kernel_initializer=self.kernel_initializer)
            x = op(x)

            # noinspection PyCallingNonCallable
            x = GroupNormalization(groups=1)(x)
            x = Activation(self.activation)(x)
            return x
        return inner

    def conv_transpose_block(self, filters):
        """For upsampling only."""
        def inner(x):
            op = self.conv_transpose(filters, self.kernel_size, strides=2, padding='same',
                                     kernel_initializer=self.kernel_initializer)
            x = op(x)

            # noinspection PyCallingNonCallable
            x = GroupNormalization(groups=1)(x)
            x = Activation(self.activation)(x)
            return x
        return inner


class NeuralOperatorSeg:
    """Architecture related to the Fourier neural operator (FNO).
    This function can be used to create the FNO, FNO-shared, FNOSeg, and HNOSeg architectures
    by different combinations of arguments.

    Please refer to our ISBI 2023 and MICCAI 2023 papers for more details.

    Args:
        image_size: Spatial image size, used with `num_input_channels` to create the input layer.
        num_input_channels: Number of input channels, used with `image_size` to create the input layer.
        filters: Number of filters for all layers except the output layer.
        num_transform_blocks: Number of transform (FNO or HNO) blocks (int).
        num_output_channels: Number of output channels of the model.
        optimizer: Optimizer.
        num_modes: Number of frequency modes (k_max). Can be an int or a list of int.
            Note that num_modes must be smaller than half of the `image_size` in each dimension.
        transform_type: Type of transformation. Can be 'Fourier' (default) or 'Hartley'.
        transform_weights_type: Type of weights in the frequency domain.
            Must be 'individual' or 'shared' (default).
        use_resize: If True (default), learnable input downsampling and output upsampling are used.
        merge_method: Method used for skip connections. Can be 'add' (default), 'concat', or None.
        use_deep_supervision: If True (default), deep supervision is used.
        activation: Activation (default: 'selu').
        output_activation: Activation for the output, usually 'softmax' (default) for segmentation.
        kernel_initializer: Kernel weights initializer (default: 'glorot_uniform').
        loss: Loss function(s). Can be a str, a callable, or a list of them (default: 'PCCLoss').
        loss_args: Optional loss function arguments.
            If it is a list, each element is only used if it is a dict (default: None).
    """
    def __init__(
            self,
            image_size,
            num_input_channels,
            filters,
            num_transform_blocks,
            num_output_channels,
            optimizer,
            num_modes,
            transform_type='Fourier',
            transform_weights_type='shared',
            use_resize=True,
            merge_method='add',
            use_deep_supervision=True,
            activation='selu',
            output_activation='softmax',
            kernel_initializer='glorot_uniform',
            loss='PCCLoss',
            loss_args=None,
    ):
        self.image_size = image_size
        self.num_input_channels = num_input_channels
        self.filters = filters
        self.num_transform_blocks = num_transform_blocks
        self.num_output_channels = num_output_channels
        self.optimizer = optimizer
        self.num_modes = num_modes
        self.transform_type = transform_type
        self.transform_weights_type = transform_weights_type
        self.use_resize = use_resize
        self.merge_method = merge_method
        self.use_deep_supervision = use_deep_supervision
        self.activation = activation
        self.output_activation = output_activation
        self.kernel_initializer = kernel_initializer

        self.loss = get_loss(loss, loss_args, custom_objects)

        assert self.transform_type in ['Fourier', 'Hartley']
        if self.merge_method is not None:
            assert self.merge_method in ['add', 'concat']

        ndim = len(self.image_size)
        self.conv = Conv2D if ndim == 2 else Conv3D
        self.conv_transpose = Conv2DTranspose if ndim == 2 else Conv3DTranspose

    def __call__(self):
        tensors = []

        inputs = Input(self.image_size + (self.num_input_channels,))

        x = inputs

        if self.use_resize:
            x = self.conv_block(self.filters, kernel_size=2, strides=2)(x)

        x = self.conv_block(self.filters)(x)
        if self.use_deep_supervision:
            tensors.append(x)

        filters = self.filters
        for _ in range(self.num_transform_blocks):
            if self.merge_method is None:
                x = self.transform_block(filters)(x)
            else:
                tmp = self.transform_block(filters)(x)
                merge = Add() if self.merge_method == 'add' else Concatenate()
                x = merge([x, tmp])

            if self.use_deep_supervision:
                tensors.append(x)

        if tensors:
            x = Concatenate()(tensors)

        if self.use_resize:
            op = self.conv_transpose(filters=self.num_output_channels, kernel_size=2, strides=2, padding='same',
                                     kernel_initializer=self.kernel_initializer)
        else:
            op = self.conv(filters=self.num_output_channels, kernel_size=1, kernel_initializer=self.kernel_initializer)
        x = op(x)

        x = spatial_padcrop(x, self.image_size)

        x = Activation(self.output_activation)(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(optimizer=self.optimizer, loss=self.loss)

        return model

    def transform_block(self, filters):
        def inner(x):
            if self.transform_type == 'Fourier':
                op = FourierOperator(filters, self.num_modes, weights_type=self.transform_weights_type)
            else:
                op = HartleyOperator(filters, self.num_modes, weights_type=self.transform_weights_type)
            x1 = op(x)

            op = self.conv(filters, 1, kernel_initializer=self.kernel_initializer)
            x2 = op(x)

            x = Add()([x1, x2])

            # noinspection PyCallingNonCallable
            x = GroupNormalization(groups=1)(x)
            x = Activation(self.activation)(x)

            return x
        return inner

    def conv_block(self, filters, kernel_size=1, strides=1):
        def inner(x):
            op = self.conv(filters, kernel_size=kernel_size, strides=strides, padding='same',
                           kernel_initializer=self.kernel_initializer)
            x = op(x)

            # noinspection PyCallingNonCallable
            x = GroupNormalization(groups=1)(x)
            x = Activation(self.activation)(x)
            return x
        return inner


class HartleyMHASeg:
    """HartleyMHA architecture.
    This architecture comprises layers that perform self-attention in the frequency domain.

    Please refer to our MICCAI 2023 paper for more details.

    Args:
        image_size: Spatial image size, used with `num_input_channels` to create the input layer.
        num_input_channels: Number of input channels, used with `image_size` to create the input layer.
        filters: Number of filters for all layers except the output layer.
        num_transform_blocks: Number of transform (Hartley MHA) blocks (int).
        num_output_channels: Number of output channels of the model.
        optimizer: Optimizer.
        num_heads: Number of attention heads.
        num_modes: Number of frequency modes (k_max). Can be an int or a list of int.
            Note that `num_modes` must be smaller than half of the input spatial size in each dimension,
            and must be divisible by `patch_size`.
        patch_size: Patch size for grouping in the frequency domain.
        attention_activation: Activation applied on the attention matrix (default: 'selu').
        use_resize: If True (default), learnable input downsampling and output upsampling are used.
        merge_method: Method used for skip connections. Can be 'add' (default), 'concat', or None.
        use_deep_supervision: If True (default), deep supervision is used.
        activation: Activation (default: 'selu').
        output_activation: Activation for the output, usually 'softmax' (default) for segmentation.
        kernel_initializer: Kernel weights initializer (default: 'glorot_uniform').
        loss: Loss function(s). Can be a str, a callable, or a list of them (default: 'PCCLoss').
        loss_args: Optional loss function arguments.
            If it is a list, each element is only used if it is a dict (default: None).
    """
    def __init__(
            self,
            image_size,
            num_input_channels,
            filters,
            num_transform_blocks,
            num_output_channels,
            optimizer,
            num_heads,
            num_modes,
            patch_size,
            attention_activation='selu',
            use_resize=True,
            merge_method='add',
            use_deep_supervision=True,
            activation='selu',
            output_activation='softmax',
            kernel_initializer='glorot_uniform',
            loss='PCCLoss',
            loss_args=None,
    ):
        self.image_size = image_size
        self.num_input_channels = num_input_channels
        self.filters = filters
        self.num_transform_blocks = num_transform_blocks
        self.num_output_channels = num_output_channels
        self.optimizer = optimizer
        self.num_heads = num_heads
        self.num_modes = num_modes
        self.patch_size = patch_size
        self.attention_activation = attention_activation
        self.use_resize = use_resize
        self.merge_method = merge_method
        self.use_deep_supervision = use_deep_supervision
        self.activation = activation
        self.output_activation = output_activation
        self.kernel_initializer = kernel_initializer

        self.loss = get_loss(loss, loss_args, custom_objects)

        if self.merge_method is not None:
            assert self.merge_method in ['add', 'concat']

        ndim = len(self.image_size)
        self.conv = Conv2D if ndim == 2 else Conv3D
        self.conv_transpose = Conv2DTranspose if ndim == 2 else Conv3DTranspose

    def __call__(self):
        tensors = []

        inputs = Input(self.image_size + (self.num_input_channels,))

        x = inputs

        if self.use_resize:
            x = self.conv_block(self.filters, kernel_size=2, strides=2)(x)

        x = self.conv_block(self.filters)(x)
        if self.use_deep_supervision:
            tensors.append(x)

        filters = self.filters
        for _ in range(self.num_transform_blocks):
            if self.merge_method is None:
                x = self.transform_block(filters)(x)
            else:
                tmp = self.transform_block(filters)(x)
                merge = Add() if self.merge_method == 'add' else Concatenate()
                x = merge([x, tmp])

            if self.use_deep_supervision:
                tensors.append(x)

        if tensors:
            x = Concatenate()(tensors)

        if self.use_resize:
            op = self.conv_transpose(filters=self.num_output_channels, kernel_size=2, strides=2, padding='same',
                                     kernel_initializer=self.kernel_initializer)
        else:
            op = self.conv(filters=self.num_output_channels, kernel_size=1, kernel_initializer=self.kernel_initializer)
        x = op(x)

        x = spatial_padcrop(x, self.image_size)

        x = Activation(self.output_activation)(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(optimizer=self.optimizer, loss=self.loss)

        return model

    def transform_block(self, filters):
        def inner(x):
            op = HartleyMultiHeadAttention(
                self.num_heads, filters, self.num_modes, self.patch_size,
                attention_activation=self.attention_activation,
            )
            x1 = op(x)

            op = self.conv(filters, 1, kernel_initializer=self.kernel_initializer)
            x2 = op(x)

            x = Add()([x1, x2])

            # noinspection PyCallingNonCallable
            x = GroupNormalization(groups=1)(x)
            x = Activation(self.activation)(x)

            return x
        return inner

    def conv_block(self, filters, kernel_size=1, strides=1):
        def inner(x):
            op = self.conv(filters, kernel_size=kernel_size, strides=strides, padding='same',
                           kernel_initializer=self.kernel_initializer)
            x = op(x)

            # noinspection PyCallingNonCallable
            x = GroupNormalization(groups=1)(x)
            x = Activation(self.activation)(x)
            return x
        return inner


def upsampling(tensors):
    """Upsamples a dict of tensors to the same spatial size.
    The tensor of key 0 is assumed to be the largest tensor and used as a reference.

    Args:
        tensors: a dict of tensors with keys as integers and values as tensors.

    Returns:
        A list of upsampled tensors.
    """
    ref_tensor = tensors[0]  # Must be the tensor with the largest size
    ref_size = tuple(ref_tensor.shape[1:-1])

    up = []
    for key in tensors:
        t = tensors[key]
        t_sz = tuple(t.shape[1:-1])
        size = [int(np.round(sz / t_sz[i])) for i, sz in enumerate(ref_size)]
        op = UpSampling2D if ref_tensor.ndim == 4 else UpSampling3D
        up.append(spatial_padcrop(op(size=size)(t), ref_size))

    return up
