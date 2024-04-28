#
# Copyright 2023 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import tensorflow as tf
from keras.layers import Layer, InputSpec
from keras import constraints
from keras import initializers
from keras import regularizers

import numpy as np

from nets.dht import dht2d, dht3d

__author__ = 'Ken C. L. Wong'


class HartleyMultiHeadAttention(Layer):
    """A Keras layer that applies the Hartley multi-head attention.
    The input tensor is Hartley transformed and multi-head self-attention is applied
    in the frequency domain. The inverse Hartley transform converts the results back
    to the spatial domain.

    Args:
        num_heads: Number of attention heads.
        key_dim: Size of each attention head for query and key.
        num_modes: Number of frequency modes (k_max). Can be an int or a list of int.
            Note that `num_modes` must be smaller than half of the input spatial size in each dimension,
            and must be divisible by `patch_size`.
        patch_size: Patch size for grouping in the frequency domain.
        attention_activation: Activation applied on the attention matrix (default: 'selu').
        value_dim: Size of each attention head for value. If None (default), `key_dim` is used.
        use_bias: If True, biases are added to the query, value, key, and output tensors (default: False).
        kernel_initializer: Kernel weights initializer (default: 'glorot_uniform').
        bias_initializer: Bias weights initializer (default: 'zeros').
        kernel_regularizer: Kernel regularizer (default: None).
        bias_regularizer: Bias regularizer (default: None).
        kernel_constraint: Kernel constraint (default: None).
        bias_constraint: Bias constraint (default: None).
        trainable: If True (default), the layer is trainable.
        name: Optional name for the instance (default: None).
        **kwargs: Optional keyword arguments.
    """
    def __init__(self,
                 num_heads,
                 key_dim,
                 num_modes,
                 patch_size,
                 attention_activation='selu',
                 value_dim=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.num_modes = num_modes
        self.patch_size = patch_size
        self.attention_activation = attention_activation
        self.value_dim = value_dim or key_dim
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.kernel_query = None
        self.kernel_key = None
        self.kernel_value = None
        self.kernel_out = None

        self.bias_query = None
        self.bias_key = None
        self.bias_value = None
        self.bias_out = None

    def build(self, input_shape):
        if not isinstance(input_shape[0], (tuple, tf.TensorShape)):  # Single input
            query_shape = key_shape = value_shape = input_shape
        elif len(input_shape) == 2:
            query_shape = input_shape[0]
            key_shape = value_shape = input_shape[1]
        elif len(input_shape) == 3:
            query_shape, key_shape, value_shape = input_shape
        else:
            raise ValueError('Invalid inputs.')

        ndim = len(query_shape)

        # Ensures self.num_modes be a tuple
        if np.isscalar(self.num_modes):
            self.num_modes = (self.num_modes,) * (ndim - 2)
        else:
            assert len(self.num_modes) == ndim - 2
            self.num_modes = tuple(self.num_modes)

        # Ensures proper modes range
        if ndim == 4:
            s0, s1 = query_shape[1:-1]
            modes_0, modes_1 = self.num_modes
            assert s0 >= 2 * modes_0 and s1 >= 2 * modes_1
        else:
            s0, s1, s2 = query_shape[1:-1]
            modes_0, modes_1, modes_2 = self.num_modes
            assert s0 >= 2 * modes_0 and s1 >= 2 * modes_1 and s2 >= 2 * modes_2

        self.kernel_query = self.add_weight(
            name='kernel_query',
            shape=(query_shape[-1], self.key_dim, self.num_heads),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        self.kernel_key = self.add_weight(
            name='kernel_key',
            shape=(key_shape[-1], self.key_dim, self.num_heads),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        self.kernel_value = self.add_weight(
            name='kernel_value',
            shape=(value_shape[-1], self.value_dim, self.num_heads),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        self.kernel_out = self.add_weight(
            name='kernel_out',
            shape=(self.value_dim * self.num_heads, self.value_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias_query = self.add_weight(
                name='bias_query',
                shape=(self.key_dim, self.num_heads) + (1,) * (ndim - 2),  # 1's are necessary for broadcast
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)

            self.bias_key = self.add_weight(
                name='bias_key',
                shape=(self.key_dim, self.num_heads) + (1,) * (ndim - 2),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)

            self.bias_value = self.add_weight(
                name='bias_value',
                shape=(self.value_dim, self.num_heads) + (1,) * (ndim - 2),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)

            self.bias_out = self.add_weight(
                name='bias_out',
                shape=(self.value_dim,) + (1,) * (ndim - 2),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)

        self.input_spec = InputSpec(ndim=ndim)
        self.built = True

    def call(self, inputs):
        if not isinstance(inputs, (tuple, list)):  # Single input
            query = key = value = self.dht(inputs)  # (B, C, spatial)
        elif len(inputs) == 2:
            query = self.dht(inputs[0])
            key = value = self.dht(inputs[1])
        elif len(inputs) == 3:
            query = self.dht(inputs[0])
            key = self.dht(inputs[1])
            value = self.dht(inputs[2])
        else:
            raise ValueError('Invalid inputs.')

        ndim = query.ndim  # Input ndim
        spatial_shape = tuple(query.shape[2:])  # Input spatial shape

        if ndim == 4:
            query = self._freq_conv2d(self.kernel_query, query)  # (B, C, HEADS, H, W)
            key = self._freq_conv2d(self.kernel_key, key)
            value = self._freq_conv2d(self.kernel_value, value)
        else:
            query = self._freq_conv3d(self.kernel_query, query)  # (B, C, HEADS, D, H, W)
            key = self._freq_conv3d(self.kernel_key, key)
            value = self._freq_conv3d(self.kernel_value, value)

        if self.use_bias:
            query = query + self.bias_query
            key = key + self.bias_key
            value = value + self.bias_value

        # Dimension reduction by grouping, (B, C * prod(patch_size), HEADS, num_d, num_h, num_w)
        if ndim == 4:
            query = grouping2d(query, self.patch_size)
            key = grouping2d(key, self.patch_size)
            value = grouping2d(value, self.patch_size)
        else:
            query = grouping3d(query, self.patch_size)
            key = grouping3d(key, self.patch_size)
            value = grouping3d(value, self.patch_size)

        spatial_shape_freq = tuple(query.shape[3:])  # Spatial shape before flattening

        query = self._spatial_flatten(query)  # (B, C * prod(patch_size), HEADS, num_d * num_h * num_w)
        key = self._spatial_flatten(key)
        value = self._spatial_flatten(value)

        att = tf.einsum('bchq,bchk->bhqk', query, key)
        att = att / tf.sqrt(float(key.shape[1]))
        if self.attention_activation is not None:
            activation = getattr(tf.nn, self.attention_activation)
            att = activation(att)

        output = tf.einsum('bhqk,bchk->bchq', att, value)
        output = tf.reshape(output, (-1,) + tuple(output.shape[1:3]) + spatial_shape_freq)

        # Get back the original spatial shape of query, (B, C, HEADS, D, H, W)
        if ndim == 4:
            output = ungrouping2d(output, self.value_dim, self.patch_size)
        else:
            output = ungrouping3d(output, self.value_dim, self.patch_size)

        shape = tuple(output.shape)
        output = tf.reshape(output, (-1,) + (shape[1] * shape[2],) + shape[3:])

        equation = 'io,bihw->bohw' if ndim == 4 else 'io,bidhw->bodhw'
        output = tf.einsum(equation, self.kernel_out, output)
        if self.use_bias:
            output = output + self.bias_out

        output = self._inverse2d(output, spatial_shape) if ndim == 4 else self._inverse3d(output, spatial_shape)

        return output

    @staticmethod
    def dht(x):
        # Convert to channel-first as dht(fft) only works on the innermost dimensions
        ndim = x.ndim
        perm = [0, ndim - 1] + list(range(1, ndim - 1))  # (B, C, spatial)
        x = tf.transpose(x, perm=perm)

        if ndim == 4:
            return dht2d(x)
        return dht3d(x)

    def _freq_conv2d(self, kernel, x):
        equation = 'ioz,bihw->bozhw'  # z: heads
        modes_0, modes_1 = self.num_modes

        ll = tf.einsum(equation, kernel, x[..., :modes_0, :modes_1])
        lh = tf.einsum(equation, kernel, x[..., :modes_0, -modes_1:])
        hl = tf.einsum(equation, kernel, x[..., -modes_0:, :modes_1])
        hh = tf.einsum(equation, kernel, x[..., -modes_0:, -modes_1:])

        low = tf.concat([ll, lh], axis=-1)
        high = tf.concat([hl, hh], axis=-1)
        return tf.concat([low, high], axis=-2)

    def _freq_conv3d(self, kernel, x):
        equation = 'ioz,bidhw->bozdhw'  # z: heads
        modes_0, modes_1, modes_2 = self.num_modes

        lll = tf.einsum(equation, kernel, x[..., :modes_0, :modes_1, :modes_2])
        lhl = tf.einsum(equation, kernel, x[..., :modes_0, -modes_1:, :modes_2])
        hll = tf.einsum(equation, kernel, x[..., -modes_0:, :modes_1, :modes_2])
        hhl = tf.einsum(equation, kernel, x[..., -modes_0:, -modes_1:, :modes_2])
        llh = tf.einsum(equation, kernel, x[..., :modes_0, :modes_1, -modes_2:])
        lhh = tf.einsum(equation, kernel, x[..., :modes_0, -modes_1:, -modes_2:])
        hlh = tf.einsum(equation, kernel, x[..., -modes_0:, :modes_1, -modes_2:])
        hhh = tf.einsum(equation, kernel, x[..., -modes_0:, -modes_1:, -modes_2:])

        # Combine along spatial dim 2
        ll = tf.concat([lll, llh], axis=-1)
        lh = tf.concat([lhl, lhh], axis=-1)
        hl = tf.concat([hll, hlh], axis=-1)
        hh = tf.concat([hhl, hhh], axis=-1)

        # Combine along spatial dim 1
        low = tf.concat([ll, lh], axis=-2)
        high = tf.concat([hl, hh], axis=-2)

        # Combine along spatial dim 0
        return tf.concat([low, high], axis=-3)

    def _inverse2d(self, x, spatial_shape):
        s0, s1 = spatial_shape
        modes_0, modes_1 = self.num_modes
        ndim = len(spatial_shape) + 2

        ll = x[..., :modes_0, :modes_1]
        lh = x[..., :modes_0, -modes_1:]
        hl = x[..., -modes_0:, :modes_1]
        hh = x[..., -modes_0:, -modes_1:]

        # Padding
        pad_shape = tf.concat([tf.shape(x)[:2], [modes_0, s1 - 2 * modes_1]], axis=0)
        pad_zeros = tf.zeros(pad_shape, dtype=x.dtype)
        low = tf.concat([ll, pad_zeros, lh], axis=-1)
        high = tf.concat([hl, pad_zeros, hh], axis=-1)

        pad_shape = tf.concat([tf.shape(x)[:2], [s0 - 2 * modes_0, s1]], axis=0)
        pad_zeros = tf.zeros(pad_shape, dtype=x.dtype)
        x = tf.concat([low, pad_zeros, high], axis=-2)

        x = dht2d(x, is_inverse=True)

        # Convert back to channel-last
        perm = [0] + list(range(2, ndim)) + [1]  # (b, spatial, c)
        x = tf.transpose(x, perm=perm)

        return x

    def _inverse3d(self, x, spatial_shape):
        s0, s1, s2 = spatial_shape
        modes_0, modes_1, modes_2 = self.num_modes
        ndim = len(spatial_shape) + 2

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
        pad_shape = tf.concat([tf.shape(x)[:2], [modes_0, modes_1, s2 - 2 * modes_2]], axis=0)
        pad_zeros = tf.zeros(pad_shape, dtype=x.dtype)
        ll = tf.concat([lll, pad_zeros, llh], axis=-1)
        lh = tf.concat([lhl, pad_zeros, lhh], axis=-1)
        hl = tf.concat([hll, pad_zeros, hlh], axis=-1)
        hh = tf.concat([hhl, pad_zeros, hhh], axis=-1)

        # Padding along spatial dim 1, shape = (b, c, modes_0, s1, s2)
        pad_shape = tf.concat([tf.shape(x)[:2], [modes_0, s1 - 2 * modes_1, s2]], axis=0)
        pad_zeros = tf.zeros(pad_shape, dtype=x.dtype)
        low = tf.concat([ll, pad_zeros, lh], axis=-2)
        high = tf.concat([hl, pad_zeros, hh], axis=-2)

        # Padding along spatial dim 0, shape = (b, c, s0, s1, s2)
        pad_shape = tf.concat([tf.shape(x)[:2], [s0 - 2 * modes_0, s1, s2]], axis=0)
        pad_zeros = tf.zeros(pad_shape, dtype=x.dtype)
        x = tf.concat([low, pad_zeros, high], axis=-3)

        x = dht3d(x, is_inverse=True)

        # Convert back to channel-last
        perm = [0] + list(range(2, ndim)) + [1]  # (b, spatial, c)
        x = tf.transpose(x, perm=perm)

        return x

    @staticmethod
    def _spatial_flatten(x, has_batch_axis=True):
        in_shape = tuple(x.shape)
        if has_batch_axis:
            shape = (-1,) + in_shape[1:3] + (np.prod(in_shape[3:]),)
        else:
            shape = in_shape[:2] + (np.prod(in_shape[2:]),)
        return tf.reshape(x, shape)

    def get_config(self):
        config = {
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'num_modes': self.num_modes,
            'patch_size': self.patch_size,
            'attention_activation': self.attention_activation,
            'value_dim': self.value_dim,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}


def grouping2d(x, patch_size):
    """Groups pixels into patches.

    Args:
        x: Input tensor of shape (batch_size, num_channels, num_heads, height, width).
        patch_size: Patch size.

    Returns:
        A reshaped tensor with each new pixel as a concatenation of the original pixels,
        with shape (batch_size, num_grouped_channels, num_heads, num_patches_h, num_patches_w).
    """
    assert len(patch_size) == 2

    patch_h, patch_w = patch_size
    _, c, z, h, w = x.shape  # Channel first in frequency domain, z is num_heads

    assert h % patch_h == 0 and w % patch_w == 0
    num_h = h // patch_h
    num_w = w // patch_w

    x = tf.reshape(
        x, (-1, c, z, num_h, patch_h, num_w, patch_w)
    )
    x = tf.transpose(x, (0, 1, 4, 6, 2, 3, 5))
    x = tf.reshape(x, (-1, c * patch_h * patch_w, z, num_h, num_w))

    return x


def ungrouping2d(x, num_channels, patch_size):
    """Ungroups patches back to pixels.

    Args:
        x: Input tensor of shape (batch_size, num_grouped_channels, num_heads, num_patches_h, num_patches_w).
        num_channels: Number of original channels before grouping.
        patch_size: Patch size used for grouping.

    Returns:
        A reshaped tensor with the original shape before grouping,
        with shape (batch_size, num_channels, num_heads, height, width).
    """
    assert len(patch_size) == 2

    patch_h, patch_w = patch_size
    c = num_channels
    z, num_h, num_w = x.shape[2:]

    x = tf.reshape(x, (-1, c, patch_h, patch_w, z, num_h, num_w))
    x = tf.transpose(x, (0, 1, 4, 5, 2, 6, 3))
    x = tf.reshape(x, (-1, c, z, num_h * patch_h, num_w * patch_w))

    return x


def grouping3d(x, patch_size):
    """Groups pixels into patches.

    Args:
        x: Input tensor of shape (batch_size, num_channels, num_heads, depth, height, width).
        patch_size: Patch size.

    Returns:
        A reshaped tensor with each new pixel as a concatenation of the original pixels,
        with shape (batch_size, num_grouped_channels, num_heads, num_patches_d, num_patches_h, num_patches_w).
    """
    assert len(patch_size) == 3

    patch_d, patch_h, patch_w = patch_size
    _, c, z, d, h, w = x.shape  # Channel first in frequency domain, z is num_heads

    assert d % patch_d == 0 and h % patch_h == 0 and w % patch_w == 0
    num_d = d // patch_d
    num_h = h // patch_h
    num_w = w // patch_w

    x = tf.reshape(
        x, (-1, c, z, num_d, patch_d, num_h, patch_h, num_w, patch_w)
    )
    x = tf.transpose(x, (0, 1, 4, 6, 8, 2, 3, 5, 7))
    x = tf.reshape(x, (-1, c * patch_d * patch_h * patch_w, z, num_d, num_h, num_w))

    return x


def ungrouping3d(x, num_channels, patch_size):
    """Ungroups patches back to pixels.

    Args:
        x: Input tensor of shape (batch_size, num_grouped_channels, num_heads, num_patches_d,
            num_patches_h, num_patches_w).
        num_channels: Number of original channels before grouping.
        patch_size: Patch size used for grouping.

    Returns:
        A reshaped tensor with the original shape before grouping,
        with shape (batch_size, num_channels, num_heads, depth, height, width).
    """
    assert len(patch_size) == 3

    patch_d, patch_h, patch_w = patch_size
    c = num_channels
    z, num_d, num_h, num_w = x.shape[2:]

    x = tf.reshape(x, (-1, c, patch_d, patch_h, patch_w, z, num_d, num_h, num_w))
    x = tf.transpose(x, (0, 1, 5, 6, 2, 7, 3, 8, 4))
    x = tf.reshape(x, (-1, c, z, num_d * patch_d, num_h * patch_h, num_w * patch_w))

    return x
