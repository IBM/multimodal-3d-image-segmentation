#
# Copyright 2023 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import backend
from tensorflow.python.ops import nn

import numpy as np

from nets.dht import dht2d, dht3d

__author__ = 'Ken C. L. Wong'


class HartleyOperator(Layer):
    """A Keras layer that applies the convolution theorem through the Hartley transform.
    The input tensor is Hartley transformed, modified by the learnable weights in the
    frequency domain, and inverse Hartley transformed back to the spatial domain.

    Args:
        filters: Number of output channels of this layer.
        num_modes: Number of frequency modes (k_max). Can be an int or a list of int.
            Note that `num_modes` must be smaller than half of the input spatial size in each dimension.
        use_bias: If True, learned bias is added to the output tensor (default: False).
        kernel_initializer: Kernel weights initializer (default: 'glorot_uniform').
        bias_initializer: Bias weights initializer (default: 'zeros').
        kernel_regularizer: Kernel regularizer (default: None).
        bias_regularizer: Bias regularizer (default: None).
        kernel_constraint: Kernel constraint (default: None).
        bias_constraint: Bias constraint (default: None).
        weights_type: Type of weights in the frequency domain.
            Must be 'individual' or 'shared' (default).
        trainable: If True (default), the layer is trainable.
        name: Optional name for the instance (default: None).
        **kwargs: Optional keyword arguments.
    """
    def __init__(self,
                 filters,
                 num_modes,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 weights_type='shared',
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)

        self.filters = filters
        self.num_modes = num_modes
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.weights_type = weights_type
        assert self.weights_type in ['individual', 'shared']

        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        ndim = len(input_shape)
        channel_axis = ndim - 1
        num_input_channels = input_shape[channel_axis]

        if np.isscalar(self.num_modes):
            self.num_modes = (self.num_modes,) * (ndim - 2)
        else:
            assert len(self.num_modes) == ndim - 2
            self.num_modes = tuple(self.num_modes)

        if self.weights_type == 'shared':
            kernel_shape = (num_input_channels, self.filters)
        else:
            kernel_shape = tuple(np.array(self.num_modes) * 2) + (num_input_channels, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)

        self.input_spec = InputSpec(ndim=ndim,
                                    axes={channel_axis: num_input_channels})
        self.built = True

    def call(self, inputs, **kwargs):
        if backend.ndim(inputs) == 4:
            x = self._call2d(inputs)
        else:
            x = self._call3d(inputs)

        if self.use_bias:
            x = nn.bias_add(x, self.bias)

        return x

    def _call2d(self, inputs):
        s0, s1 = backend.int_shape(inputs)[1:-1]  # Spatial size
        modes_0, modes_1 = self.num_modes
        ndim = backend.ndim(inputs)

        assert s0 >= 2 * modes_0 and s1 >= 2 * modes_1

        # Convert to channel-first as dht(fft) only works on the innermost dimensions
        perm = [0, ndim - 1] + list(range(1, ndim - 1))  # (b, c, spatial)
        x = tf.transpose(inputs, perm=perm)

        x = dht2d(x)

        if self.weights_type == 'shared':
            kernel = self.kernel
            equation = 'io,bihw->bohw'
            ll = tf.einsum(equation, kernel, x[..., :modes_0, :modes_1])
            lh = tf.einsum(equation, kernel, x[..., :modes_0, -modes_1:])
            hl = tf.einsum(equation, kernel, x[..., -modes_0:, :modes_1])
            hh = tf.einsum(equation, kernel, x[..., -modes_0:, -modes_1:])
        else:
            kernel = self.kernel
            kernel_reverse = get_reverse(kernel, [0, 1])
            x_reverse = get_reverse(x, [-2, -1])
            equation = 'hwio,bihw->bohw'
            ll = hartley_conv(equation,
                              kernel[:modes_0, :modes_1], kernel_reverse[:modes_0, :modes_1],
                              x[..., :modes_0, :modes_1], x_reverse[..., :modes_0, :modes_1])
            lh = hartley_conv(equation,
                              kernel[:modes_0, -modes_1:], kernel_reverse[:modes_0, -modes_1:],
                              x[..., :modes_0, -modes_1:], x_reverse[..., :modes_0, -modes_1:])
            hl = hartley_conv(equation,
                              kernel[-modes_0:, :modes_1], kernel_reverse[-modes_0:, :modes_1],
                              x[..., -modes_0:, :modes_1], x_reverse[..., -modes_0:, :modes_1])
            hh = hartley_conv(equation,
                              kernel[-modes_0:, -modes_1:], kernel_reverse[-modes_0:, -modes_1:],
                              x[..., -modes_0:, -modes_1:], x_reverse[..., -modes_0:, -modes_1:])

        # Padding
        pad_shape = [tf.shape(x)[0], self.filters, modes_0, s1 - 2 * modes_1]
        pad_zeros = tf.zeros(pad_shape, dtype=x.dtype)
        low = tf.concat([ll, pad_zeros, lh], axis=-1)
        high = tf.concat([hl, pad_zeros, hh], axis=-1)

        pad_shape = [tf.shape(x)[0], self.filters, s0 - 2 * modes_0, s1]
        pad_zeros = tf.zeros(pad_shape, dtype=x.dtype)
        x = tf.concat([low, pad_zeros, high], axis=-2)

        x = dht2d(x, is_inverse=True)

        # Convert back to channel-last
        perm = [0] + list(range(2, ndim)) + [1]  # (b, spatial, c)
        x = tf.transpose(x, perm=perm)

        return x

    def _call3d(self, inputs):
        s0, s1, s2 = backend.int_shape(inputs)[1:-1]  # Spatial size
        modes_0, modes_1, modes_2 = self.num_modes
        ndim = backend.ndim(inputs)

        assert s0 >= 2 * modes_0 and s1 >= 2 * modes_1 and s2 >= 2 * modes_2

        # Convert to channel-first as dht(fft) only works on the innermost dimensions
        perm = [0, ndim - 1] + list(range(1, ndim - 1))  # (b, c, spatial)
        x = tf.transpose(inputs, perm=perm)

        x = dht3d(x)

        if self.weights_type == 'shared':
            kernel = self.kernel
            equation = 'io,bidhw->bodhw'
            lll = tf.einsum(equation, kernel, x[..., :modes_0, :modes_1, :modes_2])
            lhl = tf.einsum(equation, kernel, x[..., :modes_0, -modes_1:, :modes_2])
            hll = tf.einsum(equation, kernel, x[..., -modes_0:, :modes_1, :modes_2])
            hhl = tf.einsum(equation, kernel, x[..., -modes_0:, -modes_1:, :modes_2])
            llh = tf.einsum(equation, kernel, x[..., :modes_0, :modes_1, -modes_2:])
            lhh = tf.einsum(equation, kernel, x[..., :modes_0, -modes_1:, -modes_2:])
            hlh = tf.einsum(equation, kernel, x[..., -modes_0:, :modes_1, -modes_2:])
            hhh = tf.einsum(equation, kernel, x[..., -modes_0:, -modes_1:, -modes_2:])
        else:
            kernel = self.kernel
            kernel_reverse = get_reverse(kernel, [0, 1, 2])
            x_reverse = get_reverse(x, [-3, -2, -1])
            equation = 'dhwio,bidhw->bodhw'
            lll = hartley_conv(
                equation,
                kernel[:modes_0, :modes_1, :modes_2], kernel_reverse[:modes_0, :modes_1, :modes_2],
                x[..., :modes_0, :modes_1, :modes_2], x_reverse[..., :modes_0, :modes_1, :modes_2]
            )
            lhl = hartley_conv(
                equation,
                kernel[:modes_0, -modes_1:, :modes_2], kernel_reverse[:modes_0, -modes_1:, :modes_2],
                x[..., :modes_0, -modes_1:, :modes_2], x_reverse[..., :modes_0, -modes_1:, :modes_2]
            )
            hll = hartley_conv(
                equation,
                kernel[-modes_0:, :modes_1, :modes_2], kernel_reverse[-modes_0:, :modes_1, :modes_2],
                x[..., -modes_0:, :modes_1, :modes_2], x_reverse[..., -modes_0:, :modes_1, :modes_2]
            )
            hhl = hartley_conv(
                equation,
                kernel[-modes_0:, -modes_1:, :modes_2], kernel_reverse[-modes_0:, -modes_1:, :modes_2],
                x[..., -modes_0:, -modes_1:, :modes_2], x_reverse[..., -modes_0:, -modes_1:, :modes_2]
            )
            llh = hartley_conv(
                equation,
                kernel[:modes_0, :modes_1, -modes_2:], kernel_reverse[:modes_0, :modes_1, -modes_2:],
                x[..., :modes_0, :modes_1, -modes_2:], x_reverse[..., :modes_0, :modes_1, -modes_2:]
            )
            lhh = hartley_conv(
                equation,
                kernel[:modes_0, -modes_1:, -modes_2:], kernel_reverse[:modes_0, -modes_1:, -modes_2:],
                x[..., :modes_0, -modes_1:, -modes_2:], x_reverse[..., :modes_0, -modes_1:, -modes_2:]
            )
            hlh = hartley_conv(
                equation,
                kernel[-modes_0:, :modes_1, -modes_2:], kernel_reverse[-modes_0:, :modes_1, -modes_2:],
                x[..., -modes_0:, :modes_1, -modes_2:], x_reverse[..., -modes_0:, :modes_1, -modes_2:]
            )
            hhh = hartley_conv(
                equation,
                kernel[-modes_0:, -modes_1:, -modes_2:], kernel_reverse[-modes_0:, -modes_1:, -modes_2:],
                x[..., -modes_0:, -modes_1:, -modes_2:], x_reverse[..., -modes_0:, -modes_1:, -modes_2:]
            )

        # Padding needs to be done manually as ifft only pads at the end

        # Padding along spatial dim 2, shape = (b, c, modes_0, modes_1, s2)
        pad_shape = [tf.shape(x)[0], self.filters, modes_0, modes_1, s2 - 2 * modes_2]
        pad_zeros = tf.zeros(pad_shape, dtype=x.dtype)
        ll = tf.concat([lll, pad_zeros, llh], axis=-1)
        lh = tf.concat([lhl, pad_zeros, lhh], axis=-1)
        hl = tf.concat([hll, pad_zeros, hlh], axis=-1)
        hh = tf.concat([hhl, pad_zeros, hhh], axis=-1)

        # Padding along spatial dim 1, shape = (b, c, modes_0, s1, s2)
        pad_shape = [tf.shape(x)[0], self.filters, modes_0, s1 - 2 * modes_1, s2]
        pad_zeros = tf.zeros(pad_shape, dtype=x.dtype)
        low = tf.concat([ll, pad_zeros, lh], axis=-2)
        high = tf.concat([hl, pad_zeros, hh], axis=-2)

        # Padding along spatial dim 0, shape = (b, c, s0, s1, s2)
        pad_shape = [tf.shape(x)[0], self.filters, s0 - 2 * modes_0, s1, s2]
        pad_zeros = tf.zeros(pad_shape, dtype=x.dtype)
        x = tf.concat([low, pad_zeros, high], axis=-3)

        x = dht3d(x, is_inverse=True)

        # Convert back to channel-last
        perm = [0] + list(range(2, ndim)) + [1]  # (b, spatial, c)
        x = tf.transpose(x, perm=perm)

        return x

    def get_config(self):
        config = {
            'filters': self.filters,
            'num_modes': self.num_modes,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'weights_type': self.weights_type,
        }
        base_config = super().get_config()
        return {**base_config, **config}


def hartley_conv(equation, kernel, kernel_reverse, x, x_reverse):
    """Applies Hartley convolution theorem in the frequency domain.

    Args:
        equation: An equation for tf.einsum which describes how kernel and data (x) interact.
        kernel: A kernel.
        kernel_reverse: kernel with the frequency axes reversed.
        x: Data
        x_reverse: x with the frequency axes reversed.

    Returns:
        A tensor in the frequency domain.
    """
    h1 = tf.einsum(equation, kernel, x + x_reverse)
    h2 = tf.einsum(equation, kernel_reverse, x - x_reverse)
    return (h1 + h2) * 0.5


def get_reverse(x, axis):
    """Get x[N-k] by 'reverse then roll by 1'.
    'reverse' converts x[k] to x[N-1-k], then 'roll by 1' changes x[N-1-k] to x[N-k] as x[0] = x[N].

    Args:
        x: Input tensor.
        axis: Which axes to reverse and roll. Must be a list or tuple.

    Returns:
        x[N-k]
    """
    assert isinstance(axis, (list, tuple))
    shift = [1] * len(axis)
    return tf.roll(tf.reverse(x, axis), shift, axis)
