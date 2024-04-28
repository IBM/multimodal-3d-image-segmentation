#
# Copyright 2023 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Discrete Hartley transforms implemented by FFT.

Author: Ken C. L. Wong
"""

import numpy as np
import tensorflow as tf

__author__ = 'Ken C. L. Wong'


def standardize_type_fft(x):
    """Standardizes data for TensorFlow FFT.

    Args:
        x: An ndarray or TensorFlow tensor to be standardized.

    Returns:
        A complex64 tensor or a float32 ndarray.
    """
    if tf.is_tensor(x):
        if x.dtype in ['float32', 'float64']:  # Must be converted to complex for TF tensor
            x = tf.complex(x, tf.zeros_like(x))
        x = tf.cast(x, tf.complex64)
    else:
        x = np.asarray(x, dtype=np.float32)  # TF fft converts ndarray automatically
    return x


def dht2d(x, is_inverse=False):
    """Computes discrete Hartley transform over the innermost dimensions.

    The FFT of TensorFlow is used. Note that inverse DHT is DHT divided by the number of
    elements in the innermost dimensions.

    Args:
        x: Input tensor/ndarray.
        is_inverse: True if inverse DHT is desired.

    Returns:
        The DHT output.
    """
    x = standardize_type_fft(x)
    x_fft = tf.signal.fft2d(x)
    x_hart = tf.math.real(x_fft) - tf.math.imag(x_fft)

    if is_inverse:
        x_hart = x_hart / tf.cast(tf.reduce_prod(x_hart.shape[-2:]), tf.float32)

    return x_hart


def dht3d(x, is_inverse=False):
    """Computes discrete Hartley transform over the innermost dimensions.

    The FFT of TensorFlow is used. Note that inverse DHT is DHT divided by the number of
    elements in the innermost dimensions.

    Args:
        x: Input tensor/ndarray.
        is_inverse: True if inverse DHT is desired.

    Returns:
        The DHT output.
    """
    x = standardize_type_fft(x)
    x_fft = tf.signal.fft3d(x)
    x_hart = tf.math.real(x_fft) - tf.math.imag(x_fft)

    if is_inverse:
        x_hart = x_hart / tf.cast(tf.reduce_prod(x_hart.shape[-3:]), tf.float32)

    return x_hart
