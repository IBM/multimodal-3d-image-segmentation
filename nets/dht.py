#
# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Discrete Hartley transforms implemented by FFT.

Author: Ken C. L. Wong
"""

import torch

__author__ = 'Ken C. L. Wong'


def dhtn(x, dim, is_inverse=False):
    """Computes (inverse) discrete Hartley transform (DHT) over the innermost dimensions.

    torch.fft.fftn is used.

    Args:
        x: Input tensor.
        dim: Dimensions to be transformed.
        is_inverse: If True, inverse transform is performed (default: False).

    Returns:
        The DHT output.
    """
    # Using DHT with 1/N normalization (and inverse without normalization)
    # allows transforms of different image sizes to have similar intensity
    # ranges in the frequency domain, which is beneficial for super-resolution
    norm = 'backward' if is_inverse else 'forward'
    x_fft = torch.fft.fftn(x, dim=dim, norm=norm)
    x_hart = x_fft.real - x_fft.imag

    return x_hart


def dht2(x, is_inverse=False):
    """Computes (inverse) discrete Hartley transform (DHT) over the innermost dimensions.

    torch.fft.fftn is used.

    Args:
        x: Input tensor.
        is_inverse: If True, inverse transform is performed (default: False).

    Returns:
        The DHT output.
    """
    return dhtn(x, dim=(-2, -1), is_inverse=is_inverse)


def dht3(x, is_inverse=False):
    """Computes (inverse) discrete Hartley transform (DHT) over the innermost dimensions.

    torch.fft.fftn is used.

    Args:
        x: Input tensor.
        is_inverse: If True, inverse transform is performed (default: False).

    Returns:
        The DHT output.
    """
    return dhtn(x, dim=(-3, -2, -1), is_inverse=is_inverse)
