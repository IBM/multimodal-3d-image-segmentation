#
# Copyright 2023 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Helper functions for creating architectures.

Author: Ken C. L. Wong
"""

from tensorflow.keras.layers import Cropping2D, ZeroPadding2D, Cropping3D, ZeroPadding3D
from tensorflow.keras import backend
from tensorflow.keras import initializers, regularizers, constraints, losses

import numpy as np
from typing import List, Tuple, Union

__author__ = 'Ken C. L. Wong'


def spatial_padcrop(x, target_shape):
    """Performs spatial cropping and/or padding.
    Nothing is done if the shapes are already matched.

    Args:
        x: The tensor to be reshaped.
        target_shape: Target shape.

    Returns:
        A reshaped tensor.
    """
    ndim = backend.ndim(x)
    padding, cropping = get_spatial_padcrop(x, target_shape)

    if np.sum(padding) != 0:
        op = ZeroPadding2D if ndim == 4 else ZeroPadding3D
        x = op(padding)(x)

    if np.sum(cropping) != 0:
        op = Cropping2D if ndim == 4 else Cropping3D
        x = op(cropping)(x)

    return x


def get_spatial_padcrop(x, target_shape):
    """Computes the amount needed to be padded and cropped.
    Returned values can be used by Keras layers or backend functions.

    Args:
        x: The tensor to be reshaped.
        target_shape: Target shape.

    Returns:
        The padding and cropping lists.
    """
    if len(target_shape) == backend.ndim(x):
        target_shape = np.array(target_shape[1:-1])
    shape = np.array(backend.int_shape(x)[1:-1])

    ndim = len(shape)
    zeros = (0, 0)  # Lower and upper

    if np.array_equal(target_shape, shape):
        return [zeros] * ndim, [zeros] * ndim

    diff = target_shape - shape

    # Regardless of dimension, at most one padding and one cropping is enough
    padding = []
    cropping = []
    for d in diff:
        if d >= 0:
            cropping.append(zeros)
            q = d // 2
            if d % 2 == 0:
                padding.append((q, q))
            else:
                padding.append((q, q + 1))
        else:
            padding.append(zeros)
            d = -d
            q = d // 2
            if d % 2 == 0:
                cropping.append((q, q))
            else:
                cropping.append((q, q + 1))

    return padding, cropping


def get_loss(loss, loss_args: Union[dict, Tuple[dict], List[dict]] = None, custom_objects=None):
    """Gets callable loss functions.
    This is needed to process custom losses.

    Args:
        loss: Loss function(s). Can be a str, function, or a list of them.
        loss_args: Optional loss function arguments.
            If it is a list, each element is only used if it is a dict.
        custom_objects: Custom objects.

    Returns:
        A callable loss function, or a list of callable lost functions.
    """
    # Convert to list
    if not isinstance(loss, (list, tuple)):
        loss = [loss]
        if loss_args is not None:
            assert not isinstance(loss_args, (list, tuple))
            loss_args = [loss_args]

    loss_fns = []
    for i, ls in enumerate(loss):
        if loss_args is not None and isinstance(loss_args[i], dict):
            ls_args = loss_args[i]
            if callable(ls):
                ls = ls(**ls_args)
            elif isinstance(ls, str):
                ls = {'class_name': ls, 'config': ls_args}
            else:
                raise ValueError(f'{ls} is not a valid loss function.')
        loss_fns.append(get_single_loss(ls, custom_objects))

    if len(loss_fns) == 1:
        return loss_fns[0]
    else:
        return loss_fns


def get_single_loss(identifier, custom_objects=None):
    return get_deserialized_object(identifier, 'loss', custom_objects=custom_objects)


def get_deserialized_object(identifier, module_name, custom_objects=None):
    assert module_name in ['initializer', 'regularizer', 'constraint', 'loss']
    if module_name == 'initializer':
        deserialize = initializers.deserialize
    elif module_name == 'regularizer':
        deserialize = regularizers.deserialize
    elif module_name == 'constraint':
        deserialize = constraints.deserialize
    else:
        deserialize = losses.deserialize

    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier, custom_objects=custom_objects)
    elif isinstance(identifier, str):
        return deserialize(identifier, custom_objects=custom_objects)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret %s identifier: ' % module_name +
                         str(identifier))
