#
# Copyright 2023 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Custom Keras loss functions.

Author: Ken C. L. Wong
"""

from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from tensorflow.keras import backend

__author__ = 'Ken C. L. Wong'


def corrcoef(y_true, y_pred):
    """Computes the Pearson's correlation coefficients.

    Args:
        y_true: One-hot ground truth labels.
        y_pred: Prediction scores.

    Returns:
        Pearson's correlation coefficients with shape (batch_size, num_labels).
    """
    ndim = backend.ndim(y_true)
    axis = list(range(ndim))[1:-1]  # Spatial dimensions

    assert ndim in [3, 4, 5]

    y_true = y_true - backend.mean(y_true, axis=axis, keepdims=True)
    y_pred = y_pred - backend.mean(y_pred, axis=axis, keepdims=True)

    tp = backend.sum(y_true * y_pred, axis=axis)
    tt = backend.sum(backend.square(y_true), axis=axis)
    pp = backend.sum(backend.square(y_pred), axis=axis)

    output = tp / backend.sqrt(tt * pp + backend.epsilon())

    return output


def pcc_loss(y_true, y_pred):
    """Loss function based on the Pearson's correlation coefficient (PCC).

    Please refer to our MLMI 2022 paper for more details:
    Wong, K.C.L., Moradi, M. (2022). 3D Segmentation with Fully Trainable Gabor Kernels
    and Pearson’s Correlation Coefficient. In: Machine Learning in Medical Imaging. MLMI 2022.
    https://doi.org/10.1007/978-3-031-21014-3_6

    Args:
        y_true: One-hot ground truth labels.
        y_pred: Prediction scores.

    Returns:
        The PCC loss with shape (batch_size,).
    """
    output = corrcoef(y_true, y_pred)  # (-1, 1)
    output = (output + 1) * 0.5  # (0, 1)
    output = 1 - output
    return backend.mean(output, axis=-1)


class PCCLoss(LossFunctionWrapper):
    """Loss function based on the Pearson's correlation coefficient (PCC).

    Please refer to our MLMI 2022 paper for more details:
    Wong, K.C.L., Moradi, M. (2022). 3D Segmentation with Fully Trainable Gabor Kernels
    and Pearson’s Correlation Coefficient. In: Machine Learning in Medical Imaging. MLMI 2022.
    https://doi.org/10.1007/978-3-031-21014-3_6

    Args:
        reduction: Type of `tf.keras.losses.Reduction` to apply to loss. Default value is `AUTO`.
        name: Optional name for the instance.
    """
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='pcc_loss'):
        super().__init__(
            pcc_loss,
            name=name,
            reduction=reduction
        )


def dice_coef(y_true, y_pred):
    """Computes the (soft) Dice coefficients.

    Args:
        y_true: One-hot ground truth labels.
        y_pred: Prediction scores.

    Returns:
        The (soft) Dice coefficients with shape (batch_size, num_labels).
    """
    ndim = backend.ndim(y_true)
    axis = list(range(ndim))[1:-1]  # Spatial dimensions

    assert ndim in [3, 4, 5]

    intersection = backend.sum(y_true * y_pred, axis=axis)
    union = backend.sum(y_true + y_pred, axis=axis)
    return 2. * intersection / (union + backend.epsilon())


def dice_loss(y_true, y_pred):
    """Dice loss.

    Args:
        y_true: One-hot ground truth labels.
        y_pred: Prediction scores.

    Returns:
        The Dice loss with shape (batch_size,).
    """
    output = dice_coef(y_true, y_pred)
    output = 1 - output
    return backend.mean(output, axis=-1)


class DiceLoss(LossFunctionWrapper):
    """Dice loss.

    Args:
        reduction: Type of `tf.keras.losses.Reduction` to apply to loss. Default value is `AUTO`.
        name: Optional name for the instance.
    """
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='dice_loss'):
        super().__init__(
            dice_loss,
            name=name,
            reduction=reduction
        )


def combine_losses(y_true, y_pred, losses, loss_weights=None):
    """Combines multiple losses into one.

    Args:
        y_true: Ground truth labels.
        y_pred: Prediction scores.
        losses: A list of callable loss functions.
        loss_weights: A list of weights.

    Returns:
        A combined loss.
    """
    assert isinstance(losses, (list, tuple))

    loss_vals = [backend.mean(ls(y_true, y_pred)) for ls in losses]

    output = 0
    if loss_weights is not None:
        for val, weight in zip(loss_vals, loss_weights):
            output += weight * val
    else:
        for val in loss_vals:
            output += val

    return output


class CombineLosses(LossFunctionWrapper):
    """Combines multiple losses into one.

    Args:
        losses: A list of callable loss functions.
        loss_weights: A list of weights.
        reduction: Type of `tf.keras.losses.Reduction` to apply to loss. Default value is `AUTO`.
        name: Optional name for the instance.
    """
    def __init__(self,
                 losses,
                 loss_weights=None,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='combine_losses'):
        super().__init__(
            combine_losses,
            name=name,
            reduction=reduction,
            losses=losses,
            loss_weights=loss_weights
        )
