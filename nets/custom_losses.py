#
# Copyright 2023 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Custom Keras loss functions.

Author: Ken C. L. Wong
"""

from keras.src.losses.losses import LossFunctionWrapper, Loss
import tensorflow as tf

__author__ = 'Ken C. L. Wong'


def corrcoef(y_true, y_pred):
    """Computes the Pearson's correlation coefficients.

    Args:
        y_true: One-hot ground truth labels.
        y_pred: Prediction scores.

    Returns:
        Pearson's correlation coefficients with shape (batch_size, num_labels).
    """
    ndim = y_true.ndim
    axis = list(range(ndim))[1:-1]  # Spatial dimensions

    assert ndim in [3, 4, 5]

    y_true = y_true - tf.reduce_mean(y_true, axis=axis, keepdims=True)
    y_pred = y_pred - tf.reduce_mean(y_pred, axis=axis, keepdims=True)

    tp = tf.reduce_sum(y_true * y_pred, axis=axis)
    tt = tf.reduce_sum(tf.square(y_true), axis=axis)
    pp = tf.reduce_sum(tf.square(y_pred), axis=axis)

    output = tp / tf.sqrt(tt * pp + 1e-7)

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
    return tf.reduce_mean(output, axis=-1)


class PCCLoss(LossFunctionWrapper):
    """Loss function based on the Pearson's correlation coefficient (PCC).

    Please refer to our MLMI 2022 paper for more details:
    Wong, K.C.L., Moradi, M. (2022). 3D Segmentation with Fully Trainable Gabor Kernels
    and Pearson’s Correlation Coefficient. In: Machine Learning in Medical Imaging. MLMI 2022.
    https://doi.org/10.1007/978-3-031-21014-3_6

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the instance.
    """
    def __init__(self,
                 reduction='sum_over_batch_size',
                 name='pcc_loss'):
        super().__init__(
            pcc_loss,
            name=name,
            reduction=reduction
        )

    def get_config(self):
        return Loss.get_config(self)


def dice_coef(y_true, y_pred):
    """Computes the (soft) Dice coefficients.

    Args:
        y_true: One-hot ground truth labels.
        y_pred: Prediction scores.

    Returns:
        The (soft) Dice coefficients with shape (batch_size, num_labels).
    """
    ndim = y_true.ndim
    axis = list(range(ndim))[1:-1]  # Spatial dimensions

    assert ndim in [3, 4, 5]

    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true + y_pred, axis=axis)
    return 2. * intersection / (union + 1e-7)


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
    return tf.reduce_mean(output, axis=-1)


class DiceLoss(LossFunctionWrapper):
    """Dice loss.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the instance.
    """
    def __init__(self,
                 reduction='sum_over_batch_size',
                 name='dice_loss'):
        super().__init__(
            dice_loss,
            name=name,
            reduction=reduction
        )

    def get_config(self):
        return Loss.get_config(self)
