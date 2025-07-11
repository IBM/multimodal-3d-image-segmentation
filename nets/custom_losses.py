#
# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Custom loss functions.

Author: Ken C. L. Wong
"""

import torch
from torch.nn import Module

__author__ = 'Ken C. L. Wong'


def corrcoef(y_pred, y_true):
    """Computes the Pearson's correlation coefficients.

    Args:
        y_pred: Prediction scores.
        y_true: One-hot ground truth labels.

    Returns:
        Pearson's correlation coefficients with shape (batch_size, num_labels).
    """
    ndim = y_true.ndim
    axis = list(range(ndim))[2:]  # Spatial dimensions

    assert ndim in (3, 4, 5)

    y_true = y_true - torch.mean(y_true, dim=axis, keepdim=True)
    y_pred = y_pred - torch.mean(y_pred, dim=axis, keepdim=True)

    tp = torch.sum(y_true * y_pred, dim=axis)
    tt = torch.sum(torch.square(y_true), dim=axis)
    pp = torch.sum(torch.square(y_pred), dim=axis)

    output = tp / torch.sqrt(tt * pp + 1e-7)

    return output


class PCCLoss(Module):
    """Loss function based on the Pearson's correlation coefficient (PCC).

    Please refer to our MLMI 2022 paper for more details:
    Wong, K.C.L., Moradi, M. (2022). 3D Segmentation with Fully Trainable Gabor Kernels
    and Pearson’s Correlation Coefficient. In: Machine Learning in Medical Imaging. MLMI 2022.
    https://doi.org/10.1007/978-3-031-21014-3_6
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(y_pred, y_true):
        """Loss function based on the Pearson's correlation coefficient (PCC).

        Args:
            y_pred: Prediction scores.
            y_true: One-hot ground truth labels.

        Returns:
            The PCC loss.
        """
        output = corrcoef(y_pred, y_true)  # (-1, 1)
        output = (output + 1) * 0.5  # (0, 1)
        output = 1 - output
        return torch.mean(output)


def dice_coef(y_pred, y_true):
    """Computes the (soft) Dice coefficients.

    Args:
        y_pred: Prediction scores.
        y_true: One-hot ground truth labels.

    Returns:
        The (soft) Dice coefficients with shape (batch_size, num_labels).
    """
    ndim = y_true.ndim
    axis = list(range(ndim))[2:]  # Spatial dimensions

    assert ndim in (3, 4, 5)

    intersection = torch.sum(y_true * y_pred, dim=axis)
    union = torch.sum(y_true + y_pred, dim=axis)
    return 2. * intersection / (union + 1e-7)


class DiceLoss(Module):
    """Dice loss."""
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(y_pred, y_true):
        """Dice loss.

        Args:
            y_pred: Prediction scores.
            y_true: One-hot ground truth labels.

        Returns:
            The Dice loss.
        """
        output = dice_coef(y_pred, y_true)
        output = 1 - output
        return torch.mean(output)


class ExpDiceLoss(Module):
    """Exponential Dice loss."""
    def __init__(self, exp=0.3):
        super().__init__()
        self.exp = exp

    def forward(self, y_pred, y_true):
        """Exponential Dice loss.

        Args:
            y_pred: Prediction scores.
            y_true: One-hot ground truth labels.

        Returns:
            The exponential Dice loss.
        """
        output = dice_coef(y_pred, y_true)
        output = torch.clamp(output, 1e-7, 1.0 - 1e-7)
        output = torch.pow(-torch.log(output), self.exp)
        return torch.mean(output)
