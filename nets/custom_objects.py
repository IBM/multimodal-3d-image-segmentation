#
# Copyright 2023 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Collections of different custom objects for Keras.

Author: Ken C. L. Wong
"""

from nets.custom_losses import CombineLosses, PCCLoss, DiceLoss
from nets.fourier_operator import FourierOperator
from nets.hartley_operator import HartleyOperator
from nets.hartley_mha import HartleyMultiHeadAttention

__author__ = 'Ken C. L. Wong'


custom_objects = {
    'CombineLosses': CombineLosses,
    'PCCLoss': PCCLoss,
    'DiceLoss': DiceLoss,
    'FourierOperator': FourierOperator,
    'HartleyOperator': HartleyOperator,
    'HartleyMultiHeadAttention': HartleyMultiHeadAttention,
}
