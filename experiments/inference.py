#
# Copyright 2023 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""This module is for performing inference of a trained model.

Author: Ken C. L. Wong
"""

import copy
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import time
from functools import partial

import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # Run faster without memory leak
from tensorflow.keras.models import load_model, Model

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from data_io.input_data import InputData
import nets
from nets.custom_objects import custom_objects
from utils import remap_labels, normalize_modalities, save_output
from utils import get_config, save_config, get_data_lists, read_img

__author__ = 'Ken C. L. Wong'


def run(config_args):
    """Runs an experiment.
    Using the hyperparameters provided in `config_args`, this function reads a pre-trained model
    and performs inference on testing data.

    Args:
        config_args: A dict of configurations.
    """
    target_dir = os.path.expanduser(config_args['main']['target_dir'])

    os.environ['CUDA_VISIBLE_DEVICES'] = config_args['main']['visible_devices']
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    #
    # Create InputData

    input_lists = copy.deepcopy(config_args['input_lists'])
    data_dir = os.path.expanduser(input_lists.get('data_dir'))
    data_lists_test = get_data_lists(input_lists.get('data_lists_test_paths'), data_dir)

    input_args = copy.deepcopy(config_args['input_args'])
    input_args['reader'] = read_img
    if input_args.pop('use_data_normalization', True):
        x_processing = partial(normalize_modalities, mask_val=0)  # Assume background value of 0
    else:
        x_processing = None

    input_data = InputData(data_lists_test=data_lists_test, x_processing=x_processing, **input_args)

    #
    # Load trained model

    model_path = os.path.join(target_dir, 'model/model.h5')
    model = load_model(model_path, custom_objects=custom_objects)

    # If testing image size is different from the model's,
    # we create a new model with the same model parameters.
    num_input_channels = len(input_args['idx_x_modalities'])
    test_image_size = input_data.get_test_image_size()
    if test_image_size != model.input_shape[1:-1]:
        model_args = copy.deepcopy(config_args['model'])
        model_args['image_size'] = test_image_size
        model_new = create_model(model_args, num_input_channels)
        model_new.set_weights(model.get_weights())
        model = model_new
        print(f'\nModel is rebuilt for image size {test_image_size}.\n')

    #
    # Inference

    output_dir = os.path.join(target_dir, config_args['test']['output_folder'])
    os.makedirs(output_dir, exist_ok=True)
    save_config(config_args, output_dir)

    label_mapping = config_args['test'].get('label_mapping')
    output_origin = config_args['test'].get('output_origin')
    inference(model, input_data, output_dir, label_mapping, output_origin)


def inference(
        model: Model,
        input_data: InputData,
        output_dir,
        label_mapping=None,
        output_origin=None,
):
    """This function performs prediction on testing data.

    Args:
        model: A trained model.
        input_data: InputData.
        output_dir: Output directory.
        label_mapping: A dict for label mapping if given (default: None).
        output_origin: Output origin (default: None).
    """
    test_num_batches = input_data.get_test_num_batches()

    print('test_num_batches:', test_num_batches)
    print()
    print('Testing started')

    start_time = time.time()

    assert input_data.batch_size == 1, 'A batch size of 1 is required to save the outputs one-by-one.'

    predict_times = []
    data_lists_test = input_data.data_lists_test
    n_batches = 0
    for x in input_data.get_test_flow():
        s_time = time.time()
        y_pred = model.predict_on_batch(x)
        e_time = time.time()
        if n_batches != 0:
            predict_times.append(e_time - s_time)

        y_pred = y_pred.argmax(-1).astype(np.int16)[0]
        if label_mapping is not None:
            y_pred = remap_labels(y_pred, label_mapping)

        save_output(y_pred, data_lists_test, n_batches, output_dir, output_origin)

        n_batches += 1
        if n_batches == test_num_batches:
            break

    end_time = time.time()

    input_data.stop_enqueuers()

    print()
    print(output_dir)
    print(f'Time used: {end_time - start_time:.2f} seconds.')
    print(f'Average prediction time: {np.mean(predict_times)}')
    with open(os.path.join(output_dir, 'time_used.txt'), 'w') as f:
        print(f'Time used: {end_time - start_time:.2f} seconds.', file=f)
        print(f'Average prediction time: {np.mean(predict_times)}', file=f)


def create_model(model_args, num_input_channels):
    """Creates a model from hyperparameters.

    Args:
        model_args: Model specific hyperparameters (dict).
        num_input_channels: The number of input channels obtained from InputData.

    Returns:
        A compiled Keras model.
    """
    model_args = copy.deepcopy(model_args)
    model_args['num_input_channels'] = num_input_channels
    builder_name = model_args.pop('builder_name')
    model_builder = getattr(nets, builder_name)
    model_args['optimizer'] = 'Adamax'  # Optimizer is no use in testing
    model = model_builder(**model_args)()
    return model


if __name__ == '__main__':
    run(get_config(sys.argv[1]))
