#
# Copyright 2023 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""This module contains the procedures of training and testing a model.

Author: Ken C. L. Wong
"""

import copy
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # noqa: E402
import numpy as np
from functools import partial

import tensorflow as tf
# tf.compat.v1.disable_eager_execution()  # noqa: E402 Run faster without memory leak
from keras import optimizers
from keras.models import load_model
from keras.optimizers import schedules

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))  # noqa: E402
from data_io.input_data import InputData
import nets
from nets.custom_objects import custom_objects
from experiments.train_test import training, testing, statistics, statistics_regional
from utils import get_config, save_config, get_data_lists, normalize_modalities, read_img

__author__ = 'Ken C. L. Wong'


def create_model(model_args, num_input_channels, optimizer_name, optimizer_args, scheduler_args=None,
                 steps_per_epoch=None):
    """Creates a model from hyperparameters.

    Args:
        model_args: Model specific hyperparameters (dict).
        num_input_channels: The number of input channels obtained from InputData.
        optimizer_name: The name of the optimizer (str).
        optimizer_args: Optimizer arguments (dict).
        scheduler_args: Learning rate scheduler arguments (dict).
            `scheduler_args['scheduler_name']` contains the name of the scheduler, e.g., 'CosineDecayRestarts'.
            If None (default), no scheduler is used and 'learning_rate' should be specified in `optimizer_args`.
        steps_per_epoch: Steps per epoch (default: None), should be the same as the number of training batches
            per epoch. Used to compute the `decay_steps` of the scheduler.

    Returns:
        A compiled Keras model.
    """
    model_args = copy.deepcopy(model_args)
    model_args['num_input_channels'] = num_input_channels
    builder_name = model_args.pop('builder_name')
    model_builder = getattr(nets, builder_name)
    model_args['optimizer'] = get_optimizer(optimizer_name, optimizer_args, scheduler_args, steps_per_epoch)
    model = model_builder(**model_args)()
    return model


def get_optimizer(optimizer_name, optimizer_args, scheduler_args=None, steps_per_epoch=None):
    """Gets the optimizer.

    Args:
        optimizer_name: The name of the optimizer (str).
        optimizer_args: Optimizer arguments (dict).
        scheduler_args: Learning rate scheduler arguments (dict).
            `scheduler_args['scheduler_name']` contains the name of the scheduler, e.g., 'CosineDecayRestarts'.
            If None (default), no scheduler is used and 'learning_rate' should be specified in `optimizer_args`.
        steps_per_epoch: Steps per epoch (default: None), should be the same as the number of training batches
            per epoch. Used to compute the `decay_steps` of the scheduler.

    Returns:
        An optimizer.
    """
    if scheduler_args is not None:
        scheduler = get_scheduler(scheduler_args, steps_per_epoch)
        optimizer_args = copy.deepcopy(optimizer_args)
        optimizer_args['learning_rate'] = scheduler
    return getattr(optimizers, optimizer_name)(**optimizer_args)


def get_scheduler(scheduler_args, steps_per_epoch):
    """Gets the learning rate scheduler.

    Args:
        scheduler_args: Learning rate scheduler arguments (dict).
            `scheduler_args['scheduler_name']` contains the name of the scheduler, e.g., 'CosineDecayRestarts'.
        steps_per_epoch: Steps per epoch, should be the same as the number of training batches
            per epoch. Used with `scheduler_args['decay_epochs']` to compute the `decay_steps` of the scheduler.

    Returns:
        A learning rate scheduler.
    """
    scheduler_args = copy.deepcopy(scheduler_args)
    scheduler = scheduler_args.pop('scheduler_name')
    decay_epochs = scheduler_args.pop('decay_epochs', None)
    if decay_epochs is not None:
        assert steps_per_epoch is not None
        decay_steps = decay_epochs * steps_per_epoch
        if scheduler == 'CosineDecayRestarts':
            scheduler_args['first_decay_steps'] = decay_steps
        else:
            scheduler_args['decay_steps'] = decay_steps
    return getattr(schedules, scheduler)(**scheduler_args)


def run(config_args):
    """Runs an experiment.
    Using the hyperparameters provided in `config_args`, this function trains a model or reads
    a pre-trained model. Testing is performed on the trained model and the results statistics
    are computed.

    Args:
        config_args: A dict of configurations.
    """
    output_dir = os.path.expanduser(config_args['main']['output_dir'])

    os.environ['CUDA_VISIBLE_DEVICES'] = config_args['main']['visible_devices']
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    #
    # Create InputData as a sample generator

    input_lists = copy.deepcopy(config_args['input_lists'])
    data_dir = os.path.expanduser(input_lists.get('data_dir'))
    data_lists_train = get_data_lists(input_lists.get('data_lists_train_paths'), data_dir)
    data_lists_valid = get_data_lists(input_lists.get('data_lists_valid_paths'), data_dir)
    data_lists_test = get_data_lists(input_lists.get('data_lists_test_paths'), data_dir)

    input_args = copy.deepcopy(config_args['input_args'])
    if input_args.pop('use_data_normalization', True):
        x_processing = partial(normalize_modalities, mask_val=0)  # Assume background value of 0
    else:
        x_processing = None

    input_data = None
    transform_kwargs = config_args.get('augmentation')
    if config_args['main']['is_train'] or config_args['main']['is_test']:
        input_data = InputData(reader=read_img,
                               data_lists_train=data_lists_train,
                               data_lists_valid=data_lists_valid,
                               data_lists_test=data_lists_test,
                               x_processing=x_processing,
                               transform_kwargs=transform_kwargs,
                               **input_args)

    #
    # Train or read model

    num_input_channels = len(input_args['idx_x_modalities'])
    model = None
    if config_args['main']['is_train']:
        # To avoid accidental overwriting of an existing model
        if os.path.exists(output_dir):
            raise RuntimeError(f'output_dir already exists! \n{output_dir}')

        os.makedirs(output_dir)
        save_config(config_args, output_dir)

        optimizer_args = copy.deepcopy(config_args['optimizer'])
        optimizer_name = optimizer_args.pop('optimizer_name')

        # Learning rate scheduler
        scheduler_args = steps_per_epoch = None
        if 'scheduler' in config_args:
            scheduler_args = copy.deepcopy(config_args['scheduler'])
            steps_per_epoch = input_data.get_train_num_batches()

        model_args = copy.deepcopy(config_args['model'])
        model_args['image_size'] = input_data.get_train_image_size()
        model = create_model(model_args, num_input_channels, optimizer_name, optimizer_args, scheduler_args,
                             steps_per_epoch)

        train_args = copy.deepcopy(config_args['train'])
        train_args['model'] = model
        train_args['input_data'] = input_data
        train_args['output_dir'] = output_dir

        # Train model
        model = training(**train_args)

    elif config_args['main']['is_test']:
        model_path = os.path.join(output_dir, 'model/model.keras')
        model = load_model(model_path, custom_objects=custom_objects)

        # If testing image size is different from the model's
        test_image_size = input_data.get_test_image_size()
        if test_image_size != model.input_shape[1:-1]:
            model_args = copy.deepcopy(config_args['model'])
            model_args['image_size'] = test_image_size
            model_new = create_model(model_args, num_input_channels, 'Adamax', {})  # Optimizer is no use in testing
            model_new.set_weights(model.get_weights())
            model = model_new
            print(f'\nModel is rebuilt for image size {test_image_size}.\n')

    if not config_args['main']['is_test'] and not config_args['main']['is_statistics']:
        return

    #
    # Testing

    test_args = copy.deepcopy(config_args['test'])
    test_dir = os.path.join(output_dir, test_args.pop('output_folder', 'test'))
    if 'is_print' not in test_args and 'train' in config_args:
        is_print = config_args['train'].get('is_print', True)
    else:
        is_print = test_args.get('is_print', True)

    y_true = None
    y_pred = None
    if config_args['main']['is_test']:
        test_args['model'] = model
        test_args['input_data'] = input_data
        test_args['output_dir'] = test_dir
        test_args['is_print'] = is_print
        y_true, y_pred = testing(**test_args)

    if config_args['main']['is_statistics']:
        idx_y_modalities = input_args.get('idx_y_modalities')
        if idx_y_modalities:
            if not config_args['main']['is_test']:  # Load from existing test results
                results = np.load(os.path.join(str(test_dir), 'y_true_pred.npz'))
                y_true, y_pred = results['y_true'], results['y_pred']
            idx_y = idx_y_modalities[0]
            statistics(y_true, y_pred, data_lists_test[idx_y], test_dir, is_print)
            statistics_regional(y_true, y_pred, data_lists_test[idx_y], test_dir, is_print)
        else:
            print('Statistics cannot be computed without valid idx_y_modalities (ground truths).')


if __name__ == '__main__':
    run(get_config(sys.argv[1]))
