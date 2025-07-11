#
# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""This module contains the procedures of training and testing a model.

Author: Ken C. L. Wong
"""

import copy
import sys
import os
from functools import partial
import SimpleITK as sitk
import torch

from multimodal_3d_image_segmentation.experiments.data_io.input_data import InputData
from multimodal_3d_image_segmentation import nets
from multimodal_3d_image_segmentation.nets import custom_losses
from multimodal_3d_image_segmentation.experiments.train_test import training, testing
from multimodal_3d_image_segmentation.experiments.metrics import statistics_regional
from multimodal_3d_image_segmentation.experiments.utils import (get_config, save_config, get_data_lists,
                                                                normalize_modalities, read_img)

__author__ = 'Ken C. L. Wong'


def run(config_args):  # noqa
    """Runs an experiment.
    Using the hyperparameters provided in `config_args`, this function trains a model or reads
    a pre-trained model. Testing is performed on the trained model and the results statistics
    are computed.

    Args:
        config_args: A dict of configurations.
    """
    output_dir = os.path.expanduser(config_args['main']['output_dir'])
    torch.cuda.set_device(int(config_args['main']['visible_devices']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #
    # Create InputData as a sample generator

    input_lists = copy.deepcopy(config_args['input_lists'])
    data_dir = os.path.expanduser(input_lists.get('data_dir'))
    data_lists_train = get_data_lists(input_lists.get('data_lists_train_paths'), data_dir)
    data_lists_valid = get_data_lists(input_lists.get('data_lists_valid_paths'), data_dir)
    data_lists_test = get_data_lists(input_lists.get('data_lists_test_paths'), data_dir)

    input_args = copy.deepcopy(config_args['input_args'])
    if input_args.pop('use_data_normalization', True):
        mask_val = input_args.pop('mask_val', 0)
        clip_val = input_args.pop('clip_val', None)
        x_processing = partial(normalize_modalities, mask_val=mask_val, clip_val=clip_val)
    else:
        x_processing = None

    input_data = None
    transform_args = config_args.get('augmentation')
    if config_args['main']['is_train'] or config_args['main']['is_test']:
        input_data = InputData(reader=read_img,
                               data_lists_train=data_lists_train,
                               data_lists_valid=data_lists_valid,
                               data_lists_test=data_lists_test,
                               x_processing=x_processing,
                               transform_kwargs=transform_args,
                               **input_args)

    #
    # Train or read model

    model = None
    if config_args['main']['is_train']:
        # To avoid overwriting an existing model
        if os.path.exists(output_dir) and not config_args['main'].get('is_continue', False):
            raise RuntimeError(f'output_dir already exists! \n{output_dir}')

        os.makedirs(output_dir, exist_ok=True)
        save_config(config_args, output_dir)

        model_args = copy.deepcopy(config_args['model'])
        model_args['in_channels'] = input_data.get_num_x_modalities()
        model_args['ndim'] = len(input_data.get_train_image_size()) + 2
        model_args['device'] = device
        model_name = model_args.pop('model_name')
        model = getattr(nets, model_name)(**model_args)

        optimizer_args = copy.deepcopy(config_args['optimizer'])
        optimizer_name = optimizer_args.pop('optimizer_name')
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_args)

        # Learning rate scheduler
        scheduler = None
        if 'scheduler' in config_args:
            scheduler_args = copy.deepcopy(config_args['scheduler'])
            scheduler_name = scheduler_args.pop('scheduler_name')
            if scheduler_name == 'CosineAnnealingWarmRestarts':
                if 'T_0' not in scheduler_args and 'restart_epochs' not in scheduler_args:
                    scheduler_args['T_0'] = input_data.get_train_num_batches() * config_args['train']['num_epochs']
                elif 'restart_epochs' in scheduler_args:
                    scheduler_args['T_0'] = input_data.get_train_num_batches() * scheduler_args.pop('restart_epochs')
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_args)

        loss_args = copy.deepcopy(config_args['loss'])
        loss_name = loss_args.pop('loss_name')
        if hasattr(custom_losses, loss_name):
            loss_fn = getattr(custom_losses, loss_name)(**loss_args)
        else:
            loss_fn = getattr(torch.nn, loss_name)(**loss_args)

        train_args = copy.deepcopy(config_args['train'])
        train_args['model'] = model
        train_args['input_data'] = input_data
        train_args['output_dir'] = output_dir
        train_args['loss_fn'] = loss_fn
        train_args['optimizer'] = optimizer
        train_args['scheduler'] = scheduler
        train_args['device'] = device

        # Train model
        model = training(**train_args)

    elif config_args['main']['is_test']:
        model_path = os.path.join(output_dir, 'model/model.pt')

        model_args = copy.deepcopy(config_args['model'])
        model_args['in_channels'] = input_data.get_num_x_modalities()
        model_args['ndim'] = len(input_data.get_test_image_size()) + 2
        model_args['device'] = device
        model_name = model_args.pop('model_name')
        model = getattr(nets, model_name)(**model_args)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

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

    if config_args['main']['is_test']:
        test_args['model'] = model
        test_args['input_data'] = input_data
        test_args['output_dir'] = test_dir
        test_args['is_print'] = is_print
        test_args['device'] = device
        testing(**test_args)

    if config_args['main']['is_statistics']:
        idx_y_modalities = input_args.get('idx_y_modalities')
        if idx_y_modalities:
            if is_print:
                print('\nComputing statistics')

            idx_y = idx_y_modalities[0]
            y_list_test = data_lists_test[idx_y]

            ids = [fn.split('/')[-2] for fn in y_list_test]
            fn_true = [os.path.join(str(test_dir), 'images', f'{i}_true.nii.gz') for i in ids]
            fn_pred = [os.path.join(str(test_dir), 'images', f'{i}_pred.nii.gz') for i in ids]
            y_true = [sitk.GetArrayFromImage(sitk.ReadImage(fn)) for fn in fn_true]
            y_pred = [sitk.GetArrayFromImage(sitk.ReadImage(fn)) for fn in fn_pred]
            assert len(y_true) == len(y_pred)

            if is_print:
                print(f'There are {len(y_true)} samples loaded.')

            use_surface_dice = True
            use_hd95 = True
            region_names = region_labels = None
            if 'statistics' in config_args:
                use_surface_dice = config_args['statistics'].get('use_surface_dice', True)
                use_hd95 = config_args['statistics'].get('use_hd95', True)
                region_names = config_args['statistics'].get('region_names', None)
                region_labels = config_args['statistics'].get('region_labels', None)

            nproc = config_args['input_args']['num_workers']
            if is_print:
                print('-------- Regional result statistics --------')
            statistics_regional(y_true, y_pred, y_list_test, test_dir, region_names, region_labels, is_print,
                                use_surface_dice=use_surface_dice, use_hd95=use_hd95, nproc=nproc)
        else:
            print('Statistics cannot be computed without valid idx_y_modalities (ground truths).')


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')
    config_args = get_config(sys.argv[1])
    run(config_args)
