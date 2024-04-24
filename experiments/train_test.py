#
# Copyright 2023 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""This module contains functions for model training and testing.

Author: Ken C. L. Wong
"""

import os
import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import time
from os.path import join
import pandas as pd

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model

from data_io.input_data import InputData
from utils import remap_labels, to_categorical, save_output, save_model_summary


__author__ = 'Ken C. L. Wong'


def training(
        model: Model,
        input_data: InputData,
        output_dir,
        label_mapping=None,
        num_epochs=100,
        selection_epoch_portion=0.8,
        is_save_model=True,
        is_plot_model=False,
        is_print=True,
        plot_epoch_portion=None,
):
    """Trains a model.

    Args:
        model: A model to be trained.
        input_data: InputData.
        output_dir: Output directory, should already be created by the calling function.
        label_mapping: A dict for label mapping if given (default: None).
        num_epochs: Number of epochs (default: 100).
        selection_epoch_portion: The models after this portion of num_epochs are
            candidates for the final model (default: 0.8).
        is_save_model: The trained model is saved if True (default).
        is_plot_model: Plots the model architecture if True (default: False).
        is_print: Print info or not (default: True).
        plot_epoch_portion: The losses after this portion of num_epochs are plotted if not None (default: None).

    Returns:
        The trained model.
    """
    if os.path.exists(join(output_dir, 'stdout.txt')):
        raise RuntimeError('stdout.txt already exists!')  # Avoid accidents

    num_epochs = int(num_epochs)
    train_num_batches = input_data.get_train_num_batches()
    valid_num_batches = input_data.get_valid_num_batches()

    if is_print:
        print('\ntrain_num_batches:', train_num_batches)
        print('valid_num_batches:', valid_num_batches)
        print()
    with open(join(output_dir, 'stdout.txt'), 'a') as f:
        print('train_num_batches:', train_num_batches, file=f)
        print('valid_num_batches:', valid_num_batches, file=f)
        print(file=f)

    # Save model summary
    save_model_summary(model, join(output_dir, 'model_summary.txt'))
    if is_plot_model:
        plot_model(model, show_shapes=True, show_layer_names=True, to_file=join(output_dir, 'model.pdf'))

    train_flow = input_data.get_train_flow(shuffle=True, seed=np.random.randint(0, 10000))
    valid_flow = input_data.get_valid_flow()

    num_labels = model.output_shape[-1]

    if is_print:
        print('Training started')

    start_time = time.time()

    # Epoch average loss
    train_loss = []
    valid_loss = []

    min_loss = float('inf')
    best_epoch = None
    best_weights = None
    for epoch in range(num_epochs):
        #
        # Training phase

        train_loss_epoch = []
        n_batches = 0
        for x, y in train_flow:
            if label_mapping is not None:
                y = remap_labels(y, label_mapping)
            assert y.shape[-1] == 1, 'Can only handle single label per pixel.'
            y = to_categorical(y[..., 0], num_labels)

            loss = model.train_on_batch(x, y)
            train_loss_epoch.append(loss)

            n_batches += 1
            if n_batches == train_num_batches:
                break

        train_loss.append(np.mean(train_loss_epoch))

        if is_print:
            print('\n-------------------------')
            print(f'Epoch: {epoch}')
            print(f'train_loss: {train_loss[-1]}')
        with open(join(output_dir, 'stdout.txt'), 'a') as f:
            print('\n-------------------------', file=f)
            print(f'Epoch: {epoch}', file=f)
            print(f'train_loss: {train_loss[-1]}', file=f)

        #
        # Validation phase

        valid_loss_epoch = []
        n_batches = 0
        for x, y in valid_flow:
            if label_mapping is not None:
                y = remap_labels(y, label_mapping)
            assert y.shape[-1] == 1, 'Can only handle single label per pixel.'
            y = to_categorical(y[..., 0], num_labels)

            loss = model.test_on_batch(x, y)
            valid_loss_epoch.append(loss)

            n_batches += 1
            if n_batches == valid_num_batches:
                break

        valid_loss.append(np.mean(valid_loss_epoch))

        if is_print:
            print(f'valid_loss: {valid_loss[-1]}')
        with open(join(output_dir, 'stdout.txt'), 'a') as f:
            print(f'valid_loss: {valid_loss[-1]}', file=f)

        selection_epoch = int(num_epochs * selection_epoch_portion)
        if (epoch > selection_epoch or epoch == num_epochs - 1) and valid_loss[-1] < min_loss:
            min_loss = valid_loss[-1]
            best_epoch = epoch
            best_weights = model.get_weights()
            if is_save_model:
                save_model(model, join(output_dir, 'model', 'model.h5'))

    end_time = time.time()

    input_data.stop_enqueuers()

    if best_weights is not None:
        model.set_weights(best_weights)
    else:  # num_epochs == 0, i.e., no training
        if is_save_model:
            save_model(model, join(output_dir, 'model', 'model.h5'))

    # Plot losses
    start_epoch = int(num_epochs * plot_epoch_portion) if plot_epoch_portion is not None else 0
    losses = [train_loss, valid_loss]
    styles = ['r', 'b--']
    labels = ['Train loss', 'Valid loss']
    output_file = join(output_dir, 'plot_loss.pdf')
    plot_losses(num_epochs, start_epoch, losses, styles, labels, output_file)

    if is_print:
        print(f'\nTime used: {end_time - start_time:.2f} seconds.')
        print(f'Best epoch: {best_epoch}')
        print(f'Min loss: {min_loss}')
    with open(join(output_dir, 'stdout.txt'), 'a') as f:
        print(f'\nTime used: {end_time - start_time:.2f} seconds.', file=f)
        print(f'Best epoch: {best_epoch}', file=f)
        print(f'Min loss: {min_loss}', file=f)

    return model


def plot_losses(num_epochs, start_epoch, losses, styles, labels, output_file):
    """Plots the evolutions of losses."""
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)

    x = np.arange(num_epochs)[start_epoch:]
    for i in range(len(losses)):
        ax.plot(x, losses[i][start_epoch:], styles[i], label=labels[i])

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    ax.tick_params(labelsize=20)
    plt.grid(which='both')

    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper right', fancybox=True, framealpha=0.8, ncol=1)
    for label in legend.get_texts():
        label.set_fontsize(20)
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)


def testing(
        model: Model,
        input_data: InputData,
        output_dir,
        label_mapping=None,
        save_image=False,
        output_origin=None,
        is_print=True,
):
    """Performs prediction on testing data.

    Args:
        model: A trained model.
        input_data: InputData.
        output_dir: Output directory (full path).
        label_mapping: A dict for label mapping (default: None).
        save_image: True if saving images (default: False).
        output_origin: Output origin for nifty saving (default: None).
        is_print: Print info or not (default: True).

    Returns:
        All ground truths (y_true) and predictions (y_pred).
    """
    os.makedirs(output_dir, exist_ok=True)

    test_num_batches = input_data.get_test_num_batches()
    data_lists_test = input_data.data_lists_test

    if is_print:
        print('test_num_batches:', test_num_batches)
        print()

    test_flow = input_data.get_test_flow()

    if is_print:
        print('Testing started')

    start_time = time.time()

    predict_times = []
    y_true = []
    y_pred = []
    n_batches = 0
    for xy in test_flow:
        if isinstance(xy, (tuple, list)):
            x, y = xy
            y_true.append(np.asarray(y, dtype=np.int16)[..., 0])  # Last dimension of size 1 is ignored
        else:
            x = xy

        s_time = time.time()
        yp = model.predict_on_batch(x)
        e_time = time.time()
        if n_batches != 0:
            predict_times.append(e_time - s_time)
        y_pred.append(yp)

        n_batches += 1
        if n_batches == test_num_batches:
            break

    end_time = time.time()

    input_data.stop_enqueuers()

    y_true = np.concatenate(y_true) if y_true else None
    y_pred = np.concatenate(y_pred)

    # Change to int label and remap if needed
    y_pred = y_pred.argmax(-1).astype(np.int16)  # Last dimension is gone
    if label_mapping is not None:
        y_pred = remap_labels(y_pred, label_mapping)

    if save_image:
        for i, y in enumerate(y_pred):
            save_output(y, data_lists_test, i, os.path.join(output_dir, 'images'), output_origin, '_pred')
        if y_true.size:
            for i, y in enumerate(y_true):
                save_output(y, data_lists_test, i, os.path.join(output_dir, 'images'), output_origin, '_true')

    np.savez_compressed(join(output_dir, 'y_true_pred.npz'), y_true=y_true, y_pred=y_pred)

    if is_print:
        print(f'\nTime used: {end_time - start_time:.2f} seconds.')
        print(f'Average prediction time: {np.mean(predict_times)}')

    with open(os.path.join(output_dir, 'prediction_time.txt'), 'w') as f:
        print(f'Average prediction time: {np.mean(predict_times)}', file=f)

    return y_true, y_pred


def statistics(y_true, y_pred, y_list_test, output_dir, is_print=True):
    """Computes and saves the statistics on given predictions and ground truths.
    Sample-wise results are saved to a csv file, while average results are saved to a txt file.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_list_test: List of filenames correspond to the samples.
        output_dir: Output directory (full path).
        is_print: Print info if True (default).
    """
    dice_all = dice_coef(y_true, y_pred)  # (num_samples, num_labels)

    num_labels = dice_all.shape[-1]
    ids = pd.DataFrame([os.path.basename(fn) for fn in y_list_test])
    df = [ids] + [pd.DataFrame(dice_all[:, i]) for i in range(num_labels)]
    header = ['ID'] + [f'Label {lab}' for lab in np.unique(y_true)]

    output_file = os.path.join(output_dir, 'results.csv')
    pd.concat(df, axis=1).to_csv(output_file, sep=str('\t'), header=header, index=False, float_format=str('%.6f'))

    dice_all = np.ma.array(dice_all, mask=np.isnan(dice_all))
    dice_mean = list(dice_all.mean(0).filled(np.nan))
    dice_std = list(dice_all.std(0).filled(np.nan))

    if is_print:
        print()
        print('-------- Result statistics --------')
        print(f'dice_mean: {dice_mean}')
        print(f'dice_std: {dice_std}')

    with open(os.path.join(output_dir, 'average_results.txt'), 'w') as f:
        print('-------- Result statistics --------', file=f)
        print(f'dice_mean: {dice_mean}', file=f)
        print(f'dice_std: {dice_std}', file=f)
        print(file=f)


def statistics_regional(y_true, y_pred, y_list_test, output_dir, is_print=True):
    """Computes and saves the statistics on given predictions and ground truths.
    Labels are grouped into BraTS regions of 'whole tumor', 'tumor core', and 'enhancing tumor'.
    Sample-wise results are saved to a csv file, while average results are saved to a txt file.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_list_test: List of filenames correspond to the samples.
        output_dir: Output directory (full path).
        is_print: Print info if True (default).
    """
    region_names = ['background', 'whole tumor', 'tumor core', 'enhancing tumor']
    region_labels = [
        [0],
        [1, 2, 4],
        [1, 4],
        [4],
    ]

    def get_labels_union(y, target_labels):
        output = None
        for lab in target_labels:
            if output is None:
                output = (y == lab)
            else:
                output = output | (y == lab)
        return np.asarray(output, dtype=int)

    dice_all = []
    for labs in region_labels:
        yt = get_labels_union(y_true, labs)
        yp = get_labels_union(y_pred, labs)
        dice_all.append(dice_coef(yt, yp, labels=[1]))
    dice_all = np.concatenate(dice_all, axis=1)  # (num_samples, num_labels)

    num_labels = dice_all.shape[-1]
    ids = pd.DataFrame([os.path.basename(fn) for fn in y_list_test])
    df = [ids] + [pd.DataFrame(dice_all[:, i]) for i in range(num_labels)]
    header = ['ID'] + region_names

    output_file = os.path.join(output_dir, 'results_regional.csv')
    pd.concat(df, axis=1).to_csv(output_file, sep=str('\t'), header=header, index=False, float_format=str('%.6f'))

    dice_all = np.ma.array(dice_all, mask=np.isnan(dice_all))
    dice_mean = list(dice_all.mean(0).filled(np.nan))
    dice_std = list(dice_all.std(0).filled(np.nan))

    if is_print:
        print()
        print('-------- Regional result statistics --------')
        print(f'region_names: {region_names}')
        print(f'dice_mean: {dice_mean}')
        print(f'dice_std: {dice_std}')

    with open(os.path.join(output_dir, 'average_results_regional.txt'), 'w') as f:
        print('-------- Regional result statistics --------', file=f)
        print(f'region_names: {region_names}', file=f)
        print(f'dice_mean: {dice_mean}', file=f)
        print(f'dice_std: {dice_std}', file=f)
        print(file=f)


def dice_coef(y_true, y_pred, labels=None, is_average=False):
    """Computes the Dice coefficients of specified labels.

    Args:
        y_true: Ground truths. Can be bhw or bdhw with or without the last channel of size 1.
        y_pred: Predictions. Can be bhw or bdhw with or without last channel of size 1.
        labels: Labels for which the Dice coefficients are computed.
        is_average: If True, the averaged Dice coefficients are returned with shape (num_labels,).
            Otherwise, returns a 2D array of shape (b, num_labels) (default: False).

    Returns:
        The Dice coefficients of the labels.
    """
    y_true = y_true.reshape(len(y_true), -1)  # (b, num_pixels)
    y_pred = y_pred.reshape(len(y_pred), -1)  # (b, num_pixels)
    assert y_true.shape == y_pred.shape

    if labels is None:
        labels = np.unique(y_true)

    # Compute Dice coefficients
    dice_all = []
    for y_true_img, y_pred_img in zip(y_true, y_pred):  # Loop through images
        dice = []
        for label in labels:
            y_true_bin = (y_true_img == label)
            y_pred_bin = (y_pred_img == label)
            intersection = np.count_nonzero(y_true_bin & y_pred_bin)
            y_true_count = np.count_nonzero(y_true_bin)
            y_pred_count = np.count_nonzero(y_pred_bin)
            if y_true_count:
                dice.append(2 * intersection / (y_true_count + y_pred_count))
            else:
                dice.append(np.nan)  # label does not exist in y_true
        dice_all.append(dice)

    dice_all = np.ma.array(dice_all, mask=np.isnan(dice_all))  # (b, num_labels)

    if is_average:
        return dice_all.mean(0).filled(np.nan)
    else:
        return dice_all.filled(np.nan)


def save_model(model, output_path):
    """Saves a Keras model.

    Args:
        model: The model to be saved.
        output_path: The full file path.
    """
    dirname = os.path.dirname(output_path)
    os.makedirs(dirname, exist_ok=True)

    if os.path.exists(output_path):
        os.remove(output_path)  # To avoid occasional h5py crashing when overwriting

    model.save(str(output_path))  # h5py compares with str while output_dir is unicode
