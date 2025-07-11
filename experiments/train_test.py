#
# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""This module contains functions for model training and testing.

Author: Ken C. L. Wong
"""
import copy
import os
import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import time
from os.path import join
import re

import torch
import torchview

from .data_io.input_data import InputData
from .utils import remap_labels, to_categorical, save_output, save_model_summary


__author__ = 'Ken C. L. Wong'


def training(
        model: torch.nn.Module,
        input_data: InputData,
        output_dir,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        label_mapping=None,
        num_epochs=100,
        selection_epoch_portion=0.8,
        checkpoint_epoch=10,
        is_plot_model=False,
        is_print=True,
        plot_epoch_portion=None,
        use_autocast=False,
        device=None,
):
    """Trains a model.

    Args:
        model: A model to be trained.
        input_data: InputData.
        output_dir: Output directory, should already be created by the calling function.
        loss_fn: The loss function.
        optimizer: The optimizer.
        scheduler: A learning rate scheduler.
        label_mapping: A dict for label mapping if given (default: None).
        num_epochs: Number of epochs (default: 100).
        selection_epoch_portion: The models after this portion of num_epochs are
            candidates for the final model (default: 0.8).
        checkpoint_epoch: The number of epochs to save checkpoint (default: 10).
            Checkpoint is also saved with the current best model.
        is_plot_model: Plots the model architecture if True (default: False).
        is_print: Print info or not (default: True).
        plot_epoch_portion: The losses after this portion of num_epochs are plotted if not None (default: None).
        use_autocast: If True, PyTorch autocast is used (default: False).
            Autocast may raise RuntimeError with cuFFT.
        device: Device index to select, e.g., 'cuda', 'cpu' (default: None).

    Returns:
        The trained model.
    """
    model_dir = join(output_dir, 'model')
    model_path = join(model_dir, 'model.pt')
    chkpt_path = join(model_dir, 'checkpoint.pt')
    stdout_file = join(output_dir, 'stdout.txt')
    os.makedirs(model_dir, exist_ok=True)

    scaler = torch.amp.GradScaler() if use_autocast else None
    model.to(device)

    if os.path.exists(chkpt_path):
        start_epoch, min_loss, best_epoch = load_checkpoint(chkpt_path, model, optimizer, scheduler, scaler, device)
        start_epoch += 1
        if start_epoch >= num_epochs:
            raise RuntimeError(f'Checkpoint detected, but start_epoch ({start_epoch}) >= num_epochs ({num_epochs})')
        if is_print:
            print(f'Checkpoint loaded for epoch {start_epoch}')

        # Remove stdout.txt contents after the last checkpoint
        with open(stdout_file) as f:
            lines = f.readlines()[::-1]
        idx = None
        for i in range(len(lines)):
            if 'checkpoint' in lines[i]:
                idx = i
                break
        lines = lines[idx:][::-1]
        with open(stdout_file, 'w') as f:
            f.writelines(lines)
    else:
        start_epoch = 0
        min_loss = float('inf')
        best_epoch = None

        train_num_batches = input_data.get_train_num_batches()
        valid_num_batches = input_data.get_valid_num_batches()
        if is_print:
            print('\ntrain_num_batches:', train_num_batches)
            print('valid_num_batches:', valid_num_batches)
            print()
        with open(stdout_file, 'a') as f:
            print('train_num_batches:', train_num_batches, file=f)
            print('valid_num_batches:', valid_num_batches, file=f)
            print(file=f)

        # Save model summary. Use a copy of the model to avoid modifying the model.
        input_size = (1, model.in_channels) + input_data.get_train_image_size()
        save_model_summary(copy.deepcopy(model), input_size, join(output_dir, 'model_summary.txt'))
        if is_plot_model:
            graph = torchview.draw_graph(copy.deepcopy(model), input_size=input_size, device='meta')
            graph.visual_graph.render(filename='model_graph', directory=output_dir, cleanup=True, format='pdf')

    train_flow = input_data.get_train_flow(shuffle=True)
    valid_flow = input_data.get_valid_flow()

    num_labels = model.out_channels  # Please provide out_channels in your model

    if use_autocast:
        assert device is not None
        device = torch.device(device)

    if is_print:
        print('Training started')
        print(output_dir)

    start_time = time.time()

    # Epoch average loss
    for epoch in range(start_epoch, num_epochs):
        #
        # Training phase

        model.train()
        train_loss_epoch = []
        for x, y in train_flow:
            x = x.to(device)
            y = y.to(device)

            if label_mapping is not None:
                y = remap_labels(y, label_mapping)
            y = to_categorical(y, num_labels)

            if use_autocast:
                with torch.autocast(device_type=device.type):
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)
            else:
                y_pred = model(x)
                loss = loss_fn(y_pred, y)

            train_loss_epoch.append(loss.item())

            optimizer.zero_grad()
            if use_autocast:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

        train_loss = np.mean(train_loss_epoch)
        if is_print:
            print('\n-------------------------')
            print(f'Epoch: {epoch}')
            print(f'train_loss: {train_loss}')
        with open(stdout_file, 'a') as f:
            print('\n-------------------------', file=f)
            print(f'Epoch: {epoch}', file=f)
            print(f'train_loss: {train_loss}', file=f)

        #
        # Validation phase

        model.eval()
        valid_loss_epoch = []
        for x, y in valid_flow:
            x = x.to(device)
            y = y.to(device)

            if label_mapping is not None:
                y = remap_labels(y, label_mapping)
            y = to_categorical(y, num_labels)

            with torch.no_grad():
                if use_autocast:
                    with torch.autocast(device_type=device.type):
                        y_pred = model(x)
                        loss = loss_fn(y_pred, y)
                else:
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)

                valid_loss_epoch.append(loss.item())

        valid_loss = np.mean(valid_loss_epoch)
        if is_print:
            print(f'valid_loss: {valid_loss}')
        with open(stdout_file, 'a') as f:
            print(f'valid_loss: {valid_loss}', file=f)

        if (epoch + 1) % checkpoint_epoch == 0:
            save_checkpoint(chkpt_path, epoch, model, optimizer, scheduler, min_loss, best_epoch, scaler)
            if is_print:
                print('Standard checkpoint saved.')
            with open(stdout_file, 'a') as f:
                print('Standard checkpoint saved.', file=f)

        selection_epoch = int(num_epochs * selection_epoch_portion)
        if (epoch > selection_epoch or epoch == num_epochs - 1) and valid_loss < min_loss:
            min_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            if (epoch + 1) % checkpoint_epoch != 0:  # Avoid saving twice
                save_checkpoint(chkpt_path, epoch, model, optimizer, scheduler, min_loss, best_epoch, scaler)
                if is_print:
                    print('Best checkpoint saved.')
                with open(stdout_file, 'a') as f:
                    print('Best checkpoint saved.', file=f)

    end_time = time.time()

    if best_epoch is not None:
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    else:  # num_epochs == 0, i.e., no training
        torch.save(model.state_dict(), model_path)

    # Plot losses
    start_plot_epoch = int(num_epochs * plot_epoch_portion) if plot_epoch_portion is not None else 0
    losses = get_losses_from_file(stdout_file)
    styles = ['r', 'b--']
    labels = ['Train loss', 'Valid loss']
    output_file = join(output_dir, 'plot_loss.pdf')
    plot_losses(num_epochs, start_plot_epoch, losses, styles, labels, output_file)

    if is_print:
        print(f'\nTime used: {end_time - start_time:.2f} seconds.')
        print(f'Best epoch: {best_epoch}')
        print(f'Min loss: {min_loss}')
    with open(stdout_file, 'a') as f:
        print(f'\nTime used: {end_time - start_time:.2f} seconds.', file=f)
        print(f'Best epoch: {best_epoch}', file=f)
        print(f'Min loss: {min_loss}', file=f)

    return model


def save_checkpoint(chkpt_path, epoch, model, optimizer, scheduler, min_loss, best_epoch, scaler):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'min_loss': min_loss,
        'best_epoch': best_epoch,
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    torch.save(checkpoint, chkpt_path)


def load_checkpoint(chkpt_path, model, optimizer, scheduler, scaler, device):
    checkpoint = torch.load(chkpt_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint['epoch']
    min_loss = checkpoint['min_loss']
    best_epoch = checkpoint['best_epoch']
    return epoch, min_loss, best_epoch


def get_losses_from_file(filename):
    with open(filename) as f:
        lines = f.readlines()

    train_loss = []
    valid_loss = []
    for ln in lines:
        if 'train_loss' in ln:
            train_loss.append(float(re.findall('train_loss: (.+)', ln)[0]))
        elif 'valid_loss' in ln:
            valid_loss.append(float(re.findall('valid_loss: (.+)', ln)[0]))

    assert len(train_loss) == len(valid_loss)
    return train_loss, valid_loss


def plot_losses(num_epochs, start_plot_epoch, losses, styles, labels, output_file):
    """Plots the evolutions of losses."""
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)

    x = np.arange(num_epochs)[start_plot_epoch:]
    for i in range(len(losses)):
        ax.plot(x, losses[i][start_plot_epoch:], styles[i], label=labels[i])

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
        model: torch.nn.Module,
        input_data: InputData,
        output_dir,
        label_mapping=None,
        output_origin=None,
        is_print=True,
        use_autocast=False,
        device=None,
):
    """Performs prediction on testing data.

    Args:
        model: A trained model.
        input_data: InputData.
        output_dir: Output directory (full path).
        label_mapping: A dict for label mapping (default: None).
        output_origin: Output origin (x, y, z) for nifty saving (default: None).
        is_print: Print info or not (default: True).
        use_autocast: If True, PyTorch autocast is used (default: False).
            Autocast may raise RuntimeError with cuFFT.
        device: Device index to select, e.g., 'cuda', 'cpu' (default: None).

    Returns:
        All ground truths (y_true) and predictions (y_pred).
    """
    assert input_data.batch_size == 1
    os.makedirs(output_dir, exist_ok=True)

    test_num_batches = input_data.get_test_num_batches()
    data_lists_test = input_data.data_lists_test

    if is_print:
        print('test_num_batches:', test_num_batches)
        print()

    if use_autocast:
        assert device is not None
        if not isinstance(device, torch.device):
            device = torch.device(device)

    test_flow = input_data.get_test_flow()
    model.to(device)
    model.eval()

    if is_print:
        print('Testing started')
        print(output_dir)

    start_time = time.time()

    predict_times = []
    for i, xy in enumerate(test_flow):
        s_time = time.time()

        y_true = None
        if isinstance(xy, (tuple, list)):
            x, y = xy
            y_true = np.asarray(y, dtype=np.uint8)[0, 0]  # (1, 1, D, H, W) to (D, H, W)
        else:
            x = xy
        x = x.to(device)

        with torch.no_grad():
            if use_autocast:
                with torch.autocast(device_type=device.type):
                    yp = model(x)
            else:
                yp = model(x)

        y_pred = np.asarray(yp.detach().to('cpu'))  # (1, C, D, H, W)

        e_time = time.time()

        if y_true is not None:
            save_output(y_true, data_lists_test, i, os.path.join(output_dir, 'images'), output_origin, '_true')
        y_pred = y_pred.argmax(1).astype(np.uint8)[0]
        if label_mapping is not None:
            y_pred = remap_labels(y_pred, label_mapping)
        save_output(y_pred, data_lists_test, i, os.path.join(output_dir, 'images'), output_origin, '_pred')

        if i != 0:  # Skip the first iteration which involves model initialization
            predict_times.append(e_time - s_time)

    end_time = time.time()

    if is_print:
        print(f'\nTime used: {end_time - start_time:.2f} seconds.')
        print(f'Average prediction time: {np.mean(predict_times)}')
        print(f'max_memory_reserved: {torch.cuda.max_memory_reserved(device) / 1024 ** 2:.2f} MiB')
        print(f'max_memory_allocated: {torch.cuda.max_memory_allocated(device) / 1024 ** 2:.2f} MiB')
    with open(os.path.join(output_dir, 'prediction_time_memory.txt'), 'w') as f:
        print(f'Average prediction time: {np.mean(predict_times)}', file=f)
        print(f'max_memory_reserved: {torch.cuda.max_memory_reserved(device) / 1024 ** 2:.2f} MiB', file=f)
        print(f'max_memory_allocated: {torch.cuda.max_memory_allocated(device) / 1024 ** 2:.2f} MiB', file=f)
