#
# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import numpy as np
import pandas as pd
import os
import itertools
import SimpleITK as sitk
from collections import defaultdict
import scipy
from multiprocessing import Pool
from functools import partial

from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance, compute_robust_hausdorff

# https://numpy.org/devdocs/release/2.0.0-notes.html#representation-of-numpy-scalars-changed
np.set_printoptions(legacy="1.25")


def compute_regional_metrics(y_true, y_pred, spacing=None, labels=None, use_surface_dice=True, use_hd95=True):
    """Compute the metrics for a case in a region.

    Args:
        y_true: The ground truth.
        y_pred: The prediction.
        spacing: The pixel spacing (default: None).
        labels: The labels of a region (default: None).
        use_surface_dice: If True, surface Dice is computed (default: True).
        use_hd95: If True, HD95 is computed (default: True).

    Returns:
        A dict of computed metrics {name: value}.
    """
    assert y_true.shape == y_pred.shape

    outputs = {}

    # Get boolean masks using regional labels
    y_true_bin = get_labels_union(y_true, labels)
    y_pred_bin = get_labels_union(y_pred, labels)

    outputs['dice'] = dice_binary(y_true_bin, y_pred_bin)
    if use_surface_dice:
        outputs['surface_dice'] = surface_dice_binary(y_true_bin, y_pred_bin, spacing)
    if use_hd95:
        outputs['hd95'] = hd95_binary(y_true_bin, y_pred_bin, spacing)

    return outputs


def statistics_regional(y_true, y_pred, y_list_test, output_dir, region_names=None, region_labels=None, is_print=True,
                        suffix='_regional', use_surface_dice=True, use_hd95=True, nproc=None):
    """Computes and saves the statistics on given predictions and ground truths.
    Labels are grouped into regions using regional labels.
    Sample-wise results are saved to a csv file, while average results are saved to a txt file.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_list_test: A list of image full paths correspond to the samples.
            For example, the list of full paths to the label images.
            The last folder names are used as patient IDs.
            Image spacings are obtained from the images to compute surface Dice and HD95.
        output_dir: Output directory (full path).
        region_names: List of names of different regions (default: None).
            If None, the names are inferred from `region_labels`.
        region_labels: List of label lists, each contains the labels of a region (default: None).
            This is needed as a region may correspond to multiple labels.
            If None, the unique labels of `y_true` are used.
        is_print: Print information or not (default: True).
        suffix: Suffix appended to the output file names (default: '_regional').
        use_surface_dice: If True, surface Dice is also computed (default: True).
        use_hd95: If True, HD95 is also computed (default: True).
        nproc: Number of processes for parallel processing (default: None).
    """
    if region_labels is None:
        region_labels = list(itertools.chain.from_iterable([np.unique(yt) for yt in y_true]))
        region_labels = np.unique(region_labels)
        print('Warning: as region_labels is not provided, each label is treated as a region.')

    if region_names is None:
        region_names = [f'Label {str(lab)}' for lab in region_labels]
        print(f'Warning: as region_names is not provided, {region_names} are used.')

    assert len(region_names) == len(region_labels)

    spacing = [None] * len(y_true)
    if use_surface_dice or use_hd95:
        spacing = [sitk.ReadImage(fn).GetSpacing()[::-1] for fn in y_list_test]
        print('Spacings are obtained from image files.')

    metrics_all = defaultdict(list)  # Keys are metric names
    for labs in region_labels:
        metrics = defaultdict(list)
        if nproc is not None:
            partial_fn = partial(compute_regional_metrics, labels=labs, use_surface_dice=use_surface_dice,
                                 use_hd95=use_hd95)
            with Pool(processes=nproc) as pool:
                results = pool.starmap(partial_fn, zip(y_true, y_pred, spacing))  # List of dict
            for res in results:
                for k, v in res.items():
                    metrics[k].append(v)
        else:
            for yt, yp, sp in zip(y_true, y_pred, spacing):  # Loop through samples
                for k, v in compute_regional_metrics(yt, yp, sp, labs, use_surface_dice, use_hd95).items():
                    metrics[k].append(v)
        for k, v in metrics.items():
            metrics_all[k].append(np.array(v)[:, None])

    metrics_all = {k: np.concatenate(v, axis=1) for k, v in metrics_all.items()}  # (num_samples, num_labels)
    num_labels = metrics_all['dice'].shape[1]
    ids = pd.DataFrame([fn.split('/')[-2] for fn in y_list_test] + ['End'])  # Folder names as ids

    # Save all sample results to csv
    df = [ids]
    for k in metrics_all:
        df += [pd.DataFrame(metrics_all[k][:, i]) for i in range(num_labels)]
    header = ['ID'] + [' '.join(tmp) for tmp in itertools.product(list(metrics_all.keys()), region_names)]
    output_file = os.path.join(output_dir, f'results{suffix}.csv')
    pd.concat(df, axis=1).to_csv(output_file, sep=str('\t'), header=header, index=False, float_format=str('%.6f'))

    # Overall results
    with open(os.path.join(output_dir, f'average_results{suffix}.txt'), 'w') as f:
        print(f'region_names: {region_names}', file=f)
    if is_print:
        print()
        print(f'region_names: {region_names}')
    for k, v in metrics_all.items():
        scores = np.ma.array(v, mask=np.isnan(v) | np.isinf(v))
        mean = list(scores.mean(0).filled(np.nan))
        std = list(scores.std(0).filled(np.nan))
        with open(os.path.join(output_dir, f'average_results{suffix}.txt'), 'a') as f:
            print(f'{k}_mean: {mean}', file=f)
            print(f'{k}_std: {std}', file=f)
        if is_print:
            print(f'{k}_mean: {mean}')
            print(f'{k}_std: {std}')


def dice_binary(y_true_bin, y_pred_bin):
    intersection = np.count_nonzero(y_true_bin & y_pred_bin)
    y_true_count = np.count_nonzero(y_true_bin)
    y_pred_count = np.count_nonzero(y_pred_bin)
    if y_true_count == 0:  # Label does not exist
        return np.nan
    return 2 * intersection / (y_true_count + y_pred_count)


def surface_dice_binary(y_true_bin, y_pred_bin, spacing):
    if np.count_nonzero(y_true_bin) == 0:  # Label does not exist
        return np.nan
    dist = compute_surface_distances(y_true_bin, y_pred_bin, spacing)
    return compute_surface_dice_at_tolerance(dist, np.mean(spacing))


def hd95_binary(y_true_bin, y_pred_bin, spacing):
    if np.count_nonzero(y_true_bin) == 0:  # Label does not exist
        return np.nan
    y_pred_bin = scipy.ndimage.binary_opening(y_pred_bin)  # Reduce noise from prediction
    dist = compute_surface_distances(y_true_bin, y_pred_bin, spacing)
    return compute_robust_hausdorff(dist, 95)


def get_labels_union(y, target_labels):
    if np.isscalar(target_labels):
        target_labels = [target_labels]

    output = None
    for lab in target_labels:
        if output is None:
            output = (y == lab)
        else:
            output |= (y == lab)
    return output
