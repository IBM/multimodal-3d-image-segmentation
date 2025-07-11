#
# Copyright 2023 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import os
import sys
import copy
import numpy as np
from natsort import os_sorted

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
from experiments.utils import get_config, save_config

__author__ = 'Ken C. L. Wong'


def partitioning(
        base_path: str,
        modalities,
        ext,
        train_fraction=0.,
        valid_fraction=0.,
        test_fraction=0.,
        remove_str='',
        seed=None
):
    """Creates the data lists for training, validation, and testing for each modality.
        It is assumed that all modalities share the same sample IDs.

        This version is for BraTS 2019.

    Args:
        base_path: The full path that contains all required data.
        modalities: List of str of modalities.
        ext: Image file extension, e.g. nii.gz
        train_fraction: Fraction for training in [0, 1] (default: 0).
        valid_fraction: Fraction for validation in [0, 1] (default: 0).
        test_fraction: Fraction for testing in [0, 1] (default: 0).
        remove_str: String to be removed from the file paths in the lists (default: '').
        seed: For the random number generator to produce the same randomization, if not None (default: None).

    Returns:
        dict for training, validation, and testing. Key for modality; val for list of file paths.
            val can be an empty list if the corresponding fraction is 0.

    """
    assert 0.9999 < train_fraction + valid_fraction + test_fraction < 1.0001

    # In case '~' is used to indicate the home directory
    base_path = os.path.expanduser(base_path)
    remove_str = os.path.expanduser(remove_str)

    # In BraTS'19, the folder names are the patient IDs
    ids = os_sorted(os.listdir(base_path))
    ids = [i for i in ids if os.path.isdir(os.path.join(base_path, i))]
    num_samples = len(ids)

    # Get the ids for different partitions
    thres1 = round(train_fraction * num_samples)
    thres2 = round((train_fraction + valid_fraction) * num_samples)
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids)
    train_ids = os_sorted(ids[:thres1])
    valid_ids = os_sorted(ids[thres1:thres2])
    test_ids = os_sorted(ids[thres2:])

    prefix = base_path.replace(remove_str, '')
    train_dict = {}
    valid_dict = {}
    test_dict = {}
    for m in modalities:
        # For BraTS'19, ids are the folder names and file name prefixes
        train_partition = [os.path.join(prefix, i, f'{i}_{m}.{ext}') for i in train_ids]
        valid_partition = [os.path.join(prefix, i, f'{i}_{m}.{ext}') for i in valid_ids]
        test_partition = [os.path.join(prefix, i, f'{i}_{m}.{ext}') for i in test_ids]

        # Ensure no overlapping between partitions
        assert np.all(np.isin(train_partition, valid_partition, invert=True))
        assert np.all(np.isin(train_partition, test_partition, invert=True))
        assert np.all(np.isin(test_partition, valid_partition, invert=True))

        train_dict[m] = train_partition
        valid_dict[m] = valid_partition
        test_dict[m] = test_partition

    return train_dict, valid_dict, test_dict


def merge_dict(dict_all, adict):
    if dict_all is None:
        dict_all = adict
    else:
        for m, ls in adict.items():
            dict_all[m] = dict_all[m] + ls
    return dict_all


def save_files(dict_all, output_dir, suffix):
    for m, ls in dict_all.items():
        if not ls:
            continue
        output_path = os.path.join(output_dir, f'{m}_{suffix}.txt')
        ls = [ln + '\n' for ln in ls]
        with open(output_path, 'w') as f:
            f.writelines(ls)


def main(config_file):
    config_args = get_config(config_file)

    partition_args = copy.deepcopy(config_args['partitioning'])
    base_paths = partition_args.pop('base_paths')

    train_dict_all = None
    valid_dict_all = None
    test_dict_all = None
    for base_path in base_paths:
        train_dict, valid_dict, test_dict = partitioning(base_path, **partition_args)
        train_dict_all = merge_dict(train_dict_all, train_dict)
        valid_dict_all = merge_dict(valid_dict_all, valid_dict)
        test_dict_all = merge_dict(test_dict_all, test_dict)

    output_dir = os.path.expanduser(config_args['io']['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    save_config(config_args, output_dir)

    train_fraction = partition_args['train_fraction']
    save_files(train_dict_all, output_dir, f'train-{train_fraction}')

    valid_fraction = partition_args['valid_fraction']
    save_files(valid_dict_all, output_dir, f'valid-{valid_fraction}')

    test_fraction = partition_args['test_fraction']
    save_files(test_dict_all, output_dir, f'test-{test_fraction}')

    print('Done!\n')


if __name__ == '__main__':
    main(sys.argv[1])
