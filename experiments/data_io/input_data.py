#
# Copyright 2023 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import math

from keras.src.trainers.data_adapters.py_dataset_adapter import PyDatasetAdapter

from .dataset import MultimodalImageDataset, ImageTransform


__author__ = 'Ken C. L. Wong'


class InputData(object):
    """Organized input data for creating generators.
    With workers >= 1, the generators are threaded thus run in parallel
    with other processes.
    Note that the label images are also considered as a modality associated with `idx_y_modalities`.

    Args:
        reader: For reading a sample from a file (default: None).
        data_lists_train: A list of filename lists. Each filename list contains the full paths
            to the data samples of a modality (default: None).
        data_lists_valid: A list of filename lists. Each filename list contains the full paths
            to the data samples of a modality (default: None).
        data_lists_test: A list of filename lists. Each filename list contains the full paths
            to the data samples of a modality (default: None).
        idx_x_modalities: Indexes to x (input) modalities in data lists (default: None).
        idx_y_modalities: Indexes to y (output) modalities in data lists (default: None).
            If it is None, the generators only generate x.
        x_processing: A function that performs custom processing on x, e.g. data normalization (default: None).
        batch_size: Batch size (default: 1).
        max_queue_size: Maximum queue size (default: 1).
        workers: Number of parallel processes (default: 1).
            If 0, no threading or multiprocessing is used.
        use_multiprocessing: Use multiprocessing if True, otherwise threading (default: False).
        transform_kwargs: Dict of keyword arguments for ImageTransform (default: None).
            See `.dataset.ImageTransform` for details.
    """
    def __init__(self,
                 reader=None,
                 data_lists_train=None,
                 data_lists_valid=None,
                 data_lists_test=None,
                 idx_x_modalities=None,
                 idx_y_modalities=None,
                 x_processing=None,
                 batch_size=1,
                 max_queue_size=1,
                 workers=1,
                 use_multiprocessing=False,
                 transform_kwargs=None,
                 ):
        self.reader = reader or (lambda x: x)
        self.data_lists_train = data_lists_train
        self.data_lists_valid = data_lists_valid
        self.data_lists_test = data_lists_test
        self.idx_x_modalities = idx_x_modalities
        self.idx_y_modalities = idx_y_modalities
        self.x_processing = x_processing
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.transform_kwargs = transform_kwargs

    def _get_flow(self, data_lists, shuffle=False, transform_kwargs=None):
        transform = ImageTransform(**transform_kwargs) if transform_kwargs is not None else None
        dataset = MultimodalImageDataset(
            data_lists,
            self.batch_size,
            reader=self.reader,
            idx_x_modalities=self.idx_x_modalities,
            idx_y_modalities=self.idx_y_modalities,
            x_processing=self.x_processing,
            transform=transform,
            workers=self.workers,
            use_multiprocessing=self.use_multiprocessing,
            max_queue_size=self.max_queue_size,
        )

        adapter = PyDatasetAdapter(
            dataset,
            shuffle=shuffle
        )

        return adapter

    def get_train_flow(self, shuffle=True):
        """Gets the PyDatasetAdapter for training.

        Args:
            shuffle: If True (default), the data are shuffled in each epoch.

        Returns:
            A PyDatasetAdapter that can create an iterator for one epoch by get_numpy_iterator().
        """
        return self._get_flow(self.data_lists_train, shuffle=shuffle, transform_kwargs=self.transform_kwargs)

    def get_valid_flow(self):
        """Gets the PyDatasetAdapter for validation.
        No data shuffling and augmentation.

        Returns:
            A PyDatasetAdapter that can create an iterator for one epoch by get_numpy_iterator().
        """
        return self._get_flow(self.data_lists_valid)

    def get_test_flow(self):
        """Gets the PyDatasetAdapter for testing.
        No data shuffling and augmentation.

        Returns:
            A PyDatasetAdapter that can create an iterator for one epoch by get_numpy_iterator().
        """
        return self._get_flow(self.data_lists_test)

    def _get_num_batches(self, data):
        if data is None:
            return 0

        num_samples = len(data[0])
        return math.ceil(num_samples / self.batch_size)

    def get_train_num_batches(self):
        data = self.data_lists_train
        return self._get_num_batches(data)

    def get_valid_num_batches(self):
        data = self.data_lists_valid
        return self._get_num_batches(data)

    def get_test_num_batches(self):
        data = self.data_lists_test
        return self._get_num_batches(data)

    def _get_image_size(self, data):
        if data is None:
            return None
        return self.reader(data[0][0]).shape

    def get_train_image_size(self):
        return self._get_image_size(self.data_lists_train)

    def get_valid_image_size(self):
        return self._get_image_size(self.data_lists_valid)

    def get_test_image_size(self):
        return self._get_image_size(self.data_lists_test)
