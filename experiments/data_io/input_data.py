#
# Copyright 2023 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import math

from .datagenerator import DataGenerator
from tensorflow.keras.utils import Sequence, OrderedEnqueuer

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
        generator_kwargs: Dict of keyword arguments for DataGenerator (default: None).
            E.g., arguments for data augmentation.
            See DataGenerator for details.
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
                 generator_kwargs=None,
                 ):
        self.reader = reader
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
        self.generator_kwargs = generator_kwargs

        self.train_enqueuer = None
        self.valid_enqueuer = None
        self.test_enqueuer = None

    def _get_flow(self, data_lists, shuffle=False, generator_kwargs=None, seed=None):
        generator_kwargs = generator_kwargs or {}
        generator = DataGenerator(**generator_kwargs).flow(
            reader=self.reader,
            data_lists=data_lists,
            idx_x_modalities=self.idx_x_modalities,
            idx_y_modalities=self.idx_y_modalities,
            x_processing=self.x_processing,
            batch_size=self.batch_size, shuffle=shuffle, seed=seed)

        generator, enqueuer = get_threaded_generator(generator, max_queue_size=self.max_queue_size,
                                                     workers=self.workers,
                                                     use_multiprocessing=self.use_multiprocessing)
        return generator, enqueuer

    def get_train_flow(self, shuffle=True, seed=None):
        """Gets the generator for training.

        Args:
            shuffle: If True (default), the data are shuffled in each epoch.
            seed: For controlling the randomness (default: None).
                Warning: may not work as expected.

        Returns:
            A generator that loops indefinitely.
        """
        generator, self.train_enqueuer = self._get_flow(
            self.data_lists_train, shuffle=shuffle, generator_kwargs=self.generator_kwargs,
            seed=seed
        )
        return generator

    def get_valid_flow(self):
        """Gets the generator for validation.
        No data shuffling and augmentation.

        Returns:
            A generator that loops indefinitely.
        """
        generator, self.valid_enqueuer = self._get_flow(
            self.data_lists_valid, shuffle=False, generator_kwargs=None, seed=None
        )
        return generator

    def get_test_flow(self):
        """Gets the generator for testing.
        No data shuffling and augmentation.

        Returns:
            A generator that loops indefinitely.
        """
        generator, self.test_enqueuer = self._get_flow(
            self.data_lists_test, shuffle=False, generator_kwargs=None, seed=None
        )
        return generator

    def _get_num_batches(self, data):
        if data is None:
            return 0

        num_samples = len(data[0])
        return int(math.ceil(num_samples / self.batch_size))

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

    def stop_enqueuers(self):
        """Stops running threads if necessary."""
        if self.train_enqueuer is not None:
            self.train_enqueuer.stop()
            self.train_enqueuer = None
        if self.valid_enqueuer is not None:
            self.valid_enqueuer.stop()
            self.valid_enqueuer = None
        if self.test_enqueuer is not None:
            self.test_enqueuer.stop()
            self.test_enqueuer = None


def get_threaded_generator(generator, max_queue_size=1, workers=1, use_multiprocessing=False):
    """Gets a threaded generator which runs in parallel with other processes.

    Args:
        generator: A generator.
        max_queue_size: Maximum queue size (default: 1).
        workers: Number of parallel processes (default: 1).
            If 0, no threading or multiprocessing is used.
        use_multiprocessing: Use multiprocessing if True, otherwise threading (default: False).

    Returns:
        A threaded generator and an enqueuer which controls the thread.
    """
    if not isinstance(generator, Sequence):
        raise ValueError('Must use a generator inherited from the keras.utils.Sequence class.')

    enqueuer = None
    if workers > 0:
        enqueuer = OrderedEnqueuer(generator, use_multiprocessing=use_multiprocessing)
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()
    else:
        output_generator = iter(generator)

    return output_generator, enqueuer
