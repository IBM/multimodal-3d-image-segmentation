#
# Copyright 2023 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import numpy as np
import SimpleITK as sitk

from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import Iterator

__author__ = 'Ken C. L. Wong'


class DataGenerator(object):
    """Generates minibatches of data.
    Works for both 2D and 3D images.
    For the arguments, the default value of None means no action is performed.
    Note that the channel-last format is assumed.

    Args:
        rotation_range: A scalar for 2D images and a list of length 3 (depth, height, width) for 3D images,
            in degrees (0 to 180) (default: None).
        shift_range: A list of length N for ND images, fraction of total size (default: None).
        zoom_range: Amount of zoom. A sequence of two (e.g., [0.7, 1.2]) (default: None).
        flip: A list of length N for ND images, boolean values indicating whether to randomly flip the
            corresponding axes or not (default: None).
        cval: Value used for points outside the boundaries after transformation (default: 0).
        augmentation_probability: Probability of performing augmentation for each sample (default: 1.0).
        ndim: Expected dimension of input matrix, 4 for 2D images and 5 for 3D images (default: 5).
    """
    def __init__(self,
                 rotation_range=None,
                 shift_range=None,
                 zoom_range=None,
                 flip=None,
                 cval=0.,
                 augmentation_probability=1.0,
                 ndim=5,
                 ):
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.flip = flip
        self.cval = cval
        self.augmentation_probability = augmentation_probability
        self.ndim = ndim

        self.channel_axis = self.ndim - 1
        self.size_axis = list(range(1, self.ndim - 1))

    def random_transform(self, x, y=None, seed=None):
        """Randomly augments a single image tensor.

        Args:
            x: 3D or 4D tensor, a single image with channels.
            y: The corresponding observation (default: None).
            seed: random seed (default: None).

        Returns:
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_size_axis = np.array(self.size_axis) - 1
        img_channel_axis = self.channel_axis - 1

        rng = np.random.default_rng(seed)

        if rng.binomial(1, self.augmentation_probability):
            # Rotation
            theta = None
            if self.rotation_range is not None:
                if np.ndim(self.rotation_range) == 0:
                    assert self.ndim == 4
                    if self.rotation_range:
                        theta = np.pi / 180 * rng.uniform(-self.rotation_range, self.rotation_range)
                    else:
                        theta = 0
                else:
                    assert len(self.rotation_range) == 3
                    theta = []
                    for rot in self.rotation_range:
                        theta.append(np.pi / 180 * rng.uniform(-rot, rot) if rot else 0)

            # Shift
            shift = None
            if self.shift_range is not None:
                assert len(self.shift_range) == self.ndim - 2
                shift = []
                for i, s in enumerate(self.shift_range):
                    shift.append(rng.uniform(-s, s) * x.shape[img_size_axis[i]] if s else 0)

            # Zoom
            zoom = None
            if self.zoom_range is not None:
                if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
                    zoom = 1
                else:
                    zoom = rng.uniform(self.zoom_range[0], self.zoom_range[1])

            # Create transformation matrix

            transform_matrix = None

            # Rotation
            if theta is not None:
                if np.ndim(theta) == 0 and theta != 0:
                    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                                [np.sin(theta), np.cos(theta), 0],
                                                [0, 0, 1]])
                    transform_matrix = rotation_matrix
                elif any(th != 0 for th in theta):
                    theta = theta[::-1]
                    cd = np.cos(theta[0])
                    sd = np.sin(theta[0])
                    ch = np.cos(theta[1])
                    sh = np.sin(theta[1])
                    cw = np.cos(theta[2])
                    sw = np.sin(theta[2])
                    rotation_matrix = np.array(
                        [[ch * cw, -cd * sw + sd * sh * cw, sd * sw + cd * sh * cw, 0],
                         [ch * sw, cd * cw + sd * sh * sw, -sd * cw + cd * sh * sw, 0],
                         [-sh, sd * ch, cd * ch, 0],
                         [0, 0, 0, 1]]
                    )
                    transform_matrix = rotation_matrix

            # Shift
            if shift is not None and any(sh != 0 for sh in shift):
                shift = shift[::-1]
                shift = np.asarray(shift)
                shift_matrix = np.eye(self.ndim - 1)
                shift_matrix[:-1, -1] = shift
                transform_matrix = shift_matrix if transform_matrix is None else np.dot(shift_matrix, transform_matrix)

            # Zoom
            if zoom is not None and zoom != 1:
                zoom_matrix = np.eye(self.ndim - 1)
                zoom_matrix[:-1, :-1] = np.eye(self.ndim - 2) * zoom
                transform_matrix = zoom_matrix if transform_matrix is None else np.dot(zoom_matrix, transform_matrix)

            if transform_matrix is not None:
                x = apply_transform(x, transform_matrix, img_channel_axis, cval=self.cval)
                if y is not None:
                    y = apply_transform(y, transform_matrix, img_channel_axis, cval=self.cval)

            if self.flip is not None:
                assert len(self.flip) == self.ndim - 2
                for i, fp in enumerate(self.flip):
                    if fp and rng.random() < 0.5:
                        x = flip_axis(x, img_size_axis[i])  # Warning: this function returns a view
                        if y is not None:
                            y = flip_axis(y, img_size_axis[i])  # Warning: this function returns a view

        if y is None:
            return x
        return x, y

    def flow(self,
             reader=None,
             data_lists=None,
             idx_x_modalities=None,
             idx_y_modalities=None,
             x_processing=None,
             batch_size=1,
             shuffle=False,
             seed=None):
        """Returns a generator of batches.
        The generator is indefinite, repeats after looping through all data.

        Args:
            reader: For reading a sample from a file (default: None).
            data_lists: A list of lists, each list contains the full paths to the data samples
                of a modality (default: None).
            idx_x_modalities: Indexes to x (input) modalities (default: None).
            idx_y_modalities: Indexes to y (output) modalities (default: None).
            x_processing: A function that performs custom processing on x, e.g. data normalization (default: None).
            batch_size: Batch size (default: 1).
            shuffle: Performs shuffling or not (default: False).
            seed: For controlling the randomness (default: None). Warning: may not work as expected.

        Returns:
            An indefinite generator.
        """
        return DataIterator(
            self,
            reader,
            data_lists,
            idx_x_modalities,
            idx_y_modalities,
            x_processing,
            batch_size,
            shuffle,
            seed
        )


class DataIterator(Iterator):
    """An iterator that generates batches.

    Args:
        data_generator: For data pre-processing.
        reader: For reading a sample from a file (default: None).
        data_lists: A list of lists, each list contains the full paths to the data samples
            of a modality (default: None).
        idx_x_modalities: Indexes to x (input) modalities (default: None).
        idx_y_modalities: Indexes to y (output) modalities (default: None).
        x_processing: A function that performs custom processing on x, e.g. data normalization (default: None).
        batch_size: Batch size (default: 1).
        shuffle: Performs shuffling or not (default: False).
        seed: For controlling the randomness (default: None). Warning: may not work as expected.
    """
    def __init__(self,
                 data_generator: DataGenerator,
                 reader=None,
                 data_lists=None,
                 idx_x_modalities=None,
                 idx_y_modalities=None,
                 x_processing=None,
                 batch_size=1,
                 shuffle=False,
                 seed=None
                 ):
        self.data_generator = data_generator
        self.reader = reader
        self.data_lists = data_lists
        self.idx_x_modalities = idx_x_modalities
        self.idx_y_modalities = idx_y_modalities
        self.x_processing = x_processing

        assert self.idx_x_modalities, 'Invalid idx_x_modalities.'

        # Get the data information
        num_samples, self.num_modalities, x_shape, y_shape = self._get_info(self.data_lists, self.reader)

        self.x_shape = x_shape + (len(self.idx_x_modalities),)
        self.y_shape = y_shape + (len(self.idx_y_modalities),) if self.idx_y_modalities else None

        super().__init__(num_samples, batch_size, shuffle, seed)

    def _get_info(self, list_of_data, reader=None):
        # Ensure all modalities have the same num_samples
        num_samples = len(list_of_data[0])
        for data in list_of_data:
            assert num_samples == len(data)

        # Create a dummy reader if needed
        reader = reader or (lambda a: a)

        num_modalities = len(list_of_data)

        x_shape = reader(list_of_data[self.idx_x_modalities[0]][0]).shape
        y_shape = reader(list_of_data[self.idx_y_modalities[0]][0]).shape if self.idx_y_modalities else None

        return num_samples, num_modalities, x_shape, y_shape

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.x_shape, dtype=backend.floatx())
        batch_y = np.zeros((len(index_array),) + self.y_shape, dtype=backend.floatx()) if self.y_shape else None

        # The shape is channel-first for faster value assignments, values are overwritten in each index_array loop
        x = np.zeros(self.x_shape[-1:] + self.x_shape[:-1])
        y = np.zeros(self.y_shape[-1:] + self.y_shape[:-1]) if self.y_shape else None

        # build batch of data
        for i, idx in enumerate(index_array):
            batch_x[i] = self._get_sample(idx, self.idx_x_modalities, x)
            if self.x_processing is not None:
                batch_x[i] = self.x_processing(batch_x[i])
            if batch_y is not None:
                batch_y[i] = self._get_sample(idx, self.idx_y_modalities, y)
                batch_x[i], batch_y[i] = self.data_generator.random_transform(batch_x[i], batch_y[i])
            else:
                batch_x[i] = self.data_generator.random_transform(batch_x[i])

        if batch_y is None:
            return batch_x
        return batch_x, batch_y

    def _get_sample(self, idx, idx_modalities, xy):
        offset = 0
        for m in range(self.num_modalities):
            if idx_modalities is not None and m not in idx_modalities:
                continue
            xy[offset] = self.reader(self.data_lists[m][idx])
            offset += 1
        return np.moveaxis(xy, 0, -1)


def transform_matrix_offset_center(matrix, img_size):
    offset = np.array(img_size) / 2.0 + 0.5
    offset_matrix = np.eye(matrix.shape[0])
    offset_matrix[:-1, -1] = offset
    reset_matrix = np.eye(matrix.shape[0])
    reset_matrix[:-1, -1] = -offset
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_axis=-1, cval=0.):
    """Applies the image transformation specified by a matrix.

    Args:
        x: A 3D or 4D numpy array, a single image with channels.
        transform_matrix: A numpy array specifying the geometric transformation, 3D or 4D.
        channel_axis: Index of the channel axis (default: -1).
        cval: Value used for points outside the boundaries (default: 0).

    Returns:
        The transformed version of the input.
    """
    img_size = x.shape[1:][::-1] if channel_axis == 0 else x.shape[:-1][::-1]
    transform_matrix = transform_matrix_offset_center(transform_matrix, img_size)
    final_affine_matrix = transform_matrix[:-1, :-1]
    final_offset = transform_matrix[:-1, -1]

    transform = sitk.AffineTransform(final_affine_matrix.flatten(), final_offset)
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetDefaultPixelValue(cval)
    resample.SetTransform(transform)

    x = np.moveaxis(x, channel_axis, 0)
    channel_images = []
    for x_channel in x:
        image = sitk.GetImageFromArray(x_channel)
        resample.SetSize(image.GetSize())
        resample.SetOutputSpacing(image.GetSpacing())
        resample.SetOutputOrigin(image.GetOrigin())
        image = resample.Execute(image)
        channel_images.append(sitk.GetArrayFromImage(image))

    x = np.stack(channel_images, axis=0)
    x = np.moveaxis(x, 0, channel_axis)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x
