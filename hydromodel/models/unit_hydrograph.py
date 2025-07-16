"""
Author: Wenyu Ouyang
Date: 2025-07-08 19:01:27
LastEditTime: 2025-07-09 09:18:21
LastEditors: Wenyu Ouyang
Description: Unit hydrograph functions
FilePath: /hydromodel/hydromodel/models/unit_hydrograph.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import itertools
import logging
import numpy as np


def uh_conv(x, uh, truncate=True):
    """
    Function for convolution calculation supporting different array dimensions

    Parameters
    ----------
    x
        input array for convolution:
        - 1D: [seq] - sequence data
        - 2D: [seq, batch] - sequence data with batch dimension
        - 3D: [seq, batch, feature] - sequence data with batch and feature dims
    uh
        unit hydrograph array:
        - 1D: [len_uh] - for 1D input
        - 2D: [len_uh, batch] - for 2D input
        - 3D: [len_uh, batch, feature] - for 3D input
    truncate : bool, optional
        If True (default), truncate convolution result to original sequence length.
        If False, return full convolution result (changes output shape).

    Returns
    -------
    np.array
        convolution result. If truncate=True, same shape as x.
        If truncate=False, sequence dimension length = len(x) + len(uh) - 1.
    """
    x = np.asarray(x)
    uh = np.asarray(uh)

    if x.ndim == 1:
        # 1D case: [seq]
        if uh.ndim != 1:
            logging.error("For 1D input x, uh should also be 1D")
            return np.zeros_like(x)
        # Handle empty arrays
        if len(x) == 0 or len(uh) == 0:
            return np.zeros_like(x)

        conv_result = np.convolve(x, uh)
        return conv_result[: len(x)] if truncate else conv_result

    elif x.ndim == 2:
        return _uh_conv_2d(x, uh, truncate)
    elif x.ndim == 3:
        return _uh_conv_3d(x, uh, truncate)
    else:
        logging.error(
            f"Unsupported array dimension: {x.ndim}D. "
            f"Only 1D, 2D, 3D are supported."
        )
        return np.zeros_like(x)


def _uh_conv_2d(x, uh, truncate=True):
    """2D case: [seq, batch]"""
    seq_length, batch_size = x.shape
    if uh.ndim != 2 or uh.shape[1] != batch_size:
        logging.error("For 2D input x [seq, batch], uh should be [len_uh, batch]")
        return np.zeros_like(x)

    # Handle empty arrays
    if seq_length == 0 or uh.shape[0] == 0:
        return np.zeros_like(x)

    # Calculate output shape
    if truncate:
        output_shape = x.shape
        outputs = np.zeros_like(x)
    else:
        conv_length = seq_length + uh.shape[0] - 1
        output_shape = (conv_length, batch_size)
        outputs = np.zeros(output_shape, dtype=x.dtype)

    for i in range(batch_size):
        conv_result = np.convolve(x[:, i], uh[:, i])
        outputs[:, i] = conv_result[:seq_length] if truncate else conv_result
    return outputs


def _uh_conv_3d(x, uh, truncate=True):
    """3D case: [seq, batch, feature]"""
    seq_length, batch_size, feature_size = x.shape
    if uh.ndim != 3 or uh.shape[1:] != (batch_size, feature_size):
        logging.error(
            "For 3D input x [seq, batch, feature], "
            "uh should be [len_uh, batch, feature]"
        )
        return np.zeros_like(x)

    # Handle empty arrays
    if seq_length == 0 or uh.shape[0] == 0:
        return np.zeros_like(x)

    # Calculate output shape
    if truncate:
        output_shape = x.shape
        outputs = np.zeros_like(x)
    else:
        conv_length = seq_length + uh.shape[0] - 1
        output_shape = (conv_length, batch_size, feature_size)
        outputs = np.zeros(output_shape, dtype=x.dtype)

    for i, j in itertools.product(range(batch_size), range(feature_size)):
        conv_result = np.convolve(x[:, i, j], uh[:, i, j])
        outputs[:, i, j] = conv_result[:seq_length] if truncate else conv_result
    return outputs
