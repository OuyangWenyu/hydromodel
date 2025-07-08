"""
Author: Wenyu Ouyang
Date: 2025-07-08 19:01:27
LastEditTime: 2025-07-08 19:51:19
LastEditors: Wenyu Ouyang
Description: Unit hydrograph functions
FilePath: /hydromodel/hydromodel/models/unit_hydrograph.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import logging
import numpy as np


def uh_conv(x, uh):
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

    Returns
    -------
    np.array
        convolution result with same shape as x
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
        return np.convolve(x, uh)[: len(x)]

    elif x.ndim == 2:
        # 2D case: [seq, batch]
        seq_length, batch_size = x.shape
        if uh.ndim != 2 or uh.shape[1] != batch_size:
            logging.error("For 2D input x [seq, batch], uh should be [len_uh, batch]")
            return np.zeros_like(x)

        # Handle empty arrays
        if seq_length == 0 or uh.shape[0] == 0:
            return np.zeros_like(x)

        outputs = np.zeros_like(x)
        for i in range(batch_size):
            outputs[:, i] = np.convolve(x[:, i], uh[:, i])[:seq_length]
        return outputs

    elif x.ndim == 3:
        # 3D case: [seq, batch, feature]
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

        outputs = np.zeros_like(x)
        for i in range(batch_size):
            for j in range(feature_size):
                conv_result = np.convolve(x[:, i, j], uh[:, i, j])
                outputs[:, i, j] = conv_result[:seq_length]
        return outputs

    else:
        logging.error(
            f"Unsupported array dimension: {x.ndim}D. "
            f"Only 1D, 2D, 3D are supported."
        )
        return np.zeros_like(x)
