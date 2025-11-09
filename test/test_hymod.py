"""
Author: Wenyu Ouyang
Date: 2023-06-02 09:30:36
LastEditTime: 2025-08-19 10:13:46
LastEditors: Wenyu Ouyang
Description: Test case for HYMOD model
FilePath: \hydromodel\test\test_hymod.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
import pytest

from hydromodel.models.hymod import hymod


@pytest.fixture()
def params():
    return np.array([0.39359, 0.01005, 0.20831, 0.75010, 0.48652]).reshape(
        1, 5
    )


def test_hymod(p_and_e, params, qobs, warmup_length):
    qsim, _ = hymod(p_and_e, params, warmup_length=warmup_length)
    np.testing.assert_array_equal(
        qsim.shape,
        (qobs.shape[0] - warmup_length, qobs.shape[1], qobs.shape[2]),
    )
