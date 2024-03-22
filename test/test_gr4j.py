"""
Author: Wenyu Ouyang
Date: 2023-06-02 09:30:36
LastEditTime: 2024-03-22 20:21:30
LastEditors: Wenyu Ouyang
Description: Test case for GR4J model
FilePath: \hydro-model-xaj\test\test_gr4j.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
import pytest

from hydromodel.models.gr4j import gr4j


@pytest.fixture()
def params():
    # all parameters are in range [0,1]
    return np.tile([0.5], (1, 4))


def test_gr4j(p_and_e, params, qobs, warmup_length):
    qsim, _ = gr4j(p_and_e, params, warmup_length=warmup_length)
    np.testing.assert_array_equal(
        qsim.shape, (qobs.shape[0] - warmup_length, qobs.shape[1], qobs.shape[2])
    )
