"""
Author: Wenyu Ouyang
Date: 2023-06-02 09:30:36
LastEditTime: 2024-03-22 11:07:00
LastEditors: Wenyu Ouyang
Description: Test case for GR4J model
FilePath: \hydro-model-xaj\test\test_gr4j.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
import pandas as pd
import pytest

from hydrodataset import Camels
from hydromodel import SETTING
from hydromodel.models.gr4j import gr4j


@pytest.fixture()
def warmup_length():
    return 30


@pytest.fixture()
def camels():
    # for methods testing, we simply use the CAMELS dataset
    root_dir = SETTING["local_data_path"]["datasets-origin"]
    return Camels(os.path.join(root_dir, "camels", "camels_us"))


@pytest.fixture()
def basin_area(camels):
    attr = camels.read_attributes(["01013500"], ["area"])
    return attr["area"].values[0]


@pytest.fixture()
def p_and_e(camels):
    p_and_e = camels.read_ts_xrdataset(
        ["01013500"], ["2010-01-01", "2014-01-01"], ["prcp", "PET"]
    )
    # three dims: sequence (time), batch (basin), feature (variable)
    return np.expand_dims(p_and_e.values, axis=1)


@pytest.fixture()
def qobs(basin_area, camels):
    qobs_ = camels.read_ts_xrdataset(
        ["01013500"], ["2010-01-01", "2014-01-01"], ["streamflow"]
    )
    # we use pint package to handle the unit conversion
    # trans unit to mm/day
    return qobs_ * 1e-3 / (basin_area * km2tom2) * mtomm * daytos


@pytest.fixture()
def params():
    # all parameters are in range [0,1]
    return np.tile([0.5], (1, 4))


def test_gr4j(p_and_e, params):
    qsim = gr4j(p_and_e, params, warmup_length=10)
    np.testing.assert_array_equal(qsim.shape, (1817, 1, 1))
