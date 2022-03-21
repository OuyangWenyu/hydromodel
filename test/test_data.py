import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

import definitions
from hydromodel.utils import hydro_utils


@pytest.fixture()
def txt_file():
    return os.path.join(
        definitions.ROOT_DIR, "hydromodel", "example", "01013500_lump_p_pe_q.txt"
    )


@pytest.fixture()
def json_file():
    return os.path.join(definitions.ROOT_DIR, "hydromodel", "example", "data_info.json")


@pytest.fixture()
def npy_file():
    return os.path.join(
        definitions.ROOT_DIR, "hydromodel", "example", "basins_lump_p_pe_q.npy"
    )


def test_save_data(txt_file, json_file, npy_file):
    data = pd.read_csv(txt_file)
    print(data.columns)
    # Note: The units are all mm/day! For streamflow, data is divided by basin's area
    variables = ["prcp(mm/day)", "petfao56(mm/day)", "streamflow(mm/day)"]
    data_info = OrderedDict(
        {
            "time": data["date"].values.tolist(),
            "basin": ["01013500"],
            "variable": variables,
        }
    )
    hydro_utils.serialize_json(data_info, json_file)
    # 1 ft3 = 0.02831685 m3
    ft3tom3 = 2.831685e-2
    # 1 km2 = 10^6 m2
    km2tom2 = 1e6
    # 1 m = 1000 mm
    mtomm = 1000
    # 1 day = 24 * 3600 s
    daytos = 24 * 3600
    # trans ft3/s to mm/day
    basin_area = 2252.7
    data[variables[-1]] = (
        data[["streamflow(ft3/s)"]].values
        * ft3tom3
        / (basin_area * km2tom2)
        * mtomm
        * daytos
    )
    df = data[variables]
    hydro_utils.serialize_numpy(np.expand_dims(df.values, axis=1), npy_file)


def test_load_data(txt_file, npy_file):
    data_ = pd.read_csv(txt_file)
    df = data_[["prcp(mm/day)", "petfao56(mm/day)"]]
    data = hydro_utils.unserialize_numpy(npy_file)[:, :, :2]
    np.testing.assert_array_equal(data, np.expand_dims(df.values, axis=1))
