"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2024-03-21 18:41:03
LastEditors: Wenyu Ouyang
Description: Test for data preprocess
FilePath: \hydro-model-xaj\test\test_data.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import fnmatch
import socket
from datetime import datetime
import pathlib

from hydroutils import hydro_file

import definitions
from hydromodel.utils import hydro_utils
from hydromodel.data.data_preprocess import (
    cross_valid_data,
    split_train_test,
)


# @pytest.fixture()
# def txt_file():
#     return os.path.join(
#         definitions.ROOT_DIR, "hydromodel", "example", "01013500_lump_p_pe_q.txt"
#     )


# @pytest.fixture()
# def json_file():
#     return os.path.join(definitions.ROOT_DIR, "hydromodel", "example", "data_info.json")


# @pytest.fixture()
# def npy_file():
#     return os.path.join(
#         definitions.ROOT_DIR, "hydromodel", "example", "basins_lump_p_pe_q.npy"
#     )

txt_file = pathlib.Path(
    "/home/ldaning/code/biye/hydro-model-xaj/hydromodel/example/wuxi.csv"
)
forcing_data = pathlib.Path(
    "/home/ldaning/code/biye/hydro-model-xaj/hydromodel/example/wuxi.csv"
)
json_file = pathlib.Path(
    "/home/ldaning/code/biye/hydro-model-xaj/hydromodel/example/model_run_wuxi7/data_info.json"
)
npy_file = pathlib.Path(
    "/home/ldaning/code/biye/hydro-model-xaj/hydromodel/example/model_run_wuxi7/data_info.npy"
)


# def test_save_data(txt_file, json_file, npy_file):
data = pd.read_csv(txt_file)
# Note: The units are all mm/day! For streamflow, data is divided by basin's area
# variables = ["prcp(mm/day)", "petfao56(mm/day)", "streamflow(mm/day)"]
variables = ["prcp(mm/hour)", "pev(mm/hour)", "streamflow(m3/s)"]
data_info = OrderedDict(
    {
        "time": data["date"].values.tolist(),
        "basin": ["wuxi"],
        "variable": variables,
        "area": ["1992.62"],
    }
)
hydro_utils.serialize_json(data_info, json_file)
# 1 ft3 = 0.02831685 m3
# ft3tom3 = 2.831685e-2

# 1 km2 = 10^6 m2
km2tom2 = 1e6
# 1 m = 1000 mm
mtomm = 1000
# 1 day = 24 * 3600 s
# daytos = 24 * 3600
hourtos = 3600
# trans ft3/s to mm/day
# basin_area = 2055.56
basin_area = 1992.62
data[variables[-1]] = (
    data[["streamflow(m3/s)"]].values
    # * ft3tom3
    / (basin_area * km2tom2)
    * mtomm
    * hourtos
)
df = data[variables]
hydro_utils.serialize_numpy(np.expand_dims(df.values, axis=1), npy_file)


# def test_load_data(txt_file, npy_file):
#     data_ = pd.read_csv(txt_file)
#     df = data_[["prcp(mm/day)", "petfao56(mm/day)"]]
#     data = hydro_utils.unserialize_numpy(npy_file)[:, :, :2]
#     np.testing.assert_array_equal(data, np.expand_dims(df.values, axis=1))


# start_train = datetime(2014, 5, 1, 1)
# end_train = datetime(2020, 1, 1, 7)
# start_test = datetime(2020, 1, 1, 8)
# end_test = datetime(2021, 10, 11, 23)
# train_period = ["2014-05-01 09:00:00", "2019-01-01 08:00:00"]
test_period = ["2019-01-01 07:00:00", "2021-10-12 09:00:00"]
# test_period = ["2019-01-01 08:00:00", "2021-10-11 09:00:00"]
train_period = ["2014-05-01 09:00:00", "2019-01-01 07:00:00"]
period = ["2014-05-01 09:00:00", "2021-10-12 09:00:00"]
cv_fold = 1
warmup_length = 365

# if not (cv_fold > 1):
#     # no cross validation
#     periods = np.sort(
#         [train_period[0], train_period[1], test_period[0], test_period[1]]
#     )
#     print(periods)
if cv_fold > 1:
    cross_valid_data(json_file, npy_file, period, warmup_length, cv_fold)
else:
    split_train_test(json_file, npy_file, train_period, test_period)


kfold = [
    int(f_name[len("data_info_fold") : -len("_test.json")])
    for f_name in os.listdir(os.path.dirname(txt_file))
    if fnmatch.fnmatch(f_name, "*_fold*_test.json")
]
kfold = np.sort(kfold)
for fold in kfold:
    print(f"Start to calibrate the {fold}-th fold")
    train_data_info_file = os.path.join(
        os.path.dirname(forcing_data), f"data_info_fold{str(fold)}_train.json"
    )
    train_data_file = os.path.join(
        os.path.dirname(forcing_data), f"data_info_fold{str(fold)}_train.npy"
    )
    test_data_info_file = os.path.join(
        os.path.dirname(forcing_data), f"data_info_fold{str(fold)}_test.json"
    )
    test_data_file = os.path.join(
        os.path.dirname(forcing_data), f"data_info_fold{str(fold)}_test.npy"
    )
    if (
        os.path.exists(train_data_info_file) is False
        or os.path.exists(train_data_file) is False
        or os.path.exists(test_data_info_file) is False
        or os.path.exists(test_data_file) is False
    ):
        raise FileNotFoundError(
            "The data files are not found, please run datapreprocess4calibrate.py first."
        )
    data_train = hydro_utils.unserialize_numpy(train_data_file)
    print(data_train.shape)
    data_test = hydro_utils.unserialize_numpy(test_data_file)
    data_info_train = hydro_utils.unserialize_json_ordered(train_data_info_file)
    data_info_test = hydro_utils.unserialize_json_ordered(test_data_info_file)
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    # one directory for one model + one hyperparam setting and one basin
    save_dir = os.path.join(
        os.path.dirname(forcing_data),
        current_time + "_" + socket.gethostname() + "_fold" + str(fold),
    )
