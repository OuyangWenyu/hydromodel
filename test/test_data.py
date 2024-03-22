"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2024-03-22 09:26:38
LastEditors: Wenyu Ouyang
Description: Test for data preprocess
FilePath: \hydro-model-xaj\test\test_data.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from hydrodataset import Camels

from hydromodel import SETTING


def test_load_dataset():
    dataset_dir = SETTING["local_data_path"]["datasets-origin"]
    camels = Camels(dataset_dir)
    data = camels.read_ts_xrdataset(
        ["01013500"], ["2014-05-01 09:00:00", "2019-01-01 07:00:00"], "streamflow"
    )
    print(data)


def test_read_your_own_data():
    pass
