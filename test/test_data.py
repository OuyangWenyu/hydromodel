"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2024-03-25 11:29:10
LastEditors: Wenyu Ouyang
Description: Test for data preprocess
FilePath: \hydro-model-xaj\test\test_data.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os

from hydrodataset import Camels

from hydromodel import SETTING


def test_load_dataset():
    dataset_dir = SETTING["local_data_path"]["datasets-origin"]
    camels = Camels(os.path.join(dataset_dir, "camels", "camels_us"))
    data = camels.read_ts_xrdataset(
        ["01013500"], ["2010-01-01", "2014-01-01"], ["streamflow"]
    )
    print(data)
