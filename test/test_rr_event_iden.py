"""
Author: Wenyu Ouyang
Date: 2023-10-28 09:23:22
LastEditTime: 2024-03-22 09:32:32
LastEditors: Wenyu Ouyang
Description: Test for rainfall-runoff event identification
FilePath: \hydro-model-xaj\test\test_rr_event_iden.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pandas as pd

from hydromodel import SETTING
from hydromodel.datasets.dmca_esr import rainfall_runoff_event_identify


def test_rainfall_runoff_event_identify():
    rain = pd.read_csv(
        os.path.join(
            SETTING["local_data_path"]["root"],
            "hydromodel",
            "example",
            "daily_rainfall_27071.txt",
        ),
        header=None,
        sep="\\s+",
    )
    flow = pd.read_csv(
        os.path.join(
            SETTING["local_data_path"]["root"],
            "hydromodel",
            "example",
            "daily_flow_27071.txt",
        ),
        header=None,
        sep="\\s+",
    )
    BEGINNING_RAIN, END_RAIN, BEGINNING_FLOW, END_FLOW = rainfall_runoff_event_identify(
        rain.iloc[:, -1], flow.iloc[:, -1]
    )
    assert BEGINNING_RAIN == 0
    assert END_RAIN == 1
    assert BEGINNING_FLOW == 0
    assert END_FLOW == 1
