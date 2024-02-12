"""
Author: Wenyu Ouyang
Date: 2023-10-28 09:23:22
LastEditTime: 2024-02-12 16:17:26
LastEditors: Wenyu Ouyang
Description: Test for rainfall-runoff event identification
FilePath: \hydromodel\test\test_rr_event_iden.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pandas as pd
import definitions
from hydromodel.utils.dmca_esr import rainfall_runoff_event_identify


def test_rainfall_runoff_event_identify():
    rain = pd.read_csv(
        os.path.join(
            definitions.ROOT_DIR, "hydromodel", "example", "daily_rainfall_27071.txt"
        ),
        header=None,
        sep="\\s+",
    )
    flow = pd.read_csv(
        os.path.join(
            definitions.ROOT_DIR, "hydromodel", "example", "daily_flow_27071.txt"
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
