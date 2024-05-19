"""
Author: Wenyu Ouyang
Date: 2024-03-22 17:45:18
LastEditTime: 2024-05-19 11:57:04
LastEditors: Wenyu Ouyang
Description: Test case for calibrate
FilePath: \hydromodel\test\test_calibrate.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
from hydromodel.trainers.calibrate_ga import calibrate_by_ga
from hydromodel.trainers.calibrate_sceua import calibrate_by_sceua


@pytest.fixture()
def db_dir():
    the_dir = os.path.join(
        os.path.abspath(os.path.join(__file__, "..", "..")), "result", "test_camels_us"
    )
    if not os.path.exists(the_dir):
        os.makedirs(the_dir)
    return the_dir


def test_calibrate_xaj_sceua(basins, p_and_e, qobs, warmup_length, db_dir):
    # just for testing, so the hyper-param is chosen for quick running
    calibrate_by_sceua(
        basins,
        p_and_e,
        qobs,
        os.path.join(db_dir, "sceua_xaj"),
        warmup_length,
        model={
            "name": "xaj_mz",
            "source_type": "sources",
            "source_book": "HF",
            "time_interval_hours": 1,
        },
        algorithm={
            "name": "SCE_UA",
            "random_seed": 1234,
            "rep": 5,
            "ngs": 7,
            "kstop": 3,
            "peps": 0.1,
            "pcento": 0.1,
        },
        loss={
            "type": "time_series",
            "obj_func": "RMSE",
            "events": None,
        },
    )


def test_calibrate_xaj_ga(p_and_e, qobs, warmup_length, db_dir):
    calibrate_by_ga(
        p_and_e,
        qobs,
        deap_dir=os.path.join(db_dir, "ga_xaj"),
        warmup_length=warmup_length,
    )
