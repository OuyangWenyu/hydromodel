"""
Author: Wenyu Ouyang
Date: 2022-12-08 09:24:54
LastEditTime: 2024-03-22 20:58:24
LastEditors: Wenyu Ouyang
Description: some util funcs for hydro model
FilePath: \hydro-model-xaj\test\test_show_results.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from hydromodel.datasets.data_postprocess import read_save_sceua_calibrated_params
from hydromodel.models.xaj import xaj
from hydromodel.trainers.calibrate_sceua import calibrate_by_sceua
from hydromodel.trainers.train_utils import show_calibrate_result, show_test_result


from hydroutils import hydro_time


def test_show_calibrate_sceua_result(p_and_e, qobs, warmup_length, db_name, basin_area):
    sampler = calibrate_by_sceua(
        p_and_e,
        qobs,
        db_name,
        warmup_length,
        model={
            "name": "xaj_mz",
            "source_type": "sources",
            "source_book": "HF",
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
    )
    train_period = hydro_time.t_range_days(["2012-01-01", "2017-01-01"])
    show_calibrate_result(
        sampler.setup,
        db_name,
        warmup_length=warmup_length,
        save_dir=db_name,
        basin_id="basin_id",
        train_period=train_period,
        basin_area=basin_area,
    )


def test_show_test_result(p_and_e, qobs, warmup_length, db_name, basin_area):
    params = read_save_sceua_calibrated_params("basin_id", db_name, db_name)
    qsim, _ = xaj(
        p_and_e,
        params,
        warmup_length=warmup_length,
        name="xaj_mz",
        source_type="sources",
        source_book="HF",
    )

    qsim = units.convert_unit(
        qsim,
        unit_now="mm/day",
        unit_final=units.unit["streamflow"],
        basin_area=basin_area,
    )
    qobs = units.convert_unit(
        qobs[warmup_length:, :, :],
        unit_now="mm/day",
        unit_final=units.unit["streamflow"],
        basin_area=basin_area,
    )
    test_period = hydro_time.t_range_days(["2012-01-01", "2017-01-01"])
    show_test_result(
        "basin_id", test_period[warmup_length:], qsim, qobs, save_dir=db_name
    )
