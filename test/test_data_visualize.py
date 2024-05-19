"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2024-05-19 11:57:19
LastEditors: Wenyu Ouyang
Description: Test for results visualization
FilePath: \hydromodel\test\test_data_visualize.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import matplotlib.pyplot as plt
import spotpy
from spotpy.examples.spot_setup_hymod_python import spot_setup as hymod_setup

from hydroutils import hydro_time

from hydromodel.datasets.data_visualize import show_events_result, show_ts_result
from hydromodel.models.xaj import xaj
from hydromodel.trainers.calibrate_sceua import calibrate_by_sceua
from hydromodel.trainers.evaluate import _read_save_sceua_calibrated_params


def test_run_hymod_calibration():
    # a case from spotpy example
    setup = hymod_setup(spotpy.objectivefunctions.rmse)

    # 创建SCE-UA算法的sampler
    sampler = spotpy.algorithms.sceua(setup, dbname="test/SCEUA_hymod", dbformat="csv")

    # 设置校准参数
    repetitions = 5000  # 最大迭代次数

    # 运行sampler
    sampler.sample(repetitions, ngs=7, kstop=3, peps=0.1, pcento=0.1)

    # 从CSV文件加载结果
    results = spotpy.analyser.load_csv_results("test/SCEUA_hymod")

    # 绘制目标函数的变化
    plt.figure(figsize=(9, 5))
    plt.plot(results["like1"])
    plt.ylabel("RMSE")
    plt.xlabel("Iteration")
    plt.show()

    # 寻找最佳模型运行结果
    bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)
    best_model_run = results[bestindex]

    # 提取并绘制最佳模型运行的结果
    fields = [word for word in best_model_run.dtype.names if word.startswith("sim")]
    best_simulation = list(best_model_run[fields])

    plt.figure(figsize=(16, 9))
    plt.plot(
        best_simulation,
        color="black",
        linestyle="solid",
        label="Best objf.=" + str(bestobjf),
    )
    plt.plot(setup.evaluation(), "r.", markersize=3, label="Observation data")
    plt.xlabel("Number of Observation Points")
    plt.ylabel("Discharge [l s-1]")
    plt.legend(loc="upper right")
    plt.show()


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
    )
    train_period = hydro_time.t_range_days(["2012-01-01", "2017-01-01"])
    show_events_result(
        sampler.setup,
        db_name,
        warmup_length=warmup_length,
        save_dir=db_name,
        basin_id="basin_id",
        train_period=train_period,
        basin_area=basin_area,
    )


def test_show_test_result(p_and_e, qobs, warmup_length, db_name, basin_area):
    params = _read_save_sceua_calibrated_params("basin_id", db_name, db_name)
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
    show_ts_result(
        "basin_id", test_period[warmup_length:], qsim, qobs, save_dir=db_name
    )
