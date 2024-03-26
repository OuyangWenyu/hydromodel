"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2024-03-26 17:01:09
LastEditors: Wenyu Ouyang
Description: Test for data preprocess
FilePath: \hydro-model-xaj\test\test_data_postprocess.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spotpy
from spotpy.examples.spot_setup_hymod_python import spot_setup as hymod_setup
from trainers.evaluate import read_save_sceua_calibrated_params


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


def test_read_save_sceua_calibrated_params(tmpdir):
    # Create a temporary directory for testing
    temp_dir = tmpdir.mkdir("test_data")

    # Generate some dummy data
    results = np.array(
        [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
        dtype=[("par1", int), ("par2", int), ("par3", int)],
    )
    spotpy.analyser.load_csv_results = lambda _: results
    spotpy.analyser.get_minlikeindex = lambda _: (0, 0)

    # Call the function
    basin_id = "test_basin"
    save_dir = temp_dir
    sceua_calibrated_file_name = "test_results.csv"
    result = read_save_sceua_calibrated_params(
        basin_id, save_dir, sceua_calibrated_file_name
    )

    # Check if the file is saved correctly
    expected_file_path = os.path.join(save_dir, basin_id + "_calibrate_params.txt")
    assert os.path.exists(expected_file_path)

    # Check if the saved file contains the expected data
    expected_data = pd.DataFrame([(1, 2, 3)], columns=["par1", "par2", "par3"])
    saved_data = pd.read_csv(expected_file_path)
    pd.testing.assert_frame_equal(saved_data, expected_data)

    # Check if the returned result is correct
    expected_result = np.array([(1, 2, 3)])
    np.testing.assert_array_equal(result, expected_result)
