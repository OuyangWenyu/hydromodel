"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2024-03-26 11:56:33
LastEditors: Wenyu Ouyang
Description: Test for data preprocess
FilePath: \hydro-model-xaj\test\test_data_postprocess.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import spotpy
from spotpy.examples.spot_setup_hymod_python import spot_setup as hymod_setup
import matplotlib.pyplot as plt


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
