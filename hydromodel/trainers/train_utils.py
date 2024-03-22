"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2024-03-22 20:07:08
LastEditors: Wenyu Ouyang
Description: Plots for calibration and testing results
FilePath: \hydro-model-xaj\hydromodel\trainers\train_utils.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from matplotlib import pyplot as plt
import spotpy
import pandas as pd
import os
import numpy as np

from hydroutils import hydro_file, hydro_stat


def plot_sim_and_obs(
    date,
    sim,
    obs,
    save_fig,
    xlabel="Date",
    ylabel=None,
):
    # matplotlib.use("Agg")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.subplots()
    ax.plot(
        date,
        sim,
        color="black",
        linestyle="solid",
        label="Simulation",
    )
    ax.plot(
        date,
        obs,
        "r.",
        markersize=3,
        label="Observation",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_fig, bbox_inches="tight")
    # plt.cla()
    plt.close()


def plot_train_iteration(likelihood, save_fig):
    # matplotlib.use("Agg")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.subplots()
    ax.plot(likelihood)
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Iteration")
    plt.savefig(save_fig, bbox_inches="tight")
    # plt.cla()
    plt.close()


def show_calibrate_result(
    spot_setup,
    sceua_calibrated_file,
    warmup_length,
    save_dir,
    basin_id,
    train_period,
    result_unit="mm/hour",
    basin_area=None,
    prcp=None,
):
    """
    Plot all year result to see the effect of optimized parameters

    Parameters
    ----------
    spot_setup
        Spotpy's setup class instance
    sceua_calibrated_file
        the result file saved after optimizing
    basin_id
        id of the basin
    train_period
        the period of training data
    result_unit
        the unit of result, default is mm/day, we will convert it to m3/s
    basin_area
        the area of the basin, its unit must be km2

    Returns
    -------
    None
    """
    # Load the results gained with the sceua sampler, stored in SCEUA_xaj.csv
    # results = []
    # for chunk in pd.read_csv(sceua_calibrated_file, chunksize=100000 ):
    #  results.append(chunk)
    # results = pd.concat(results)
    results = spotpy.analyser.load_csv_results(sceua_calibrated_file)  # 读取结果
    # Plot how the objective function was minimized during sampling
    if not os.path.exists(save_dir):  # 绘制采样过程中目标函数的最小化情况
        os.makedirs(save_dir)
    plot_train_iteration(
        results["like1"],
        os.path.join(save_dir, "train_iteration.png"),  # 绘制迭代中的RMSE
    )
    # Plot the best model run
    # Find the run_id with the minimal objective function value
    bestindex, bestobjf = spotpy.analyser.get_minlikeindex(
        results
    )  # 绘制最佳模型图并找到run—id
    # Select best model run
    best_model_run = results[bestindex]  # 选择最佳模型结果
    # Filter results for simulation results #最佳模型模拟结果
    fields = [word for word in best_model_run.dtype.names if word.startswith("sim")]
    best_simulation = list(best_model_run[fields])
    convert_unit_sim = units.convert_unit(
        np.array(best_simulation).reshape(1, -1),
        # np.array(list(map(float, best_simulation)), dtype=float).reshape(1, -1),
        result_unit,
        units.unit["streamflow"],
        basin_area=basin_area,
    )
    convert_unit_obs = units.convert_unit(
        np.array(spot_setup.evaluation()).reshape(1, -1),
        result_unit,
        units.unit["streamflow"],
        basin_area=basin_area,
    )

    # save calibrated results of calibration period      #保存率定的结果
    train_result_file = os.path.join(
        save_dir,
        "train_qsim_" + spot_setup.model["name"] + "_" + str(basin_id) + ".csv",
    )
    pd.DataFrame(convert_unit_sim.reshape(-1, 1)).to_csv(
        train_result_file,
        sep=",",
        index=False,
        header=False,
    )
    # calculation rmse、nashsutcliffe and bias for training period
    stat_error = hydro_stat.stat_error(
        convert_unit_obs,
        convert_unit_sim,
    )
    print("Training Metrics:", basin_id, stat_error)
    hydro_file.serialize_json_np(
        stat_error, os.path.join(save_dir, "train_metrics.json")
    )

    # 循还画图
    time = pd.read_excel(
        "D:/研究生/毕业论文/new毕业论文/预答辩/碧流河水库/站点信息/洪水率定时间.xlsx"
    )
    calibrate_starttime = pd.to_datetime("2012-06-10 0:00:00")
    calibrate_endtime = pd.to_datetime("2019-12-31 23:00:00")
    basin_area = float(basin_area)
    best_simulation = [
        x * (basin_area * 1000000 / 1000 / 3600) for x in best_simulation
    ]
    obs = [x * (basin_area * 1000000 / 1000 / 3600) for x in spot_setup.evaluation()]
    time["starttime"] = pd.to_datetime(time["starttime"])
    time["endtime"] = pd.to_datetime(time["endtime"])
    Prcp_list = []
    W_obs_list = []
    W_sim_list = []
    W_bias_abs_list = []
    W_bias_rela_list = []
    Q_max_obs_list = []
    Q_max_sim_list = []
    Q_bias_rela_list = []
    time_bias_list = []
    DC_list = []
    ID_list = []
    for i, row in time.iterrows():
        # for i in range(len(time)):
        if row["starttime"] < calibrate_endtime:
            # if(time["starttime",0]<calibrate_endtime):
            start_num = (
                row["starttime"]
                - calibrate_starttime
                - pd.Timedelta(hours=warmup_length)
            ) / pd.Timedelta(hours=1)
            end_num = (
                row["endtime"] - calibrate_starttime - pd.Timedelta(hours=warmup_length)
            ) / pd.Timedelta(hours=1)
            start_period = (row["endtime"] - calibrate_starttime) / pd.Timedelta(
                hours=1
            )
            end_period = (row["endtime"] - calibrate_starttime) / pd.Timedelta(hours=1)
            start_period = int(start_period)
            end_period = int(end_period)
            start_num = int(start_num)
            end_num = int(end_num)
            t_range_train_changci = pd.date_range(
                row["starttime"], row["endtime"], freq="H"
            )
            save_fig = os.path.join(save_dir, "train_results" + str(i) + ".png")
            best_simulation_changci = best_simulation[start_num : end_num + 1]
            plot_sim_and_obs(
                t_range_train_changci,
                best_simulation[start_num : end_num + 1],
                obs[start_num : end_num + 1],
                prcp[start_num : end_num + 1],
                save_fig,
            )
            Prcp = sum(prcp[start_num : end_num + 1])
            W_obs = (
                sum(obs[start_num : end_num + 1]) * 3600 * 1000 / basin_area / 1000000
            )
            W_sim = sum(best_simulation_changci) * 3600 * 1000 / basin_area / 1000000
            W_bias_abs = W_sim - W_obs
            W_bias_rela = W_bias_abs / W_obs
            Q_max_obs = np.max(obs[start_num : end_num + 1])
            Q_max_sim = np.max(best_simulation_changci)
            Q_bias_rela = (Q_max_sim - Q_max_obs) / Q_max_obs
            t1 = np.argmax(best_simulation_changci)
            t2 = np.argmax(obs[start_num : end_num + 1])
            time_bias = t1 - t2
            DC = NSE(obs[start_num : end_num + 1], best_simulation_changci)
            ID = row["starttime"].strftime("%Y%m%d")
            Prcp_list.append(Prcp)
            W_obs_list.append(W_obs)
            W_sim_list.append(W_sim)
            W_bias_abs_list.append(W_bias_abs)
            W_bias_rela_list.append(W_bias_rela)
            Q_max_obs_list.append(Q_max_obs)
            Q_max_sim_list.append(Q_max_sim)
            Q_bias_rela_list.append(Q_bias_rela)
            time_bias_list.append(time_bias)

            DC_list.append(DC)
            ID_list.append(ID)

    bias = pd.DataFrame(
        {
            "Prcp(mm)": Prcp_list,
            "W_obs(mm)": W_obs_list,
            "W_sim(mm)": W_sim_list,
            "W_bias_abs": W_bias_abs_list,
            "W_bias_rela": W_bias_rela_list,
            "Q_max_obs(m3/s)": Q_max_obs_list,
            "Q_max_sim(m3/s)": Q_max_sim_list,
            "Q_bias_rela": Q_bias_rela_list,
            "time_bias": time_bias_list,
            "DC": DC_list,
            "ID": ID_list,
        }
    )
    bias.to_csv(
        os.path.join(
            "D:/研究生/毕业论文/new毕业论文/预答辩/碧流河水库/站点信息/train_metrics.csv"
        )
    )
    t_range_train = pd.to_datetime(train_period[warmup_length:]).values.astype(
        "datetime64[h]"
    )
    save_fig = os.path.join(save_dir, "train_results.png")  # 生成结果图
    plot_sim_and_obs(t_range_train, best_simulation, obs, prcp[:], save_fig)


def show_test_result(basin_id, test_date, qsim, obs, save_dir):
    stat_error = hydro_stat.stat_error(obs.reshape(1, -1), qsim.reshape(1, -1))
    print("Test Metrics:", basin_id, stat_error)
    hydro_file.serialize_json_np(
        stat_error, os.path.join(save_dir, "test_metrics.json")
    )
    time = pd.read_excel(
        "D:/研究生/毕业论文/new毕业论文/预答辩/碧流河水库/站点信息/洪水率定时间.xlsx"
    )
    test_starttime = pd.to_datetime("2020-01-01 00:00:00")
    test_endtime = pd.to_datetime("2022-08-31 23:00:00")
    # for i in range(len(time)):
    #     if(test_starttime<time.iloc[i,0]<test_endtime):
    #             start_num = (time.iloc[i,0]-test_starttime-pd.Timedelta(hours=warmup_length))/pd.Timedelta(hours=1)
    #             end_num = (time.iloc[i,1]-test_starttime-pd.Timedelta(hours=warmup_length))/pd.Timedelta(hours=1)
    #             start_period = (time.iloc[i,0]-test_starttime)/pd.Timedelta(hours=1)
    #             end_period = (time.iloc[i,1]-test_starttime)/pd.Timedelta(hours=1)
    #             start_period = int(start_period)
    #             end_period = int(end_period)
    #             start_num = int(start_num)
    #             end_num = int(end_num)
    #             t_range_test_changci = pd.to_datetime(test_date[start_period:end_period]).values.astype("datetime64[h]")
    #             save_fig = os.path.join(save_dir, "test_results"+str(i)+".png")
    #             plot_sim_and_obs(t_range_test_changci, qsim.flatten()[start_num:end_num],obs.flatten()[start_num:end_num], prcp[start_num:end_num],save_fig)
    Prcp_list = []
    W_obs_list = []
    W_sim_list = []
    W_bias_abs_list = []
    W_bias_rela_list = []
    Q_max_obs_list = []
    Q_max_sim_list = []
    Q_bias_rela_list = []
    time_bias_list = []
    DC_list = []
    ID_list = []
    for i, row in time.iterrows():
        if test_starttime < row["starttime"] < test_endtime:
            start_num = (
                row["starttime"] - test_starttime - pd.Timedelta(hours=warmup_length)
            ) / pd.Timedelta(hours=1)
            end_num = (
                row["endtime"] - test_starttime - pd.Timedelta(hours=warmup_length)
            ) / pd.Timedelta(hours=1)
            start_period = (row["endtime"] - test_starttime) / pd.Timedelta(hours=1)
            end_period = (row["endtime"] - test_starttime) / pd.Timedelta(hours=1)
            start_period = int(start_period)
            end_period = int(end_period)
            start_num = int(start_num)
            end_num = int(end_num)
            t_range_train_changci = pd.date_range(
                row["starttime"], row["endtime"], freq="H"
            )
            save_fig = os.path.join(save_dir, "test_results" + str(i) + ".png")
            plot_sim_and_obs(
                t_range_train_changci,
                qsim.flatten()[start_num : end_num + 1],
                obs.flatten()[start_num : end_num + 1],
                prcp[start_num : end_num + 1],
                save_fig,
            )
            Prcp = sum(prcp[start_num : end_num + 1])
            W_obs = sum(obs.flatten()[start_num : end_num + 1])
            W_sim = sum(qsim.flatten()[start_num : end_num + 1])
            W_bias_abs = W_sim - W_obs
            W_bias_rela = W_bias_abs / W_obs
            Q_max_obs = np.max(obs[start_num : end_num + 1])
            Q_max_sim = np.max(qsim.flatten()[start_num : end_num + 1])
            Q_bias_rela = (Q_max_sim - Q_max_obs) / Q_max_obs
            t1 = np.argmax(qsim.flatten()[start_num : end_num + 1])
            t2 = np.argmax(obs[start_num : end_num + 1])
            time_bias = t1 - t2
            DC = NSE(
                obs.flatten()[start_num : end_num + 1],
                qsim.flatten()[start_num : end_num + 1],
            )
            ID = row["starttime"].strftime("%Y%m%d")
            Prcp_list.append(Prcp)
            W_obs_list.append(W_obs)
            W_sim_list.append(W_sim)
            W_bias_abs_list.append(W_bias_abs)
            W_bias_rela_list.append(W_bias_rela)
            Q_max_obs_list.append(Q_max_obs)
            Q_max_sim_list.append(Q_max_sim)
            Q_bias_rela_list.append(Q_bias_rela)
            time_bias_list.append(time_bias)
            DC_list.append(DC)
            ID_list.append(ID)

    bias = pd.DataFrame(
        {
            "Prcp(mm)": Prcp_list,
            "W_obs(mm)": W_obs_list,
            "W_sim(mm)": W_sim_list,
            "W_bias_abs": W_bias_abs_list,
            "W_bias_rela": W_bias_rela_list,
            "Q_max_obs(m3/s)": Q_max_obs_list,
            "Q_max_sim(m3/s)": Q_max_sim_list,
            "Q_bias_rela": Q_bias_rela_list,
            "time_bias": time_bias_list,
            "DC": DC_list,
            "ID": ID_list,
        }
    )
    bias.to_csv(
        os.path.join(
            "D:/研究生/毕业论文/new毕业论文/预答辩/碧流河水库/站点信息/test_metrics.csv"
        )
    )

    save_fig = os.path.join(save_dir, "test_results.png")

    plot_sim_and_obs(
        test_date[365:],
        qsim.flatten(),
        obs.flatten(),
        prcp[:],
        save_fig,
    )


def NSE(obs, mol):
    numerator = 0
    denominator = 0
    meangauge = 0
    count = 0
    for i in range(len(obs)):
        if obs[i] >= 0:
            numerator += pow(abs(mol[i]) - obs[i], 2)
            meangauge += obs[i]
            count += 1
    meangauge = meangauge / count
    for i in range(len(obs)):
        if obs[i] >= 0:
            denominator += pow(obs[i] - meangauge, 2)
    return 1 - numerator / denominator
