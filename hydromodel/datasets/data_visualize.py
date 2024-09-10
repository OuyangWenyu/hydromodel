"""Show results of calibration and validation."""

import os
from matplotlib import dates, pyplot as plt
import numpy as np
import pandas as pd

from hydroutils import hydro_file, hydro_stat, hydro_plot
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

# 新增读取降雨数据的函数，根据流域 ID 读取相关 csv 文件
def read_rainfall_data(basin_id, start_time, end_time):
    print(f"Reading rainfall data for {basin_id} from {start_time} to {end_time}")
    rainfall_csv_path = f"/ftproot/basins-interim/timeseries/1D/{basin_id}.csv"
    rainfall_data = pd.read_csv(rainfall_csv_path, parse_dates=["time"])
    rainfall_data = rainfall_data.set_index("time")
    rainfall_filtered = rainfall_data[start_time:end_time]
    # 检查读取的数据
    print(rainfall_filtered["total_precipitation_hourly"].head())
    return rainfall_filtered["total_precipitation_hourly"]

def plot_precipitation(date, basin_id, start_time, end_time, ax=None):
    # 读取降雨数据
    precipitation = read_rainfall_data(basin_id, start_time, end_time)
    
    # 检查是否有降雨数据
    print(f"Precipitation data for {basin_id}:")
    print(precipitation.head())

    # 如果没有传入外部子图，则创建一个
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 4))

    # 绘制降雨数据为柱状图，使用 precipitation.index 作为横坐标
    ax.bar(precipitation.index, precipitation.values, color="blue", label="Precipitation", width=0.8)
    
    # 设置x轴和y轴的标签
    ax.set_xlabel('Date')  # 改为标记为 "Date"
    ax.set_ylabel('Precipitation (mm/d)', color='black')  # 改为黑色标记
    
    # y轴逆置，确保仅执行一次逆置
    ax.invert_yaxis()
    
    # 设置x轴显示格式，显示年月
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))

    plt.xticks(rotation=45)  # 旋转x轴标签以避免重叠

    # 设置图例
    ax.legend(loc="lower right")
    
    return ax

def plot_sim_and_obs(date, sim, obs, ax=None, xlabel="Date", ylabel="Streamflow (m³/s)"):
    # 如果没有传入外部子图，则创建一个
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 4))
    # 绘制模拟值和观测值
    ax.plot(date, sim, color="black", linestyle="solid", label="Simulation")
    ax.plot(date, obs, "r.", markersize=3, label="Observation")
    # 设置x轴显示格式，显示年月
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax.set_xlabel(xlabel)  # 横轴标记仍为 "Date"
    ax.set_ylabel(ylabel)  # 纵轴标记改为 "Streamflow"
    ax.legend(loc="upper right")
    return ax

def plot_combined_figure(date, sim, obs, save_fig, basin_id, start_time, end_time):
    # 创建图形对象，包含两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    # 上图：绘制降雨数据
    plot_precipitation(date, basin_id, start_time, end_time, ax=ax1)
    # 下图：绘制模拟值与观测值的对比
    plot_sim_and_obs(date, sim, obs, ax=ax2)
    # 保存图形
    plt.tight_layout()
    plt.savefig(save_fig, bbox_inches="tight")
    plt.close()


# def plot_sim_and_obs(
#     date,
#     sim,
#     obs,
#     save_fig,
#     xlabel="Date",
#     ylabel=None,
# ):
#     # matplotlib.use("Agg")
#     fig = plt.figure(figsize=(9, 6))
#     ax = fig.subplots()
#     ax.plot(
#         date,
#         sim,
#         color="black",
#         linestyle="solid",
#         label="Simulation",
#     )
#     ax.plot(
#         date,
#         obs,
#         "r.",
#         markersize=3,
#         label="Observation",
#     )
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     plt.legend(loc="upper right")
#     plt.tight_layout()
#     plt.savefig(save_fig, bbox_inches="tight")
#     # plt.cla()
#     plt.close()


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


def plot_rr_events(rr_events, rain, flow, save_dir=None):
    for i in range(len(rr_events)):
        beginning_time = rr_events["BEGINNING_RAIN"].iloc[i]
        end_time = rr_events["END_FLOW"].iloc[i]  # Ensure this column exists

        # Filter data for the specific time period
        filtered_rain_data = rain.sel(time=slice(beginning_time, end_time))
        filter_flow_data = flow.sel(time=slice(beginning_time, end_time))

        # Plotting
        hydro_plot.plot_rainfall_runoff(
            filtered_rain_data.time.values,
            filtered_rain_data.values,
            [filter_flow_data.values],
            title=f"Rainfall-Runoff Event {i}",
            leg_lst=["Flow"],
            xlabel="Time",
            ylabel="Flow (mm/h)",
        )
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_fig = os.path.join(save_dir, f"rr_event_{i}.png")
            plt.savefig(save_fig, bbox_inches="tight")


def show_events_result(
    warmup_length,
    save_dir,
    train_period,
    basin_area=None,
    prcp=None,
):
    """
    Plot all events result to see the effect of optimized parameters

    Parameters
    ----------
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
    # TODO: not finished
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


def show_ts_result(basin_id, test_date, qsim, obs, save_dir):
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
