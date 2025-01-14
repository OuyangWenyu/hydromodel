"""Show results of calibration and validation."""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from hydroutils import hydro_file, hydro_stat


def plot_precipitation(precipitation, ax=None):
    """
    Plots precipitation data from an xarray.DataArray.

    Parameters
    ----------
    precipitation : xarray.DataArray
        The precipitation data with time as the coordinate.
    ax : matplotlib.axes._axes.Axes, optional
        The matplotlib axis on which to plot. If None, a new figure and axis are created.

    Returns
    -------
    ax : matplotlib.axes._axes.Axes
        The axis with the plotted data.
    """
    # If no axis is provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 4))

    # Extract time and precipitation values from the xarray.DataArray
    time = precipitation.time.values
    values = precipitation.values

    # Plot the precipitation data as a bar chart
    ax.bar(
        time,  # Use time as the x-axis
        values,  # Use precipitation values as the y-axis
        color="blue",
        label="Precipitation",
        width=0.8,
    )

    # Set the x and y axis labels
    ax.set_xlabel("Date")
    ax.set_ylabel("Precipitation (mm/d)", color="black")

    # Invert the y-axis
    ax.invert_yaxis()

    # Format the x-axis to display year and month
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))

    # Rotate the x-axis labels to avoid overlap
    plt.xticks(rotation=45)

    # Add a legend
    ax.legend(loc="lower right")

    return ax


def plot_sim_and_obs_streamflow(
    date, sim, obs, ax=None, xlabel="Date", ylabel="Streamflow (m³/s)"
):
    # If no external subplot is provided, create a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(date, sim, color="black", linestyle="solid", label="Simulation")
    ax.plot(date, obs, "r.", markersize=3, label="Observation")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    return ax


def plot_sim_and_obs(
    date,
    prcp,
    sim,
    obs,
    save_fig,
    xlabel="Date",
    ylabel=None,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    # Plot precipitation data on the upper subplot
    plot_precipitation(prcp, ax=ax1)

    # Plot the comparison between simulated and observed values
    plot_sim_and_obs_streamflow(date, sim, obs, ax=ax2, xlabel=xlabel, ylabel=ylabel)
    plt.tight_layout()
    plt.savefig(save_fig, bbox_inches="tight")
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


# TODO: Following functions are not used in the current version of the code, maybe useful in the future
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
