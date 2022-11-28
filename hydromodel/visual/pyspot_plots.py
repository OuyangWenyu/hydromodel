"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2022-11-17 10:39:58
LastEditors: Wenyu Ouyang
Description: Plots for calibration and testing results
FilePath: \hydro-model-xaj\hydromodel\visual\pyspot_plots.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import spotpy
from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np
from hydromodel.utils import stat
from hydromodel.utils import hydro_utils


def show_calibrate_result(
    spot_setup,
    sceua_calibrated_file,
    warmup_length,
    save_dir,
    basin_id,
    train_period,
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

    Returns
    -------
    None
    """
    # Load the results gained with the sceua sampler, stored in SCEUA_xaj.csv
    results = spotpy.analyser.load_csv_results(sceua_calibrated_file)
    # Plot how the objective function was minimized during sampling
    plot_train_iteration(
        results["like1"], os.path.join(save_dir, "train_iteration.png")
    )
    # Plot the best model run
    # Find the run_id with the minimal objective function value
    bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)
    # Select best model run
    best_model_run = results[bestindex]
    # Filter results for simulation results
    fields = [word for word in best_model_run.dtype.names if word.startswith("sim")]
    best_simulation = list(best_model_run[fields])
    # calculation rmse、nashsutcliffe and bias for training period
    stat_error = stat.statError(
        np.array(spot_setup.evaluation()).reshape(1, -1),
        np.array(best_simulation).reshape(1, -1),
    )
    print("Training Metrics:", basin_id, stat_error)
    hydro_utils.serialize_json_np(
        stat_error, os.path.join(save_dir, "train_metrics.json")
    )
    t_range_train = pd.to_datetime(train_period[warmup_length:]).values.astype(
        "datetime64[D]"
    )
    save_fig = os.path.join(save_dir, "train_results.png")
    plot_sim_and_obs(t_range_train, best_simulation, spot_setup.evaluation(), save_fig)


def show_test_result(basin_id, test_date, qsim, obs, save_dir):
    stat_error = stat.statError(obs.reshape(1, -1), qsim.reshape(1, -1))
    print("Test Metrics:", basin_id, stat_error)
    hydro_utils.serialize_json_np(
        stat_error, os.path.join(save_dir, "test_metrics.json")
    )
    save_fig = os.path.join(save_dir, "test_results.png")
    plot_sim_and_obs(
        test_date,
        qsim.flatten(),
        obs.flatten(),
        save_fig,
        ylabel="Streamflow ($m^3/s$)",
    )


def plot_train_iteration(likelihood, save_fig):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.subplots()
    ax.plot(likelihood)
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Iteration")
    plt.savefig(save_fig, bbox_inches="tight")


def plot_sim_and_obs(
    date, sim, obs, save_fig, xlabel="Date", ylabel="Streamflow(mm/day)"
):
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
