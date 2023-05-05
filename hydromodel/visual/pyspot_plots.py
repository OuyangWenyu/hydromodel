"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2022-12-08 11:21:22
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
from hydromodel.utils import hydro_constant, stat
from hydromodel.utils import hydro_utils
from hydromodel.visual.hydro_plot import plot_sim_and_obs, plot_train_iteration


def show_calibrate_result(
    spot_setup,
    sceua_calibrated_file,
    warmup_length,
    save_dir,
    basin_id,
    train_period,
    result_unit="mm/day",
    basin_area=None,
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
    results = spotpy.analyser.load_csv_results(sceua_calibrated_file)
    # Plot how the objective function was minimized during sampling
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
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
    convert_unit_sim = hydro_constant.convert_unit(
        np.array(best_simulation).reshape(1, -1),
        result_unit,
        hydro_constant.unit["streamflow"],
        basin_area=basin_area,
    )
    convert_unit_obs = hydro_constant.convert_unit(
        np.array(spot_setup.evaluation()).reshape(1, -1),
        result_unit,
        hydro_constant.unit["streamflow"],
        basin_area=basin_area,
    )
    # save calibrated results of calibration period
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
    stat_error = stat.statError(
        convert_unit_obs,
        convert_unit_sim,
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
    )
