import spotpy
from matplotlib import pyplot as plt
import pandas as pd
import definitions
import os
import pathlib
import numpy as np
from hydromodel.utils import stat
from hydromodel.utils import hydro_utils


def show_calibrate_result(
    spot_setup,
    sceua_calibrated_file,
    warmup_length,
    basin_id,
    train_period,
    flow_unit="mm day-1",
):
    """
    Plot all year result to see the effect of optimized parameters

    Parameters
    ----------
    spot_setup
        Spotpy's setup class instance
    sceua_calibrated_file
        the result file saved after optimizing
    flow_unit
        unit of streamflow

    Returns
    -------
    None
    """
    # Load the results gained with the sceua sampler, stored in SCEUA_xaj.csv
    results = spotpy.analyser.load_csv_results(sceua_calibrated_file)
    # Plot how the objective function was minimized during sampling
    fig = plt.figure(1, figsize=(9, 6))
    plt.plot(results["like1"])
    plt.ylabel("RMSE")
    plt.xlabel("Iteration")
    plt.savefig("..\\example\\" + basin_id + "\\" + "RMSE.png", bbox_inches="tight")
    # Plot the best model run
    # Find the run_id with the minimal objective function value
    bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)
    # Select best model run
    best_model_run = results[bestindex]
    # Filter results for simulation results
    fields = [word for word in best_model_run.dtype.names if word.startswith("sim")]
    best_simulation = list(best_model_run[fields])
    # calculation train‘s rmse、nashsutcliffe and bias
    StatError = stat.statError(
        np.array(spot_setup.evaluation()).reshape(1, -1),
        np.array(best_simulation).reshape(1, -1),
    )
    print(StatError)
    test_data = pd.read_csv(
        os.path.join(
            definitions.ROOT_DIR,
            "hydromodel",
            "example",
            basin_id,
            basin_id + "_lump_p_pe_q.txt",
        )
    )
    date_year = pd.to_datetime(test_data["date"]).dt.year
    date = pd.to_datetime(test_data["date"]).values.astype("datetime64[D]")
    t_range_train = hydro_utils.t_range_days(train_period)
    [C, ind1, ind2] = np.intersect1d(date, t_range_train, return_indices=True)
    year_unique = date_year[warmup_length : ind1[-1]].unique()
    for i in year_unique:
        year_index = np.where(date_year[warmup_length : ind1[-1]] == i)
        fig = plt.figure(figsize=(9, 6))
        ax = plt.subplot(1, 1, 1)
        ax.plot(
            best_simulation[year_index[0][0] : year_index[0][-1]],
            color="black",
            linestyle="solid",
            label="Best objf.=" + str(bestobjf),
        )
        ax.plot(
            spot_setup.evaluation()[year_index[0][0] : year_index[0][-1]],
            "r.",
            markersize=3,
            label="Observation data",
        )
        plt.xlabel("Number of Observation Points")
        plt.ylabel("Discharge [" + flow_unit + "]")
        plt.legend(loc="upper right")
        plt.title(i)
        plt.tight_layout()
        plt.savefig(
            "..\\example\\" + basin_id + "\\" + str(i) + ".png", bbox_inches="tight"
        )
    plt.show()


def show_test_result(qsim, obs, warmup_length, basin_id):
    eva = obs[warmup_length:, :, :]
    pd.DataFrame(eva.reshape(-1, 1)).to_csv(
        "..\\example\\" + str(basin_id) + "\\" + str(basin_id) + "_eva.txt",
        sep=",",
        index=False,
        header=True,
    )
    stat_error = stat.statError(eva.reshape(1, -1), qsim.reshape(1, -1))
    print(stat_error)
    f = open(r"..\\example\\" + basin_id + "\\" + basin_id + "_zhibiao.txt", "w")
    print(stat_error, file=f)
    f.close()
    fig = plt.figure(1, figsize=(9, 6))
    ax = plt.subplot(1, 1, 1)
    ax.plot(qsim.flatten(), color="black", linestyle="solid", label="simulation data")
    ax.plot(eva.flatten(), "r.", markersize=3, label="Observation data")
    plt.legend(loc="upper right")
    plt.savefig("..\\example\\" + basin_id + "\\test.png", bbox_inches="tight")
    plt.show()


def show_sceua_rmse_iteration(path):
    path = pathlib.Path(path)
    all_basins_files = [file for file in path.iterdir() if file.is_dir()]
    for i in all_basins_files:
        basin_files = os.listdir(i)
        basin_id = basin_files[10][0:5]
        results = pd.read_csv(os.path.join(i, basin_files[10]))
        fig = plt.figure(1, figsize=(9, 6))
        plt.plot(results["like1"])
        plt.ylabel("RMSE")
        plt.xlabel("Iteration")
        plt.savefig(
            "E:\\owen\\code\\hydro-model-xaj\\result\\"
            + "CSL"
            + "\\"
            + "RMSE_Iteration"
            + "\\"
            + basin_id
            + "_RMSE_Iteration.png",
            bbox_inches="tight",
        )
        plt.show()


# path=os.path.join(definitions.ROOT_DIR, "hydromodel", "example","CSL")
# show_sceua_rmse_iteration(path)


def show_sceua_rmse_time(path):
    path = pathlib.Path(path)
    all_basins_files = [file for file in path.iterdir() if file.is_dir()]
    for i in all_basins_files:
        basin_files = os.listdir(i)
        basin_id = basin_files[10][0:5]
        results = pd.read_csv(os.path.join(i, basin_files[10]))
        time = pd.read_csv(os.path.join(i, basin_files[11]), header=None)
        time["rmse"] = results["like1"]
        time.columns = ["time", "rmse"]
        fig = plt.figure(1, figsize=(9, 6))
        # plt.plot(results["like1"])
        # plt.ylabel("RMSE")
        # plt.xlabel("Iteration")
        # plt.savefig(
        #     "E:\\owen\\code\\hydro-model-xaj\\result\\" + "CSL" + "\\" + "RMSE_Iteration" + "\\" + basin_id + "_RMSE_Iteration.png",
        #     bbox_inches='tight')
        # plt.show()
