"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2024-09-17 15:13:57
LastEditors: Wenyu Ouyang
Description: Plots for calibration and testing results
FilePath: \hydromodel\hydromodel\trainers\evaluate.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import yaml
import numpy as np
import pandas as pd
import xarray as xr

from hydroutils import hydro_stat
from hydrodatasource.utils.utils import streamflow_unit_conv

from hydromodel.datasets import *
from hydromodel.datasets.data_preprocess import (
    get_basin_area,
    _get_pe_q_from_ts,
)
from hydromodel.models.model_config import read_model_param_dict
from hydromodel.models.model_dict import MODEL_DICT


class Evaluator:
    def __init__(self, cali_dir, param_dir=None, eval_dir=None):
        """_summary_

        Parameters
        ----------
        cali_dir : _type_
            calibration directory
        param_dir : str
            parameters directory
        eval_dir : _type_
            evaluation directory
        """
        if param_dir is None:
            param_dir = cali_dir
        if eval_dir is None:
            eval_dir = cali_dir
        cali_config = read_yaml_config(os.path.join(cali_dir, "config.yaml"))
        self.config = cali_config
        self.data_type = cali_config["data_type"]
        self.data_dir = cali_config["data_dir"]
        self.model_info = cali_config["model"]
        self.save_dir = eval_dir
        self.params_dir = param_dir
        self.param_range_file = cali_config["param_range_file"]
        if not os.path.exists(param_dir):
            os.makedirs(param_dir)
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

    def predict(self, ds):
        """predict the streamflow of all basins in ds

        Parameters
        ----------
        ds : xr.Dataset
            the input dataset

        Returns
        -------
        tuple
            qsim, qobs
        """
        model_info = self.model_info
        p_and_e, _ = _get_pe_q_from_ts(ds)
        basins = ds["basin"].data.astype(str)
        params = _read_all_basin_params(basins, self.params_dir)
        qsim, etsim = MODEL_DICT[model_info["name"]](
            p_and_e,
            params,
            # we set the warmup_length=0 but later we get results from warmup_length to the end to evaluate
            warmup_length=0,
            **model_info,
            **{"param_range_file": self.param_range_file},
        )
        qsim, qobs, etsim = self._convert_streamflow_units(ds, qsim, etsim)
        return qsim, qobs, etsim

    def save_results(self, ds, qsim, qobs, etsim):
        """save the evaluation results

        Parameters
        ----------
        ds : xr.Dataset
            input dataset
        qsim : xr.Dataset
            simulated streamflow with unit of m^3/s
        qobs : xr.Dataset
            streamflow observation with unit of m^3/s
        etsim : xr.Dataset
            simulated evapotranspiration with unit of mm/time_unit(d, h, etc.)
        """
        basins = ds["basin"].data.astype(str)
        self._summarize_parameters(basins)
        self._renormalize_params(basins)
        self._save_evaluate_results(qsim, qobs, etsim, ds)
        self._summarize_metrics(basins)

    def _convert_streamflow_units(self, test_data, qsim, etsim):
        """convert the streamflow units to m^3/s and save all variables to xr.Dataset

        Parameters
        ----------
        test_data : xr.Dataset
            _description_
        qsim : np.ndarray
            simulated streamflow
        etsim : np.ndarray
            simulated evapotranspiration

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset, xr.Dataset]
            ds_simflow, ds_obsflow, ds_et -- we use unified name for variables hence save them to different datasets
        """
        data_type = self.data_type
        data_dir = self.data_dir
        times = test_data["time"].data
        basins = test_data["basin"].data
        flow_name = remove_unit_from_name(FLOW_NAME)
        et_name = remove_unit_from_name(ET_NAME)
        flow_dataarray = xr.DataArray(
            qsim.squeeze(-1),
            coords=[("time", times), ("basin", basins)],
            name=flow_name,
        )
        flow_dataarray.attrs["units"] = test_data[flow_name].attrs["units"]
        et_dataarray = xr.DataArray(
            etsim.squeeze(-1),
            coords=[("time", times), ("basin", basins)],
            name=et_name,
        )
        # etsim's unit is same as flow's unit -- mm/time_unit(d, h, etc.)
        et_dataarray.attrs["units"] = test_data[flow_name].attrs["units"]
        ds_et = xr.Dataset()
        ds_et[et_name] = et_dataarray
        ds = xr.Dataset()
        ds[flow_name] = flow_dataarray
        target_unit = "m^3/s"
        basin_area = get_basin_area(basins, data_type, data_dir)
        ds_simflow = streamflow_unit_conv(
            ds, basin_area, target_unit=target_unit, inverse=True
        )
        ds_obsflow = streamflow_unit_conv(
            test_data[[flow_name]], basin_area, target_unit=target_unit, inverse=True
        )
        return ds_simflow, ds_obsflow, ds_et

    def _summarize_parameters(self, basin_ids):
        """
        output parameters of all basins to one file

        Parameters
        ----------
        param_dir
            the directory where we save params
        model_name
            the name of the model

        Returns
        -------

        """
        param_dir = self.params_dir
        model_name = self.model_info["name"]
        params = []
        model_param_dict = read_model_param_dict(self.param_range_file)
        for basin_id in basin_ids:
            columns = model_param_dict[model_name]["param_name"]
            params_txt = pd.read_csv(
                os.path.join(param_dir, basin_id + "_calibrate_params.txt")
            )
            params_df = pd.DataFrame(params_txt.values.T, columns=columns)
            params.append(params_df)
        params_dfs = pd.concat(params, axis=0)
        params_dfs.index = basin_ids
        print(params_dfs)
        params_csv_file = os.path.join(param_dir, "basins_norm_params.csv")
        params_dfs.to_csv(params_csv_file, sep=",", index=True, header=True)

    def _renormalize_params(self, basin_ids):
        param_dir = self.params_dir
        model_name = self.model_info["name"]
        renormalization_params = []
        model_param_dict = read_model_param_dict(self.param_range_file)
        for basin_id in basin_ids:
            params = np.loadtxt(
                os.path.join(param_dir, basin_id + "_calibrate_params.txt")
            )[1:].reshape(1, -1)
            param_ranges = model_param_dict[model_name]["param_range"]
            xaj_params = [
                (value[1] - value[0]) * params[:, i] + value[0]
                for i, (key, value) in enumerate(param_ranges.items())
            ]
            xaj_params_ = np.array([x for j in xaj_params for x in j])
            params_df = pd.DataFrame(xaj_params_.T)
            renormalization_params.append(params_df)
        renormalization_params_dfs = pd.concat(renormalization_params, axis=1)
        renormalization_params_dfs.index = model_param_dict[model_name]["param_name"]
        renormalization_params_dfs.columns = basin_ids
        print(renormalization_params_dfs)
        params_csv_file = os.path.join(param_dir, "basins_denorm_params.csv")
        renormalization_params_dfs.transpose().to_csv(
            params_csv_file, sep=",", index=True, header=True
        )

    def _summarize_metrics(self, basin_ids):
        """
        output all results' metrics of basins to one file

        Parameters
        ----------
        basin_ids
            the ids of basins

        Returns
        -------

        """
        result_dir = self.save_dir
        model_name = self.model_info["name"]
        file_path = os.path.join(result_dir, f"{model_name}_evaluation_results.nc")
        ds = xr.open_dataset(file_path)
        # for metrics, warmup_length should be considered
        warmup_length = self.config["warmup"]
        qobs = ds["qobs"].transpose("basin", "time").to_numpy()[:, warmup_length:]
        qsim = ds["qsim"].transpose("basin", "time").to_numpy()[:, warmup_length:]
        test_metrics = hydro_stat.stat_error(
            qobs,
            qsim,
        )
        metric_dfs_test = pd.DataFrame(test_metrics, index=basin_ids)
        metric_file_test = os.path.join(result_dir, "basins_metrics.csv")
        metric_dfs_test.to_csv(metric_file_test, sep=",", index=True, header=True)

    def _save_evaluate_results(self, qsim, qobs, etsim, obs_ds):
        result_dir = self.save_dir
        model_name = self.model_info["name"]
        ds = xr.Dataset()

        # 添加 qsim 和 qobs
        ds["qsim"] = qsim["flow"]
        ds["qobs"] = qobs["flow"]

        # 添加 prcp 和 pet
        ds["prcp"] = obs_ds["prcp"]
        ds["pet"] = obs_ds["pet"]
        ds["etsim"] = etsim["et"]

        # 保存为 .nc 文件
        file_path = os.path.join(result_dir, f"{model_name}_evaluation_results.nc")
        ds.to_netcdf(file_path)

        print(f"Results saved to: {file_path}")

    def load_results(self):
        result_dir = self.save_dir
        model_name = self.model_info["name"]
        file_path = os.path.join(result_dir, f"{model_name}_evaluation_results.nc")
        return xr.open_dataset(file_path)


def _load_csv_results_pandas(sceua_calibrated_file_name):
    return pd.read_csv(sceua_calibrated_file_name + ".csv")


def _get_minlikeindex_pandas(results_df, like_index=1, verbose=True):
    """
    Get the minimum objectivefunction of your result DataFrame

    :results_df: Expects a pandas DataFrame with a "like" column for objective functions
    :type: DataFrame

    :return: Index of the position in the DataFrame with the minimum objective function
        value and the value of the minimum objective function
    :rtype: int and float
    """
    # Extract the 'like' column based on the like_index
    like_column = f"like{str(like_index)}"
    likes = results_df[like_column].values

    # Find the minimum value in the 'like' column
    minimum = np.nanmin(likes)
    value = str(round(minimum, 4))
    index = np.where(likes == minimum)
    text2 = " has the lowest objective function with: "
    textv = f"Run number {str(index[0][0])}{text2}{value}"

    if verbose:
        print(textv)

    return index[0][0], minimum


def _read_save_sceua_calibrated_params(basin_id, save_dir, sceua_calibrated_file_name):
    """
    read the parameters' file generated by spotpy SCE-UA when finishing calibration

    We also save the parameters of the best model run to a file

    Parameters
    ----------
    basin_id
        id of a basin
    save_dir
        the directory where we save params
    sceua_calibrated_file_name
        the parameters' file generated by spotpy SCE-UA when finishing calibration

    Returns
    -------

    """
    results = _load_csv_results_pandas(sceua_calibrated_file_name)
    # Index of the position in the results array with the minimum objective function
    bestindex, bestobjf = _get_minlikeindex_pandas(results)
    # the following code is from spotpy but its performance is not good so we use pandas to replace it
    # results = spotpy.analyser.load_csv_results(sceua_calibrated_file_name)
    # bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)
    best_model_run = results.iloc[bestindex]
    fields = [word for word in best_model_run.index if word.startswith("par")]
    best_calibrate_params = pd.DataFrame(
        [best_model_run[fields].values], columns=fields
    )
    save_file = os.path.join(save_dir, basin_id + "_calibrate_params.txt")
    # to keep consistent with the original code, we save the best parameters to a txt file
    best_calibrate_params.T.to_csv(save_file, index=False, columns=None)
    # Return the best result as a single row
    return best_calibrate_params.to_numpy().reshape(1, -1)


def _read_all_basin_params(basins, param_dir):
    params_list = []
    for basin_id in basins:
        db_name = os.path.join(param_dir, basin_id)
        # Read parameters for each basin
        basin_params = _read_save_sceua_calibrated_params(basin_id, param_dir, db_name)
        # Ensure basin_params is one-dimensional
        basin_params = basin_params.flatten()
        params_list.append(basin_params)
    return np.vstack(params_list)


def read_yaml_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config
