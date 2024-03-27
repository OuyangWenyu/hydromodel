"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2024-03-27 16:18:48
LastEditors: Wenyu Ouyang
Description: Plots for calibration and testing results
FilePath: \hydro-model-xaj\hydromodel\trainers\evaluate.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import pathlib
import pandas as pd
import os
import numpy as np
import xarray as xr
import spotpy
import yaml

from hydroutils import hydro_file, hydro_stat
from hydrodata.utils.utils import streamflow_unit_conv

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
        qsim, _ = MODEL_DICT[model_info["name"]](
            p_and_e,
            params,
            # we set the warmup_length=0 but later we get results from warmup_length to the end to evaluate
            warmup_length=0,
            **model_info,
            **{"param_range_file": self.param_range_file},
        )
        qsim, qobs = self._convert_streamflow_units(ds, qsim)
        return qsim, qobs

    def save_results(self, ds, qsim, qobs):
        """save the evaluation results

        Parameters
        ----------
        ds : _type_
            input dataset
        qsim : _type_
            _description_
        qobs : _type_
            _description_
        """
        basins = ds["basin"].data.astype(str)
        self._summarize_parameters(basins)
        self._renormalize_params(basins)
        self._save_evaluate_results(qsim, qobs, ds)
        self._summarize_metrics(basins)

    def _convert_streamflow_units(self, test_data, qsim):
        data_type = self.data_type
        data_dir = self.data_dir
        times = test_data["time"].data
        basins = test_data["basin"].data
        flow_name = remove_unit_from_name(FLOW_NAME)
        flow_dataarray = xr.DataArray(
            qsim.squeeze(-1),
            coords=[("time", times), ("basin", basins)],
            name=flow_name,
        )
        flow_dataarray.attrs["units"] = test_data[flow_name].attrs["units"]
        ds = xr.Dataset()
        ds[flow_name] = flow_dataarray
        target_unit = "m^3/s"
        basin_area = get_basin_area(data_type, data_dir, basins)
        ds_simflow = streamflow_unit_conv(
            ds, basin_area, target_unit=target_unit, inverse=True
        )
        ds_obsflow = streamflow_unit_conv(
            test_data[[flow_name]], basin_area, target_unit=target_unit, inverse=True
        )
        return ds_simflow, ds_obsflow

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

    def _save_evaluate_results(self, qsim, qobs, obs_ds):
        result_dir = self.save_dir
        model_name = self.model_info["name"]
        ds = xr.Dataset()

        # 添加 qsim 和 qobs
        ds["qsim"] = qsim["flow"]
        ds["qobs"] = qobs["flow"]

        # 添加 prcp 和 pet
        ds["prcp"] = obs_ds["prcp"]
        ds["pet"] = obs_ds["pet"]

        # 保存为 .nc 文件
        file_path = os.path.join(result_dir, f"{model_name}_evaluation_results.nc")
        ds.to_netcdf(file_path)

        print(f"Results saved to: {file_path}")

    def load_results(self):
        result_dir = self.save_dir
        model_name = self.model_info["name"]
        file_path = os.path.join(result_dir, f"{model_name}_evaluation_results.nc")
        return xr.open_dataset(file_path)


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
    results = spotpy.analyser.load_csv_results(sceua_calibrated_file_name)
    bestindex, bestobjf = spotpy.analyser.get_minlikeindex(
        results
    )  # 结果数组中具有最小目标函数的位置的索引
    best_model_run = results[bestindex]
    fields = [word for word in best_model_run.dtype.names if word.startswith("par")]
    best_calibrate_params = pd.DataFrame(list(best_model_run[fields]))
    save_file = os.path.join(save_dir, basin_id + "_calibrate_params.txt")
    best_calibrate_params.to_csv(save_file, sep=",", index=False, header=True)
    return np.array(best_calibrate_params).reshape(1, -1)  # 返回一列最佳的结果


def _read_all_basin_params(basins, param_dir):
    params_list = []
    for basin_id in basins:
        db_name = os.path.join(param_dir, basin_id)
        # 读取每个流域的参数
        basin_params = _read_save_sceua_calibrated_params(basin_id, param_dir, db_name)
        # 确保basin_params是一维的
        basin_params = basin_params.flatten()
        params_list.append(basin_params)
    return np.vstack(params_list)


def read_and_save_et_ouputs(result_dir, fold: int):
    # TODO: not finished yet after we refactor the code
    prameter_file = os.path.join(result_dir, "basins_params.csv")
    param_values = pd.read_csv(prameter_file, index_col=0)
    basins_id = param_values.columns.values
    args_file = os.path.join(result_dir, "args.json")
    args = hydro_file.unserialize_json(args_file)
    warmup_length = args["warmup_length"]
    model_func_param = args["model"]
    exp_dir = pathlib.Path(result_dir).parent
    data_info_train = hydro_file.unserialize_json(
        exp_dir.joinpath(f"data_info_fold{fold}_train.json")
    )
    data_info_test = hydro_file.unserialize_json(
        exp_dir.joinpath(f"data_info_fold{fold}_test.json")
    )
    train_period = data_info_train["time"]
    test_period = data_info_test["time"]
    # TODO: basins_lump_p_pe_q_fold NAME need to be unified
    train_np_file = os.path.join(exp_dir, f"data_info_fold{fold}_train.npy")
    test_np_file = os.path.join(exp_dir, f"data_info_fold{fold}_test.npy")
    # train_np_file = os.path.join(exp_dir, f"basins_lump_p_pe_q_fold{fold}_train.npy")
    # test_np_file = os.path.join(exp_dir, f"basins_lump_p_pe_q_fold{fold}_test.npy")
    train_data = np.load(train_np_file)
    test_data = np.load(test_np_file)
    es_test = []
    es_train = []
    for i in range(len(basins_id)):
        _, e_train = xaj(
            train_data[:, :, 0:2],
            param_values[basins_id[i]].values.reshape(1, -1),
            warmup_length=warmup_length,
            **model_func_param,
        )
        _, e_test = xaj(
            test_data[:, :, 0:2],
            param_values[basins_id[i]].values.reshape(1, -1),
            warmup_length=warmup_length,
            **model_func_param,
        )
        es_train.append(e_train.flatten())
        es_test.append(e_test.flatten())
    df_e_train = pd.DataFrame(
        np.array(es_train).T, columns=basins_id, index=train_period[warmup_length:]
    )
    df_e_test = pd.DataFrame(
        np.array(es_test).T, columns=basins_id, index=test_period[warmup_length:]
    )
    etsim_train_save_path = os.path.join(result_dir, "basin_etsim_train.csv")
    etsim_test_save_path = os.path.join(result_dir, "basin_etsim_test.csv")
    df_e_train.to_csv(etsim_train_save_path)
    df_e_test.to_csv(etsim_test_save_path)


def read_yaml_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config
