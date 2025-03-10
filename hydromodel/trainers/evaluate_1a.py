'''
Author: zhuanglaihong
Date: 2025-03-04 23:51:53
LastEditTime: 2025-03-04 23:53:47
LastEditors: zhuanglaihong
Description: 
FilePath: /zlh/hydromodel/hydromodel/trainers/evaluate_gr1a.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
'''


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
    def __init__(self, cali_dir, param_dir=None, eval_dir=None, warmup_length=0):
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
        self. warmup_length=warmup_length
        self.param_range_file = cali_config["param_range_file"]
        if not os.path.exists(param_dir):
            os.makedirs(param_dir)
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

    def predict(self, ds):
        """predict the streamflow of all basins in ds"""
        model_info = self.model_info
        p_and_e, _ = _get_pe_q_from_ts(ds)
        basins = ds["basin"].data.astype(str)
        params = _read_all_basin_params(basins, self.params_dir)
        
        # GR1A模型只返回年径流，不返回蒸散发
        qsim = MODEL_DICT[model_info["name"]](
            p_and_e,
            params,
            warmup_length=self.warmup_length,
            **model_info,
            **{"param_range_file": self.param_range_file},
        )
        
        # 创建与qsim相同形状的空etsim数组
        print(f"qsim type: {type(qsim)}")
        print(f"qsim shape: {np.array(qsim).shape if hasattr(qsim, 'shape') else 'unknown'}")
        
        
        qsim, qobs, _ = self._convert_streamflow_units(ds, qsim, None)
        return qsim, qobs, None  # 返回None代替etsim

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
        self._save_evaluate_results(qsim, qobs, ds)  # 移除etsim参数
        self._summarize_metrics(basins)

    def _convert_streamflow_units(self, test_data, qsim, etsim):
        """convert the streamflow units to m^3/s and save all variables to xr.Dataset"""
        data_type = self.data_type
        data_dir = self.data_dir
        times = test_data["time"].data
        basins = test_data["basin"].data
        flow_name = remove_unit_from_name(FLOW_NAME)
        
        # GR1A模型返回的是tuple，第一个元素是年径流
        qsim = qsim[0] if isinstance(qsim, tuple) else qsim
        qsim = np.array(qsim, dtype=float)
        
        # 处理多余的维度
        qsim = qsim.squeeze()  # 移除所有长度为1的维度
        
        # 计算年数和流域数
        days_per_year = 365
        n_years = len(times) // days_per_year
        n_basins = len(basins)
        
        #print("Array info:")
        #print("qsim size:", qsim.size)
        #print("Expected size:", n_years * days_per_year * n_basins)
        #print("n_years:", n_years)
        #print("n_basins:", n_basins)
        
        # 确保数据长度正确
        if qsim.size != n_years * days_per_year * n_basins:
            # 如果数据长度不匹配，可能需要截断或填充
            expected_length = n_years * days_per_year * n_basins
            if qsim.size > expected_length:
                qsim = qsim[:expected_length]
            else:
                # 如果数据不够，用0填充
                padded_qsim = np.zeros(expected_length)
                padded_qsim[:qsim.size] = qsim
                qsim = padded_qsim
        
        # 重塑数组为年尺度，注意处理流域维度
        qsim = qsim.reshape(n_years * days_per_year, n_basins)
        # 重塑为三维数组 (年份, 天数, 流域)
        qsim = qsim.reshape(n_years, days_per_year, n_basins)
        # 计算年总量
        qsim = qsim.sum(axis=1)  # 现在形状应该是 (n_years, n_basins)
        
        #print("Data shapes after reshape:")
        #print("qsim shape:", qsim.shape)
        
        # 创建年尺度的时间坐标
        start_year = pd.Timestamp(times[0]).year
        year_coords = [pd.Timestamp(f"{start_year + i}-01-01") for i in range(n_years)]
        
        #print("Time coordinates info:")
        #print("Number of years:", len(year_coords))
        #print("First year:", year_coords[0])
        #print("Last year:", year_coords[-1])
        
        # 创建 DataArray
        flow_dataarray = xr.DataArray(
            qsim,
            coords={
                'time': year_coords,
                'basin': basins
            },
            dims=['time', 'basin'],
            name=flow_name,
        )
        # 设置正确的单位属性
        flow_dataarray.attrs["units"] = "mm/year"  # GR1A模型输出的是年径流深度
        
        ds = xr.Dataset({flow_name: flow_dataarray})
        
        # 转换单位：从mm/year到m³/s
        target_unit = "m^3/s"
        basin_area = get_basin_area(basins, data_type, data_dir)  # km²
        basin_area = np.array(basin_area)
        conversion_factor = 0.001 * (basin_area * 1e6)  # 转换为立方米每年
        # 首先将mm/year转换为m³/year
        # 1 mm = 0.001 m
        # area需要从km²转换为m²
        # 确保conversion_factor的维度与qsim匹配
        conversion_factor = np.array(conversion_factor).reshape(1, -1)  # 添加时间维度
        
        # 创建模拟流量的数据集副本
        ds_simflow = ds.copy()
        ds_simflow[flow_name].values = ds[flow_name].values * conversion_factor
        
        # 将m³/year转换为m³/s
        seconds_per_year = 365 * 24 * 3600
        ds_simflow[flow_name].values = ds_simflow[flow_name].values / seconds_per_year
        ds_simflow[flow_name].attrs["units"] = "m^3/s"
        
        # 处理观测数据
        obs_annual = []
        for year_coord in year_coords:
            year = year_coord.year
            try:
                year_data = test_data.sel(time=str(year), method='nearest')
                year_sum = year_data[flow_name].sum(dim='time').values
            except Exception as e:
                print(f"Warning: Error processing year {year}: {str(e)}")
                year_sum = np.zeros_like(basins, dtype=float)
            obs_annual.append(year_sum)
        
        obs_annual = np.array(obs_annual)
        #print("Observation data shape:", obs_annual.shape)
        
        obs_dataarray = xr.DataArray(
            obs_annual,
            coords={
                'time': year_coords,
                'basin': basins
            },
            dims=['time', 'basin'],
            name=flow_name,
        )
        obs_dataarray.attrs["units"] = "m^3/s"  # 观测数据已经是m³/s单位
        
        ds_obsflow = xr.Dataset({flow_name: obs_dataarray})
        
        return ds_simflow, ds_obsflow, None

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
        """保存评估结果，将日尺度数据转换为年尺度进行对比"""
        result_dir = self.save_dir
        model_name = self.model_info["name"]
        ds = xr.Dataset()

        # 添加原始日尺度数据
        ds["qsim_daily"] = qsim["flow"]
        ds["qobs_daily"] = qobs["flow"]
        ds["prcp"] = obs_ds["prcp"]
        ds["pet"] = obs_ds["pet"]
        
        # 将日尺度数据转换为年尺度
        # 获取时间步长设置
        days_per_year = self.model_info.get("days_per_year", 365)
        
        # 创建年份标签
        time_coords = obs_ds.coords["time"].values
        start_year = pd.Timestamp(time_coords[0]).year
        years = range(start_year, start_year + len(time_coords) // days_per_year)
        
        # 转换为年尺度数据
        qsim_annual = []
        qobs_annual = []
        prcp_annual = []
        pet_annual = []
        
        for i, year in enumerate(years):
            start_idx = i * days_per_year
            end_idx = (i + 1) * days_per_year
            
            if end_idx <= len(time_coords):
                qsim_annual.append(ds["qsim_daily"].isel(time=slice(start_idx, end_idx)).sum(dim="time").values)
                qobs_annual.append(ds["qobs_daily"].isel(time=slice(start_idx, end_idx)).sum(dim="time").values)
                prcp_annual.append(ds["prcp"].isel(time=slice(start_idx, end_idx)).sum(dim="time").values)
                pet_annual.append(ds["pet"].isel(time=slice(start_idx, end_idx)).sum(dim="time").values)
        
        # 创建年尺度数据集
        year_coords = [pd.Timestamp(f"{year}-01-01") for year in years]
        
        ds["qsim"] = xr.DataArray(
            np.array(qsim_annual),
            coords={"time": year_coords, "basin": obs_ds.coords["basin"].values},
            dims=["time", "basin"]
        )
        
        ds["qobs"] = xr.DataArray(
            np.array(qobs_annual),
            coords={"time": year_coords, "basin": obs_ds.coords["basin"].values},
            dims=["time", "basin"]
        )
        
        ds["prcp_annual"] = xr.DataArray(
            np.array(prcp_annual),
            coords={"time": year_coords, "basin": obs_ds.coords["basin"].values},
            dims=["time", "basin"]
        )
        
        ds["pet_annual"] = xr.DataArray(
            np.array(pet_annual),
            coords={"time": year_coords, "basin": obs_ds.coords["basin"].values},
            dims=["time", "basin"]
        )

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
