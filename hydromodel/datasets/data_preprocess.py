"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2024-03-25 17:19:38
LastEditors: Wenyu Ouyang
Description: preprocess data for models in hydro-model-xaj
FilePath: \hydro-model-xaj\hydromodel\datasets\data_preprocess.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from collections import OrderedDict
import xarray as xr

from hydroutils import hydro_time, hydro_file

from hydromodel import CACHE_DIR
from hydromodel.datasets import *


def check_tsdata_format(file_path):
    """
    Checks the time-series data for required and optional columns
    used in hydrological modeling.

    Parameters
    ----------
    file_path : str
        Path to the hydrological data file.

    Returns
    -------
    bool
        True if the data file format is correct, False otherwise.
    """
    # prcp means precipitation, pet means potential evapotranspiration, flow means streamflow
    required_columns = [
        remove_unit_from_name(TIME_NAME),
        remove_unit_from_name(PRCP_NAME),
        remove_unit_from_name(PET_NAME),
        remove_unit_from_name(FLOW_NAME),
    ]
    # et means evapotranspiration, node_flow means upstream streamflow
    # node1 means the first upstream node, node2 means the second upstream node, etc.
    # these nodes are the nearest upstream nodes of the target node
    # meaning: if node1_flow, node2_flow, and more upstream nodes are parellel.
    # No serial relationship
    optional_columns = [
        remove_unit_from_name(ET_NAME),
        remove_unit_from_name(NODE_FLOW_NAME),
    ]

    try:
        data = pd.read_csv(file_path)

        data_columns = [remove_unit_from_name(col) for col in data.columns]
        # Check required columns
        missing_required_columns = [
            column for column in required_columns if column not in data_columns
        ]

        if missing_required_columns:
            print(
                f"Missing required columns in file: {file_path}: {missing_required_columns}"
            )
            return False

        # Check optional columns
        for column in optional_columns:
            if column not in data_columns:
                print(
                    f"Optional column '{column}' not found in file: {file_path}, but it's okay."
                )

        # Check node_flow columns (flexible number of nodes)
        node_flow_columns = [
            col for col in data.columns if re.match(r"node\d+_flow", col)
        ]
        if not node_flow_columns:
            print(f"No 'node_flow' columns found in file: {file_path}, but it's okay.")

        # Check time format and sorting
        time_parsed = False
        for time_format in POSSIBLE_TIME_FORMATS:
            try:
                data[TIME_NAME] = pd.to_datetime(data[TIME_NAME], format=time_format)
                time_parsed = True
                break
            except ValueError:
                continue

        if not time_parsed:
            print(f"Time format is incorrect in file: {file_path}")
            return False

        if not data[TIME_NAME].is_monotonic_increasing:
            print(f"Data is not sorted by time in file: {file_path}")
            return False

        # Check for consistent time intervals
        time_differences = (
            data[TIME_NAME].diff().dropna()
        )  # Calculate differences and remove NaN
        if not all(time_differences == time_differences.iloc[0]):
            print(f"Time series is not at consistent intervals in file: {file_path}")
            return False

        return True

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False


def check_basin_attr_format(file_path):
    """
    Checks the basin attributes data for required columns.

    Parameters
    ----------
    file_path : str
        Path to the basin attributes data file.

    Returns
    -------
    bool
        True if the basin attributes file format is correct, False otherwise.
    """
    required_columns = [ID_NAME, NAME_NAME, AREA_NAME]

    try:
        data = pd.read_csv(file_path)

        if missing_required_columns := [
            col for col in required_columns if col not in data.columns
        ]:
            print(
                f"Missing required columns in basin attributes file: {file_path}: {missing_required_columns}"
            )
            return False

        # Additional checks (e.g., datatype checks, non-empty rows) can be added here

        return True

    except Exception as e:
        print(f"Error reading basin attributes file {file_path}: {e}")
        return False


def check_folder_contents(folder_path, basin_attr_file="basin_attributes.csv"):
    """
    Checks all time series data files in a folder and a single basin attributes file.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the time series data files.
    basin_attr_file : str
        Filename of the basin attributes file, default is "basin_attributes.csv".

    Returns
    -------
    bool
        True if all files in the folder and the basin attributes file are correct, False otherwise.
    """
    # 检查流域属性文件
    if not check_basin_attr_format(os.path.join(folder_path, basin_attr_file)):
        return False

    # 获取流域ID列表
    basin_ids = pd.read_csv(
        os.path.join(folder_path, basin_attr_file), dtype={ID_NAME: str}
    )[ID_NAME].tolist()

    # 检查每个流域的时序文件
    for basin_id in basin_ids:
        file_name = f"basin_{basin_id}.csv"
        file_path = os.path.join(folder_path, file_name)

        if not os.path.exists(file_path):
            print(f"Missing time series data file for basin {basin_id}: {file_path}")
            return False

        if not check_tsdata_format(file_path):
            print(f"Time series data format check failed for file: {file_path}")
            return False

    return True


def process_and_save_data_as_nc(
    folder_path,
    save_folder=CACHE_DIR,
    nc_attrs_file="attributes.nc",
    nc_ts_file="timeseries.nc",
):
    # 验证文件夹内容
    if not check_folder_contents(folder_path):
        print("Folder contents validation failed.")
        return False

    # 读取流域属性
    basin_attr_file = os.path.join(folder_path, "basin_attributes.csv")
    basin_attrs = pd.read_csv(basin_attr_file)

    # 创建属性数据集
    ds_attrs = xr.Dataset.from_dataframe(basin_attrs.set_index(ID_NAME))
    new_column_names = {}
    units = {}

    for col in basin_attrs.columns:
        new_name = remove_unit_from_name(col)
        unit = get_unit_from_name(col)
        new_column_names[col] = new_name
        if unit:
            units[new_name] = unit

    basin_attrs.rename(columns=new_column_names, inplace=True)

    # 创建不带单位的数据集
    ds_attrs = xr.Dataset.from_dataframe(basin_attrs.set_index(ID_NAME))

    # 为有单位的变量添加单位属性
    for var_name, unit in units.items():
        ds_attrs[var_name].attrs["units"] = unit
    # 初始化时序数据集
    ds_ts = xr.Dataset()

    # 初始化用于保存单位的字典
    units = {}

    # 获取流域ID列表
    basin_ids = basin_attrs[ID_NAME].tolist()

    # 为每个流域读取并处理时序数据
    for i, basin_id in enumerate(basin_ids):
        file_name = f"basin_{basin_id}.csv"
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path)
        for time_format in POSSIBLE_TIME_FORMATS:
            try:
                data[TIME_NAME] = pd.to_datetime(data[TIME_NAME], format=time_format)
                break
            except ValueError:
                continue
        # 在处理第一个流域时构建单位字典
        if i == 0:
            for col in data.columns:
                new_name = remove_unit_from_name(col)
                if unit := get_unit_from_name(col):
                    units[new_name] = unit

        # 修改列名以移除单位
        renamed_columns = {col: remove_unit_from_name(col) for col in data.columns}
        data.rename(columns=renamed_columns, inplace=True)

        # 将 DataFrame 转换为 xarray Dataset
        ds_basin = xr.Dataset.from_dataframe(data.set_index(TIME_NAME))

        # 为每个变量设置单位属性
        for var in ds_basin.data_vars:
            if var in units:
                ds_basin[var].attrs["units"] = units[var]
        # 添加 basin 坐标
        ds_basin = ds_basin.expand_dims({"basin": [basin_id]})
        # 合并到主数据集
        ds_ts = xr.merge([ds_ts, ds_basin], compat="no_conflicts")

    # 保存为 NetCDF 文件
    ds_attrs.to_netcdf(os.path.join(save_folder, nc_attrs_file))
    ds_ts.to_netcdf(os.path.join(save_folder, nc_ts_file))

    return True


def split_train_test(json_file, npy_file, train_period, test_period):
    """
    Split all data to train and test parts with same format

    Parameters
    ----------
    json_file
        dict file of all data
    npy_file
        numpy file of all data
    train_period
        training period
    test_period
        testing period

    Returns
    -------
    None
    """
    data = hydro_file.unserialize_numpy(npy_file)
    data_info = hydro_file.unserialize_json(json_file)
    date_lst = pd.to_datetime(data_info["time"]).values.astype("datetime64[D]")
    t_range_train = hydro_time.t_range_days(train_period)
    t_range_test = hydro_time.t_range_days(test_period)
    _, ind1, ind2 = np.intersect1d(date_lst, t_range_train, return_indices=True)
    _, ind3, ind4 = np.intersect1d(date_lst, t_range_test, return_indices=True)
    data_info_train = OrderedDict(
        {
            "time": [str(t)[:10] for t in hydro_time.t_range_days(train_period)],
            # TODO: for time, more detailed time is needed, so we need to change the format of time
            # "time": [str(t)[:16] for t in hydro_time.t_range_days(train_period)],
            "basin": data_info["basin"],
            "variable": data_info["variable"],
            "area": data_info["area"],
        }
    )
    data_info_test = OrderedDict(
        {
            "time": [str(t)[:10] for t in hydro_time.t_range_days(test_period)],
            # TODO: for time, more detailed time is needed, so we need to change the format of time
            # "time": [str(t)[:16] for t in hydro_time.t_range_days(test_period)],
            "basin": data_info["basin"],
            "variable": data_info["variable"],
            "area": data_info["area"],
        }
    )
    # unify it with cross validation case, so we add a 'fold0'
    train_json_file = json_file.parent.joinpath(json_file.stem + "_fold0_train.json")
    train_npy_file = json_file.parent.joinpath(npy_file.stem + "_fold0_train.npy")
    hydro_file.serialize_json(data_info_train, train_json_file)
    hydro_file.serialize_numpy(data[ind1, :, :], train_npy_file)
    test_json_file = json_file.parent.joinpath(json_file.stem + "_fold0_test.json")
    test_npy_file = json_file.parent.joinpath(npy_file.stem + "_fold0_test.npy")
    hydro_file.serialize_json(data_info_test, test_json_file)
    hydro_file.serialize_numpy(data[ind3, :, :], test_npy_file)


def cross_valid_data(json_file, npy_file, period, warmup, cv_fold, time_unit="h"):
    """
    Split all data to train and test parts with same format

    Parameters
    ----------
    json_file
        dict file of all data
    npy_file
        numpy file of all data
    period
        the whole period
    warmup
        warmup period length
    cv_fold
        number of folds

    Returns
    -------
    None
    """
    data = hydro_file.unserialize_numpy(npy_file)
    data_info = hydro_file.unserialize_json(json_file)
    date_lst = pd.to_datetime(data_info["time"]).values.astype("datetime64[D]")
    date_wo_warmup = date_lst[warmup:]
    kf = KFold(n_splits=cv_fold, shuffle=False)
    for i, (train, test) in enumerate(kf.split(date_wo_warmup)):
        train_period = date_wo_warmup[train]
        test_period = date_wo_warmup[test]
        train_period_warmup = np.arange(
            train_period[0] - np.timedelta64(warmup, time_unit), train_period[0]
        )
        test_period_warmup = np.arange(
            test_period[0] - np.timedelta64(warmup, time_unit), test_period[0]
        )
        t_range_train = np.concatenate((train_period_warmup, train_period))
        t_range_test = np.concatenate((test_period_warmup, test_period))
        _, ind1, ind2 = np.intersect1d(date_lst, t_range_train, return_indices=True)
        _, ind3, ind4 = np.intersect1d(date_lst, t_range_test, return_indices=True)
        data_info_train = OrderedDict(
            {
                "time": [
                    np.datetime_as_string(d, unit=time_unit) for d in t_range_train
                ],
                "basin": data_info["basin"],
                "variable": data_info["variable"],
                "area": data_info["area"],
            }
        )
        data_info_test = OrderedDict(
            {
                "time": [
                    np.datetime_as_string(d, unit=time_unit) for d in t_range_test
                ],
                "basin": data_info["basin"],
                "variable": data_info["variable"],
                "area": data_info["area"],
            }
        )
        train_json_file = json_file.parent.joinpath(
            json_file.stem + "_fold" + str(i) + "_train.json"
        )
        train_npy_file = json_file.parent.joinpath(
            npy_file.stem + "_fold" + str(i) + "_train.npy"
        )
        hydro_file.serialize_json(data_info_train, train_json_file)
        hydro_file.serialize_numpy(data[ind1, :, :], train_npy_file)
        test_json_file = json_file.parent.joinpath(
            json_file.stem + "_fold" + str(i) + "_test.json"
        )
        test_npy_file = json_file.parent.joinpath(
            npy_file.stem + "_fold" + str(i) + "_test.npy"
        )
        hydro_file.serialize_json(data_info_test, test_json_file)
        hydro_file.serialize_numpy(data[ind3, :, :], test_npy_file)
