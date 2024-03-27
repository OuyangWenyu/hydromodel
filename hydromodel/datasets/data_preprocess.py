"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2024-03-27 14:30:15
LastEditors: Wenyu Ouyang
Description: preprocess data for models in hydro-model-xaj
FilePath: \hydro-model-xaj\hydromodel\datasets\data_preprocess.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import os
import re
from hydrodataset import Camels
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xarray as xr

from hydrodata.utils.utils import streamflow_unit_conv

from hydromodel import CACHE_DIR, SETTING
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


def split_train_test(ts_data, train_period, test_period):
    """
    Split all data to train and test parts with same format

    Parameters
    ----------
    ts_data: xr.Dataset
        time series data
    train_period
        training period
    test_period
        testing period

    Returns
    -------
    tuple of xr.Dataset
        A tuple of xr.Dataset for training and testing data
    """
    # Convert date strings to pandas datetime objects
    train_start, train_end = pd.to_datetime(train_period[0]), pd.to_datetime(
        train_period[1]
    )
    test_start, test_end = pd.to_datetime(test_period[0]), pd.to_datetime(
        test_period[1]
    )

    # Select data for training and testing periods
    train_data = ts_data.sel(time=slice(train_start, train_end))
    test_data = ts_data.sel(time=slice(test_start, test_end))

    return train_data, test_data


def validate_freq(freq):
    """
    Validate if the freq string is a valid pandas frequency.

    Parameters
    ----------
    freq : str
        Frequency string to validate.

    Returns
    -------
    bool
        True if the freq string is valid, False otherwise.
    """
    try:
        pd.to_timedelta("1" + freq)
        return True
    except ValueError:
        return False


def cross_valid_data(ts_data, period, warmup, cv_fold, freq="1D"):
    """
    Split all data to train and test parts with same format for cross validation.

    Parameters
    ----------
    ts_data : xr.Dataset
        time series data.
    period : tuple of str
        The whole period in the format ("start_date", "end_date").
    warmup : int
        Warmup period length in days.
    cv_fold : int
        Number of folds for cross-validation.
    freq : str
        len of one period.

    Returns
    -------
    list of tuples
        Each tuple contains training and testing datasets for a fold.
    """
    if not validate_freq(freq):
        raise ValueError(
            "Time unit must be number with either 'Y','M','W','D','h','m' or 's', such as 3D."
        )

    # Convert the whole period to pandas datetime
    start_date, end_date = pd.to_datetime(period[0]), pd.to_datetime(period[1])
    date_lst = pd.date_range(start=start_date, end=end_date, freq=freq)
    date_rm_warmup = date_lst[warmup:]

    # Initialize lists to store train and test datasets for each fold
    train_test_data = []

    # KFold split
    kf = KFold(n_splits=cv_fold, shuffle=False)
    for train_index, test_index in kf.split(date_rm_warmup):
        train_period = date_rm_warmup[train_index]
        test_period = date_rm_warmup[test_index]
        # Create warmup periods using the specified frequency
        train_period_warmup = pd.date_range(
            end=train_period[0], periods=warmup + 1, freq=freq
        )[:-1]
        test_period_warmup = pd.date_range(
            end=test_period[0], periods=warmup + 1, freq=freq
        )[:-1]

        # Select data from ts_data based on train and test periods
        train_data = ts_data.sel(time=train_period.union(train_period_warmup))
        test_data = ts_data.sel(time=test_period.union(test_period_warmup))

        # Add the datasets to the list
        train_test_data.append((train_data, test_data))

    return train_test_data


def get_basin_area(data_type, data_dir, basin_ids) -> xr.Dataset:
    """_summary_

    Parameters
    ----------
    data_type : _type_
        _description_
    data_dir : _type_
        _description_
    basin_ids : _type_
        _description_

    Returns
    -------
    xr.Dataset
        _description_
    """
    area_name = remove_unit_from_name(AREA_NAME)
    if data_type == "camels":
        camels_data_dir = os.path.join(
            SETTING["local_data_path"]["datasets-origin"], "camels", data_dir
        )
        camels = Camels(camels_data_dir)
        basin_area = camels.read_area(basin_ids)
    elif data_type == "owndata":
        attr_data = xr.open_dataset(os.path.join(data_dir, "attributes.nc"))
        # to guarantee the column name is same as the column name in the time series data
        basin_area = attr_data[[area_name]].rename({"id": "basin"})
    return basin_area


def get_ts_from_diffsource(data_type, data_dir, periods, basin_ids):
    """Get time series data from different sources and unify the format and unit of streamflow.

    Parameters
    ----------
    data_type
        The type of the data source, 'camels' or 'owndata'
    data_dir
        The directory of the data source
    periods
        The periods of the time series data
    basin_ids
        The ids of the basins

    Returns
    -------
    xr.Dataset
        The time series data

    Raises
    ------
    NotImplementedError
        The data type is not 'camels' or 'owndata'
    """
    prcp_name = remove_unit_from_name(PRCP_NAME)
    pet_name = remove_unit_from_name(PET_NAME)
    flow_name = remove_unit_from_name(FLOW_NAME)
    basin_area = get_basin_area(data_type, data_dir, basin_ids)
    if data_type == "camels":
        camels_data_dir = os.path.join(
            SETTING["local_data_path"]["datasets-origin"], "camels", data_dir
        )
        camels = Camels(camels_data_dir)
        ts_data = camels.read_ts_xrdataset(
            basin_ids, periods, ["prcp", "PET", "streamflow"]
        )
        # trans unit to mm/day
        qobs_ = ts_data[["streamflow"]]
        target_unit = ts_data["prcp"].attrs.get("units", "unknown")
        r_mmd = streamflow_unit_conv(qobs_, basin_area, target_unit=target_unit)
        ts_data[flow_name] = r_mmd["streamflow"]
        ts_data[flow_name].attrs["units"] = target_unit
        ts_data = ts_data.rename({"PET": pet_name})
        # ts_data = ts_data.drop_vars('streamflow')
    elif data_type == "owndata":
        ts_data = xr.open_dataset(os.path.join(data_dir, "timeseries.nc"))
        target_unit = ts_data[prcp_name].attrs.get("units", "unknown")
        qobs_ = ts_data[[flow_name]]
        r_mmd = streamflow_unit_conv(qobs_, basin_area, target_unit=target_unit)
        ts_data[flow_name] = r_mmd[flow_name]
        ts_data[flow_name].attrs["units"] = target_unit
    else:
        raise NotImplementedError(
            "You should set the data type as 'camels' or 'owndata'"
        )

    return ts_data


def _get_pe_q_from_ts(ts_xr_dataset):
    """Transform the time series data to the format that can be used in the calibration process

    Parameters
    ----------
    ts_xr_dataset : xr.Dataset
        The time series data

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The tuple contains the precipitation and evaporation data and the observed streamflow data
    """
    prcp_name = remove_unit_from_name(PRCP_NAME)
    pet_name = remove_unit_from_name(PET_NAME)
    flow_name = remove_unit_from_name(FLOW_NAME)
    p_and_e = (
        ts_xr_dataset[[prcp_name, pet_name]].to_array().to_numpy().transpose(2, 1, 0)
    )
    qobs = np.expand_dims(ts_xr_dataset[flow_name].to_numpy().transpose(1, 0), axis=2)

    return p_and_e, qobs


def cross_val_split_tsdata(
    data_type, data_dir, cv_fold, train_period, test_period, periods, warmup, basin_ids
):
    ts_data = get_ts_from_diffsource(data_type, data_dir, periods, basin_ids)
    if cv_fold <= 1:
        # no cross validation
        periods = np.sort(
            [train_period[0], train_period[1], test_period[0], test_period[1]]
        )
        train_and_test_data = split_train_test(ts_data, train_period, test_period)
    else:
        # cross validation
        train_and_test_data = cross_valid_data(ts_data, periods, warmup, cv_fold)
    return train_and_test_data
