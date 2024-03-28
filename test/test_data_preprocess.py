from hydrodataset import Camels
import numpy as np
import pytest
import os
import pandas as pd
import xarray as xr

from hydromodel import SETTING
from hydromodel.datasets import *
from hydromodel.datasets.data_preprocess import (
    process_and_save_data_as_nc,
    split_train_test,
    check_tsdata_format,
    check_basin_attr_format,
    check_folder_contents,
    cross_valid_data,
)


@pytest.fixture()
def basin_attrs_file(tmp_path):
    # Create a temporary CSV file with required columns
    file_path = tmp_path / "basin_attributes.csv"
    data = pd.DataFrame(
        {
            ID_NAME: [1, 2, 3],
            NAME_NAME: ["Basin A", "Basin B", "Basin C"],
            AREA_NAME: [100, 200, 300],
        }
    )
    data.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture()
def all_data_dir(tmp_path):
    # Create time series data files for basins 1, 2, and 3
    for basin_id in [1, 2, 3]:
        file_name = f"basin_{basin_id}.csv"
        file_path = tmp_path / file_name

        data = pd.DataFrame(
            {
                TIME_NAME: [
                    "2022-01-01 00:00:00",
                    "2022-01-02 00:00:00",
                    "2022-01-03 00:00:00",
                ],
                PET_NAME: [1, 2, 3],
                PRCP_NAME: [4, 5, 6],
                FLOW_NAME: [7, 8, 9],
                ET_NAME: [10, 11, 12],
                NODE_FLOW_NAME: [13, 14, 15],
            }
        )
        data.to_csv(file_path, index=False)

    return str(tmp_path)


def test_check_basin_attributes_format_with_valid_file(basin_attrs_file):
    assert check_basin_attr_format(basin_attrs_file) == True


def test_check_basin_attributes_format_with_missing_columns(basin_attrs_file):
    # Remove the 'name' column from the file
    data = pd.read_csv(basin_attrs_file)
    data.drop(columns=["name"], inplace=True)
    data.to_csv(basin_attrs_file, index=False)

    assert check_basin_attr_format(basin_attrs_file) == False


def test_check_basin_attributes_format_with_invalid_file(basin_attrs_file):
    # Write invalid data to the file
    with open(basin_attrs_file, "w") as f:
        f.write("Invalid data")

    assert check_basin_attr_format(basin_attrs_file) == False


def test_check_your_own_data(all_data_dir):
    """
    Test to check the format of hydrological modeling data.
    """
    # Define a sample file path
    file_path = os.path.join(all_data_dir, "basin_1.csv")

    # Check the format of hydrological modeling data
    result = check_tsdata_format(file_path)

    # Assert that the result is True
    assert result

    # Clean up the sample file
    os.remove(file_path)


def test_check_your_own_data_missing_required_columns(tmpdir):
    """
    Test to check the format of hydrological modeling data with missing required columns.
    """
    # Define a sample file path
    file_path = os.path.join(str(tmpdir), "hydro_data.csv")

    # Create a sample file with missing required columns
    sample_data = pd.DataFrame(
        {
            PRCP_NAME: [4, 5, 6],
            FLOW_NAME: [7, 8, 9],
            ET_NAME: [10, 11, 12],
            NODE_FLOW_NAME: [13, 14, 15],
        }
    )
    sample_data.to_csv(file_path, index=False)

    # Check the format of hydrological modeling data
    result = check_tsdata_format(file_path)

    # Assert that the result is False
    assert not result

    # Clean up the sample file
    os.remove(file_path)


def test_check_your_own_data_missing_optional_columns(tmpdir):
    """
    Test to check the format of hydrological modeling data with missing optional columns.
    """
    # Define a sample file path
    file_path = os.path.join(str(tmpdir), "hydro_data.csv")

    # Create a sample file with missing optional columns
    sample_data = pd.DataFrame(
        {
            TIME_NAME: [
                "2022-01-01 00:00:00",
                "2022-01-02 00:00:00",
                "2022-01-03 00:00:00",
            ],
            PET_NAME: [1, 2, 3],
            PRCP_NAME: [4, 5, 6],
            FLOW_NAME: [7, 8, 9],
        }
    )
    sample_data.to_csv(file_path, index=False)

    # Check the format of hydrological modeling data
    result = check_tsdata_format(file_path)

    # Assert that the result is True
    assert result

    # Clean up the sample file
    os.remove(file_path)


def test_check_your_own_data_invalid_file(tmpdir):
    """
    Test to check the format of an invalid hydrological modeling data file.
    """
    # Define a sample file path
    file_path = os.path.join(str(tmpdir), "hydro_data.csv")

    # Create an invalid file (not a CSV)
    with open(file_path, "w") as f:
        f.write("This is not a valid CSV file.")

    # Check the format of hydrological modeling data
    result = check_tsdata_format(file_path)

    # Assert that the result is False
    assert not result

    # Clean up the sample file
    os.remove(file_path)


def test_check_folder_contents_with_valid_files(basin_attrs_file, all_data_dir):
    assert check_folder_contents(all_data_dir, basin_attrs_file) == True


def test_check_folder_contents_with_missing_time_series_data_file(
    basin_attrs_file, tmp_path
):
    # 创建一个模拟的时序数据文件，然后删除它，模拟缺失的文件场景
    file_path = tmp_path / "basin_2.csv"
    with open(file_path, "w") as f:
        f.write("Dummy data")
    os.remove(file_path)

    # 调用检查函数，确保它正确地返回 False
    assert check_folder_contents(tmp_path, basin_attrs_file) == False


def test_check_folder_contents_with_invalid_time_series_data_file(
    basin_attrs_file, tmp_path
):
    # Create an invalid time series data file for basin 1
    file_path = tmp_path / "basin_1.csv"
    with open(file_path, "w") as f:
        f.write("Invalid data")

    assert check_folder_contents(tmp_path, basin_attrs_file) == False


def test_check_folder_contents_with_missing_basin_attributes_file(
    tmp_path, basin_attrs_file
):
    # Remove basin attributes file
    os.remove(basin_attrs_file)

    assert check_folder_contents(tmp_path) == False


def test_check_folder_contents_with_missing_basin_attributes_column(
    basin_attrs_file, all_data_dir
):
    # Remove 'name' column from basin attributes file
    data = pd.read_csv(basin_attrs_file)
    data.drop(columns=["name"], inplace=True)
    data.to_csv(basin_attrs_file, index=False)

    assert check_folder_contents(all_data_dir, basin_attrs_file) == False


def test_process_and_save_data_as_nc_with_valid_data(all_data_dir, basin_attrs_file):
    # Create a temporary folder for testing
    folder_path = os.path.join(all_data_dir, "test_folder")
    os.makedirs(folder_path)

    # Call the function to process and save the data as NetCDF files
    result = process_and_save_data_as_nc(all_data_dir, folder_path)

    # Assert that the function returns True
    assert result

    # Assert that the NetCDF files are created
    attrs_file = os.path.join(folder_path, "attributes.nc")
    ts_file = os.path.join(folder_path, "timeseries.nc")
    assert os.path.exists(attrs_file)
    assert os.path.exists(ts_file)

    # 使用 xarray 加载 NetCDF 文件
    ds_attrs = xr.open_dataset(attrs_file)
    ds_ts = xr.open_dataset(ts_file)

    # 确保加载的数据集不为空
    assert ds_attrs is not None
    assert ds_ts is not None


def test_load_dataset():
    dataset_dir = SETTING["local_data_path"]["datasets-origin"]
    camels = Camels(os.path.join(dataset_dir, "camels", "camels_us"))
    data = camels.read_ts_xrdataset(
        ["01013500"], ["2010-01-01", "2014-01-01"], ["streamflow"]
    )
    print(data)


@pytest.fixture
def ts_data_tmp(periods=10):
    basins = ["basin1", "basin2", "basin3"]
    return xr.Dataset(
        {
            "flow": (("time", "basin"), np.random.rand(periods, 3)),
            "prcp": (("time", "basin"), np.random.rand(periods, 3)),
        },
        coords={
            "time": pd.date_range(start="2022-01-01", periods=periods),
            "basin": basins,
        },
    )


def test_cross_valid_data(ts_data_tmp):
    period = ("2022-01-01", "2022-01-10")
    warmup = 3
    cv_fold = 3

    train_test_data = cross_valid_data(ts_data_tmp, period, warmup, cv_fold)

    assert len(train_test_data) == cv_fold


def test_split_train_test(ts_data_tmp):
    # Define the train and test periods
    train_period = ("2022-01-01", "2022-01-05")
    test_period = ("2022-01-06", "2022-01-10")

    # Call the function to split the data
    train_data, test_data = split_train_test(ts_data_tmp, train_period, test_period)

    # Assert that the train and test data have the correct length and shape
    basins = ["basin1", "basin2", "basin3"]
    assert len(train_data.time) == 5 and train_data.flow.shape == (5, len(basins))
    assert len(test_data.time) == 5 and test_data.flow.shape == (5, len(basins))
