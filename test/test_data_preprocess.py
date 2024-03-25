import pytest
import os
import pandas as pd
import xarray as xr

from hydromodel.datasets import *
from hydromodel.datasets.data_preprocess import process_and_save_data_as_nc
from hydromodel.datasets.data_preprocess import check_tsdata_format
from hydromodel.datasets.data_preprocess import check_basin_attr_format
from hydromodel.datasets.data_preprocess import check_folder_contents


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
    assert os.path.exists(os.path.join(folder_path, "attributes.nc"))
    assert os.path.exists(os.path.join(folder_path, "timeseries.nc"))
