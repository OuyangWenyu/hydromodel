import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

import definitions
from hydromodel.utils import hydro_utils

@pytest.fixture()
def basin_id():
    return 1439500

@pytest.fixture()
def basin_area(basin_id):
    basin_areas = OrderedDict({
        "1439500": 306,
        "6888500": 843,
        "8104900": 348,
        "9510200": 428,
    })
    basin_area=basin_areas.get(str(basin_id))
    return basin_area


@pytest.fixture()
def txt_file(basin_id):
    basin_mean_forcing = os.path.join(definitions.DATASET_DIR, 'camels_test', 'basin_mean_forcing','era5')
    basin_output_data_csv= pd.read_csv(
        os.path.join(definitions.DATASET_DIR, 'camels_test','streamflow', str(basin_id)+ '.csv'))
    basin_output_data=basin_output_data_csv[['Year', 'Mnth', 'Day','streamflow(cfs)']]
    basin_output_data['DATE'] =pd.to_datetime(basin_output_data[['Year', 'Mnth', 'Day']])
    basin_output_data= basin_output_data[['DATE','streamflow(cfs)']]
    basin_txt = pd.read_csv(os.path.join(basin_mean_forcing,str(basin_id)+ '_lump_era5_land_forcing.txt'), sep=" ")
    columns = ['Year', 'Mnth', 'Day', 'potential_evaporation_hourly', 'total_precipitation_hourly']
    basin_input_data = basin_txt[columns]
    basin_input_data['DATE'] = pd.to_datetime(
        dict(year=basin_input_data.Year, month=basin_input_data.Mnth, day=basin_input_data.Day)).dt.date.astype(
        'str')
    basin_input_data_ = basin_input_data[['DATE', 'total_precipitation_hourly', 'potential_evaporation_hourly']]
    basin_test_data = pd.merge(basin_input_data_, basin_output_data, on='DATE', how='inner')
    basin_test_data.rename(
        columns={"DATE": "date", "total_precipitation_hourly": "prcp(m/day)", "potential_evaporation_hourly": "petfao56(m/day)",
                 "'streamflow(cfs)'": "streamflow(cfs)"}, inplace=True)
    txt_filename_path = os.path.join(definitions.ROOT_DIR , "hydromodel", "example", "camels_test",str(basin_id))
    if not os.path.exists(txt_filename_path):
        os.makedirs(txt_filename_path)
    basin_test_data.to_csv('..\\hydromodel\\example\\camels_test\\'+str(basin_id)+'\\'+ str(basin_id)+'_lump_p_pe_q.txt', sep=',',
                           index=False, header=True)
    return os.path.join(definitions.ROOT_DIR, "hydromodel", "example","camels_test",str(basin_id),str(basin_id)+'_lump_p_pe_q.txt')


@pytest.fixture()
def json_file(basin_id):
    return os.path.join(definitions.ROOT_DIR, "hydromodel", "example","camels_test",str(basin_id) , 'data_info.json')


@pytest.fixture()
def npy_file(basin_id):
    return os.path.join(definitions.ROOT_DIR, "hydromodel", "example", "camels_test",str(basin_id), 'basins_lump_p_pe_q.npy')


def test_save_data(basin_id,basin_area,txt_file, json_file, npy_file):
    data = pd.read_csv(txt_file)
    print(data.columns)
    # Note: The units are all mm/day! For streamflow, data is divided by basin's area
    variables = ['prcp(mm/day)', 'petfao56(mm/day)', 'streamflow(mm/day)']
    data_info = OrderedDict({"time": data['date'].values.tolist(), "basin": [basin_id], "variable": variables})
    hydro_utils.serialize_json(data_info, json_file)
    # 1 ft3 = 0.02831685 m3
    ft3tom3 = 2.831685e-2
    # 1 km2 = 10^6 m2
    km2tom2 = 1e6
    # 1 m = 1000 mm
    mtomm = 1000
    # 1 day = 24 * 3600 s
    daytos = 24 * 3600
    # trans ft3/s to mm/day
    data[variables[-1]] = data[['streamflow(m3/s)']].values * mtomm * daytos * ft3tom3 / (basin_area * km2tom2)
    data[variables[0]] = data[['prcp(m/day)']].values * 1000
    data[variables[1]] = data[['petfao56(m/day)']].values * 1000*(-1)
    df = data[variables]
    hydro_utils.serialize_numpy(np.expand_dims(df.values, axis=1), npy_file)


def test_load_data(txt_file, npy_file):
    data_ = pd.read_csv(txt_file)
    df = data_[['prcp(m/day)', 'petfao56(m/day)']]
    df['petfao56(m/day)']=df[['petfao56(m/day)']].values*(-1)
    data = hydro_utils.unserialize_numpy(npy_file)[:, :, :2]
    np.testing.assert_array_equal(data, np.expand_dims(df.values*1000, axis=1))

