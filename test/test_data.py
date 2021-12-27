import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

import definitions
from hydromodel.utils import hydro_utils


@pytest.fixture()
def txt_file():
    root_dir = definitions.ROOT_DIR
    print(root_dir)
    test_txt = pd.read_csv(os.path.join(root_dir,'camels_mr','basin_mean_forcing','61019_lump_era5_land_forcing.txt'),sep=" ")
    test_csv= pd.read_csv(os.path.join(root_dir,'camels_mr','streamflow','61019.csv'))
    test_csv['DATE']=test_csv['DATE'].astype('str')
    columns=['Year','Mnth','Day','potential_evaporation','total_precipitation']
    test_data=test_txt[columns]
    test_data['DATE']=pd.to_datetime(dict(year=test_data.Year, month=test_data.Mnth,day=test_data.Day)).dt.date.astype('str')
    test_data_=test_data[['DATE','total_precipitation','potential_evaporation']]
    test_data_con=pd.merge(test_data_,test_csv,on='DATE',how='inner')
    test_data_con.rename(columns={"DATE": "date", "total_precipitation": "prcp(m/day)","potential_evaporation": "petfao56(m/day)","Q": "streamflow(m3/s)"}, inplace=True)
    test_data_con.to_csv('D:\Codes\hydro-model-xaj\hydromodel\data\\61019\\61019_lump_p_pe_q.txt', sep=',', index=False,header=True)
    # test_data_con.to_csv('D:\Codes\hydro-model-xaj\hydromodel\data\\61019\\61019_lump_p_pe_q.csv', sep=',', index=False,
    #                      header=True)
    return os.path.join(definitions.ROOT_DIR, "hydromodel", "data", '61019','61019_lump_p_pe_q.txt')


@pytest.fixture()
def json_file():
    return os.path.join(definitions.ROOT_DIR, "hydromodel", 'data', '61019','data_info.json')


@pytest.fixture()
def npy_file():
    return os.path.join(definitions.ROOT_DIR, "hydromodel", 'data', '61019', 'basins_lump_p_pe_q.npy')


def test_save_data(txt_file, json_file, npy_file):
    data = pd.read_csv(txt_file)
    print(data.columns)
    # Note: The units are all mm/day! For streamflow, data is divided by basin's area
    variables = ['prcp(mm/day)', 'petfao56(mm/day)', 'streamflow(mm/day)']
    data_info = OrderedDict({"time": data['date'].values.tolist(), "basin": ["61019"], "variable": variables})
    hydro_utils.serialize_json(data_info, json_file)
    # 1 ft3 = 0.02831685 m3
    # ft3tom3 = 2.831685e-2
    # 1 km2 = 10^6 m2
    km2tom2 = 1e6
    # 1 m = 1000 mm
    mtomm = 1000
    # 1 day = 24 * 3600 s
    daytos = 24 * 3600
    # trans m3/s to mm/day
    basin_area = 22944.1491
    data[variables[-1]] = data[['streamflow(m3/s)']].values / (basin_area * km2tom2) * mtomm * daytos
    data[variables[0]]=data[['prcp(m/day)']].values*1000
    data[variables[1]] = data[['petfao56(m/day)']].values * 1000
    df = data[variables]
    hydro_utils.serialize_numpy(np.expand_dims(df.values, axis=1), npy_file)
    # # 1 ft3 = 0.02831685 m3
    # # ft3tom3 = 2.831685e-2
    # # 1 km2 = 10^6 m2
    # km2tom2 = 1e6
    # # 1 m = 1000 mm
    # mtomm = 1000
    # # 1 day = 24 * 3600 s
    # daytos = 24 * 3600
    # # trans ft3/s to mm/day
    # basin_area = 22944.1491
    # data[variables[-1]] = data[['streamflow(ft3/s)']].values * ft3tom3 / (basin_area * km2tom2) * mtomm * daytos
    # df = data[variables]
    # hydro_utils.serialize_numpy(np.expand_dims(df.values, axis=1), npy_file)


def test_load_data(txt_file, npy_file):
    data_ = pd.read_csv(txt_file)
    df = data_[['prcp(m/day)', 'petfao56(m/day)']]
    data = hydro_utils.unserialize_numpy(npy_file)[:, :, :2]
    np.testing.assert_array_equal(data, np.expand_dims(df.values*1000, axis=1))
