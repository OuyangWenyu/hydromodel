import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import spotpy

import definitions
from hydromodel.calibrate.calibrate_sceua import calibrate_by_sceua, SpotSetup
from hydromodel.calibrate.calibrate_xaj_ga import calibrate_xaj_ga
from hydromodel.visual.pyspot_plots import show_calibrate_result,show_test_result
from hydromodel.models.xaj import xaj, uh_gamma, uh_conv
from hydromodel.utils import hydro_utils
from collections import OrderedDict


@pytest.fixture()
def basin_id():
    return 9510200


@pytest.fixture()
def basin_area(basin_id):
    basin_areas = OrderedDict({
        "1439500": 306,
        "6888500": 843,
        "8104900": 348,
        "9510200": 428,
        "61019": 22944.1491,
        "61561": 8750.7865,
        "61215": 134545.7288,
        "92355": 1586.9165,
        "92198": 75720.9026,
        "61277": 1771.0947,
        "61566": 12549.3679,
        "61141": 30512.6020,
        "96012": 68117.1423,
        "61239": 3385.4424,
        "92353": 12931.5138,
        "61417": 76257.9366,
        "92200": 72457.5961,
        "92199": 75997.3976,
        "92352": 18839.4601,
        "61205": 125647.9689,
        "92354": 2395.8437,
        "96001": 22567.2369,
    })
    basin_area=basin_areas.get(str(basin_id))
    return basin_area



@pytest.fixture()
def t_range_test():
    return ["2014-01-01", "2020-01-01"]
    # return ["2019-10-01","2021-10-01"]

@pytest.fixture()
def warmup_length():
    return 365

@pytest.fixture()
def npy_file(basin_id):
    return os.path.join(definitions.ROOT_DIR, "data", str(basin_id), 'basins_lump_p_pe_q.npy')

@pytest.fixture()
def split_train_test(basin_id,t_range_test):
    txt_file = os.path.join(definitions.ROOT_DIR, "data",str(basin_id),str(basin_id)+'_lump_p_pe_q.txt')
    data = pd.read_csv(txt_file)
    date = pd.to_datetime(data['date']).values.astype('datetime64[D]')
    t_range_test_ = hydro_utils.t_range_days(t_range_test)
    [C, ind1, ind2] = np.intersect1d(date, t_range_test_, return_indices=True)
    return ind1


@pytest.fixture()
def p_and_e(npy_file,split_train_test,warmup_length):
    p_and_e_df = hydro_utils.unserialize_numpy(npy_file)[:, :, :2]
    p_and_e_train = p_and_e_df[0:split_train_test[0]]
    p_and_e_test = p_and_e_df[split_train_test[0] - warmup_length:]
    return p_and_e_train, p_and_e_test


@pytest.fixture()
def qobs(npy_file,split_train_test,warmup_length):
    qobs_ = hydro_utils.unserialize_numpy(npy_file)[:, :, 2]
    qobs_train = qobs_[0:split_train_test[0]]
    qobs_test = qobs_[split_train_test[0] - warmup_length:]
    return np.expand_dims(qobs_train, axis=1), np.expand_dims(qobs_test, axis=1)


@pytest.fixture()
def params():
    # all parameters are in range [0,1]
    return np.tile([0.5], (1, 14))


@pytest.fixture()
def calibrate_params(basin_id):
    SCEUA_xaj_params=pd.read_csv(os.path.join(definitions.ROOT_DIR, "test","SCEUA_xaj.csv"))
    bestindex = SCEUA_xaj_params["like1"].idxmin()
    calibrate_params=SCEUA_xaj_params.iloc[bestindex, 1:15]
    calibrate_params.to_csv('..\\data\\' + str(basin_id) + '\\' + str(basin_id) + '_calibrate_params.txt', sep=',',index=False, header=True)
    return np.array(SCEUA_xaj_params.iloc[bestindex, 1:15]).reshape(1,14)


def test_uh_gamma():
    # repeat for 20 periods and add one dim as feature: time_seq=20, batch=10, feature=1
    routa = np.tile(2.5, (20, 10, 1))
    routb = np.tile(3.5, (20, 10, 1))
    uh = uh_gamma(routa, routb, len_uh=15)
    np.testing.assert_almost_equal(uh[:, 0, :], np.array(
        [[0.0069], [0.0314], [0.0553], [0.0738], [0.0860], [0.0923], [0.0939], [0.0919], [0.0875], [0.0814],
         [0.0744], [0.0670], [0.0597], [0.0525], [0.0459]]), decimal=3)


def test_uh():
    uh_from_gamma = np.tile(1, (5, 3, 1))
    rf = np.arange(30).reshape(10, 3, 1) / 100
    qs = uh_conv(rf, uh_from_gamma)
    np.testing.assert_almost_equal(np.array([[0.0000, 0.0100, 0.0200],
                                             [0.0300, 0.0500, 0.0700],
                                             [0.0900, 0.1200, 0.1500],
                                             [0.1800, 0.2200, 0.2600],
                                             [0.3000, 0.3500, 0.4000],
                                             [0.4500, 0.5000, 0.5500],
                                             [0.6000, 0.6500, 0.7000],
                                             [0.7500, 0.8000, 0.8500],
                                             [0.9000, 0.9500, 1.0000],
                                             [1.0500, 1.1000, 1.1500]]), qs[:, :, 0], decimal=3)


# def test_xaj(p_and_e, params, warmup_length):
#     qsim = xaj(p_and_e, params, warmup_length=warmup_length)
#     np.testing.assert_array_equal(qsim.shape[0], p_and_e.shape[0] - warmup_length)


def test_calibrate_xaj_sceua(p_and_e, qobs, warmup_length):
    calibrate_by_sceua(p_and_e[0], qobs[0], warmup_length)


def test_show_calibrate_sceua_result(p_and_e, qobs, warmup_length,basin_id,split_train_test):
    spot_setup = SpotSetup(p_and_e[0], qobs[0], warmup_length, obj_func=spotpy.objectivefunctions.rmse)
    show_calibrate_result(spot_setup, "SCEUA_xaj",warmup_length,basin_id,split_train_test)
    plt.show()


def test_show_calibrate_result_xaj(p_and_e,calibrate_params, warmup_length, qobs,basin_id,basin_area):
    qsim_ = xaj(p_and_e[1],calibrate_params, warmup_length=warmup_length)
    # 1 ft3 = 0.02831685 m3
    ft3tom3 = 2.831685e-2
    # 1 km2 = 10^6 m2
    km2tom2 = 1e6
    # 1 m = 1000 mm
    mtomm = 1000
    # 1 day = 24 * 3600 s
    daytos = 24 * 3600
    qsim = qsim_ * basin_area * km2tom2 /(mtomm * daytos * ft3tom3)
    qobs_= qobs[1]* basin_area * km2tom2 /(mtomm * daytos * ft3tom3)
    show_test_result(qsim, qobs_, warmup_length,basin_id)
    plt.show()


# def test_calibrate_xaj_ga(p_and_e, qobs, warmup_length):
#     calibrate_xaj_ga(p_and_e, qobs, warmup_length, run_counts=5, pop_num=50)
