import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import spotpy

import definitions
from hydromodel.calibrate.calibrate_hymod_sceua import calibrate_hymod_sceua, Spot4HymodSetup
from hydromodel.models.hymod import hymod
from hydromodel.visual.pyspot_plots import show_calibrate_result


@pytest.fixture()
def basin_area():
    # # the area of basin 01013500 is 2252.7; unit km2
    # the area of a basin from hymod example, unit km2
    return 1.783


@pytest.fixture()
def test_data():
    root_dir = definitions.ROOT_DIR
    return pd.read_csv(os.path.join(root_dir, "hydromodel", "example", 'hymod_input.csv'), sep=";")
    # return pd.read_csv(os.path.join(root_dir, "hydromodel","example", '01013500_lump_p_pe_q.txt'))


@pytest.fixture()
def qobs(test_data):
    return np.expand_dims(test_data[['Discharge[ls-1]']].values, axis=1)
    # qobs = np.expand_dims(test_data[['streamflow(ft3/s)']].values, axis=0)

    # 1 ft3 = 0.02831685 m3
    # ft3tom3 = 2.831685e-2
    # trans ft3/s to l/s
    # return qobs * ft3tom3 * 1000


@pytest.fixture()
def p_and_e(test_data):
    p_and_e_df = test_data[['rainfall[mm]', 'TURC [mm d-1]']]
    # p_and_e_df = test_data[['prcp(mm/day)', 'petfao56(mm/day)']]
    # three dims: batch (basin), sequence (time), feature (variable)
    return np.expand_dims(p_and_e_df.values, axis=1)


@pytest.fixture()
def params():
    return np.array([[197.4], [0.1191], [0.2854], [0.07526], [0.533]])


def test_hymod(p_and_e, params):
    qsim = hymod(p_and_e[:, :, 0:1], p_and_e[:, :, 1:2], params[0], params[1], params[2], params[3], params[4])
    np.testing.assert_almost_equal(qsim[:5, 0, 0],
                                   [0.0001236909371541905, 0.00014062802170871676, 0.00021116540384905405,
                                    0.0002213706112236017, 0.00019674672561765445])


def test_calibrate_hymod_sceua(p_and_e, qobs, basin_area):
    calibrate_hymod_sceua(p_and_e, qobs, basin_area)


def test_show_hymod_calibrate_sceua_result(p_and_e, qobs, basin_area):
    spot_setup = Spot4HymodSetup(p_and_e, qobs, basin_area, obj_func=spotpy.objectivefunctions.rmse)
    show_calibrate_result(spot_setup, "SCEUA_hymod", "l s-1")
    plt.show()
