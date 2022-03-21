import os

import numpy as np
import pandas as pd
import pytest
import spotpy
from matplotlib import pyplot as plt
import definitions
from hydromodel.calibrate.calibrate_sceua import calibrate_by_sceua, SpotSetup
from hydromodel.models.gr4j import gr4j
from hydromodel.visual.pyspot_plots import show_calibrate_result


@pytest.fixture()
def basin_area():
    # the area of basin 01013500, unit km2
    # basin_area = 2252.7
    return 1.783


@pytest.fixture()
def warmup_length():
    return 30


@pytest.fixture()
def the_data():
    root_dir = definitions.ROOT_DIR
    # test_data = pd.read_csv(os.path.join(root_dir, "hydromodel", "example", '01013500_lump_p_pe_q.txt'))
    return pd.read_csv(
        os.path.join(root_dir, "hydromodel", "example", "hymod_input.csv"), sep=";"
    )


@pytest.fixture()
def p_and_e(the_data):
    # p_and_e_df = test_data[['prcp(mm/day)', 'petfao56(mm/day)']]
    # three dims: sequence (time), batch (basin), feature (variable)
    # p_and_e = np.expand_dims(p_and_e_df.values, axis=1)
    p_and_e_df = the_data[["rainfall[mm]", "TURC [mm d-1]"]]
    return np.expand_dims(p_and_e_df.values, axis=1)


@pytest.fixture()
def qobs(basin_area, the_data):
    # 1 ft3 = 0.02831685 m3
    ft3tom3 = 2.831685e-2
    # 1 km2 = 10^6 m2
    km2tom2 = 1e6
    # 1 m = 1000 mm
    mtomm = 1000
    # 1 day = 24 * 3600 s
    daytos = 24 * 3600
    # qobs_ = np.expand_dims(test_data[['streamflow(ft3/s)']].values, axis=1)
    # trans ft3/s to mm/day
    # return qobs_ * ft3tom3 / (basin_area * km2tom2) * mtomm * daytos

    qobs_ = np.expand_dims(the_data[["Discharge[ls-1]"]].values, axis=1)
    # trans l/s to mm/day
    return qobs_ * 1e-3 / (basin_area * km2tom2) * mtomm * daytos


@pytest.fixture()
def params():
    # all parameters are in range [0,1]
    return np.tile([0.5], (1, 4))


def test_gr4j(p_and_e, params):
    qsim = gr4j(p_and_e, params, warmup_length=10)
    np.testing.assert_array_equal(qsim.shape, (1817, 1, 1))


def test_calibrate_gr4j_sceua(p_and_e, qobs, warmup_length):
    calibrate_by_sceua(
        p_and_e,
        qobs,
        warmup_length,
        model="gr4j",
        random_seed=2000,
        rep=5000,
        ngs=7,
        kstop=3,
        peps=0.1,
        pcento=0.1,
    )


def test_show_calibrate_sceua_result(p_and_e, qobs, warmup_length):
    spot_setup = SpotSetup(
        p_and_e,
        qobs,
        warmup_length,
        model="gr4j",
        obj_func=spotpy.objectivefunctions.rmse,
    )
    show_calibrate_result(spot_setup, "SCEUA_gr4j")
    plt.show()
