import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spotpy

import definitions
from src.calibrate.calibrate_hymod_sceua import calibrate_hymod_sceua, Spot4HymodSetup
from src.visual.pyspot_plots import show_calibrate_result


class XajTestCase(unittest.TestCase):

    def setUp(self):
        root_dir = definitions.ROOT_DIR
        test_data = pd.read_csv(os.path.join(root_dir, "example", '01013500_lump_p_pe_q.txt'))
        p_and_e_df = test_data[['prcp(mm/day)', 'petfao56(mm/day)']]
        qobs = np.expand_dims(test_data[['streamflow(ft3/s)']].values, axis=0)
        # the area of basin 01013500, unit km2
        self.basin_area = 2252.7

        # test_data = pd.read_csv(os.path.join(root_dir, "example", 'USGS_02430680_combined.csv'), skiprows=1)
        # p_and_e_df = test_data[['P', 'PE']]
        # qobs = np.expand_dims(test_data[['Q']].values, axis=0)
        # # the areas of basins 01013500, 02430680 are 2252.7, 342.2 respectively; unit km2
        # self.basin_area = 342.2

        # 1 ft3 = 0.02831685 m3
        ft3tom3 = 2.831685e-2
        # trans ft3/s to l/s
        self.qobs = qobs * ft3tom3 * 1000

        # test_data = pd.read_csv(os.path.join(root_dir, "example", 'hymod_input.csv'), sep=";")
        # p_and_e_df = test_data[['rainfall[mm]', 'TURC [mm d-1]']]
        # self.qobs = np.expand_dims(test_data[['Discharge[ls-1]']].values, axis=0)
        # self.basin_area = 1.783

        # three dims: batch (basin), sequence (time), feature (variable)
        self.p_and_e = np.expand_dims(p_and_e_df.values, axis=0)

    def tearDown(self):
        pass

    def test_calibrate_hymod_sceua(self):
        calibrate_hymod_sceua(self.p_and_e, self.qobs, self.basin_area)

    def test_show_hymod_calibrate_sceua_result(self):
        spot_setup = Spot4HymodSetup(self.p_and_e, self.qobs, self.basin_area, obj_func=spotpy.objectivefunctions.rmse)
        show_calibrate_result(spot_setup, "SCEUA_hymod", "l s-1")
        plt.show()


if __name__ == '__main__':
    unittest.main()
