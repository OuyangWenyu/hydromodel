import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import definitions
from src.calibrate.calibrate_xaj_ga import calibrate_xaj_ga
from src.calibrate.calibrate_xaj_sceua import calibrate_xaj_sceua, show_calibrate_result
from src.xaj.xaj import xaj


class XajTestCase(unittest.TestCase):

    def setUp(self):
        root_dir = definitions.ROOT_DIR
        test_data = pd.read_csv(os.path.join(root_dir, "example", '01013500_lump_p_pe_q.txt'))
        p_and_e_df = test_data[['prcp(mm/day)', 'petfao56(mm/day)']].iloc[:730]
        # three dims: batch (basin), sequence (time), feature (variable)
        self.p_and_e = np.expand_dims(p_and_e_df.values, axis=0)
        qobs = np.expand_dims(test_data[['streamflow(ft3/s)']].iloc[:730].values, axis=0)
        # the area of basin 01013500, unit km2
        basin_area = 2252.7
        # 1 ft3 = 0.02831685 m3
        ft3tom3 = 2.831685e-2
        # 1 km2 = 10^6 m2
        km2tom2 = 1e6
        # 1 m = 1000 mm
        mtomm = 1000
        # 1 day = 24 * 3600 s
        daytos = 24 * 3600
        # trans ft3/s to mm/day
        self.qobs = qobs * ft3tom3 / (basin_area * km2tom2) * mtomm * daytos
        self.params = {
            "B": 0.3,
            "IM": 0.03,
            "UM": 20,
            "LM": 70,
            "DM": 70,
            "C": 0.15,
            "SM": 20,
            "EX": 1.2,
            "KI": 0.4,
            "KG": 0.3,
            "CS": 0.2,
            "CI": 0.5,
            "CG": 0.99,
        }

        self.states = {
            "WU0": 0.60 * self.params['UM'],
            "WL0": 0.60 * self.params['LM'],
            "WD0": 0.60 * self.params['DM'],
            "S0": 0.60 * self.params['SM'],
            "FR0": 0.02,
            "QS0": 0.01,
            "QI0": 0.01,
            "QG0": 0.01,
        }

    def tearDown(self):
        pass

    def test_xaj(self):
        qsim = xaj(self.p_and_e, self.params, self.states, source_type="sources5mm")
        # assert_array_almost_equal(qsim, self.qsim, 3)

    def test_calibrate_xaj_ga(self):
        calibrate_xaj_ga(self.p_and_e, self.qobs, init_states=self.states, run_counts=500, pop_num=50)

    def test_calibrate_xaj_sceua(self):
        calibrate_xaj_sceua(self.p_and_e, self.qobs, init_states=self.states)

    def test_show_calibrate_sceua_result(self):
        show_calibrate_result(self.p_and_e, self.qobs, init_states=self.states)
        plt.show()


if __name__ == '__main__':
    unittest.main()
