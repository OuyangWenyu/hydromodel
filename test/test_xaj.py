import os
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd

import definitions
from src.calibrate.calibrate_xaj import calibrate_xaj
from src.calibrate.stat import statNse
from src.xaj.xaj import xaj


class XajTestCase(unittest.TestCase):

    def setUp(self):
        root_dir = definitions.ROOT_DIR
        test_data = pd.read_csv(os.path.join(root_dir, "example", 'USGS_02430680_combined.csv'), skiprows=1)
        sims = pd.read_csv(os.path.join(root_dir, "example", 'sims.csv'), header=None, skiprows=1)
        p_and_e_df = test_data[['P', 'PE']].iloc[:730]
        # three dims: batch (basin), sequence (time), feature (variable)
        self.p_and_e = np.expand_dims(p_and_e_df.values, axis=0)
        self.qobs = np.expand_dims(test_data[['Q']].iloc[:730].values, axis=0)
        self.qsim = np.expand_dims(sims.values, axis=0)
        self.params = {
            "B": 0.2,
            "IM": 0.01,
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

    def test_calibrate_xaj(self):
        calibrate_xaj(self.p_and_e, self.qobs, init_states=self.states, run_counts=2, pop_num=2)


if __name__ == '__main__':
    unittest.main()
