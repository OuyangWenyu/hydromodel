import os
import unittest

from numpy.testing import assert_array_almost_equal
import pandas as pd

import definitions
from src.gr4j.gr4j import gr4j


class Gr4jTestCase(unittest.TestCase):

    def setUp(self):
        root_dir = definitions.ROOT_DIR
        test_data = pd.read_csv(os.path.join(root_dir, "example", 'USGS_02430680_combined.csv'), skiprows=1)
        sims = pd.read_csv(os.path.join(root_dir, "example", 'sims.csv'), header=None, skiprows=1)
        self.p = test_data['P'].iloc[:730]
        self.pet = test_data['PE'].iloc[:730]
        self.qobs = test_data['Q'].iloc[:730]
        self.qsim = sims.iloc[:, 0]

    def tearDown(self):
        pass

    def test_gr4j(self):
        params = {
            'X1': 303.627616,
            'X2': 0.32238919,
            'X3': 6.49759466,
            'X4': 0.294803885
        }

        states = {
            'production_store': 0.60 * params['X1'],
            'routing_store': 0.70 * params['X3']
        }

        qsim = gr4j(self.p, self.pet, params, states)

        assert_array_almost_equal(qsim, self.qsim, 3)


if __name__ == '__main__':
    unittest.main()
