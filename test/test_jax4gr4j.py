import os
import unittest
import jax.numpy as jnp

import definitions
from src.gr4j import jax4gr4j as jg
import pandas as pd


class TestCases(unittest.TestCase):

    def test_forward_simulate(self):
        data = pd.read_csv(os.path.join(definitions.ROOT_DIR, "example", "sample.csv"))
        P = jnp.array(data['P'].values)
        E = jnp.array(data['ET'].values)
        x1 = 320.11
        x2 = 2.42
        x3 = 69.63
        x4 = 1.39
        x4_limit = 5.

        S0 = 0.6 * 320.11
        R0 = 0.7 * 69.63
        Pr0 = jnp.zeros(9)

        streamflow = jg.simulate_streamflow(P, E,
                                            S0, Pr0, R0,
                                            x1, x2, x3, x4, x4_limit)

        mae = jnp.mean(jnp.abs(streamflow - data['modeled_Q'].values))

        self.assertTrue(mae < 0.01)
