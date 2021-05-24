import os
import unittest
import numpy as np
import pandas as pd
import definitions
from src.data.data_camels import Camels
from src.pet.pet4daymet import pm_fao56
from src.utils.hydro_utils import t_range_days
import jax.numpy as jnp
from src.xaj import jax4xaj as jx
from src.xaj import np4xaj as nx
from src.xaj.data_process import read_data, init_parameters, initialize_condition


class TestXajDaily(unittest.TestCase):
    def setUp(self) -> None:
        self.camels = Camels(os.path.join(definitions.DATASET_DIR, "camels"))
        self.usgs_id_lst = ["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
                            "01055000", "01057000", "01170100"]
        self.t_range = ["1995-01-01", "2015-01-01"]

    def test_cal_pet_with_basin_mean_forcing(self):
        print("calculate pet with basin mean forcing data:")
        camels = self.camels
        usgs_id_lst = self.usgs_id_lst
        t_range = self.t_range
        var_lst = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
        forcing_data = camels.read_relevant_cols(usgs_id_lst, t_range, var_lst)
        t_min = forcing_data[:, :, 5]
        t_max = forcing_data[:, :, 4]
        # average over the daylight period of the day, W/m^2 -> average over the day, MJ m-2 day-1
        r_surf = forcing_data[:, :, 2] * forcing_data[:, :, 0] * 1e-6
        const_var_lst = ["gauge_lat", "elev_mean"]
        attr_data = camels.read_constant_cols(usgs_id_lst, const_var_lst)
        lat = attr_data[:, 0]
        # ° -> rad
        phi = lat * np.pi / 180.0
        elevation = attr_data[:, 1]
        dtype = t_min.dtype
        time_range_lst = t_range_days(t_range)
        doy_values = pd.to_datetime(time_range_lst).dayofyear.astype(dtype).values
        doy = np.expand_dims(doy_values, axis=0).repeat(len(usgs_id_lst), axis=0)
        e_a = forcing_data[:, :, 6] * 1e-3

        phi_ = np.expand_dims(phi, axis=1).repeat(len(time_range_lst), axis=1)
        elev_ = np.expand_dims(elevation, axis=1).repeat(len(time_range_lst), axis=1)
        pet_fao56 = pm_fao56(t_min, t_max, r_surf, phi_, elev_, doy, e_a=e_a)
        print(pet_fao56)

    def test_forward_simulate(self):
        dataset = self.camels
        basins_id = ["01013500"]
        t_range = ["1990-01-01", "1992-01-01"]
        basin_areas, rainfalls, pet, streamflows = read_data(dataset, basins_id, t_range)
        xaj_params = init_parameters(basins_id)
        initial_conditions = initialize_condition(basins_id)
        b = xaj_params["B"][0]
        im = xaj_params["IMP"][0]
        um = xaj_params["WUM"][0]
        lm = xaj_params["WLM"][0]
        dm = xaj_params["WDM"][0]
        c = xaj_params["B"][0]
        sm = xaj_params["SM"][0]
        ex = xaj_params["EX"][0]
        ki = xaj_params["KI"][0]
        kg = xaj_params["KG"][0]
        ci = xaj_params["CI"][0]
        cg = xaj_params["CG"][0]
        l = jnp.int32(xaj_params["L"][0])
        cs = xaj_params["CR"][0]

        wu0 = initial_conditions["WU"][0]
        wl0 = initial_conditions["WL"][0]
        wd0 = initial_conditions["WD"][0]
        s0 = initial_conditions["S0"][0]
        fr0 = initial_conditions["FR0"][0]
        qi0 = 0
        qg0 = 0
        qt_history = jnp.zeros(pet.shape[1])
        q0 = 0

        Q = jx.simulate_streamflow(rainfalls[0], pet[0], basin_areas[0],
                                   wu0, wl0, wd0, s0, fr0, qi0, qg0, qt_history, q0,
                                   b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg, l, cs)
        print(Q)
        mae = jnp.mean(jnp.abs(Q - streamflows))

        print(mae)

    def test_forward_simulate_np(self):
        dataset = self.camels
        basins_id = ["01013500"]
        t_range = ["1990-01-01", "1992-01-01"]
        basin_areas, rainfalls, pet, streamflows = read_data(dataset, basins_id, t_range)
        xaj_params = init_parameters(basins_id)
        initial_conditions = initialize_condition(basins_id)
        b = xaj_params["B"][0]
        im = xaj_params["IMP"][0]
        um = xaj_params["WUM"][0]
        lm = xaj_params["WLM"][0]
        dm = xaj_params["WDM"][0]
        c = xaj_params["B"][0]
        sm = xaj_params["SM"][0]
        ex = xaj_params["EX"][0]
        ki = xaj_params["KI"][0]
        kg = xaj_params["KG"][0]
        ci = xaj_params["CI"][0]
        cg = xaj_params["CG"][0]
        l = int(xaj_params["L"][0])
        cs = xaj_params["CR"][0]

        wu0 = initial_conditions["WU"][0]
        wl0 = initial_conditions["WL"][0]
        wd0 = initial_conditions["WD"][0]
        s0 = initial_conditions["S0"][0]
        fr0 = initial_conditions["FR0"][0]
        qi0 = 0
        qg0 = 0
        qt_history = jnp.zeros(pet.shape[1])
        q0 = 0

        Q = nx.simulate_streamflow(rainfalls[0], pet[0], basin_areas[0],
                                   wu0, wl0, wd0, s0, fr0, qi0, qg0, qt_history, q0,
                                   b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg, l, cs)
        mae = jnp.mean(jnp.abs(Q - streamflows))

        print(mae)


if __name__ == '__main__':
    unittest.main()
