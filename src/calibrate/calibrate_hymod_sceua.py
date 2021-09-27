from typing import Union

import numpy as np
import spotpy
from spotpy.parameter import Uniform, ParameterSet
from spotpy.objectivefunctions import rmse

from src.hymod.hymod import hymod


class Spot4HymodSetup(object):
    cmax = Uniform(low=1.0, high=500, optguess=412.33)
    bexp = Uniform(low=0.1, high=2.0, optguess=0.1725)
    alpha = Uniform(low=0.1, high=0.99, optguess=0.8127)
    Ks = Uniform(low=0.001, high=0.10, optguess=0.0404)
    Kq = Uniform(low=0.1, high=0.99, optguess=0.5592)

    def __init__(self, p_and_e, qobs, basin_area, obj_func=None):
        # Just a way to keep this example flexible and applicable to various examples
        self.obj_func = obj_func
        # Transform [mm/day] into [l s-1]
        self.Factor = basin_area * 1000 * 1000 / (60 * 60 * 24)
        # Load Observation data from file
        self.Precip = p_and_e[0, :, 0]
        self.PET = p_and_e[0, :, 1]
        self.trueObs = qobs[0, :, 0]

    def simulation(self, x: ParameterSet) -> Union[list, np.array]:
        """
        run xaj model

        Parameters
        ----------
        x:
            the parameters of xaj. This function only has this one parameter.

        Returns
        -------
        list
                simulated result from xaj
        """
        # Here the model is actualy startet with one paramter combination
        data = hymod(self.Precip, self.PET, x[0], x[1], x[2], x[3], x[4])
        sim = []
        for val in data:
            sim.append(val * self.Factor)
        # The first year of simulation data is ignored (warm-up)
        return sim[365:]

    def evaluation(self) -> Union[list, np.array]:
        """
        read observation values

        Returns
        -------
        Union[list, np.array]
            observation
        """
        return self.trueObs[365:]

    def objectivefunction(self,
                          simulation: Union[list, np.array],
                          evaluation: Union[list, np.array],
                          params=None) -> float:
        """
        A user defined objective function to calculate fitness.

        Parameters
        ----------
        simulation:
            simulation results
        evaluation:
            evaluation results
        params:
            parameters leading to the simulation

        Returns
        -------
        float
            likelihood
        """
        # SPOTPY expects to get one or multiple values back,
        # that define the performance of the model run
        if not self.obj_func:
            # This is used if not overwritten by user
            like = rmse(evaluation, simulation)
        else:
            # Way to ensure flexible spot setup class
            like = self.obj_func(evaluation, simulation)
        return like


def calibrate_hymod_sceua(p_and_e, qobs, basin_area, random_state=2000):
    parallel = 'seq'  # Runs everthing in sequential mode
    np.random.seed(random_state)  # Makes the results reproduceable

    # Initialize the xaj example
    # In this case, we tell the setup which algorithm we want to use, so
    # we can use this exmaple for different algorithms
    spot_setup = Spot4HymodSetup(p_and_e, qobs, basin_area, spotpy.objectivefunctions.rmse)
    # Select number of maximum allowed repetitions
    sampler = spotpy.algorithms.sceua(spot_setup, dbname='SCEUA_hymod', dbformat='csv', random_state=random_state)
    rep = 5000
    # Start the sampler, one can specify ngs, kstop, peps and pcento id desired
    sampler.sample(rep, ngs=7, kstop=3, peps=0.1, pcento=0.1)
    print("Calibrate Finished!")
