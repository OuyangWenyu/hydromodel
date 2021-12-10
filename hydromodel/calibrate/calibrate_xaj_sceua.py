from typing import Union

import numpy as np
import spotpy
from spotpy.parameter import Uniform, ParameterSet
from spotpy.objectivefunctions import rmse

from hydromodel.models.xaj import xaj


class SpotSetup(object):
    # All parameters' range are [0,1], we will transform them to normal range in the model
    B = Uniform(low=0.0, high=1.0)
    IM = Uniform(low=0.0, high=1.0)
    UM = Uniform(low=0.0, high=1.0)
    LM = Uniform(low=0.0, high=1.0)
    DM = Uniform(low=0.0, high=1.0)
    C = Uniform(low=0.0, high=1.0)
    SM = Uniform(low=0.0, high=1.0)
    EX = Uniform(low=0.0, high=1.0)
    KI = Uniform(low=0.0, high=1.0)
    KG = Uniform(low=0.0, high=1.0)
    A = Uniform(low=0.0, high=1.0)
    THETA = Uniform(low=0.0, high=1.0)
    CI = Uniform(low=0.0, high=1.0)
    CG = Uniform(low=0.0, high=1.0)

    def __init__(self, p_and_e, qobs, warmup_length=30, obj_func=None):
        """
        Set up for Spotpy

        Parameters
        ----------
        p_and_e
            inputs of model
        qobs
            observation data
        warmup_length
            XAJ model need warmup period
        obj_func
            objective function, typically RMSE
        """
        # Just a way to keep this example flexible and applicable to various examples
        self.obj_func = obj_func
        # Load Observation data from file
        self.p_and_e = p_and_e
        # chose observation data after warmup period
        self.true_obs = qobs[warmup_length:, :, :]

    def simulation(self, x: ParameterSet) -> Union[list, np.array]:
        """
        run xaj model

        Parameters
        ----------
        x
            the parameters of xaj. This function only has this one parameter.

        Returns
        -------
        Union[list, np.array]
            simulated result from xaj
        """
        # Here the model is started with one parameter combination
        # TODO: Now ParameterSet only support one list, and we only support one basin's calibration now
        params = np.array(x).reshape(-1, 1)
        sim = xaj(self.p_and_e, params)
        return sim[:, 0, 0]

    def evaluation(self) -> Union[list, np.array]:
        """
        read observation values

        Returns
        -------
        Union[list, np.array]
            observation
        """
        # TODO: we only support one basin's calibration now
        return self.true_obs[:, 0, 0]

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


def calibrate_xaj_sceua(p_and_e, qobs, warmup_length=30, random_state=2000):
    """
    Function for calibrating hymod

    Parameters
    ----------
    p_and_e
        inputs of model
    qobs
        observation data
    warmup_length
        the length of warmup period
    random_state
        random seed

    Returns
    -------
    None
    """
    np.random.seed(random_state)  # Makes the results reproduceable

    # Initialize the xaj example
    # In this case, we tell the setup which algorithm we want to use, so
    # we can use this exmaple for different algorithms
    spot_setup = SpotSetup(p_and_e, qobs, warmup_length=warmup_length, obj_func=spotpy.objectivefunctions.rmse)
    # Select number of maximum allowed repetitions
    sampler = spotpy.algorithms.sceua(spot_setup, dbname='SCEUA_xaj', dbformat='csv', random_state=random_state)
    rep = 5000
    # Start the sampler, one can specify ngs, kstop, peps and pcento id desired
    sampler.sample(rep, ngs=7, kstop=3, peps=0.1, pcento=0.1)
    print("Calibrate Finished!")
