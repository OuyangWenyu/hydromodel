from typing import Union

import numpy as np
import spotpy
from spotpy.parameter import Uniform, ParameterSet
from spotpy.objectivefunctions import rmse

from hydromodel.models.gr4j import gr4j
from hydromodel.models.hymod import hymod
from hydromodel.models.xaj import xaj


class SpotSetup(object):
    def __init__(self, p_and_e, qobs, warmup_length=30, model="xaj", obj_func=None):
        """
        Set up for Spotpy

        Parameters
        ----------
        p_and_e
            inputs of model
        qobs
            observation data
        warmup_length
            GR4J model need warmup period
        model
            we support "gr4j", "hymod", and "xaj"
        obj_func
            objective function, typically RMSE
        """
        if model == "xaj":
            self.parameter_names = ["B", "IM", "UM", "LM", "DM", "C", "SM", "EX", "KI", "KG", "A", "THETA", "CI", "CG"]
        elif model == "gr4j":
            self.parameter_names = ["x1", "x2", "x3", "x4"]
        elif model == "hymod":
            self.parameter_names = ["cmax", "bexp", "alpha", "ks", "kq"]
        else:
            raise NotImplementedError("We don't provide this model now")
        self.model = model
        self.params = []
        for par_name in self.parameter_names:
            # All parameters' range are [0,1], we will transform them to normal range in the model
            self.params.append(Uniform(par_name, low=0.0, high=1.0))
        # Just a way to keep this example flexible and applicable to various examples
        self.obj_func = obj_func
        # Load Observation data from file
        self.p_and_e = p_and_e
        # chose observation data after warmup period
        self.true_obs = qobs[warmup_length:, :, :]
        self.warmup_length = warmup_length

    def parameters(self):
        return spotpy.parameter.generate(self.params)

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
        # parameter, 2-dim variable: [basin=1, parameter]
        params = np.array(x).reshape(1, -1)
        if self.model == "xaj":
            sim = xaj(self.p_and_e, params, warmup_length=self.warmup_length)
        elif self.model == "gr4j":
            sim = gr4j(self.p_and_e, params, warmup_length=self.warmup_length)
        elif self.model == "hymod":
            sim = hymod(self.p_and_e, params, warmup_length=self.warmup_length)
        else:
            raise NotImplementedError("We don't provide this model now")
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


def calibrate_by_sceua(p_and_e, qobs, warmup_length=30, model="xaj", random_state=2000):
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
    model
        we support "gr4j", "hymod", and "xaj"
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
    spot_setup = SpotSetup(p_and_e, qobs, warmup_length=warmup_length, model=model,
                           obj_func=spotpy.objectivefunctions.rmse)
    # Select number of maximum allowed repetitions

    sampler = spotpy.algorithms.sceua(spot_setup, dbname='SCEUA_' + model, dbformat='csv', random_state=random_state)
    rep = 5000
    # Start the sampler, one can specify ngs, kstop, peps and pcento id desired
    ngs = 20
    kstop =20
    peps = 0.1
    pcento = 0.1
    sampler.sample(rep, ngs, kstop, peps, pcento)
    print("Calibrate Finished!")
