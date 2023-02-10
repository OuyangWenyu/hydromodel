from typing import Union
import numpy as np
import spotpy
from spotpy.parameter import Uniform, ParameterSet
from spotpy.objectivefunctions import rmse
from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.models.gr4j import gr4j
from hydromodel.models.hymod import hymod
from hydromodel.models.xaj import xaj


class SpotSetup(object):
    def __init__(
        self,
        p_and_e,
        qobs,
        warmup_length=30,
        model={
            "name": "xaj_mz",
            "source_type": "sources",
            "source_book": "HF",
        },
        obj_func=None,
    ):
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
        model_func_param
            parameters of model function
        obj_func
            objective function, typically RMSE
        """
        self.parameter_names = MODEL_PARAM_DICT[model["name"]]["param_name"]
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
        # parameter, 2-dim variable: [basin=1, parameter]
        params = np.array(x).reshape(1, -1)
        if self.model["name"] in ["xaj", "xaj_mz"]:
            # xaj model's output include streamflow and evaporation now,
            # but now we only calibrate the model with streamflow
            sim, _ = xaj(
                self.p_and_e, params, warmup_length=self.warmup_length, **self.model
            )
        elif self.model["name"] == "gr4j":
            sim = gr4j(
                self.p_and_e, params, warmup_length=self.warmup_length, **self.model
            )
        elif self.model["name"] == "hymod":
            sim = hymod(
                self.p_and_e, params, warmup_length=self.warmup_length, **self.model
            )
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

    def objectivefunction(
        self,
        simulation: Union[list, np.array],
        evaluation: Union[list, np.array],
        params=None,
    ) -> float:
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


def calibrate_by_sceua(
    p_and_e,
    qobs,
    dbname,
    warmup_length=30,
    model={
        "name": "xaj_mz",
        "source_type": "sources",
        "source_book": "HF",
    },
    algorithm={
        "name": "SCE_UA",
        "random_seed": 1234,
        "rep": 1000,
        "ngs": 1000,
        "kstop": 500,
        "peps": 0.001,
        "pcento": 0.001,
    },
):
    """
    Function for calibrating model by SCE-UA

    Now we only support one basin's calibration in one sampler

    Parameters
    ----------
    p_and_e
        inputs of model
    qobs
        observation data
    dbname
        where save the result file of sampler
    warmup_length
        the length of warmup period
    model
        we support "gr4j", "hymod", and "xaj", parameters for hydro model
    calibrate_algo
        calibrate algorithm. For example, if you want to calibrate xaj model,
        and use sce-ua algorithm -- random seed=2000, rep=5000, ngs=7, kstop=3, peps=0.1, pcento=0.1

    Returns
    -------
    None
    """
    random_seed = algorithm["random_seed"]
    rep = algorithm["rep"]
    ngs = algorithm["ngs"]
    kstop = algorithm["kstop"]
    peps = algorithm["peps"]
    pcento = algorithm["pcento"]
    np.random.seed(random_seed)  # Makes the results reproduceable

    # Initialize the xaj example
    # In this case, we tell the setup which algorithm we want to use, so
    # we can use this exmaple for different algorithms
    spot_setup = SpotSetup(
        p_and_e,
        qobs,
        warmup_length=warmup_length,
        model=model,
        obj_func=spotpy.objectivefunctions.rmse,
    )
    # Select number of maximum allowed repetitions
    sampler = spotpy.algorithms.sceua(
        spot_setup,
        dbname=dbname,
        dbformat="csv",
        random_state=random_seed,
    )
    # Start the sampler, one can specify ngs, kstop, peps and pcento id desired
    sampler.sample(rep, ngs=ngs, kstop=kstop, peps=peps, pcento=pcento)
    print("Calibrate Finished!")
    return sampler
