import os
from typing import Union
import numpy as np
import spotpy
import pandas as pd
from spotpy.parameter import Uniform, ParameterSet
from hydromodel.models.model_config import read_model_param_dict
from hydromodel.models.model_dict import LOSS_DICT, MODEL_DICT


class SpotSetup(object):
    def __init__(
        self, p_and_e, qobs, warmup_length=365, model=None, param_file=None, loss=None
    ):
        """
        Set up for Spotpy
        NOTE: once for a basin in one sampler or
        for all basins in one sampler with one parameter combination

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
        param_range
            parameters range of model
        loss
            loss configs including objective function, typically RMSE
        """
        if model is None:
            model = {
                "name": "xaj_mz",
                "source_type": "sources5mm",
                "source_book": "HF",
                "time_interval_hours": 1,
            }
        if loss is None:
            loss = {
                "type": "time_series",
                "obj_func": "rmse",
                "events": None,
            }
        self.param_range_file = {"param_range_file": param_file}
        param_range = read_model_param_dict(param_file)
        self.parameter_names = param_range[model["name"]]["param_name"]
        self.model = model
        self.params = []
        self.params.extend(
            Uniform(par_name, low=0.0, high=1.0) for par_name in self.parameter_names
        )
        # Just a way to keep this example flexible and applicable to various examples
        self.loss = loss
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
        # xaj model's output include streamflow and evaporation now,
        # but now we only calibrate the model with streamflow
        sim, _ = MODEL_DICT[self.model["name"]](
            self.p_and_e,
            params,
            warmup_length=self.warmup_length,
            **self.model,
            **self.param_range_file
        )
        return sim

    def evaluation(self) -> Union[list, np.array]:
        """
        read observation values

        Returns
        -------
        Union[list, np.array]
            observation
        """
        return self.true_obs

    def objectivefunction(
        self,
        simulation: Union[list, np.array],
        evaluation: Union[list, np.array],
        params=None,  # this cannot be removed
    ) -> float:
        """
        A user defined objective function to calculate fitness.

        Parameters
        ----------
        simulation:
            simulation results
        evaluation:
            evaluation results

        Returns
        -------
        float
            likelihood
        """
        if self.loss["type"] == "time_series":
            return LOSS_DICT[self.loss["obj_func"]](evaluation, simulation)
        # for events
        time = self.loss["events"]
        if time is None:
            raise ValueError(
                "time should not be None since you choose events, otherwise choose time_series"
            )
        # TODO: not finished for events
        calibrate_starttime = pd.to_datetime("2012-06-10 0:00:00")
        calibrate_endtime = pd.to_datetime("2019-12-31 23:00:00")
        total = 0
        count = 0
        for i in range(len(time)):
            if time.iloc[i, 0] < calibrate_endtime:
                start_num = (
                    time.iloc[i, 0] - calibrate_starttime - pd.Timedelta(hours=365)
                ) / pd.Timedelta(hours=1)
                end_num = (
                    time.iloc[i, 1] - calibrate_starttime - pd.Timedelta(hours=365)
                ) / pd.Timedelta(hours=1)
                start_num = int(start_num)
                end_num = int(end_num)
                like_ = LOSS_DICT[self.loss["obj_func"]](
                    evaluation[start_num:end_num,], simulation[start_num:end_num,]
                )
                count += 1

                total += like_
        return total / count


def calibrate_by_sceua(
    basins,
    p_and_e,
    qobs,
    dbname,
    warmup_length=365,
    model=None,
    algorithm=None,
    loss=None,
    param_file=None,
):
    """
    Function for calibrating model by SCE-UA

    Now we only support one basin's calibration in one sampler

    Parameters
    ----------
    basins
        basin ids
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
    algorithm
        calibrate algorithm. For example, if you want to calibrate xaj model,
        and use sce-ua algorithm -- random seed=2000, rep=5000, ngs=7, kstop=3, peps=0.1, pcento=0.1
    loss
        loss configs for events calculation or
        just one long time-series calculation
        with an objective function, typically RMSE
    param_file
        the file of the parameter range, yaml file

    Returns
    -------
    None
    """
    if model is None:
        model = {
            "name": "xaj_mz",  # 模型
            "source_type": "sources5mm",
            "source_book": "HF",
            "time_interval_hours": 1,
        }
    if algorithm is None:
        algorithm = {
            "name": "SCE_UA",
            "random_seed": 1234,
            "rep": 1000,
            "ngs": 1000,
            "kstop": 500,
            "peps": 0.1,
            "pcento": 0.1,
        }
    if loss is None:
        loss = {
            "type": "time_series",
            "obj_func": "RMSE",
            # when "type" is "events", this is not None, but idxs of events in time series
            "events": None,
        }
    random_seed = algorithm["random_seed"]
    rep = algorithm["rep"]
    ngs = algorithm["ngs"]
    kstop = algorithm["kstop"]
    peps = algorithm["peps"]
    pcento = algorithm["pcento"]
    np.random.seed(random_seed)  # Makes the results reproduceable
    for i in range(len(basins)):
        # Initialize the xaj example
        # In this case, we tell the setup which algorithm we want to use, so
        # we can use this exmaple for different algorithms
        spot_setup = SpotSetup(
            p_and_e[:, i : i + 1, :],
            qobs[:, i : i + 1, :],
            warmup_length=warmup_length,
            model=model,
            loss=loss,
            param_file=param_file,
        )
        if not os.path.exists(dbname):
            os.makedirs(dbname)
        db_basin = os.path.join(dbname, basins[i])
        # Select number of maximum allowed repetitions
        sampler = spotpy.algorithms.sceua(
            spot_setup,
            dbname=db_basin,
            dbformat="csv",
            random_state=random_seed,
        )
        # Start the sampler, one can specify ngs, kstop, peps and pcento id desired
        sampler.sample(rep, ngs=ngs, kstop=kstop, peps=peps, pcento=pcento)
        print("Calibrate Finished!")
    return sampler
