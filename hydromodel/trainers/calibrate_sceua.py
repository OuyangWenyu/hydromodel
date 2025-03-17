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
                "kernel_size": 15,
                "time_interval_hours": 24,
            }
        if loss is None:
            loss = {
                "type": "time_series",
                "obj_func": "rmse",
                "events": None,
            }
        self.param_range_file = {"param_range_file": param_file}
        self.param_range = read_model_param_dict(param_file)
        self.parameter_names = self.param_range[model["name"]]["param_name"]
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
            **self.param_range
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
    """
    if model is None:
        model = {
            "name": "xaj_mz",
            "source_type": "sources5mm",
            "source_book": "HF",
            "kernel_size": 15,
            "time_interval_hours": 24,
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
            "events": None,
        }
    random_seed = algorithm["random_seed"]
    rep = algorithm["rep"]
    ngs = algorithm["ngs"]
    kstop = algorithm["kstop"]
    peps = algorithm["peps"]
    pcento = algorithm["pcento"]
    np.random.seed(random_seed)  # Makes the results reproduceable
    
    samplers = []
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
        
        # 修改获取最佳参数的方式
        best_params = {}
        best_params[basins[i]] = {}
        # 打印模型参数信息
        print(f"模型名称: {model['name']}")
        #print(f"参数名称列表: {spot_setup.parameter_names}")
        #print(f"参数数量: {len(spot_setup.parameter_names)}")
        # 获取数据并转换为DataFrame
        results = sampler.getdata()
        df_results = pd.DataFrame(results)
        
        # 获取最佳参数组合
        best_run = df_results.loc[df_results['like1'].idxmin()] # 目标函数最小值
        
        # 获取参数值（使用 parx1, parx2 等格式的列名）
        for j, param_name in enumerate(spot_setup.parameter_names):
            param_col = f'parx{j+1}'  # SPOTPY使用的是从1开始的索引
            best_params[basins[i]][param_name] = float(best_run[param_col])
        
        # 保存为JSON文件
        import json
        best_params_file = os.path.join(dbname, "best_params.json")
        with open(best_params_file, "w") as f:
            json.dump(best_params, f, indent=4)
        
        samplers.append(sampler)
    
    return samplers