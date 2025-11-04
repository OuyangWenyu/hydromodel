"""
Author: Wenyu Ouyang
Date: 2021-12-10 23:01:02
LastEditTime: 2024-08-15 16:38:17
LastEditors: Wenyu Ouyang
Description: Calibrate XAJ model using SCE-UA
FilePath: /hydromodel/hydromodel/trainers/calibrate_ga.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
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
        self,
        p_and_e,
        qobs,
        warmup_length=365,
        model=None,
        param_file=None,
        loss=None,
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
            models need warmup period
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
            Uniform(par_name, low=0.0, high=1.0)
            for par_name in self.parameter_names
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
            **self.param_range,
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
                    time.iloc[i, 0]
                    - calibrate_starttime
                    - pd.Timedelta(hours=365)
                ) / pd.Timedelta(hours=1)
                end_num = (
                    time.iloc[i, 1]
                    - calibrate_starttime
                    - pd.Timedelta(hours=365)
                ) / pd.Timedelta(hours=1)
                start_num = int(start_num)
                end_num = int(end_num)
                like_ = LOSS_DICT[self.loss["obj_func"]](
                    evaluation[start_num:end_num,],
                    simulation[start_num:end_num,],
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
        best_params = {basins[i]: {}}
        # 打印模型参数信息
        print(f"模型名称: {model['name']}")
        # print(f"参数名称列表: {spot_setup.parameter_names}")
        # print(f"参数数量: {len(spot_setup.parameter_names)}")
        # 获取数据并转换为DataFrame
        results = sampler.getdata()
        df_results = pd.DataFrame(results)

        # 调试：打印DataFrame的列名
        # print(f"📊 SPOTPY返回的数据列名: {list(df_results.columns)}")
        print(f"🔢 参数名称: {spot_setup.parameter_names}")

        # 获取最佳参数组合
        best_run = df_results.loc[
            df_results["like1"].idxmin()
        ]  # 目标函数最小值

        # 获取参数值 - 智能检测列名格式
        param_columns = []

        # 方法1: 尝试使用 parx1, parx2 格式
        for j in range(len(spot_setup.parameter_names)):
            param_col = f"parx{j+1}"
            if param_col in df_results.columns:
                param_columns.append(param_col)

        # 方法2: 如果方法1失败，尝试使用 par{param_name} 格式
        if len(param_columns) != len(spot_setup.parameter_names):
            param_columns = []
            for param_name in spot_setup.parameter_names:
                param_col = f"par{param_name}"
                if param_col in df_results.columns:
                    param_columns.append(param_col)

        # 方法3: 如果前面都失败，尝试直接使用参数名
        if len(param_columns) != len(spot_setup.parameter_names):
            param_columns = []
            for param_name in spot_setup.parameter_names:
                if param_name in df_results.columns:
                    param_columns.append(param_name)

        # 方法4: 如果还是失败，使用数字索引查找包含参数相关的列
        if len(param_columns) != len(spot_setup.parameter_names):
            param_columns = []
            # 查找所有以'par'开头的列
            par_cols = [
                col for col in df_results.columns if str(col).startswith("par")
            ]
            if len(par_cols) >= len(spot_setup.parameter_names):
                param_columns = sorted(par_cols)[
                    : len(spot_setup.parameter_names)
                ]

        print(f"🎯 检测到的参数列: {param_columns}")

        # 验证参数列数量
        if len(param_columns) != len(spot_setup.parameter_names):
            print(
                f"❌ 错误：参数列数量({len(param_columns)})与参数名称数量({len(spot_setup.parameter_names)})不匹配"
            )
            print(f"   可用列名: {list(df_results.columns)}")
            # 使用前N列作为参数（排除目标函数列）
            exclude_cols = [
                "like1",
                "chain",
                "simulation",
                "chain1",
            ]  # 常见的非参数列
            available_cols = [
                col for col in df_results.columns if col not in exclude_cols
            ]

            # 进一步过滤：只保留数值型列
            numeric_cols = []
            for col in available_cols:
                try:
                    pd.to_numeric(df_results[col])
                    numeric_cols.append(col)
                except:
                    continue

            param_columns = numeric_cols[: len(spot_setup.parameter_names)]
            print(f"   使用备用方案（数值型列）：{param_columns}")

            # 最后的备用方案：如果还是不够，使用前几列
            if len(param_columns) < len(spot_setup.parameter_names):
                all_cols = list(df_results.columns)
                param_columns = all_cols[: len(spot_setup.parameter_names)]
                print(f"   使用最终备用方案（前几列）：{param_columns}")

        # 获取参数值
        for j, param_name in enumerate(spot_setup.parameter_names):
            if j < len(param_columns):
                param_col = param_columns[j]
                try:
                    best_params[basins[i]][param_name] = float(
                        best_run[param_col]
                    )
                    print(
                        f"   ✅ {param_name} = {best_run[param_col]} (来自列: {param_col})"
                    )
                except Exception as e:
                    print(f"   ❌ 获取参数 {param_name} 失败: {e}")
                    best_params[basins[i]][param_name] = 0.0  # 设置默认值
            else:
                print(f"   ⚠️  参数 {param_name} 没有对应的列，使用默认值")
                best_params[basins[i]][param_name] = 0.0

        # 保存为JSON文件
        import json

        best_params_file = os.path.join(dbname, "best_params.json")
        with open(best_params_file, "w") as f:
            json.dump(best_params, f, indent=4)

        samplers.append(sampler)

    return samplers
