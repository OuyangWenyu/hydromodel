'''
Author: zhuanglaihong
Date: 2025-03-26 18:26:28
LastEditTime: 2025-03-27 09:48:04
LastEditors: zhuanglaihong
Description: GR series hydrological models (GR4J, GR5J, GR6J)
FilePath: /zlh/hydromodel/hydromodel/models/gr_model.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
'''

import math
from typing import Optional, Tuple, Dict, List
import numpy as np
from numba import jit

from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.models.xaj import uh_conv


class GRModel:
    """GR系列水文模型（GR4J/GR5J/GR6J）的统一实现"""

    def __init__(self, model_type: str = "gr4j"):
        """
        初始化GR模型

        Parameters
        ----------
        model_type : str
            model type:'gr4j', 'gr5j', or 'gr6j'
        """
        self.model_type = model_type.lower()
        if self.model_type not in ["gr4j", "gr5j", "gr6j"]:
            raise ValueError("模型类型必须是 'gr4j', 'gr5j' 或 'gr6j'")

        self.param_ranges = MODEL_PARAM_DICT[self.model_type]["param_range"]
        self.param_names = list(self.param_ranges.keys())
        self.num_params = len(self.param_names)

    @staticmethod
    @jit(nopython=True)
    def calculate_precip_store(
        s: np.ndarray, precip_net: np.ndarray, x1: np.ndarray
    ) -> np.ndarray:
        """计算进入产流水库的降水量"""
        n = x1 * (1.0 - (s / x1) ** 2) * np.tanh(precip_net / x1)
        d = 1.0 + (s / x1) * np.tanh(precip_net / x1)
        return n / d

    @staticmethod
    @jit(nopython=True)
    def calculate_evap_store(
        s: np.ndarray, evap_net: np.ndarray, x1: np.ndarray
    ) -> np.ndarray:
        """计算产流水库的蒸发损失"""
        n = s * (2.0 - s / x1) * np.tanh(evap_net / x1)
        d = 1.0 + (1.0 - s / x1) * np.tanh(evap_net / x1)
        return n / d

    @staticmethod
    @jit(nopython=True)
    def calculate_perc(current_store: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """计算产流水库的渗流量"""
        return current_store * (
            1.0 - (1.0 + (4.0 / 9.0 * current_store / x1) ** 4) ** -0.25
        )

    def production(
        self, p_and_e: np.ndarray, x1: np.ndarray, s_level: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """产流计算"""
        if s_level is None:
            s_level = 0.6 * x1

        precip_difference = p_and_e[:, 0] - p_and_e[:, 1]
        precip_net = np.maximum(precip_difference, 0.0)
        evap_net = np.maximum(-precip_difference, 0.0)

        s_level = np.clip(s_level, a_min=np.full(s_level.shape, 0.0), a_max=x1)

        precip_store = self.calculate_precip_store(s_level, precip_net, x1)
        evap_store = self.calculate_evap_store(s_level, evap_net, x1)

        s_update = s_level - evap_store + precip_store
        s_update = np.clip(s_update, a_min=np.full(s_update.shape, 0.0), a_max=x1)

        perc = self.calculate_perc(s_update, x1)
        s_update = s_update - perc

        current_runoff = perc + (precip_net - precip_store)
        return current_runoff, evap_store, s_update

    @staticmethod
    @jit(nopython=True)
    def s_curves1(t: float, x4: float) -> float:
        """UH1的S曲线"""
        if t <= 0:
            return 0
        elif t < x4:
            return (t / x4) ** 2.5
        else:
            return 1

    @staticmethod
    @jit(nopython=True)
    def s_curves2(t: float, x4: float) -> float:
        """UH2的S曲线"""
        if t <= 0:
            return 0
        elif t < x4:
            return 0.5 * (t / x4) ** 2.5
        elif t < 2 * x4:
            return 1 - 0.5 * (2 - t / x4) ** 2.5
        else:
            return 1

    def uh_gr4j(self, x4: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """生成GR4J汇流模块的卷积核"""
        uh1_ordinates = []
        uh2_ordinates = []

        for i in range(len(x4)):
            n_uh1 = int(math.ceil(x4[i]))
            n_uh2 = int(math.ceil(2.0 * x4[i]))

            uh1_ordinate = np.zeros(n_uh1)
            uh2_ordinate = np.zeros(n_uh2)

            for t in range(1, n_uh1 + 1):
                uh1_ordinate[t - 1] = self.s_curves1(t, x4[i]) - self.s_curves1(
                    t - 1, x4[i]
                )

            for t in range(1, n_uh2 + 1):
                uh2_ordinate[t - 1] = self.s_curves2(t, x4[i]) - self.s_curves2(
                    t - 1, x4[i]
                )

            uh1_ordinates.append(uh1_ordinate)
            uh2_ordinates.append(uh2_ordinate)

        return uh1_ordinates, uh2_ordinates

    def routing_gr4j(
        self,
        q9: np.ndarray,
        q1: np.ndarray,
        x2: np.ndarray,
        x3: np.ndarray,
        r_level: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """gr4j汇流计算"""
        if r_level is None:
            r_level = 0.7 * x3

        r_level = np.clip(r_level, a_min=np.full(r_level.shape, 0.0), a_max=x3)
        groundwater_ex = x2 * (r_level / x3) ** 3.5
        r_updated = np.maximum(
            np.full(r_level.shape, 0.0), r_level + q9 + groundwater_ex
        )

        qr = r_updated * (1.0 - (1.0 + (r_updated / x3) ** 4) ** -0.25)
        r_updated = r_updated - qr

        qd = np.maximum(np.full(groundwater_ex.shape, 0.0), q1 + groundwater_ex)
        q = qr + qd
        return q, r_updated

    def routing_gr5j(
        self,
        q9: np.ndarray,
        q1: np.ndarray,
        x2: np.ndarray,
        x3: np.ndarray,
        x5: np.ndarray,
        r_level: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GR5J汇流计算,增加了交换参数x5"""
        if r_level is None:
            r_level = 0.7 * x3

        r_level = np.clip(r_level, a_min=np.full(r_level.shape, 0.0), a_max=x3)

        # 计算地下水交换量，GR5J中使用x5参数
        groundwater_ex = x2 * r_level / x3 - x2 * x5

        r_updated = np.maximum(
            np.full(r_level.shape, 0.0),
            r_level + q9 + groundwater_ex,
        )

        qr = r_updated * (1.0 - (1.0 + (r_updated / x3) ** 4) ** -0.25)
        r_updated = r_updated - qr

        qd = np.maximum(np.full(groundwater_ex.shape, 0.0), q1 + groundwater_ex)
        q = qr + qd
        return q, r_updated

    def routing_gr6j(
        self,
        q9: np.ndarray,
        q1: np.ndarray,
        x2: np.ndarray,
        x3: np.ndarray,
        x5: np.ndarray,
        x6: np.ndarray,
        r_level: Optional[np.ndarray] = None,
        n_level: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """GR6J汇流计算,增加了交换参数x5和指数型水库参数x6"""
        if r_level is None:
            r_level = 0.7 * x3
        if n_level is None:
            n_level = 0.3 * x3

        r_level = np.clip(r_level, a_min=np.full(r_level.shape, 0.0), a_max=x3)

        groundwater_ex = x2 * r_level / x3 - x2 * x5

        SC = 0.4
        r_updated = np.maximum(
            np.full(r_level.shape, 0.0),
            r_level + q9 * (1 - SC) + groundwater_ex,
        )

        qr = r_updated * (1.0 - (1.0 + (r_updated / x3) ** 4) ** -0.25)
        r_updated = r_updated - qr

        # GR6J特有：指数型水库计算
        n_updated = np.clip(n_level, a_min=np.full(n_level.shape, 0.0), a_max=x3)
        n_updated = q9 * SC + n_updated
        qn = x6 * np.log(1 + np.exp(n_updated / x6))
        n_updated = n_updated - qn

        qd = np.maximum(np.full(groundwater_ex.shape, 0.0), q1 + groundwater_ex)
        q = qd + qr + qn

        return q, r_updated, n_updated

    def run(
        self,
        p_and_e: np.ndarray,
        parameters: np.ndarray,
        warmup_length: int = 0,
        return_state: bool = False,
        **kwargs
    ) -> Tuple:
        """
        Runs the GR model.

        Parameters
        ----------
        p_and_e : np.ndarray
            Input data (time, basin, variable).
        parameters : np.ndarray
            Model parameters (basin, parameter).
        warmup_length : int, optional
            Warmup period length, by default 0.
        return_state : bool, optional
            Whether to return state variables, by default False.

        Returns
        -------
        Tuple
            Simulation results.
        """
        # 参数映射
        model_params = {}
        for i, param_name in enumerate(self.param_names):
            param_range = self.param_ranges[param_name]
            model_params[param_name] = param_range[0] + parameters[:, i] * (
                param_range[1] - param_range[0]
            )

        # 预热期处理
        if warmup_length > 0:
            p_and_e_warmup = p_and_e[0:warmup_length, :, :]
            if self.model_type == "gr6j":
                _, _, s0, r0, n0 = self.run(
                    p_and_e_warmup, parameters, warmup_length=0, return_state=True
                )
                n0 = 0.3 * model_params["x3"]  # 使用x3参数的一部分作为n0的初始值
            else:
                _, _, s0, r0 = self.run(
                    p_and_e_warmup, parameters, warmup_length=0, return_state=True
                )
        else:
            s0 = 0.5 * model_params["x1"]
            r0 = 0.5 * model_params["x3"]
            if self.model_type == "gr6j":
                n0 = 0.3 * model_params["x3"]

        # 模型计算
        inputs = p_and_e[warmup_length:, :, :]
        streamflow_ = np.full(inputs.shape[:2], 0.0)
        prs = np.full(inputs.shape[:2], 0.0)
        ets = np.full(inputs.shape[:2], 0.0)

        # 产流计算
        for i in range(inputs.shape[0]):
            if i == 0:
                pr, et, s = self.production(inputs[i, :, :], model_params["x1"], s0)
            else:
                pr, et, s = self.production(inputs[i, :, :], model_params["x1"], s)
            prs[i, :] = pr
            ets[i, :] = et

        # 汇流计算
        prs_x = np.expand_dims(prs, axis=2)
        conv_q9, conv_q1 = self.uh_gr4j(model_params["x4"])
        q9 = np.full([inputs.shape[0], inputs.shape[1], 1], 0.0)
        q1 = np.full([inputs.shape[0], inputs.shape[1], 1], 0.0)

        for j in range(inputs.shape[1]):
            q9[:, j : j + 1, :] = uh_conv(
                prs_x[:, j : j + 1, :], conv_q9[j].reshape(-1, 1, 1)
            )
            q1[:, j : j + 1, :] = uh_conv(
                prs_x[:, j : j + 1, :], conv_q1[j].reshape(-1, 1, 1)
            )

        if self.model_type == "gr6j":
            n = n0

        for i in range(inputs.shape[0]):
            if self.model_type == "gr6j":
                q, r, n = (
                    self.routing_gr6j(
                        q9[i, :, 0],
                        q1[i, :, 0],
                        model_params["x2"],
                        model_params["x3"],
                        model_params["x5"],
                        model_params["x6"],
                        r0,
                        n0,
                    )
                    if i == 0
                    else self.routing_gr6j(
                        q9[i, :, 0],
                        q1[i, :, 0],
                        model_params["x2"],
                        model_params["x3"],
                        model_params["x5"],
                        model_params["x6"],
                        r,
                        n,
                    )
                )
            elif self.model_type == "gr5j":
                if i == 0:
                    q, r = self.routing_gr5j(
                        q9[i, :, 0],
                        q1[i, :, 0],
                        model_params["x2"],
                        model_params["x3"],
                        model_params["x5"],
                        r0,
                    )
                else:
                    q, r = self.routing_gr5j(
                        q9[i, :, 0],
                        q1[i, :, 0],
                        model_params["x2"],
                        model_params["x3"],
                        model_params["x5"],
                        r,
                    )
            elif i == 0:
                q, r = self.routing_gr4j(
                    q9[i, :, 0], q1[i, :, 0], model_params["x2"], model_params["x3"], r0
                )
            else:
                q, r = self.routing_gr4j(
                    q9[i, :, 0], q1[i, :, 0], model_params["x2"], model_params["x3"], r
                )
            streamflow_[i, :] = q

        streamflow = np.expand_dims(streamflow_, axis=2)
        if self.model_type == "gr6j" and return_state:
            return (streamflow, ets, s, r, n)
        else:
            return (streamflow, ets, s, r) if return_state else (streamflow, ets)
