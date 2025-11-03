"""
Author: Wenyu Ouyang
Date: 2025-07-09 09:27:32
LastEditTime: 2025-07-11 17:38:47
LastEditors: Wenyu Ouyang
Description: 降雨径流相关法(P~Pa~R三变量相关图法)模型实现
FilePath: /hydromodel_dev/hydromodel_dev/ppar.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from scipy.interpolate import interp1d


class PParCorrelationModel:
    """
    降雨径流相关法模型
    支持P~Pa~R和P+Pa~R两种方法，使用矢量化计算
    """

    def __init__(self, model_type: str = "P_Pa_R"):
        """
        初始化模型

        Args:
            model_type: 模型类型，"P_Pa_R"或"P_plus_Pa_R"
        """
        self.model_type = model_type

        # 模型参数
        self.pa_initial: float = 25.0
        self.calculated_type: int = 1
        self.im: float = 100.0
        self.kd: np.ndarray = np.array([0.93] * 12)
        self.time_interval: float = 6.0

        # P~Pa~R模型参数
        self.n: int = 0
        self.pa_values: np.ndarray = np.array([])
        self.m: int = 0
        self.pr_curves: np.ndarray = np.array([])

        # P+Pa~R模型参数
        self.pr_single_curve: np.ndarray = np.array([])

        # 输入数据
        self.flood_times: List[datetime] = []
        self.start_time: Optional[datetime] = None
        self.start_index: int = 0
        self.rainfall: np.ndarray = np.array([])

        # 输出结果
        self.pa_sim: np.ndarray = np.array([])
        self.runoff_sim: np.ndarray = np.array([])

    def set_parameters(self, params: Dict) -> None:
        """设置模型参数"""
        self.pa_initial = params.get("PA0", 25.0)
        self.calculated_type = params.get("CalculatedType", 1)
        self.im = params.get("Im", 100.0)
        self.kd = np.array(params.get("KD", [0.93] * 12))
        self.time_interval = params.get("clen", 6.0)

        if self.model_type == "P_Pa_R":
            self.n = params.get("N", 0)
            self.pa_values = np.array(params.get("Pa", []))
            self.m = params.get("M", 0)
            self.pr_curves = np.array(params.get("PR", []))
        elif self.model_type == "P_plus_Pa_R":
            self.m = params.get("M", 0)
            self.pr_single_curve = np.array(params.get("PR", []))

    def set_input_data(
        self,
        rainfall: List[float],
        times: List[datetime],
        start_time: Optional[datetime] = None,
    ) -> None:
        """设置输入数据"""
        self.rainfall = np.array(rainfall)
        self.flood_times = times
        self.start_time = start_time

        if start_time and start_time in times:
            self.start_index = times.index(start_time)
        else:
            self.start_index = 0

        # 初始化输出数组
        self._initialize_outputs()

    def _initialize_outputs(self) -> None:
        """初始化输出数组"""
        n_steps = len(self.rainfall)
        self.pa_sim = np.zeros(n_steps + 1)
        self.runoff_sim = np.zeros(n_steps)
        self.pa_sim[0] = self.pa_initial

    @staticmethod
    def linear_interpolate(x0: float, x: np.ndarray, y: np.ndarray) -> float:
        """
        线性插值（备用方法）

        Args:
            x0: 插值点
            x: x坐标数组
            y: y坐标数组

        Returns:
            插值结果
        """
        if len(x) <= 1:
            return 0.0

        # 使用scipy更高效，但保留此方法作为备用
        interp_func = interp1d(
            x, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1])
        )
        return float(interp_func(x0))

    def calculate_runoff(
        self, p: np.ndarray, p_curve: np.ndarray, r_curve: np.ndarray
    ) -> np.ndarray:
        """
        矢量化径流计算（使用scipy）

        Args:
            p: 降雨量数组
            p_curve: P曲线
            r_curve: R曲线

        Returns:
            径流量数组
        """
        # 使用scipy的高效插值
        interp_func = interp1d(
            p_curve, r_curve, kind="linear", bounds_error=False, fill_value=0.0
        )

        r_sim = interp_func(p)
        # 矢量化边界处理
        r_sim = np.clip(r_sim, 0.0, p)  # 径流不能超过降雨且不能为负
        return r_sim

    def calculate_runoff_single(
        self, p: float, p_curve: np.ndarray, r_curve: np.ndarray
    ) -> float:
        """单点径流计算"""
        interp_func = interp1d(
            p_curve, r_curve, kind="linear", bounds_error=False, fill_value=0.0
        )
        r_sim = float(interp_func(p))
        return max(0.0, min(r_sim, p))

    def update_pa_vectorized(
        self,
        pa_current: np.ndarray,
        rainfall: np.ndarray,
        times: List[datetime],
    ) -> np.ndarray:
        """
        矢量化Pa更新

        Args:
            pa_current: 当前Pa值数组
            rainfall: 降雨量数组
            times: 时间列表

        Returns:
            更新后的Pa值数组
        """
        # 提取月份信息
        months = np.array([t.month for t in times])

        # 矢量化计算ka
        ka0_values = self.kd[months - 1]
        D = 24.0 / self.time_interval
        ka_values = ka0_values ** (1.0 / D)

        # 矢量化Pa更新
        pa_new = ka_values * (pa_current + rainfall)
        return np.clip(pa_new, 0.0, self.im)

    def _interpolate_pa_curves_vectorized(
        self, p: np.ndarray, paa: float
    ) -> np.ndarray:
        """
        矢量化Pa曲线插值

        Args:
            p: 降雨量数组
            paa: 前期影响雨量

        Returns:
            径流量数组
        """
        if len(self.pa_values) == 0:
            return np.zeros_like(p)

        # 查找Pa插值区间
        if paa <= self.pa_values[0]:
            # 使用第一条曲线
            return self.calculate_runoff(
                p, self.pr_curves[0], self.pr_curves[1]
            )
        elif paa >= self.pa_values[-1]:
            # 使用最后一条曲线
            return self.calculate_runoff(
                p, self.pr_curves[0], self.pr_curves[-1]
            )
        else:
            # 在两条曲线间插值
            idx = np.searchsorted(self.pa_values, paa)
            pa1, pa2 = self.pa_values[idx - 1], self.pa_values[idx]

            r1 = self.calculate_runoff(
                p, self.pr_curves[0], self.pr_curves[idx]
            )
            r2 = self.calculate_runoff(
                p, self.pr_curves[0], self.pr_curves[idx + 1]
            )

            # 线性插值
            weight = (paa - pa1) / (pa2 - pa1)
            return r1 * (1 - weight) + r2 * weight

    def run_p_pa_r_model(self) -> None:
        """运行P~Pa~R模型（矢量化版本）"""
        if self.model_type != "P_Pa_R":
            raise ValueError("当前模型类型不是P_Pa_R")

        n_steps = len(self.rainfall)

        if self.calculated_type != 1:
            # 时段雨量模式 - 逐步计算
            for i in range(n_steps):
                runoff = self._interpolate_pa_curves_vectorized(
                    np.array([self.rainfall[i]]), self.pa_sim[i]
                )[0]
                self.runoff_sim[i] = max(0.0, min(runoff, self.rainfall[i]))

                # 更新Pa
                pa_new = self.update_pa_vectorized(
                    np.array([self.pa_sim[i]]),
                    np.array([self.rainfall[i]]),
                    [self.flood_times[i]],
                )[0]
                self.pa_sim[i + 1] = pa_new
        else:
            # 累积雨量模式 - 可以部分矢量化
            self._run_cumulative_p_pa_r()

    def _run_cumulative_p_pa_r(self) -> None:
        """累积P~Pa~R模型的矢量化实现"""
        if self.start_index == 0:
            # 从开始计算
            paa = self.pa_sim[0]
            cumulative_p = np.cumsum(self.rainfall)

            # 矢量化计算累积径流
            cumulative_r = self._interpolate_pa_curves_vectorized(
                cumulative_p, paa
            )

            # 计算时段径流
            self.runoff_sim[0] = max(
                0.0, min(cumulative_r[0], self.rainfall[0])
            )
            if len(cumulative_r) > 1:
                self.runoff_sim[1:] = np.maximum(
                    0.0, np.minimum(np.diff(cumulative_r), self.rainfall[1:])
                )
        else:
            # 分段计算
            self._run_split_cumulative_p_pa_r()

        # 矢量化更新Pa
        self._update_pa_sequence()

    def _run_split_cumulative_p_pa_r(self) -> None:
        """分段累积计算"""
        # 第一段
        if self.start_index > 0:
            paa1 = self.pa_sim[0]
            cumulative_p1 = np.cumsum(self.rainfall[: self.start_index])
            cumulative_r1 = self._interpolate_pa_curves_vectorized(
                cumulative_p1, paa1
            )

            self.runoff_sim[0] = max(
                0.0, min(cumulative_r1[0], self.rainfall[0])
            )
            if len(cumulative_r1) > 1:
                self.runoff_sim[1 : self.start_index] = np.maximum(
                    0.0,
                    np.minimum(
                        np.diff(cumulative_r1),
                        self.rainfall[1 : self.start_index],
                    ),
                )

        # 第二段
        if self.start_index < len(self.rainfall):
            paa2 = self.pa_sim[self.start_index]
            rainfall_part2 = self.rainfall[self.start_index :]
            cumulative_p2 = np.cumsum(rainfall_part2)
            cumulative_r2 = self._interpolate_pa_curves_vectorized(
                cumulative_p2, paa2
            )

            start_idx = self.start_index
            self.runoff_sim[start_idx] = max(
                0.0, min(cumulative_r2[0], rainfall_part2[0])
            )
            if len(cumulative_r2) > 1:
                self.runoff_sim[start_idx + 1 :] = np.maximum(
                    0.0, np.minimum(np.diff(cumulative_r2), rainfall_part2[1:])
                )

    def run_p_plus_pa_r_model(self) -> None:
        """运行P+Pa~R模型（矢量化版本）"""
        if self.model_type != "P_plus_Pa_R":
            raise ValueError("当前模型类型不是P_plus_Pa_R")

        n_steps = len(self.rainfall)

        if self.calculated_type != 1:
            # 时段雨量模式
            for i in range(n_steps):
                # 计算当前时段径流
                r_base = self.calculate_runoff_single(
                    self.pa_sim[i],
                    self.pr_single_curve[0],
                    self.pr_single_curve[1],
                )

                r_total = self.calculate_runoff_single(
                    self.rainfall[i] + self.pa_sim[i],
                    self.pr_single_curve[0],
                    self.pr_single_curve[1],
                )

                self.runoff_sim[i] = max(
                    0.0, min(r_total - r_base, self.rainfall[i])
                )

                # 更新Pa
                pa_new = self.update_pa_vectorized(
                    np.array([self.pa_sim[i]]),
                    np.array([self.rainfall[i]]),
                    [self.flood_times[i]],
                )[0]
                self.pa_sim[i + 1] = pa_new
        else:
            # 累积雨量模式
            self._run_cumulative_p_plus_pa_r()

    def _run_cumulative_p_plus_pa_r(self) -> None:
        """累积P+Pa~R模型的矢量化实现"""
        if self.start_index == 0:
            # 从开始计算
            pa_start = self.pa_sim[0]
            cumulative_p = np.cumsum(self.rainfall)

            # 矢量化计算
            cumulative_input = cumulative_p + pa_start
            cumulative_r = self.calculate_runoff(
                cumulative_input,
                self.pr_single_curve[0],
                self.pr_single_curve[1],
            )

            # 基础径流
            r_base = self.calculate_runoff_single(
                pa_start, self.pr_single_curve[0], self.pr_single_curve[1]
            )

            # 计算时段径流
            cumulative_r_adjusted = cumulative_r - r_base
            self.runoff_sim[0] = max(
                0.0, min(cumulative_r_adjusted[0], self.rainfall[0])
            )
            if len(cumulative_r_adjusted) > 1:
                self.runoff_sim[1:] = np.maximum(
                    0.0,
                    np.minimum(
                        np.diff(cumulative_r_adjusted), self.rainfall[1:]
                    ),
                )
        else:
            # 分段计算
            self._run_split_cumulative_p_plus_pa_r()

        # 更新Pa序列
        self._update_pa_sequence()

    def _run_split_cumulative_p_plus_pa_r(self) -> None:
        """分段累积P+Pa~R计算"""
        # 第一段
        if self.start_index > 0:
            cumulative_p1 = np.cumsum(self.rainfall[: self.start_index])
            cumulative_input1 = cumulative_p1 + self.pa_initial
            cumulative_r1 = self.calculate_runoff(
                cumulative_input1,
                self.pr_single_curve[0],
                self.pr_single_curve[1],
            )

            r_base1 = self.calculate_runoff_single(
                self.pa_initial,
                self.pr_single_curve[0],
                self.pr_single_curve[1],
            )

            cumulative_r1_adjusted = cumulative_r1 - r_base1
            self.runoff_sim[0] = max(
                0.0, min(cumulative_r1_adjusted[0], self.rainfall[0])
            )
            if len(cumulative_r1_adjusted) > 1:
                self.runoff_sim[1 : self.start_index] = np.maximum(
                    0.0,
                    np.minimum(
                        np.diff(cumulative_r1_adjusted),
                        self.rainfall[1 : self.start_index],
                    ),
                )

        # 第二段
        if self.start_index < len(self.rainfall):
            rainfall_part2 = self.rainfall[self.start_index :]
            cumulative_p2 = np.cumsum(rainfall_part2)
            pa_start2 = self.pa_sim[self.start_index]

            cumulative_input2 = cumulative_p2 + pa_start2
            cumulative_r2 = self.calculate_runoff(
                cumulative_input2,
                self.pr_single_curve[0],
                self.pr_single_curve[1],
            )

            r_base2 = self.calculate_runoff_single(
                pa_start2, self.pr_single_curve[0], self.pr_single_curve[1]
            )

            cumulative_r2_adjusted = cumulative_r2 - r_base2
            start_idx = self.start_index
            self.runoff_sim[start_idx] = max(
                0.0, min(cumulative_r2_adjusted[0], rainfall_part2[0])
            )
            if len(cumulative_r2_adjusted) > 1:
                self.runoff_sim[start_idx + 1 :] = np.maximum(
                    0.0,
                    np.minimum(
                        np.diff(cumulative_r2_adjusted), rainfall_part2[1:]
                    ),
                )

    def _update_pa_sequence(self) -> None:
        """矢量化更新整个Pa序列"""
        n_steps = len(self.rainfall)

        for i in range(n_steps):
            pa_new = self.update_pa_vectorized(
                np.array([self.pa_sim[i]]),
                np.array([self.rainfall[i]]),
                [self.flood_times[i]],
            )[0]
            self.pa_sim[i + 1] = pa_new

    def run_model(self) -> None:
        """运行模型"""
        if self.model_type == "P_Pa_R":
            self.run_p_pa_r_model()
        elif self.model_type == "P_plus_Pa_R":
            self.run_p_plus_pa_r_model()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def get_results(self) -> Dict:
        """获取计算结果"""
        total_rainfall = np.sum(self.rainfall)
        total_runoff = np.sum(self.runoff_sim)

        return {
            "rainfall": self.rainfall,
            "runoff_sim": self.runoff_sim,
            "pa_sim": self.pa_sim[:-1],  # 去掉最后一个多余的Pa值
            "total_rainfall": float(total_rainfall),
            "total_runoff": float(total_runoff),
            "runoff_coefficient": (
                float(total_runoff / total_rainfall)
                if total_rainfall > 0
                else 0.0
            ),
        }

    @property
    def summary_stats(self) -> Dict:
        """获取模型统计信息"""
        return {
            "model_type": self.model_type,
            "n_timesteps": len(self.rainfall),
            "pa_range": (
                float(np.min(self.pa_sim[:-1])),
                float(np.max(self.pa_sim[:-1])),
            ),
            "rainfall_range": (
                float(np.min(self.rainfall)),
                float(np.max(self.rainfall)),
            ),
            "runoff_range": (
                float(np.min(self.runoff_sim)),
                float(np.max(self.runoff_sim)),
            ),
        }


def create_test_data() -> Tuple[List[float], List[datetime]]:
    """创建测试数据"""
    times = [
        datetime(2021, 6, 5, 8) + pd.Timedelta(hours=6 * i) for i in range(10)
    ]
    rainfall = [0.0, 5.17, 1.71, 0.0, 0.0, 0.0, 0.63, 5.17, 11.49, 0.24]
    return rainfall, times


def example_usage() -> None:
    """示例用法"""
    print("=== 降雨径流相关法模型示例 (scipy优化版本) ===")

    # 创建测试数据
    rainfall, times = create_test_data()

    # P~Pa~R模型示例
    print("\n--- P~Pa~R模型 ---")
    model1 = PParCorrelationModel("P_Pa_R")

    # 设置参数
    params1 = {
        "PA0": 25.4,
        "CalculatedType": 1,
        "Im": 100.0,
        "KD": [0.93] * 12,
        "clen": 6.0,
        "N": 2,
        "Pa": [0, 50],
        "M": 5,
        "PR": [
            [0, 10, 20, 30, 40],  # P值
            [0, 0.5, 1.0, 1.5, 2.1],  # Pa=0时的R值
            [0, 0.7, 1.3, 1.9, 2.6],  # Pa=50时的R值
        ],
    }

    model1.set_parameters(params1)
    model1.set_input_data(rainfall, times)
    model1.run_model()
    results1 = model1.get_results()

    print(f"总降雨: {results1['total_rainfall']:.1f} mm")
    print(f"总径流: {results1['total_runoff']:.1f} mm")
    print(f"径流系数: {results1['runoff_coefficient']:.3f}")
    print(f"模型统计: {model1.summary_stats}")

    # P+Pa~R模型示例
    print("\n--- P+Pa~R模型 ---")
    model2 = PParCorrelationModel("P_plus_Pa_R")

    # 设置参数
    params2 = {
        "PA0": 25.4,
        "CalculatedType": 1,
        "Im": 100.0,
        "KD": [0.93] * 12,
        "clen": 6.0,
        "M": 5,
        "PR": [[0, 20, 40, 60, 80], [0, 1.0, 2.5, 4.5, 7.0]],  # P+Pa值  # R值
    }

    model2.set_parameters(params2)
    model2.set_input_data(rainfall, times)
    model2.run_model()
    results2 = model2.get_results()

    print(f"总降雨: {results2['total_rainfall']:.1f} mm")
    print(f"总径流: {results2['total_runoff']:.1f} mm")
    print(f"径流系数: {results2['runoff_coefficient']:.3f}")
    print(f"模型统计: {model2.summary_stats}")


if __name__ == "__main__":
    example_usage()
