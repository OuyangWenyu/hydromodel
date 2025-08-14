"""
Author: Wenyu Ouyang
Date: 2025-07-30 16:44:15
LastEditTime: 2025-08-14 11:02:50
LastEditors: Wenyu Ouyang
Description: Dahuofang Model - Python implementation based on Java version
FilePath: \hydromodel\hydromodel\models\dhf.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback


def run_dhf_single_basin(
    precipitation: np.ndarray,
    potential_evapotranspiration: np.ndarray,
    parameters: np.ndarray,
    warmup_length: int = 30,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    DHF模型运行函数，用于单个流域的径流计算

    Parameters
    ----------
    precipitation : np.ndarray
        降雨数据
    potential_evapotranspiration : np.ndarray
        潜在蒸散发数据
    parameters : np.ndarray
        模型参数数组 [S0, U0, D0, KC, KW, K2, KA, G, A, B, B0, K0, N, L, DD, CC, COE, DDL, CCL]
    warmup_length : int, optional
        预热期长度，默认30
    **kwargs
        Additional keyword arguments, including time_interval_hours (default: 1.0)

    Returns
    -------
    Dict[str, np.ndarray]
        包含模拟结果和状态变量的字典

    Notes
    -----
    输入输出单位已统一，不需要流域面积转换
    """
    # 提取参数（只包含模型参数，不包含状态变量）
    s0, u0, d0, kc, kw, k2, ka = parameters[0:7]
    g, a, b, b0, k0, n, l = parameters[7:14]
    dd, cc, coe, ddl, ccl = parameters[14:19]

    # 预热期处理 - 递归调用获取合适的初始状态
    if warmup_length > 0:
        warmup_precipitation = precipitation[:warmup_length]
        warmup_pet = potential_evapotranspiration[:warmup_length]

        # 递归调用预热期（无预热期）
        warmup_results = run_dhf_single_basin(
            warmup_precipitation,
            warmup_pet,
            parameters,
            warmup_length=0,
            **kwargs,
        )

        # 从预热结果获取最终状态作为正式计算的初始状态
        sa0 = warmup_results["sa"][-1]
        ua0 = warmup_results["ua"][-1]
        ya0 = warmup_results["ya"][-1]

        # 使用预热期后的数据进行正式计算
        precipitation = precipitation[warmup_length:]
        potential_evapotranspiration = potential_evapotranspiration[
            warmup_length:
        ]
    else:
        # 使用默认初始状态 (from Jinggang Chu)
        sa0 = 0.0
        ua0 = 0.0
        ya0 = 0.5

    time_steps = len(precipitation)

    # 模型常量
    time_interval = kwargs.get("time_interval_hours", 3.0)
    PAI = np.pi

    # 初始化状态变量
    sa = np.zeros(time_steps + 1)
    ua = np.zeros(time_steps + 1)
    ya = np.zeros(time_steps + 1)

    # 初始化产流变量
    RunoffSim = np.zeros(time_steps)
    QSim = np.zeros(time_steps)
    y0 = np.zeros(time_steps)
    PE = np.zeros(time_steps)
    yu = np.zeros(time_steps)
    yL = np.zeros(time_steps)
    y = np.zeros(time_steps)
    rL = np.zeros(time_steps)
    qs = np.zeros(time_steps)
    ql = np.zeros(time_steps)
    ET = np.zeros(time_steps)

    # 设置初始状态
    sa[0] = sa0
    ua[0] = ua0
    ya[0] = ya0

    # 限制初始状态
    if sa[0] > s0:
        sa[0] = s0
    if ua[0] > u0:
        ua[0] = u0

    # DHF产流计算
    for i in range(time_steps):
        # 限制当前状态
        if sa[i] > s0:
            sa[i] = s0
        if ua[i] > u0:
            ua[i] = u0

        # 初始化当前时段变量
        yu[i] = 0.0
        yL[i] = 0.0
        Eb = 0.0

        # 计算蒸散发
        if (
            potential_evapotranspiration is not None
            and len(potential_evapotranspiration) > i
        ):
            # 如果提供了蒸发数据，直接使用
            # print(f"使用提供的蒸发数据: {evapotranspiration[i]}")
            Ep = potential_evapotranspiration[i]
            EDt = kc * Ep

        else:
            raise ValueError("Potential evapotranspiration is required")

        PE[i] = precipitation[i] - EDt  # 净雨
        y0[i] = g * PE[i]  # 不透水面积产流
        Pc = PE[i] - y0[i]  # 净渗雨强
        ET[i] = EDt

        if Pc > 0.0:
            # 计算表层蓄水
            temp = (1 - sa[i] / s0) ** (1 / a)
            Sm = a * s0 * (1 - temp)

            if Sm + Pc < a * s0:
                temp = (1 - (Sm + Pc) / (a * s0)) ** a
                rr = Pc + sa[i] - s0 + s0 * temp
            else:
                rr = Pc - (s0 - sa[i])

            # 下层流计算
            temp = (1 - ua[i] / u0) ** (1 / b)
            un = b * u0 * (1 - temp)
            temp = (1 - ua[i] / u0) ** (u0 / (b * d0))
            dn = b * d0 * (1 - temp)

            Z1 = 1 - np.exp(-k2 * time_interval * u0 / d0)
            Z2 = 1 - np.exp(-k2 * time_interval)

            # 总流量计算
            if rr + Z2 * un < Z2 * b * u0:
                temp = (1 - (Z2 * un + rr) / (Z2 * b * u0)) ** b
                y[i] = rr + Z2 * (ua[i] - u0) + Z2 * u0 * temp
            else:
                y[i] = rr + Z2 * (ua[i] - u0)

            # 地面壤中流
            temp = (1 - ua[i] / u0) ** (u0 / d0)
            if Z1 * dn + rr < Z1 * b * d0:
                temp1 = 1 - (Z1 * dn + rr) / (Z1 * b * d0)
                temp2 = temp1**b
                yu[i] = rr - Z1 * d0 * temp + Z1 * d0 * temp2
            else:
                yu[i] = rr - Z1 * d0 * temp

            # 地下径流
            yL[i] = (y[i] - yu[i]) * kw

            # 更新状态变量
            if Sm + Pc < a * s0:
                temp1 = 1 - (Sm + Pc) / (a * s0)
                temp2 = temp1**a
                sa[i + 1] = s0 * (1 - temp2)
            else:
                sa[i + 1] = sa[i] + Pc - rr

            if sa[i + 1] > s0:
                sa[i + 1] = s0

            ua[i + 1] = ua[i] + rr - y[i]
            if ua[i + 1] > u0:
                ua[i + 1] = u0

            Eb = 0.0

        else:
            rr = 0.0
            Ec = EDt - precipitation[i]
            Eb = Eb + Ec

            # 表层可蒸发量Eu计算
            temp1 = (1 - (Eb - Ec) / (a * s0)) ** a
            temp2 = (1 - Eb / (a * s0)) ** a

            if (Eb / (a * s0) <= 0.999999) and (
                (Eb - Ec) / (a * s0) <= 0.999999
            ):
                Eu = s0 * (temp1 - temp2)
            elif (Eb / (a * s0) >= 1.00001) and (
                (Eb - Ec) / (a * s0) <= 0.999999
            ):
                Eu = s0 * temp1
            else:
                Eu = 0.00001

            if sa[i] - Eu < 0.0:
                EL = (Ec - sa[i]) * ua[i] / u0
                sa[i + 1] = 0.0
                ua[i + 1] = ua[i] - EL
                if ua[i + 1] < 0.0:
                    ua[i + 1] = 0.0
            else:
                EL = (Ec - Eu) * ua[i] / u0
                sa[i + 1] = sa[i] - Eu
                ua[i + 1] = ua[i] - EL
                if ua[i + 1] < 0.0:
                    ua[i + 1] = 0.0

            y[i] = 0.0
            y0[i] = 0.0
            yu[i] = 0.0
            yL[i] = 0.0

        # 确保边界条件
        if sa[i + 1] > s0:
            sa[i + 1] = s0
        if sa[i + 1] < 0:
            sa[i + 1] = 0
        if ua[i + 1] < 0:
            ua[i + 1] = 0
        if ua[i + 1] > u0:
            ua[i + 1] = u0

    # 计算总产流并更新ya
    for j in range(time_steps):
        RunoffSim[j] = y[j] + y0[j]
        if RunoffSim[j] < 0.0:
            RunoffSim[j] = 0.0
        rL[j] = yL[j]
        if rL[j] < 0.0:
            rL[j] = 0.0

        ya[j + 1] = (ya[j] + RunoffSim[j]) * ka
        if ya[j + 1] < 0.0:
            ya[j + 1] = 0.0

    # DHF汇流计算
    # 注意: 流域面积转换已在数据预处理中完成，这里使用标准化因子
    w0 = 1.0 / time_interval

    for i in range(time_steps):
        if ya[i] < 0.5:
            ya[i] = 0.5

        tempTm = (ya[i] + RunoffSim[i]) ** (-k0)
        LB = l / b0
        Tm = LB * tempTm  # 保持为浮点数

        TT = int(n * Tm)
        TS = int(coe * Tm)

        # 地下汇流参数计算
        if TT > 0:
            tempAAL = (PAI * coe / n) ** (ddl - 1)
            AAL = ccl / (ddl * tempAAL * np.tan(PAI * coe / n))
            K3L = 0.0
            for j in range(TT):
                tmp = (PAI * j / TT) ** ddl
                tmp1 = (np.sin(PAI * j / TT)) ** ccl
                K3L += np.exp(-AAL * tmp) * tmp1

            # 按照Java版本，没有除零检查，直接计算
            if K3L != 0:
                K3L = TT * w0 / K3L
            else:
                K3L = 0.0
        else:
            K3L = 0.0
            AAL = 0.0

        # 地表汇流参数计算
        K3 = 0.0  # 初始化K3
        if Tm > 0:
            tempAA = (PAI * coe) ** (dd - 1)
            AA = cc / (dd * tempAA * np.tan(PAI * coe))

            # Java版本：for (int j = 0; j < Tm; j++)
            j = 0
            while j < Tm:
                tmp = (PAI * j / Tm) ** dd
                tmp1 = (np.sin(PAI * j / Tm)) ** cc
                K3_old = K3  # 保存旧值用于跟踪
                K3 += np.exp(-AA * tmp) * tmp1
                j += 1  # 增加计数器

            # 按照Java版本，没有除零检查，直接计算
            if K3 != 0:
                K3_before = K3
                K3 = Tm * w0 / K3

            else:
                K3 = 0.0

        else:
            AA = 0.0

        TL = TT + TS - 1
        if TL <= 0:
            TL = 0

        # 汇流计算
        for j in range(TL):
            if i + j >= time_steps:  # 防止越界
                break

            # 地表汇流计算
            if Tm > 0:
                temp0 = PAI * j / Tm
                temp1 = temp0**dd
                temp2 = np.exp(-AA * temp1)
                temp3 = (np.sin(temp0)) ** cc
                Qs = (RunoffSim[i] - rL[i]) * K3 / Tm * temp2 * temp3
            else:
                Qs = 0.0

            if np.isnan(Qs):
                Qs = 0

            # 地下汇流计算
            if TT > 0:
                temp00 = PAI * (j - TS) / TT
                temp10 = temp00**ddl
                temp20 = np.exp(-AAL * temp10)
                temp30 = (np.sin(temp00)) ** ccl
                Ql = rL[i] * K3L / TT * temp20 * temp30
            else:
                Ql = 0.0

            # 按照Java版本的精确条件判断
            if j <= int(Tm):
                if j <= TS:
                    ql[i + j] += 0.0
                    qs[i + j] += Qs
                else:
                    qs[i + j] += Qs
                    ql[i + j] += Ql
            else:
                qs[i + j] += 0.0
                ql[i + j] += Ql

            QSim[i + j] = qs[i + j] + ql[i + j]
            if QSim[i + j] < 0.0:
                QSim[i + j] = 0.0

    # 状态变量处理（移除最后一个时步的状态，因为它对应时间步长+1）
    sa = sa[:-1]
    ua = ua[:-1]
    ya = ya[:-1]

    # 构建分水源产流数组
    r_sim = np.array([y0, yu, yL, y])

    return {
        "QSim": QSim,
        "runoffSim": RunoffSim,
        "RSim": r_sim,
        "y0": y0,
        "yu": yu,
        "yl": yL,
        "y": y,
        "pe": PE,
        "sa": sa,
        "ua": ua,
        "ya": ya,
    }


def dhf(
    p_and_e: np.ndarray,
    parameters: np.ndarray,
    warmup_length: int = 365,
    return_state: bool = False,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """
    大伙房水文模型（DHF Model）- 按照dhf.py逻辑重构版本

    Parameters
    ----------
    p_and_e : np.ndarray
        precipitation and potential evapotranspiration, 3-dim variable: [time, basin, feature=2]
        where feature=0 is precipitation, feature=1 is potential evapotranspiration
    parameters : np.ndarray
        model parameters, 2-dim variable: [basin, parameter]
        Parameters expected in order: [S0, U0, D0, K, KW, K2, KA, G, A, B, B0, K0, N, L, DD, CC, COE, DDL, CCL, SA0, UA0, YA0]
    warmup_length : int, optional
        the length of warmup period (default: 365)
    return_state : bool, optional
        if True, return internal state variables, else only return streamflow (default: False)
    **kwargs
        Additional keyword arguments, including time_interval_hours (default: 1.0)

    Returns
    -------
    result : np.ndarray or tuple
        if return_state is False: QSim array [time, basin]
        if return_state is True: tuple of (QSim, complete_results_dict)
    """

    # 获取数据维度
    time_steps, num_basins, _ = p_and_e.shape

    # 提取参数
    time_interval_hours = kwargs.get("time_interval_hours", 3.0)

    # 处理每个流域
    all_results = {}
    actual_output_length = None

    for basin_idx in range(num_basins):
        # 提取当前流域的参数和数据
        params = parameters[basin_idx, :]
        precipitation = p_and_e[:, basin_idx, 0]
        potential_evapotranspiration = p_and_e[:, basin_idx, 1]

        # 运行单个流域模型（预热期处理在run_dhf_single_basin内部完成）
        basin_results = run_dhf_single_basin(
            precipitation,
            potential_evapotranspiration,
            params,
            warmup_length=warmup_length,
            time_interval_hours=time_interval_hours,
        )

        # 获取实际输出长度
        if basin_idx == 0:
            actual_output_length = len(basin_results["QSim"])
            # 初始化结果数组
            for key, value in basin_results.items():
                if key == "RSim":
                    all_results["RSim"] = np.zeros(
                        (4, actual_output_length, num_basins)
                    )
                else:
                    all_results[key] = np.zeros(
                        (actual_output_length, num_basins)
                    )

        # 存储结果
        for key, value in basin_results.items():
            if key == "RSim":
                all_results["RSim"][:, :, basin_idx] = value
            else:
                all_results[key][:, basin_idx] = value

    if return_state:
        return all_results["QSim"], all_results
    else:
        return all_results["QSim"]


def load_dhf_data_from_json(
    json_file_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从JSON文件加载DHF模型所需的时序数据和参数

    Parameters
    ----------
    json_file_path : str
        JSON文件路径，包含时间序列、降雨数据和模型参数

    Returns
    -------
    p_and_e : np.ndarray
        降雨和蒸发数据 [time, basin=1, feature=2]
        feature=1（蒸发）将在模型中根据月蒸发量计算
    parameters : np.ndarray
        模型参数 [basin=1, parameter=22]
    """
    # 读取JSON文件
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 解析时间序列和降雨数据
    dt = json.loads(data["dt"])  # 时间序列
    rain = json.loads(data["rain"])  # 降雨数据

    # 构建p_and_e数组 [time, basin=1, feature=2]
    # feature=1（蒸发）将在模型中根据月蒸发量计算
    time_steps = len(rain)
    p_and_e = np.zeros((time_steps, 1, 2))
    p_and_e[:, 0, 0] = rain  # 降雨数据
    p_and_e[:, 0, 1] = 0.0  # 蒸发数据占位（将在模型中计算）

    # 构建参数数组 [basin=1, parameter=22]
    # 参数顺序: [S0, U0, D0, K, KW, K2, KA, G, A, B, B0, K0, N, L, DD, CC, COE, DDL, CCL, SA0, UA0, YA0]
    parameters = np.array(
        [
            [
                float(data["S0"]),
                float(data["U0"]),
                float(data["D0"]),
                float(data["K"]),
                float(data["KW"]),
                float(data["K2"]),
                float(data["KA"]),
                float(data["G"]),
                float(data["A"]),
                float(data["B"]),
                float(data["B0"]),
                float(data["K0"]),
                float(data["N"]),
                float(data["L"]),
                float(data["DD"]),
                float(data["CC"]),
                float(data["COE"]),
                float(data["DDL"]),
                float(data["CCL"]),
                float(data["SA0"]),
                float(data["UA0"]),
                float(data["YA0"]),
            ]
        ]
    )

    return p_and_e, parameters


def load_dhf_data_from_csv_and_json(
    csv_file_path: str, json_params_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从CSV文件加载降雨和蒸发数据，从JSON文件加载参数并转换为标准格式

    Parameters
    ----------
    csv_file_path : str
        CSV文件路径，包含time, prcp(mm/h), pet(mm/h), flow(m^3/s)列
    json_params_path : str
        JSON参数文件路径，包含模型参数

    Returns
    -------
    p_and_e : np.ndarray
        precipitation and evapotranspiration data [time, basin=1, feature=2]
    parameters : np.ndarray
        model parameters [basin=1, parameter=22]
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 提取降雨和蒸发数据
    precipitation = df["prcp(mm/h)"].values
    evapotranspiration = df["pet(mm/h)"].values

    # 构建p_and_e数组 [time, basin=1, feature=2]
    time_steps = len(precipitation)
    p_and_e = np.zeros((time_steps, 1, 2))
    p_and_e[:, 0, 0] = precipitation  # precipitation
    p_and_e[:, 0, 1] = evapotranspiration  # evapotranspiration

    # 从JSON文件读取参数
    with open(json_params_path, "r", encoding="utf-8") as f:
        param_data = json.load(f)

    # 构建参数数组 [basin=1, parameter=22]
    # 参数顺序: [S0, U0, D0, K, KW, K2, KA, G, A, B, B0, K0, N, L, DD, CC, COE, DDL, CCL, SA0, UA0, YA0]
    parameters = np.array(
        [
            [
                float(param_data["S0"]),
                float(param_data["U0"]),
                float(param_data["D0"]),
                float(param_data["K"]),
                float(param_data["KW"]),
                float(param_data["K2"]),
                float(param_data["KA"]),
                float(param_data["G"]),
                float(param_data["A"]),
                float(param_data["B"]),
                float(param_data["B0"]),
                float(param_data["K0"]),
                float(param_data["N"]),
                float(param_data["L"]),
                float(param_data["DD"]),
                float(param_data["CC"]),
                float(param_data["COE"]),
                float(param_data["DDL"]),
                float(param_data["CCL"]),
                float(param_data["SA0"]),
                float(param_data["UA0"]),
                float(param_data["YA0"]),
            ]
        ]
    )

    return p_and_e, parameters


def main():
    """主函数示例"""
    try:
        # 从CSV和JSON文件加载数据
        p_and_e, parameters = load_dhf_data_from_csv_and_json(
            csv_file_path="hydromodel_dev/data/DHF.csv",
            json_params_path="hydromodel_dev/data/dhf_data.json",
        )

        # 读取原始CSV文件以获取时间列和观测流量
        df_original = pd.read_csv("hydromodel_dev/data/DHF.csv")

        # 设置预热期
        warmup_length = 0  # 使用0天作为预热期

        # 使用跟踪功能运行DHF模型
        print("=== 开始计算 ===")
        q_sim, all_results = dhf(
            p_and_e=p_and_e,
            parameters=parameters,
            warmup_length=warmup_length,
            return_state=True,
        )

        # 准备结果数据 - 考虑预热期
        result_dict = {
            "time": df_original["time"]
            .iloc[warmup_length:]
            .reset_index(drop=True),
            "precipitation": df_original["prcp(mm/h)"]
            .iloc[warmup_length:]
            .reset_index(drop=True),
            "pet": df_original["pet(mm/h)"]
            .iloc[warmup_length:]
            .reset_index(drop=True),
            "flow_obs": df_original["flow(m^3/s)"]
            .iloc[warmup_length:]
            .reset_index(drop=True),
            "QSim": q_sim.flatten(),
            "runoffSim": all_results["runoffSim"].flatten(),
        }

        # 创建结果DataFrame
        result_df = pd.DataFrame(result_dict)

        # 保存为CSV文件
        output_csv_path = "hydromodel_dev/data/dhf_result.csv"
        result_df.to_csv(output_csv_path, index=False)

        # 输出计算信息
        # print("\n=== DHF模型计算完成 ===")
        # print(f"预热期长度: {warmup_length} 天")
        # print(f"输入数据形状: {p_and_e.shape}")
        # print(f"参数形状: {parameters.shape}")
        # print(f"输出径流形状: {q_sim.shape}")
        # print(f"实际计算时段数: {len(q_sim)}")

        print(f"\n结果已保存到:")
        print(f"- CSV文件: {output_csv_path}")
        print(f"注意：输出结果已去除预热期 {warmup_length} 天的数据")

    except Exception as e:
        print(f"计算过程中发生错误: {e}")

        traceback.print_exc()


if __name__ == "__main__":
    main()
