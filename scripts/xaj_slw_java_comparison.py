"""
Test cases for XAJ-SLW model
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# 添加模型路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hydromodel.models.xaj_slw import xaj_slw, load_sms_lag_data_from_json


def load_data_from_csv(csv_file_path):
    """
    从CSV文件中读取降雨和蒸发数据
    Args:
        csv_file_path: CSV文件路径
    Returns:
        p_and_e: numpy数组，shape为(n, 1, 2)，其中n为时间步数，
                第一列为降雨，第二列为蒸发
        dt: 时间序列列表
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 获取降雨和ES数据
    rain = df["rain"].values
    es = df["ES"].values
    dt = df["time"].values

    # 构建p_and_e数组
    p_and_e = np.zeros((len(rain), 1, 2))
    p_and_e[:, 0, 0] = rain

    # 设置每个时间步的ES值
    p_and_e[:, 0, 1] = es

    return p_and_e, dt.tolist()


def test_xaj_slw_with_example_data(use_csv=False):
    """
    使用示例数据测试XAJ-SLW模型，输出中间变量用于与Java版本对比
    Args:
        use_csv: 是否使用CSV文件读取降雨和蒸发数据，默认为False使用JSON文件
    """
    print("开始测试XAJ-SLW模型...")

    # 加载测试数据
    sms_json_file = "/home/zlh/hydromodeljava/src/main/resources/event24.json"
    lag_json_file = "/home/zlh/hydromodeljava/src/main/resources/event24_lag_data.json"
    csv_file = "/home/zlh/hydromodel/data/event24.csv"

    try:
        # 加载和解析数据
        with open(sms_json_file, "r", encoding="utf-8") as f:
            sms_data = json.load(f)
        with open(lag_json_file, "r", encoding="utf-8") as f:
            lag_data = json.load(f)

        # 根据数据来源选择不同的数据加载方式
        if use_csv:
            # 从CSV文件读取降雨和蒸发数据
            p_and_e, dt = load_data_from_csv(csv_file)
            time_interval = float(sms_data.get("clen", 6.0))
        else:
            # 使用原有的JSON文件方式
            rain = np.array(sms_data["rain"])
            dt = sms_data["dt"]
            es = np.array(sms_data["ES"])
            time_interval = float(sms_data.get("clen", 6.0))

            # 计算蒸发量
            evap = np.zeros_like(rain)
            for i, time_str in enumerate(dt):
                dt_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                month = dt_obj.month - 1
                days_in_month = [
                    31,
                    28,
                    31,
                    30,
                    31,
                    30,
                    31,
                    31,
                    30,
                    31,
                    30,
                    31,
                ][month]
                evap[i] = es[month] / (days_in_month * 24.0 / time_interval)

            # 构建p_and_e数组
            p_and_e = np.zeros((len(rain), 1, 2))
            p_and_e[:, 0, 0] = rain
            p_and_e[:, 0, 1] = evap

        # 构建参数数组
        parameters = np.array(
            [
                [
                    float(sms_data["WUP"]),
                    float(sms_data["WLP"]),
                    float(sms_data["WDP"]),
                    float(sms_data["SP"]),
                    float(sms_data["FRP"]),
                    float(sms_data["WM"]),
                    float(sms_data["WUMx"]),
                    float(sms_data["WLMx"]),
                    float(sms_data["K"]),
                    float(sms_data["B"]),
                    float(sms_data["C"]),
                    float(sms_data["IM"]),
                    float(sms_data["SM"]),
                    float(sms_data["EX"]),
                    float(sms_data["KG"]),
                    float(sms_data["KI"]),
                    float(lag_data["CS"]),
                    float(lag_data["CI"]),
                    float(lag_data["CG"]),
                    float(lag_data["LAG"]),
                    float(lag_data["KK"]),
                    float(lag_data["X"]),
                    float(lag_data["MP"]),
                    float(lag_data["QSP"]),
                    float(lag_data["QIP"]),
                    float(lag_data["QGP"]),
                ]
            ]
        )

        # 运行模型
        print("\n运行XAJ-SLW模型...")
        warmup_length = 480  # 预热期步长
        result = xaj_slw(
            p_and_e,
            parameters,
            warmup_length=warmup_length, 
            return_state=True,
            normalized_params=False,
            time_interval_hours=time_interval,
            basin_area=float(lag_data["F"]),
        )

        # 解析结果
        q_sim, runoff_sim, rs, ri, rg, pe, wu, wl, wd= result

        # 创建结果数据框
        df_result = pd.DataFrame(
            {
                "time": dt[warmup_length:],
                "q_sim": q_sim[:, 0, 0],
                "runoff_sim": runoff_sim[:, 0, 0],
            }
        )

        print("\n最终流量:")
        print("时间                 q_sim          runoff_sim ")
        print("-" * 35)
        for i in range(len(q_sim)):
            print(f"{dt[i+warmup_length]}  {q_sim[i,0,0]:9.6f} {runoff_sim[i,0,0]:9.6f}")
            
        inflow_df = pd.read_csv(csv_file)
        nse_from_df(df_result, inflow_df, warmup_length=480)    
            
        print("\n结果已保存到 output_csv.csv 文件中")
        df_result.to_csv('output_csv.csv', index=False)
        
        return True, df_result

    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return False, None


def calculate_nse(observed, simulated):
    """
    计算 Nash-Sutcliffe 效率系数 (NSE)
    
    Args:
        observed (np.ndarray): 观测值
        simulated (np.ndarray): 模拟值
    
    Returns:
        float: NSE 值
    """
    if len(observed) != len(simulated):
        raise ValueError("观测值和模拟值的长度必须相等")
    
    mean_observed = np.mean(observed)
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - mean_observed) ** 2)
    return 1 - (numerator / denominator)

def nse_from_df(df_result, inflow_df, warmup_length=480):
    """
    从 DataFrame 中读取数据并计算 q_sim 和 inflow 列的 NSE
    
    Args:
        df_result (pd.DataFrame): 包含 q_sim 列的结果 DataFrame
        inflow_df (pd.DataFrame): 包含 inflow 列的输入 DataFrame
        warmup_length (int): 预热期长度，默认为 480
    """
    # 确保列存在
    if 'q_sim' not in df_result.columns:
        raise ValueError("df_result 中缺少列: q_sim")
    if 'inflow' not in inflow_df.columns:
        raise ValueError("inflow_df 中缺少列: inflow")
    
    # 提取 q_sim 和 inflow 列
    q_sim = df_result['q_sim'].values
    inflow = inflow_df['inflow'].values[warmup_length:]  # 减去预热期长度
    
    # 确保长度一致
    if len(q_sim) != len(inflow):
        raise ValueError("q_sim 和 inflow 的长度不一致，无法计算 NSE")
    
    # 计算 NSE
    q_sim_nse = calculate_nse(q_sim,inflow)
    
    # 输出结果
    print(f"q_sim 列的 NSE: {q_sim_nse:.4f}")
    
if __name__ == "__main__":
    # 使用JSON文件测试
    # print("\n使用JSON文件测试:")
    # success_json, results_json = test_xaj_slw_with_example_data(use_csv=False)

    # 使用CSV文件测试
    print("\n使用CSV文件测试:")
    success_csv, results_csv = test_xaj_slw_with_example_data(use_csv=True)
