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


def test_xaj_slw_with_example_data():
    """
    使用示例数据测试XAJ-SLW模型，输出中间变量用于与Java版本对比
    """
    print("开始测试XAJ-SLW模型...")
    sms_data_path = "data/sms_3_data.json"
    lag_data_path = "data/lag_3_data.json"
    try:
        # 加载示例数据
        p_and_e, parameters, time_dates, start_time, es = load_sms_lag_data_from_json(
            sms_data_path,
            lag_data_path,
            default_evap=0.918548  # 设置默认蒸散发值
        )
        
        print("\n数据加载成功:")
        print(f"时间序列长度: {p_and_e.shape[0]}")
        print(f"流域数量: {p_and_e.shape[1]}")
        print(f"特征数量: {p_and_e.shape[2]}")
        print(f"参数数量: {parameters.shape[1]}")
        print(f"时间日期数量: {len(time_dates) if time_dates else 0}")
        print(f"开始时间: {start_time if start_time else 'None'}")
        
        print("\n输入数据概览:")
        print(f"降雨数据范围: [{p_and_e[:, 0, 0].min():.2f}, {p_and_e[:, 0, 0].max():.2f}] mm")
        print(f"蒸散发数据: {p_and_e[0, 0, 1]:.2f} mm")
        
        print("\n模型参数概览:")
        param_names = [
            "WUP", "WLP", "WDP", "SP", "FRP", "WM", "WUMx", "WLMx", "KC", "B",
            "C", "IM", "SM", "EX", "KG", "KI", "CS", "CI", "CG", "LAG", "KK",
            "X", "MP", "QSP", "QIP", "QGP"
        ]
        for i, name in enumerate(param_names):
            print(f"{name}: {parameters[0, i]:.4f}")
        
        # 运行模型
        print("\n运行XAJ-SLW模型...")
        result = xaj_slw(
            p_and_e,
            parameters,
            warmup_length=0,
            return_state=True,  # 返回所有状态变量
            normalized_params=False,
            time_interval_hours=6.0,  # 根据JSON数据中的clen参数
            area=2163.0,  # 根据JSON数据中的F参数
        )
        
        # 解析结果
        q_sim, runoff_sim, rs, ri, rg, pe, wu, wl, wd = result
        
        print("\n模拟结果概览:")
        # 生成时间序列
        with open(sms_data_path, "r") as f:
            data = json.load(f)
            time_series = pd.to_datetime(data["dt"])
        
        print("\n前10个时间步的详细结果:")
        print("时间步 | 降雨 | 蒸散发 | 净雨 | 产流 | 地表径流 | 壤中流 | 地下径流 | 流量")
        print("-" * 80)
        for i in range(min(10, len(time_series))):
            print(f"{i:2d} | {p_and_e[i, 0, 0]:6.3f} | {p_and_e[i, 0, 1]:8.3f} | {pe[i, 0, 0]:6.3f} | "
                  f"{runoff_sim[i, 0, 0]:6.3f} | {rs[i, 0, 0]:8.3f} | {ri[i, 0, 0]:8.3f} | "
                  f"{rg[i, 0, 0]:8.3f} | {q_sim[i, 0, 0]:8.3f}")
        
        # 保存数值结果
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细的中间变量结果，重点关注产汇流模型的输入输出
        results_dict = {
            # 输入数据
            "input": {
                "time": time_series.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                "rainfall": p_and_e[:, 0, 0].tolist(),
                "evapotranspiration": p_and_e[:, 0, 1].tolist(),
                "net_precipitation": pe[:, 0, 0].tolist(),
            },
            # SMS产流模型输出
            "sms_output": {
                "runoff_sim": runoff_sim[:, 0, 0].tolist(),  # 总产流量
                "surface_runoff": rs[:, 0, 0].tolist(),      # 地表径流
                "interflow": ri[:, 0, 0].tolist(),           # 壤中流
                "groundwater": rg[:, 0, 0].tolist(),         # 地下径流
                "upper_tension_water": wu[:, 0, 0].tolist(), # 上层张力水
                "lower_tension_water": wl[:, 0, 0].tolist(), # 下层张力水
                "deep_tension_water": wd[:, 0, 0].tolist(),  # 深层张力水
            },
            # LAG汇流模型输出
            "lag_output": {
                "q_sim": q_sim[:, 0, 0].tolist(),  # 模拟流量
            },
            # 模型参数（用于对比）
            "parameters": {
                "WUP": float(parameters[0, 0]),
                "WLP": float(parameters[0, 1]),
                "WDP": float(parameters[0, 2]),
                "SP": float(parameters[0, 3]),
                "FRP": float(parameters[0, 4]),
                "WM": float(parameters[0, 5]),
                "WUMx": float(parameters[0, 6]),
                "WLMx": float(parameters[0, 7]),
                "KC": float(parameters[0, 8]),
                "B": float(parameters[0, 9]),
                "C": float(parameters[0, 10]),
                "IM": float(parameters[0, 11]),
                "SM": float(parameters[0, 12]),
                "EX": float(parameters[0, 13]),
                "KG": float(parameters[0, 14]),
                "KI": float(parameters[0, 15]),
                "CS": float(parameters[0, 16]),
                "CI": float(parameters[0, 17]),
                "CG": float(parameters[0, 18]),
                "LAG": float(parameters[0, 19]),
                "KK": float(parameters[0, 20]),
                "X": float(parameters[0, 21]),
                "MP": float(parameters[0, 22]),
                "QSP": float(parameters[0, 23]),
                "QIP": float(parameters[0, 24]),
                "QGP": float(parameters[0, 25]),
            },
            # 统计信息（用于快速对比）
            "statistics": {
                "rainfall_stats": {
                    "min": float(p_and_e[:, 0, 0].min()),
                    "max": float(p_and_e[:, 0, 0].max()),
                    "mean": float(p_and_e[:, 0, 0].mean()),
                    "sum": float(p_and_e[:, 0, 0].sum()),
                },
                "evap_stats": {
                    "min": float(p_and_e[:, 0, 1].min()),
                    "max": float(p_and_e[:, 0, 1].max()),
                    "mean": float(p_and_e[:, 0, 1].mean()),
                },
                "runoff_stats": {
                    "min": float(runoff_sim[:, 0, 0].min()),
                    "max": float(runoff_sim[:, 0, 0].max()),
                    "mean": float(runoff_sim[:, 0, 0].mean()),
                    "sum": float(runoff_sim[:, 0, 0].sum()),
                },
                "flow_stats": {
                    "min": float(q_sim[:, 0, 0].min()),
                    "max": float(q_sim[:, 0, 0].max()),
                    "mean": float(q_sim[:, 0, 0].mean()),
                    "sum": float(q_sim[:, 0, 0].sum()),
                },
            }
        }
        
        with open(os.path.join(output_dir, "xaj_slw_results.json"), "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print("\n测试完成！结果已保存到results目录")
        print(f"- 数值结果: {os.path.join(output_dir, 'xaj_slw_results.json')}")
        
        return True, result
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


if __name__ == "__main__":
    success, results = test_xaj_slw_with_example_data()
