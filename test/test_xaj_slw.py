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

    # 加载测试数据
    sms_json_file = "/home/zlh/hydromodel/data/sms_3_data.json"
    lag_json_file = "/home/zlh/hydromodel/data/lag_3_data.json"

    try:
        # 加载和解析数据
        with open(sms_json_file, "r", encoding="utf-8") as f:
            sms_data = json.load(f)
        with open(lag_json_file, "r", encoding="utf-8") as f:
            lag_data = json.load(f)

        # 构建输入数据
        rain = np.array(sms_data["rain"])
        dt = sms_data["dt"]
        es = np.array(sms_data["ES"])
        time_interval = float(sms_data.get("clen", 6.0))

        # 计算蒸发量
        evap = np.zeros_like(rain)
        for i, time_str in enumerate(dt):
            dt_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            month = dt_obj.month - 1
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][
                month
            ]
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
        result = xaj_slw(
            p_and_e,
            parameters,
            warmup_length=0,
            return_state=True,
            normalized_params=False,
            time_interval_hours=time_interval,
            basin_area=float(lag_data["F"]),
            lag_initial_states={
                "qsig_initial": np.array(lag_data["QSIG"], dtype=float),
                "qx_initial": np.array(lag_data["QXSIG"], dtype=float),
            },
        )

        # 解析结果
        q_sim, runoff_sim, rs, ri, rg, pe, wu, wl, wd = result

        # 创建结果数据框
        df_result = pd.DataFrame(
            {
                "datetime": dt,
                "rs": rs[:, 0, 0],
                "ri": ri[:, 0, 0],
                "rg": rg[:, 0, 0],
                "runoff_total": runoff_sim[:, 0, 0],
                "q_sim": q_sim[:, 0, 0],
            }
        )

        # 输出结果
        # print("\n产流结果:")
        # print("\nrSim (地表径流、壤中流、地下径流):")
        # print("时间                   RS         RI         RG")
        # print("-" * 55)
        # for i in range(len(dt)):
        #     print(f"{dt[i]}  {rs[i,0,0]:9.6f}  {ri[i,0,0]:9.6f}  {rg[i,0,0]:9.6f}")

        print("\n最终流量:")
        print("时间                 q_sim")
        print("-" * 35)
        for i in range(len(dt)):
            print(f"{dt[i]}  {q_sim[i,0,0]:9.6f}")

        return True, df_result

    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return False, None


if __name__ == "__main__":
    success, results = test_xaj_slw_with_example_data()
