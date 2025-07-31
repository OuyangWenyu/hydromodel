"""
Author: Wenyu Ouyang
Date: 2025-07-17 09:06:06
LastEditTime: 2025-07-17 11:52:26
LastEditors: Wenyu Ouyang
Description: some consts
FilePath: \hydromodel_dev\hydromodel_dev\consts.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

NET_RAIN = "P_eff"
OBS_FLOW = "Q_obs_eff"
# --- 全局常量 ---
# 单位线对应的有效降雨量(mm)默认1mm
DELTA_T_HOURS = 3.0  # 时间步长(小时)
DELTA_T_SECONDS = DELTA_T_HOURS * 3600.0  # 时间步长(秒)，用于洪量计算
