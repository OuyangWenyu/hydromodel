"""
Author: Wenyu Ouyang
Date: 2025-01-19 18:05:00
LastEditTime: 2025-07-31 08:33:19
LastEditors: Wenyu Ouyang
Description: hydromodel models
FilePath: /hydromodel/hydromodel/models/__init__.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

# 核心单位线分析模块
# Define constants directly for backward compatibility
DELTA_T_HOURS = 3.0
DELTA_T_SECONDS = 10800.0

# Import optimization functions from the new trainer location
try:
    from hydromodel.trainers.unit_hydrograph_trainer import (
        objective_function_multi_event,
        optimize_uh_for_group,
    )
except ImportError:
    # Fallback for missing functions - provide dummy implementations
    def objective_function_multi_event(*args, **kwargs):
        raise NotImplementedError("Unit hydrograph optimization functions moved to trainers module")
    
    def optimize_uh_for_group(*args, **kwargs):
        raise NotImplementedError("Unit hydrograph optimization functions moved to trainers module")

# 水文模型
from .dhf import dhf  # 大伙房模型


__version__ = "0.1.0"
__author__ = "Wenyu Ouyang"
__description__ = "松辽河流域水文模型开发包 - 专注于单位线分析的简洁数据接口"

# 导出所有公共接口
__all__ = [
    # 核心单位线模块
    "objective_function_multi_event",
    "optimize_uh_for_group",
    "DELTA_T_HOURS",
    "DELTA_T_SECONDS",
    # 水文模型
    "dhf",
]
