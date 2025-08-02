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
from .unit_hydrograph import (
    objective_function_multi_event,
    optimize_uh_for_group,
)

# 水文模型
from .dhf import dhf  # 大伙房模型

# 工具函数模块
from .uh_utils import (
    # 评估工具
    evaluate_single_event,
    calculate_nse,
    calculate_volume_error,
    calculate_peak_error,
    # 分类工具
    categorize_floods_by_peak,
    save_results_to_csv,
    print_report_preview,
    print_category_statistics,
)

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
    # 工具函数
    "evaluate_single_event",
    "calculate_nse",
    "calculate_volume_error",
    "calculate_peak_error",
    "categorize_floods_by_peak",
    "setup_matplotlib",
    "save_results_to_csv",
    "print_report_preview",
    "print_category_statistics",
]
