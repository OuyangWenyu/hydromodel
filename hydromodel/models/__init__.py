"""
Author: Wenyu Ouyang
Date: 2025-01-19 18:05:00
LastEditTime: 2025-08-29 17:12:56
LastEditors: Wenyu Ouyang
Description: hydromodel models
FilePath: /hydromodel/hydromodel/models/__init__.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

# 水文模型
from .dhf import dhf  # 大伙房模型

# Java 转换的模块（尚未注册到 MODEL_DICT，保留作为独立工具模块）
from . import SMS_3, LAG_3, MSK, SMS3_LAG3_Pipeline  # noqa: F401


__version__ = "0.1.0"
__author__ = "Wenyu Ouyang"
__description__ = "Hydrological Models"

__all__ = [
    "dhf",
]
