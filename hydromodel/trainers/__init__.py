"""
Author: Wenyu Ouyang
Date: 2025-08-06
LastEditTime: 2025-08-29 17:14:18
LastEditors: Wenyu Ouyang
Description: Trainers module for hydrological model calibration
FilePath: /hydromodel/hydromodel/trainers/__init__.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

# Import traditional calibration functions for backward compatibility
from .calibrate_sceua import SpotSetup
from .calibrate_sceua import calibrate_by_sceua as calibrate_by_sceua_old

# Import unified calibration interface
from .unified_calibrate import (
    calibrate,
    ModelSetupBase,
    UnifiedModelSetup,
    DEAP_AVAILABLE,
)

# Import unified evaluation interface
from .unified_evaluate import evaluate

__all__ = [
    # Traditional interfaces
    "SpotSetup",
    "calibrate_by_sceua_old",
    # Unified interfaces
    "calibrate",
    "evaluate",
    "ModelSetupBase",
    "UnifiedModelSetup",
    "DEAP_AVAILABLE",
]
