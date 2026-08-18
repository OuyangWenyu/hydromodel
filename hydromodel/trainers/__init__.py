"""
Author: Wenyu Ouyang
Date: 2025-08-06
LastEditTime: 2025-08-29 17:14:18
LastEditors: Wenyu Ouyang
Description: Trainers module for hydrological model calibration
FilePath: /hydromodel/hydromodel/trainers/__init__.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

__all__ = []

try:
    # Import traditional calibration functions for backward compatibility.
    from .calibrate_sceua import SpotSetup
    from .calibrate_sceua import calibrate_by_sceua as calibrate_by_sceua_old

    __all__.extend(["SpotSetup", "calibrate_by_sceua_old"])
except ImportError:
    pass

try:
    # Import unified calibration interface when optional data deps are present.
    from .unified_calibrate import (
        DEAP_AVAILABLE,
        ModelSetupBase,
        UnifiedCalibrator,
        calibrate,
    )

    __all__.extend(
        ["DEAP_AVAILABLE", "ModelSetupBase", "UnifiedCalibrator", "calibrate"]
    )
except ImportError:
    DEAP_AVAILABLE = False

try:
    # Import unified evaluation interface when optional data deps are present.
    from .unified_evaluate import UnifiedEvaluator, evaluate

    __all__.extend(["UnifiedEvaluator", "evaluate"])
except ImportError:
    pass
