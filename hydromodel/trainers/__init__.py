"""
Author: Wenyu Ouyang
Date: 2025-08-06 
LastEditTime: 2025-08-06 
LastEditors: Wenyu Ouyang
Description: Trainers module for hydrological model calibration
FilePath: \hydromodel\hydromodel\trainers\__init__.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

# Import traditional calibration functions for backward compatibility
from .calibrate_sceua import SpotSetup
from .calibrate_sceua import calibrate_by_sceua as calibrate_by_sceua_old

# Import unified calibration interface
from .unified_calibrate import (
    calibrate,
    calibrate_by_sceua,
    calibrate_unit_hydrograph,
    calibrate_categorized_unit_hydrograph,
    ModelSetupBase,
    TraditionalModelSetup,
    UnitHydrographSetup,
    CategorizedUnitHydrographSetup,
    SpotpyAdapter,
)

# Import GA functions if available
try:
    from .unified_calibrate import (
        _calibrate_with_ga,
        _calibrate_traditional_with_ga,
    )
    GA_AVAILABLE = True
except ImportError:
    GA_AVAILABLE = False

__all__ = [
    # Traditional interfaces
    "SpotSetup",
    "calibrate_by_sceua_old",
    
    # Unified interfaces
    "calibrate",
    "calibrate_by_sceua", 
    "calibrate_unit_hydrograph",
    "calibrate_categorized_unit_hydrograph",
    "ModelSetupBase",
    "TraditionalModelSetup", 
    "UnitHydrographSetup",
    "CategorizedUnitHydrographSetup",
    "SpotpyAdapter",
    
    # GA availability flag
    "GA_AVAILABLE",
]