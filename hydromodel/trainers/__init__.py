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
    calibrate_with_config,
    ModelSetupBase,
    UnifiedModelSetup,
    DEAP_AVAILABLE,
)

# Import unit hydrograph training functions
from .unit_hydrograph_trainer import (
    objective_function_multi_event,
    optimize_shared_unit_hydrograph,
    evaluate_single_event_from_uh,
    print_report_preview,
    save_results_to_csv,
)

__all__ = [
    # Traditional interfaces
    "SpotSetup",
    "calibrate_by_sceua_old",
    
    # Unified interfaces
    "calibrate",
    "calibrate_with_config",
    "ModelSetupBase",
    "UnifiedModelSetup",
    "DEAP_AVAILABLE",
    
    # Unit hydrograph training functions
    "objective_function_multi_event",
    "optimize_shared_unit_hydrograph", 
    "evaluate_single_event_from_uh",
    "print_report_preview",
    "save_results_to_csv",
]