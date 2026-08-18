"""
Author: Wenyu Ouyang
Date: 2024-02-09 15:56:48
LastEditTime: 2025-10-30 21:45:00
LastEditors: Wenyu Ouyang
Description: Top-level package for hydromodel with unified interfaces
FilePath: \\hydromodel\\hydromodel\\__init__.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import importlib
import sys

from hydroutils import hydro_file

# Windows consoles default to GBK/cp936, which cannot encode the emoji used in
# progress messages (e.g. \U0001f680). Force UTF-8 output where supported so
# calibration/evaluation logs do not crash with UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

__all__ = ["CACHE_DIR"]

try:
    from .models.model_dict import (
        check_dependencies,
        describe_model,
        list_losses,
        list_models,
        resolve_loss_config,
    )

    __all__.extend(
        [
            "check_dependencies",
            "describe_model",
            "list_losses",
            "list_models",
            "resolve_loss_config",
        ]
    )
except ImportError:
    pass

_LAZY_EXPORTS = {
    "calibrate": ("hydromodel.trainers.unified_calibrate", "calibrate"),
    "simulate": ("hydromodel.trainers.unified_simulate", "simulate"),
    "UnifiedSimulator": (
        "hydromodel.trainers.unified_simulate",
        "UnifiedSimulator",
    ),
    "evaluate": ("hydromodel.trainers.unified_evaluate", "evaluate"),
    "Basin": ("hydromodel.trainers.basin", "Basin"),
    "detect_time_interval": (
        "hydroutils.hydro_units",
        "detect_time_interval",
    ),
    "get_time_interval_info": (
        "hydroutils.hydro_units",
        "get_time_interval_info",
    ),
    "validate_unit_compatibility": (
        "hydroutils.hydro_units",
        "validate_unit_compatibility",
    ),
}

__all__.extend(_LAZY_EXPORTS.keys())


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'hydromodel' has no attribute '{name}'")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

__author__ = """Wenyu Ouyang"""
__email__ = "wenyuouyang@outlook.com"
__version__ = "0.4.0"

CACHE_DIR = hydro_file.get_cache_dir()
