"""
Author: Wenyu Ouyang
Date: 2025-02-18 10:20:58
LastEditTime: 2025-08-30 08:53:43
LastEditors: Wenyu Ouyang
Description: LOSS_DICT and MODEL_DICT
FilePath: \hydromodel\hydromodel\models\model_dict.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
"""

import functools
import warnings

import numpy as np
import spotpy.objectivefunctions as spotpy_obj

from hydromodel.models.semi_xaj import semi_xaj
from hydromodel.models.xaj import xaj
from hydromodel.models.gr1a import gr1a
from hydromodel.models.gr2m import gr2m
from hydromodel.models.gr3j import gr3j
from hydromodel.models.gr4j import gr4j
from hydromodel.models.gr5j import gr5j
from hydromodel.models.gr6j import gr6j
from hydromodel.models.hymod import hymod
from hydromodel.models.unit_hydrograph import (
    unit_hydrograph,
    categorized_unit_hydrograph,
)
from hydromodel.models.dhf import dhf
from hydromodel.models.xaj_slw import xaj_slw


_MAXIMIZE_LOSSES = {
    "spotpy_kge",
    "spotpy_kge_non_parametric",
    "spotpy_lognashsutcliffe",
    "spotpy_nashsutcliffe",
}

_MAXIMIZE_LOSS_REPLACEMENTS = {
    "spotpy_kge": "KGE",
    "spotpy_lognashsutcliffe": "LOGNSE",
    "spotpy_nashsutcliffe": "NSE",
}

_USER_OBJECTIVE_MAP = {
    "RMSE": "RMSE",
    "NSE": "neg_nashsutcliffe",
    "KGE": "neg_kge",
    "LOGNSE": "neg_lognashsutcliffe",
    "MSE": "spotpy_mse",
    "MAE": "spotpy_mae",
}


def _flatten_objective_inputs(obs, sim):
    """Return 1D arrays for objective functions that cannot handle 3D data."""
    return np.asarray(obs).reshape(-1), np.asarray(sim).reshape(-1)


def _wrap_spotpy_function(name, func):
    """Wrap a spotpy objective so hydromodel's 3D arrays are accepted."""

    @functools.wraps(func)
    def wrapped(obs, sim, *args, **kwargs):
        obs_flat, sim_flat = _flatten_objective_inputs(obs, sim)
        return func(obs_flat, sim_flat, *args, **kwargs)

    wrapped.__name__ = f"spotpy_{name}"
    return wrapped


def _negated_spotpy_function(func):
    @functools.wraps(func)
    def wrapped(obs, sim, *args, **kwargs):
        obs_flat, sim_flat = _flatten_objective_inputs(obs, sim)
        return -func(obs_flat, sim_flat, *args, **kwargs)

    wrapped.__name__ = f"neg_{func.__name__}"
    return wrapped


def _auto_discover_spotpy_functions():
    """Auto-discover all callable functions in spotpy.objectivefunctions.

    Returns
    -------
    dict
        Dictionary with auto-discovered spotpy functions
    """
    auto_functions = {}

    # Get all functions from spotpy.objectivefunctions
    for attr_name in dir(spotpy_obj):
        attr = getattr(spotpy_obj, attr_name)
        # Check if it's a callable function and not private
        if (
            callable(attr)
            and not attr_name.startswith("_")
            and attr_name not in ["calculate_all_functions"]
        ):  # Exclude utility functions
            auto_functions[f"spotpy_{attr_name}"] = _wrap_spotpy_function(
                attr_name, attr
            )

    return auto_functions


def rmse43darr(obs, sim):
    """RMSE for 3D array

    Parameters
    ----------
    obs : np.ndarray
        observation data
    sim : np.ndarray
        simulation data

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    rmses = np.sqrt(np.nanmean((sim - obs) ** 2, axis=0))
    rmse = rmses.mean(axis=0)
    if np.isnan(rmse) or any(np.isnan(sim)):
        raise ValueError(
            "RMSE is nan or there are nan values in the simulation data, "
            "please check the input data."
        )
    # tolist is necessary for spotpy to get the value
    # otherwise the print will incur to an issue
    # https://github.com/thouska/spotpy/issues/319
    return rmse.tolist()


# Generate LOSS_DICT with custom functions and all spotpy functions
LOSS_DICT = {
    "RMSE": rmse43darr,
    **_auto_discover_spotpy_functions(),
}

LOSS_DICT.update(
    {
        "neg_nashsutcliffe": _negated_spotpy_function(
            spotpy_obj.nashsutcliffe
        ),
        "neg_kge": _negated_spotpy_function(spotpy_obj.kge),
        "neg_lognashsutcliffe": _negated_spotpy_function(
            spotpy_obj.lognashsutcliffe
        ),
    }
)


def resolve_loss_config(loss_config):
    """Resolve user-facing objective names to minimization loss keys.

    Hydromodel optimizers minimize objective values. User-facing metrics such as
    NSE, KGE, and LogNSE are maximized by mapping them to negated objectives.
    Existing LOSS_DICT keys remain accepted for compatibility.
    """
    resolved = dict(loss_config or {})
    obj_func = resolved.get("obj_func", "RMSE")

    if callable(obj_func):
        resolved.setdefault(
            "requested_obj_func", getattr(obj_func, "__name__", "callable")
        )
        resolved.setdefault("resolved_obj_func", obj_func)
        return resolved

    requested = str(obj_func)
    requested_upper = requested.upper()

    if requested_upper in _USER_OBJECTIVE_MAP:
        resolved_obj_func = _USER_OBJECTIVE_MAP[requested_upper]
    elif requested in LOSS_DICT:
        resolved_obj_func = requested
        if requested in _MAXIMIZE_LOSSES:
            warnings.warn(
                f"Objective '{requested}' is a higher-is-better metric but "
                "hydromodel optimizers minimize objective values. Use "
                f"'{_MAXIMIZE_LOSS_REPLACEMENTS.get(requested, requested_upper)}' "
                "or the matching negated objective for calibration.",
                RuntimeWarning,
                stacklevel=2,
            )
    elif f"spotpy_{requested.lower()}" in LOSS_DICT:
        resolved_obj_func = f"spotpy_{requested.lower()}"
        if resolved_obj_func in _MAXIMIZE_LOSSES:
            warnings.warn(
                f"Objective '{resolved_obj_func}' is a higher-is-better "
                "metric but hydromodel optimizers minimize objective values. "
                "Use "
                f"'{_MAXIMIZE_LOSS_REPLACEMENTS.get(resolved_obj_func, requested_upper)}' "
                "or the matching negated objective for calibration.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        supported = sorted(set(_USER_OBJECTIVE_MAP) | set(LOSS_DICT))
        raise KeyError(
            f"Unsupported objective function '{requested}'. "
            f"Supported values include: {', '.join(supported[:20])}"
        )

    if resolved_obj_func not in LOSS_DICT:
        raise KeyError(
            f"Resolved objective '{resolved_obj_func}' is not registered in LOSS_DICT"
        )

    resolved["requested_obj_func"] = resolved.get(
        "requested_obj_func", requested_upper
    )
    resolved["resolved_obj_func"] = resolved_obj_func
    resolved["obj_func"] = resolved_obj_func
    return resolved


MODEL_DICT = {
    "xaj_mz": xaj,
    "xaj": xaj,
    "gr4j": gr4j,
    "gr5j": gr5j,
    "gr6j": gr6j,
    "gr1a": gr1a,
    "gr2m": gr2m,
    "gr3j": gr3j,
    "hymod": hymod,
    "semi_xaj": semi_xaj,
    "unit_hydrograph": unit_hydrograph,
    "categorized_unit_hydrograph": categorized_unit_hydrograph,
    "dhf": dhf,
    "xaj_slw": xaj_slw,
}


def list_models():
    """Return registered model names."""
    return sorted(MODEL_DICT.keys())


def list_losses():
    """Return user-facing objectives and registered internal loss keys."""
    return {
        "user_objectives": sorted(_USER_OBJECTIVE_MAP.keys()),
        "registered_losses": sorted(LOSS_DICT.keys()),
        "maximize_losses": sorted(_MAXIMIZE_LOSSES),
    }


def describe_model(model_name):
    """Return model callable and parameter contract metadata."""
    if model_name not in MODEL_DICT:
        raise KeyError(f"Unsupported model: {model_name}")

    from hydromodel.models.model_config import MODEL_PARAM_DICT

    return {
        "name": model_name,
        "available": True,
        "parameters": MODEL_PARAM_DICT.get(model_name),
    }


def check_dependencies():
    """Return availability of optional calibration dependencies."""
    dependencies = {}
    for package in ["deap", "spotpy", "scipy", "xarray"]:
        try:
            __import__(package)
            dependencies[package] = True
        except ImportError:
            dependencies[package] = False
    return dependencies
