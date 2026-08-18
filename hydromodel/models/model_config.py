"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2025-08-22 11:06:58
LastEditors: Wenyu Ouyang
Description: some basic config for hydro-model-xaj models
FilePath: /hydromodel/hydromodel/models/model_config.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import copy
import os
import warnings
from collections import OrderedDict

import yaml


class ParamRangeConfigError(ValueError):
    """Raised when an explicit parameter range configuration is invalid."""


def _copy_model_param_dict(param_dict):
    return copy.deepcopy(param_dict)


def _validate_param_bounds(model, param_name, bounds):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ParamRangeConfigError(
            f"Parameter range for '{model}.{param_name}' must be [min, max]"
        )
    try:
        lower = float(bounds[0])
        upper = float(bounds[1])
    except (TypeError, ValueError) as exc:
        raise ParamRangeConfigError(
            f"Parameter range for '{model}.{param_name}' must be numeric"
        ) from exc
    if lower >= upper:
        raise ParamRangeConfigError(
            f"Parameter range for '{model}.{param_name}' must satisfy min < max"
        )
    return [lower, upper]


def validate_model_param_dict(param_dict, model_name=None):
    """Validate and normalize model parameter metadata.

    The returned param_range order always follows param_name, which is required
    because normalized parameter vectors are positional.
    """
    if not isinstance(param_dict, dict):
        raise ParamRangeConfigError(
            "Parameter range file must contain a mapping"
        )

    models = (
        [model_name] if model_name is not None else list(param_dict.keys())
    )
    normalized = {}
    for model in models:
        if model not in param_dict:
            raise ParamRangeConfigError(
                f"Parameter range does not define model '{model}'"
            )
        contents = param_dict[model]
        if not isinstance(contents, dict):
            raise ParamRangeConfigError(
                f"Parameter range for model '{model}' must be a mapping"
            )

        param_names = contents.get("param_name")
        param_ranges = contents.get("param_range")
        if not isinstance(param_names, list) or not param_names:
            raise ParamRangeConfigError(
                f"Model '{model}' must define a non-empty param_name list"
            )
        if not isinstance(param_ranges, dict):
            raise ParamRangeConfigError(
                f"Model '{model}' must define a param_range mapping"
            )

        missing = [name for name in param_names if name not in param_ranges]
        extra = [name for name in param_ranges if name not in param_names]
        if missing:
            raise ParamRangeConfigError(
                f"Model '{model}' is missing ranges for: {missing}"
            )
        if extra:
            raise ParamRangeConfigError(
                f"Model '{model}' has extra ranges not in param_name: {extra}"
            )

        normalized[model] = {
            "param_name": list(param_names),
            "param_range": OrderedDict(
                (
                    name,
                    _validate_param_bounds(model, name, param_ranges[name]),
                )
                for name in param_names
            ),
        }

    return normalized


def read_model_param_dict(file_path=None, strict=False):
    # If file_path is None, return default MODEL_PARAM_DICT
    if file_path is None:
        return _copy_model_param_dict(MODEL_PARAM_DICT)

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return validate_model_param_dict(data or {})
    except Exception as e:
        if strict:
            raise ParamRangeConfigError(
                f"Invalid param_range_file '{file_path}': {e}"
            ) from e
        warnings.warn(
            f"Invalid param_range_file '{file_path}': {e}. "
            "Using default MODEL_PARAM_DICT.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _copy_model_param_dict(MODEL_PARAM_DICT)


def resolve_model_param_config(
    model_name,
    param_range_file=None,
    fallback_dir=None,
    strict=False,
):
    """Resolve parameter ranges for a model with explicit source metadata."""
    source = "default"
    resolved_file = None

    if param_range_file is not None:
        resolved_file = str(param_range_file)
        source = "explicit"
    elif fallback_dir is not None:
        candidate = os.path.join(str(fallback_dir), "param_range.yaml")
        if os.path.exists(candidate):
            resolved_file = candidate
            source = "artifact"

    if resolved_file is None:
        warnings.warn(
            f"param_range_file not provided for model '{model_name}'. "
            "Using default MODEL_PARAM_DICT ranges.",
            RuntimeWarning,
            stacklevel=2,
        )

    param_dict = read_model_param_dict(
        resolved_file, strict=strict or resolved_file is not None
    )
    normalized = validate_model_param_dict(param_dict, model_name=model_name)
    model_param_config = normalized[model_name]

    return {
        "param_dict": param_dict,
        "model_param_config": model_param_config,
        "source": source,
        "source_path": resolved_file,
    }


def inject_model_param_config(model_params, model_name, model_param_config):
    """Inject parameter metadata using both explicit and legacy contracts."""
    injected = dict(model_params or {})
    injected[model_name] = model_param_config
    injected["param_config"] = {model_name: model_param_config}
    injected["param_name"] = model_param_config["param_name"]
    injected["param_range"] = model_param_config["param_range"]
    return injected


def get_model_param_config(model_name, kwargs=None):
    """Resolve model parameter metadata from explicit, legacy, or default kwargs."""
    kwargs = kwargs or {}

    param_config = kwargs.get("param_config")
    if isinstance(param_config, dict):
        if model_name in param_config:
            return validate_model_param_dict(
                {model_name: param_config[model_name]}, model_name=model_name
            )[model_name]
        if "param_range" in param_config:
            default_config = MODEL_PARAM_DICT.get(model_name)
            if default_config is None:
                raise ParamRangeConfigError(
                    f"No default parameter configuration for model '{model_name}'"
                )
            names = param_config.get(
                "param_name", default_config["param_name"]
            )
            return validate_model_param_dict(
                {
                    model_name: {
                        "param_name": names,
                        "param_range": param_config["param_range"],
                    }
                },
                model_name=model_name,
            )[model_name]

    explicit_range = kwargs.get("param_range")
    if isinstance(explicit_range, dict):
        default_config = MODEL_PARAM_DICT.get(model_name)
        if default_config is None:
            raise ParamRangeConfigError(
                f"No default parameter configuration for model '{model_name}'"
            )
        names = kwargs.get("param_name", default_config["param_name"])
        return validate_model_param_dict(
            {model_name: {"param_name": names, "param_range": explicit_range}},
            model_name=model_name,
        )[model_name]

    legacy_config = kwargs.get(model_name)
    if isinstance(legacy_config, dict):
        if "param_name" not in legacy_config:
            default_config = MODEL_PARAM_DICT.get(model_name, {})
            legacy_config = dict(
                param_name=default_config.get("param_name", []),
                **legacy_config,
            )
        return validate_model_param_dict(
            {model_name: legacy_config}, model_name=model_name
        )[model_name]

    warnings.warn(
        f"Parameter metadata for model '{model_name}' was not provided. "
        "Falling back to MODEL_PARAM_DICT defaults.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _copy_model_param_dict(MODEL_PARAM_DICT)[model_name]


def denormalize_parameter_dict(parameters, model_param_config):
    """Convert a normalized parameter dict to physical values."""
    param_names = model_param_config["param_name"]
    param_ranges = model_param_config["param_range"]
    denormalized = OrderedDict()
    for name in param_names:
        value = parameters[name]
        lower, upper = param_ranges[name]
        denormalized[name] = lower + float(value) * (upper - lower)
    return denormalized


def serializable_loss_config(loss_config):
    """Return a JSON-safe copy of a resolved loss configuration."""
    serializable = {}
    for key, value in loss_config.items():
        if callable(value):
            serializable[key] = getattr(value, "__name__", "callable")
        else:
            serializable[key] = value
    return serializable


def attach_parameter_contract(result, model_setup):
    """Add explicit parameter and loss metadata without removing legacy fields."""
    result = dict(result)
    model_name = model_setup.model_name
    best_params = copy.deepcopy(result.get("best_params"))
    normalized = None

    if "objective_value" in result:
        result["objective_value"] = float(result["objective_value"])

    if isinstance(best_params, dict):
        model_params = best_params.get(model_name)
        if isinstance(model_params, dict):
            normalized = {
                name: float(model_params[name])
                for name in model_setup.parameter_names
                if name in model_params
            }
            best_params[model_name] = normalized

    result["parameter_format"] = "normalized"
    result["best_params_normalized"] = (
        {model_name: normalized} if normalized is not None else None
    )
    result["best_params_denormalized"] = (
        {
            model_name: denormalize_parameter_dict(
                normalized, model_setup.model_param_config
            )
        }
        if normalized is not None
        else None
    )
    result["param_range_source"] = model_setup.param_range_source
    result["param_range_source_path"] = model_setup.param_range_source_path
    result["loss_config"] = serializable_loss_config(model_setup.loss_config)
    return result


MODEL_PARAM_DICT = {
    "xaj": {
        "param_name": [
            # Allen, R.G., L. Pereira, D. Raes, and M. Smith, 1998.
            # Crop Evapotranspiration, Food and Agriculture Organization of the United Nations,
            # Rome, Italy. FAO publication 56. ISBN 92-5-104219-5. 290p.
            "K",  # ratio of potential evapotranspiration to reference crop evaporation generally from Allen, 1998
            "B",  # The exponent of the tension water capacity curve
            "IM",  # The ratio of the impervious to the total area of the basin
            "UM",  # Tension water capacity in the upper layer
            "LM",  # Tension water capacity in the lower layer
            "DM",  # Tension water capacity in the deepest layer
            "C",  # The coefficient of deep evapotranspiration
            "SM",  # The areal mean of the free water capacity of surface soil layer
            "EX",  # The exponent of the free water capacity curve
            "KI",  # Outflow coefficients of interflow
            "KG",  # Outflow coefficients of groundwater
            "CS",  # The recession constant of channel system
            "L",  # Lag time
            "CI",  # The recession constant of the lower interflow
            "CG",  # The recession constant of groundwater storage
        ],
        "param_range": OrderedDict(
            {
                "K": [0.1, 1.0],
                "B": [0.1, 0.4],
                "IM": [0.01, 0.1],
                "UM": [0.0, 20.0],
                "LM": [60.0, 90.0],
                "DM": [60.0, 120.0],
                "C": [0.0, 0.2],
                "SM": [1, 100.0],
                # "SM": [50, 100.0],
                "EX": [1.0, 1.5],
                "KI": [0.0, 0.7],
                "KG": [0.0, 0.7],
                "CS": [0.0, 1.0],
                "L": [1.0, 10.0],  # unit is same as your time step
                "CI": [0.0, 0.9],
                "CG": [0.98, 0.998],
            }
        ),
    },
    "xaj_mz": {
        "param_name": [
            # Allen, R.G., L. Pereira, D. Raes, and M. Smith, 1998.
            # Crop Evapotranspiration, Food and Agriculture Organization of the United Nations,
            # Rome, Italy. FAO publication 56. ISBN 92-5-104219-5. 290p.
            "K",  # ratio of potential evapotranspiration to reference crop evaporation generally from Allen, 1998
            "B",  # The exponent of the tension water capacity curve
            "IM",  # The ratio of the impervious to the total area of the basin
            "UM",  # Tension water capacity in the upper layer
            "LM",  # Tension water capacity in the lower layer
            "DM",  # Tension water capacity in the deepest layer
            "C",  # The coefficient of deep evapotranspiration
            "SM",  # The areal mean of the free water capacity of surface soil layer
            "EX",  # The exponent of the free water capacity curve
            "KI",  # Outflow coefficients of interflow
            "KG",  # Outflow coefficients of groundwater
            "A",  # parameter of mizuRoute
            "THETA",  # parameter of mizuRoute
            "CI",  # The recession constant of the lower interflow
            "CG",  # The recession constant of groundwater storage
            # "KERNEL",  # kernel size of mizuRoute unit hydrograph when using convolution method
        ],
        "param_range": OrderedDict(
            {
                "K": [0.1, 1.0],
                # "K": [0.5, 1.0],
                "B": [0.1, 0.4],
                # "B": [0.2, 0.4],
                "IM": [0.01, 0.1],
                # "IM": [0.07, 0.1],
                "UM": [0.0, 20.0],
                "LM": [60.0, 90.0],
                "DM": [60.0, 120.0],
                "C": [0.0, 0.2],
                "SM": [1.0, 100.0],
                # "SM": [5, 10],
                "EX": [1.0, 1.5],
                "KI": [0.0, 0.7],
                "KG": [0.0, 0.7],
                "A": [0.0, 2.9],
                "THETA": [0.0, 6.5],
                "CI": [0.0, 0.9],
                "CG": [0.98, 0.998],
                # "KERNEL": [1, 15],
            }
        ),
    },
    "gr1a": {
        "param_name": ["x1"],
        "param_range": OrderedDict(
            {
                "x1": [0.01, 3.5],
            }
        ),
    },
    "gr2m": {
        "param_name": ["x1", "x2"],
        "param_range": OrderedDict(
            {
                "x1": [140, 2640],
                "x2": [0.21, 1.31],
            }
        ),
    },
    "gr3j": {
        "param_name": ["x1", "x2", "x3"],
        "param_range": OrderedDict(
            {
                "x1": [-5.0, 5.0],
                "x2": [0.5, 800],
                "x3": [0.3, 2.9],
            }
        ),
    },
    "gr4j": {
        "param_name": ["x1", "x2", "x3", "x4"],
        "param_range": OrderedDict(
            {
                "x1": [100.0, 1200.0],
                "x2": [-5.0, 3.0],
                "x3": [20.0, 300.0],
                "x4": [1.1, 2.9],
            }
        ),
    },
    "gr5j": {
        "param_name": ["x1", "x2", "x3", "x4", "x5"],
        "param_range": OrderedDict(
            {
                "x1": [100.0, 1200.0],
                "x2": [-5.0, 3.0],
                "x3": [20.0, 300.0],
                "x4": [1.1, 2.9],
                "x5": [0, 1],
            }
        ),
    },
    "gr6j": {
        "param_name": ["x1", "x2", "x3", "x4", "x5", "x6"],
        "param_range": OrderedDict(
            {
                "x1": [100.0, 1200.0],
                "x2": [-5.0, 3.0],
                "x3": [20.0, 300.0],
                "x4": [1.1, 2.9],
                "x5": [0, 1],
                "x6": [1, 100],
            }
        ),
    },
    "hymod": {
        "param_name": ["cmax", "bexp", "alpha", "ks", "kq"],
        "param_range": OrderedDict(
            {
                "cmax": [1.0, 500.0],
                "bexp": [0.1, 2.0],
                "alpha": [0.1, 0.99],
                "ks": [0.001, 0.10],
                "kq": [0.1, 0.99],
            }
        ),
    },
    "dhf": {
        "param_name": [
            "S0",  # 表层蓄水容量
            "U0",  # 下层蓄水容量
            "D0",  # 深层蓄水容量
            "KC",  # 蒸发系数
            "KW",  # 下层流系数
            "K2",  # 渗透系数
            "KA",  # 总径流调节系数
            "G",  # 不透水面积比例
            "A",  # 表层蓄水指数
            "B",  # 下层蓄水指数
            "B0",  # 汇流参数
            "K0",  # 汇流参数
            "N",  # 汇流参数
            "DD",  # 汇流参数
            "CC",  # 汇流参数
            "COE",  # 汇流参数
            "DDL",  # 地下汇流参数
            "CCL",  # 地下汇流参数
        ],
        "param_range": OrderedDict(
            {
                "S0": [0.0, 50.0],  # 表层蓄水容量 (mm)
                "U0": [0.0, 90.0],  # 下层蓄水容量 (mm)
                "D0": [70.0, 160.0],  # 深层蓄水容量 (mm)
                "KC": [0.1, 0.9],  # 蒸发系数
                "KW": [0.0, 1.0],  # 下层流系数
                "K2": [0.2, 0.9],  # 渗透系数
                "KA": [0.7, 1.0],  # 总径流调节系数
                "G": [0.0, 1.0],  # 不透水面积比例
                "A": [0.0, 5.0],  # 表层蓄水指数
                "B": [1.0, 3.0],  # 下层蓄水指数
                "B0": [0.1, 2.0],  # 汇流参数
                "K0": [0.0, 0.8],  # 汇流参数
                "N": [2.0, 6.0],  # 汇流参数
                "DD": [0.5, 4.0],  # 汇流参数
                "CC": [0.5, 4.0],  # 汇流参数
                "COE": [0.0, 0.8],  # 汇流参数
                "DDL": [0.5, 4.0],  # 地下汇流参数
                "CCL": [0.5, 4.0],  # 地下汇流参数
            }
        ),
    },
    "xaj_slw": {
        "param_name": [
            "WUP",
            "WLP",
            "WDP",
            "SP",
            "FRP",
            "WM",
            "WUMx",
            "WLMx",
            "K",
            "B",
            "C",
            "IM",
            "SM",
            "EX",
            "KG",
            "KI",
            "CS",
            "CI",
            "CG",
            "LAG",
            "KK",
            "X",
            "MP",
            "QSP",
            "QIP",
            "QGP",
        ],
        "param_range": OrderedDict(
            {
                # Initial states and proportions
                "WUP": [0.0, 50.0],
                "WLP": [0.0, 60.0],
                "WDP": [0.0, 150.0],
                "SP": [0.0, 10.0],
                "FRP": [0.0, 1.0],
                # Generation parameters
                "WM": [80.0, 220.0],
                "WUMx": [0.05, 0.5],
                "WLMx": [0.5, 0.95],
                "K": [0.3, 1.2],
                "B": [0.05, 0.5],
                "C": [0.05, 0.35],
                "IM": [0.005, 0.1],
                "SM": [10.0, 120.0],
                "EX": [1.0, 2.0],
                "KG": [0.05, 0.7],
                "KI": [0.05, 0.7],
                # Routing/recession
                "CS": [0.1, 0.9],
                "CI": [0.3, 0.95],
                "CG": [0.95, 0.999],
                "LAG": [0.0, 10.0],
                "KK": [1.0, 15.0],
                "X": [0.0, 0.5],
                "MP": [1.0, 5.0],
                # Initial flows for routing
                "QSP": [0.0, 50.0],
                "QIP": [0.0, 50.0],
                "QGP": [0.0, 50.0],
            }
        ),
    },
}
