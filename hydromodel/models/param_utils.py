"""
Author: Wenyu Ouyang
Date: 2025-08-19
Description: Unified parameter handling utilities for hydrological models
Copyright (c) 2025 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
from typing import Dict, Any, Union, List
from collections import OrderedDict


def detect_parameter_format(
    parameters: np.ndarray, param_ranges: Dict[str, List[float]]
) -> bool:
    """
    Detect whether parameters are normalized (0-1 range) or in original scale.

    Parameters
    ----------
    parameters : np.ndarray
        Model parameters array [basin, parameter]
    param_ranges : Dict[str, List[float]]
        Parameter ranges dictionary with min/max values for each parameter

    Returns
    -------
    bool
        True if parameters appear to be normalized (0-1 range), False otherwise
    """
    # Check if all parameters are within [0, 1] range
    # Allow small tolerance for numerical precision
    tolerance = 1e-6

    # If any parameter is outside [0, 1] with tolerance, assume original scale
    if np.any(parameters < -tolerance) or np.any(parameters > 1 + tolerance):
        return False

    # Additional check: if parameters are suspiciously close to range boundaries
    # when interpreted as original values, they're likely normalized
    param_values = list(param_ranges.values())
    for i, param_range in enumerate(param_values):
        if i >= parameters.shape[1]:
            break
        param_col = parameters[:, i]
        min_val, max_val = param_range[0], param_range[1]

        # If parameters are all very close to 0-1 range but the actual range
        # is much larger, they're likely normalized
        if max_val - min_val > 2 and np.all(param_col <= 1.1):
            return True

    return True  # Default assumption: parameters are normalized


def process_parameters(
    parameters: np.ndarray,
    param_ranges: Dict[str, List[float]],
    normalized: Union[bool, str] = "auto",
) -> np.ndarray:
    """
    Process model parameters to convert from normalized to original scale if needed.

    This function provides a unified interface for parameter handling across all models,
    supporting both normalized (0-1 range) and original scale parameters.

    Parameters
    ----------
    parameters : np.ndarray
        Model parameters array [basin, parameter]
    param_ranges : Dict[str, List[float]]
        Parameter ranges dictionary with min/max values for each parameter
    normalized : Union[bool, str], optional
        Parameter format specification:
        - "auto": Automatically detect parameter format (default)
        - True: Parameters are normalized (0-1 range), convert to original scale
        - False: Parameters are already in original scale, use as-is

    Returns
    -------
    np.ndarray
        Parameters in original scale, ready for model computation

    Examples
    --------
    >>> param_ranges = {"K": [0.1, 1.0], "B": [0.1, 0.4]}
    >>> # Normalized parameters
    >>> norm_params = np.array([[0.5, 0.8]])  # Will be converted to [0.55, 0.34]
    >>> orig_params = process_parameters(norm_params, param_ranges, normalized=True)
    >>>
    >>> # Original scale parameters
    >>> orig_params = np.array([[0.55, 0.34]])  # Will be used as-is
    >>> final_params = process_parameters(orig_params, param_ranges, normalized=False)
    """
    if parameters.shape[1] != len(param_ranges):
        raise ValueError(
            f"Parameter array has {parameters.shape[1]} columns but "
            f"param_ranges has {len(param_ranges)} parameters"
        )

    # Auto-detect parameter format if requested
    if normalized == "auto":
        normalized = detect_parameter_format(parameters, param_ranges)

    # If parameters are already in original scale, return as-is
    if not normalized:
        return parameters.copy()

    # Convert normalized parameters to original scale
    converted_params = np.zeros_like(parameters)
    param_list = list(param_ranges.values())

    for i, param_range in enumerate(param_list):
        min_val, max_val = param_range[0], param_range[1]
        converted_params[:, i] = min_val + parameters[:, i] * (
            max_val - min_val
        )

    return converted_params


def get_parameter_scales(
    param_ranges: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    """
    Extract parameter scales from param_ranges for backward compatibility.

    Parameters
    ----------
    param_ranges : Dict[str, List[float]]
        Parameter ranges dictionary

    Returns
    -------
    Dict[str, List[float]]
        Dictionary mapping parameter names to [min, max] ranges
    """
    return param_ranges.copy()


def validate_parameters(
    parameters: np.ndarray,
    param_ranges: Dict[str, List[float]],
    normalized: bool = False,
) -> bool:
    """
    Validate that parameters are within acceptable ranges.

    Parameters
    ----------
    parameters : np.ndarray
        Model parameters array [basin, parameter]
    param_ranges : Dict[str, List[float]]
        Parameter ranges dictionary
    normalized : bool, optional
        Whether parameters are normalized (default: False)

    Returns
    -------
    bool
        True if all parameters are within valid ranges
    """
    if normalized:
        # For normalized parameters, check 0-1 range
        return np.all(parameters >= 0) and np.all(parameters <= 1)
    else:
        # For original scale parameters, check against param_ranges
        param_list = list(param_ranges.values())
        for i, param_range in enumerate(param_list):
            if i >= parameters.shape[1]:
                break
            min_val, max_val = param_range[0], param_range[1]
            param_col = parameters[:, i]
            if np.any(param_col < min_val) or np.any(param_col > max_val):
                return False
        return True


def normalize_parameters(
    parameters: np.ndarray, param_ranges: Dict[str, List[float]]
) -> np.ndarray:
    """
    Convert parameters from original scale to normalized (0-1) scale.

    Parameters
    ----------
    parameters : np.ndarray
        Model parameters in original scale [basin, parameter]
    param_ranges : Dict[str, List[float]]
        Parameter ranges dictionary

    Returns
    -------
    np.ndarray
        Parameters normalized to 0-1 scale
    """
    normalized_params = np.zeros_like(parameters)
    param_list = list(param_ranges.values())

    for i, param_range in enumerate(param_list):
        min_val, max_val = param_range[0], param_range[1]
        normalized_params[:, i] = (parameters[:, i] - min_val) / (
            max_val - min_val
        )

    return normalized_params
