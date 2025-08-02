"""
Common utility functions for hydromodel package.
This module contains shared functionality used across multiple model modules.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union


def save_dataframe_to_csv(
    df: pd.DataFrame,
    filepath: str,
    metadata_lines: Optional[List[str]] = None,
    encoding: str = "utf-8",
    float_format: str = "%.6f",
    **kwargs,
) -> None:
    """
    Save DataFrame to CSV file with optional metadata header.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filepath : str
        Output file path.
    metadata_lines : list of str, optional
        Optional metadata lines to write before CSV data.
    encoding : str, optional
        File encoding (default is "utf-8").
    float_format : str, optional
        Float formatting string (default is "%.6f").
    **kwargs
        Additional arguments passed to DataFrame.to_csv().
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Default CSV parameters
    csv_kwargs = {
        "index": False,
        "encoding": encoding,
        "float_format": float_format,
        "header": True,
    }
    csv_kwargs.update(kwargs)

    if metadata_lines:
        with open(filepath, "w", encoding=encoding, newline="") as f:
            f.write("\n".join(metadata_lines) + "\n")
            df.to_csv(f, **csv_kwargs)
    else:
        df.to_csv(filepath, **csv_kwargs)


def create_output_directory(output_dir: str, verbose: bool = True) -> str:
    """
    Create output directory if it doesn't exist.

    Parameters
    ----------
    output_dir : str
        Output directory path.
    verbose : bool, optional
        Whether to print information messages (default is True).

    Returns
    -------
    str
        Path to the created directory.

    Raises
    ------
    OSError
        If directory creation fails.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"📁 Output directory created/verified: {output_dir}")
        return output_dir
    except Exception as e:
        if verbose:
            print(f"❌ Failed to create output directory: {e}")
        raise


def safe_divide(
    numerator: Union[np.ndarray, float],
    denominator: Union[np.ndarray, float],
    fill_value: float = np.nan,
) -> Union[np.ndarray, float]:
    """
    Perform safe division avoiding division by zero errors.

    Parameters
    ----------
    numerator : np.ndarray or float
        Numerator values.
    denominator : np.ndarray or float
        Denominator values.
    fill_value : float, optional
        Value to use when division by zero occurs (default is np.nan).

    Returns
    -------
    np.ndarray or float
        Division result with fill_value for invalid operations.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
        if isinstance(result, np.ndarray):
            result[~np.isfinite(result)] = fill_value
        elif not np.isfinite(result):
            result = fill_value
    return result


def format_time_range(start_time: str, end_time: str) -> str:
    """
    Format time range as a string.

    Parameters
    ----------
    start_time : str
        Start time string.
    end_time : str
        End time string.

    Returns
    -------
    str
        Formatted time range string.
    """
    return f"{start_time} to {end_time}"


def get_file_size_mb(filepath: str) -> float:
    """
    Get file size in megabytes.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    float
        File size in MB, or 0.0 if file doesn't exist.
    """
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0


def print_progress(
    current: int,
    total: int,
    prefix: str = "Progress",
    suffix: str = "Complete",
    length: int = 50,
) -> None:
    """
    Print a progress bar to console.

    Parameters
    ----------
    current : int
        Current progress value.
    total : int
        Total expected value.
    prefix : str, optional
        Prefix text (default is "Progress").
    suffix : str, optional
        Suffix text (default is "Complete").
    length : int, optional
        Length of progress bar (default is 50).
    """
    percent = (current / total) * 100
    filled_length = int(length * current // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent:.1f}% {suffix}", end="")
    if current == total:
        print()


def merge_dicts_safe(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely merge multiple dictionaries.

    Parameters
    ----------
    *dicts : dict
        Dictionaries to merge. Later dictionaries override earlier ones.

    Returns
    -------
    dict
        Merged dictionary.
    """
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result
