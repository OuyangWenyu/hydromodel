#!/usr/bin/env python3
"""
Direct comparison script for DHF Python vs Java implementation
"""
from datetime import datetime
import json
import numpy as np
import calendar
from pathlib import Path
import sys

# Add the hydromodel to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydromodel.models.dhf import dhf


def _get_pet_from_es(es, dt_list, time_interval=1.0):
    """
    Convert monthly ES (evaporation) data to hourly PET values

    Args:
        es (list): Monthly ES values [Jan, Feb, ..., Dec]
        dt_list (list): List of datetime strings in format "%Y-%m-%d %H:%M:%S"
        time_interval (float): Time interval in hours (default 1.0)

    Returns:
        list: Hourly PET values matching dt_list length
    """
    pet_list = []

    for dt_str in dt_list:
        # Parse datetime to get month
        dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        month = dt_obj.month

        # Get monthly evaporation and convert to hourly
        # Based on Java formula: Ep = ES[Month-1] / days_in_month * timeIn / 24
        monthly_et = es[month - 1]  # ES is 0-indexed for months
        days_in_month = calendar.monthrange(dt_obj.year, month)[1]
        # keep the same as Java
        days_in_month = 30
        daily_et = monthly_et / days_in_month
        hourly_et = daily_et * time_interval / 24.0

        pet_list.append(hourly_et)

    return pet_list


def load_java_data():
    """Load the Java DHF data"""
    data_dir = Path(__file__).parent.parent / "data"
    data_file = data_dir / "dhf_data.json"

    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_data(time_data):
    """Prepare data exactly as in Java"""
    # use the same time steps as in Java
    dt_list_str = json.loads(time_data["dt"])
    dt_list = [
        datetime.strptime(dt, "%Y-%m-%d %H:%M:%S") for dt in dt_list_str
    ]
    time_steps = len(dt_list)
    num_basins = 1

    prcp = json.loads(time_data["rain"])
    es = json.loads(time_data["ES"])

    # Convert ES to PET using the function
    pet = _get_pet_from_es(es, dt_list_str)

    # Create [time, basin, feature] array
    prcp_array = np.array(prcp).reshape(
        time_steps, num_basins, 1
    )  # [time, basin, feature]
    pet_array = np.array(pet).reshape(
        time_steps, num_basins, 1
    )  # [time, basin, feature]

    # Stack precipitation and PET
    p_and_e = np.concatenate(
        [prcp_array, pet_array], axis=2
    )  # [time, basin, 2]

    return p_and_e


def prepare_parameters(java_params):
    """Prepare parameters exactly as in Java"""
    # DHF parameter order (must match the model implementation)
    param_order = [
        "S0",
        "U0",
        "D0",
        "K",
        "KW",
        "K2",
        "KA",
        "G",
        "A",
        "B",
        "B0",
        "K0",
        "N",
        "DD",
        "CC",
        "COE",
        "DDL",
        "CCL",
    ]

    # Create parameters for single basin
    num_basins = 1
    parameters = np.zeros((num_basins, len(param_order)))

    for i, param_name in enumerate(param_order):
        if param_name in java_params:
            parameters[:, i] = java_params[param_name]
        else:
            parameters[:, i] = 1.0

    return parameters


def run_dhf_comparison():
    """Run DHF model with Java data and compare"""
    print("Loading Java DHF data...")
    java_data = load_java_data()

    print("Preparing input data...")
    p_and_e = prepare_data(java_data)
    parameters = prepare_parameters(java_data)
    time_interval = 1.0

    print(f"Input data shape: {p_and_e.shape}")
    print(f"Parameters shape: {parameters.shape}")
    print(f"Time interval: {time_interval} hours")

    # Print first few input values
    print(f"\nFirst 5 precipitation values: {p_and_e[:5, 0, 0]}")
    print(f"First 5 PET values: {p_and_e[:5, 0, 1]}")

    # Print parameters
    param_names = [
        "S0",
        "U0",
        "D0",
        "K",
        "KW",
        "K2",
        "KA",
        "G",
        "A",
        "B",
        "B0",
        "K0",
        "N",
        "DD",
        "CC",
        "COE",
        "DDL",
        "CCL",
    ]
    print(f"\nParameters:")
    for i, name in enumerate(param_names):
        print(f"  {name}: {parameters[0, i]}")

    print("\nRunning DHF model...")
    result = dhf(
        p_and_e=p_and_e,
        parameters=parameters,
        warmup_length=0,
        return_state=False,
        normalized_params=False,
        time_interval_hours=time_interval,
        main_channel_length=8.0,
        basin_area=35.0,
    )

    print(f"\n=== DHF Python Results ===")
    print(f"Output shape: {result.shape}")
    print(f"Output range: {np.min(result):.6f} - {np.max(result):.6f}")
    print(f"First 10 output values:")
    for i in range(min(10, result.shape[0])):
        print(f"  Step {i+1}: {result[i, 0, 0]:.6f}")

    # Also run with return_state=True to see internal variables
    print("\nRunning with return_state=True...")
    results = dhf(
        p_and_e=p_and_e,
        parameters=parameters,
        warmup_length=720,
        return_state=True,
        normalized_params=False,
        time_interval_hours=time_interval,
        main_channel_length=8.0,
        basin_area=35.0,
    )

    q_sim, runoff_sim, y0, yu, yl, y, sa, ua, ya = results

    print(f"\nInternal state variables (first 5 values):")
    print(f"  q_sim (discharge): {q_sim[:5, 0, 0]}")
    print(f"  runoff_sim (total runoff): {runoff_sim[:5, 0, 0]}")
    print(f"  y0 (impervious runoff): {y0[:5, 0, 0]}")
    print(f"  yu (surface runoff): {yu[:5, 0, 0]}")
    print(f"  yl (subsurface runoff): {yl[:5, 0, 0]}")
    print(f"  y (total runoff): {y[:5, 0, 0]}")
    print(f"  sa (surface storage): {sa[:5, 0, 0]}")
    print(f"  ua (subsurface storage): {ua[:5, 0, 0]}")
    print(f"  ya (precedent rain): {ya[:5, 0, 0]}")
    return result


if __name__ == "__main__":
    result = run_dhf_comparison()
