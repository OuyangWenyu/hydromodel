r"""
Author: zhuanglaihong
Date: 2025-11-02
LastEditTime: 2025-11-02 21:00:00
LastEditors: zhuanglaihong
Description: Simple script demonstrating how to use UnifiedSimulator for model simulation
FilePath: \hydromodel\scripts\run_xaj_simulate.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.

This script is a minimal example showing how to:
1. Load data and parameters
2. Create a UnifiedSimulator instance
3. Run simulation
4. Visualize or save basic results

Users can modify this script as a template for their own simulation needs.
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

# Add hydromodel to path
repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)

from hydromodel.datasets.unified_data_loader import UnifiedDataLoader  # noqa: E402
from hydromodel.trainers.unified_simulate import UnifiedSimulator  # noqa: E402
from hydromodel.models.model_config import read_model_param_dict  # noqa: E402


def load_parameters_from_csv(csv_file: str, param_names: list) -> OrderedDict:
    """
    Load best parameters from SCE-UA calibration results.

    NOTE: This function is ONLY compatible with SCE-UA algorithm output format.
    For other algorithms (GA, DDS, etc.), use YAML parameter files instead.

    Parameters
    ----------
    csv_file : str
        Path to *_sceua.csv file (SCE-UA specific format)
    param_names : list
        List of parameter names

    Returns
    -------
    OrderedDict
        Parameter dictionary
    """
    df = pd.read_csv(csv_file)
    if "like1" not in df.columns or len(df) == 0:
        raise ValueError(f"Invalid SCE-UA results file: {csv_file}")

    # Get best run (minimum loss)
    best_run = df.loc[df["like1"].idxmin()]
    params = OrderedDict()

    # Try par{name} format (e.g., parK, parB, ...)
    for name in param_names:
        col = f"par{name}"
        if col in df.columns:
            params[name] = float(best_run[col])
        else:
            raise ValueError(f"Parameter column '{col}' not found in CSV")

    return params


def load_parameters_from_yaml(yaml_file: str, model_name: str = "xaj") -> OrderedDict:
    """
    Load parameters from YAML file (supports both param_range and params formats).

    Supported formats:

    Format 1 - Single values (params):
    xaj:
      param_name: [K, B, ...]
      params:
        K: 0.75
        B: 0.25
        ...

    Format 2 - Ranges (param_range):
    xaj:
      param_name: [K, B, ...]
      param_range:
        K: [0.1, 1.0]
        B: [0.1, 0.4]
        ...

    When using param_range format, the function extracts the middle value of each range.

    Parameters
    ----------
    yaml_file : str
        Path to YAML parameter file
    model_name : str
        Model name (default: "xaj")

    Returns
    -------
    OrderedDict
        Parameter dictionary
    """
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if model_name not in data:
        raise ValueError(
            f"Model '{model_name}' not found in parameter file. "
            f"Available models: {list(data.keys())}"
        )

    model_data = data[model_name]
    params = OrderedDict()

    # Check which format is used
    if "params" in model_data:
        # Format 1: Single parameter values
        param_values = model_data["params"]
        param_names = model_data.get("param_name", param_values.keys())

        for param_name in param_names:
            if param_name not in param_values:
                raise ValueError(f"Parameter '{param_name}' not found in params")

            value = param_values[param_name]
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Invalid value for {param_name}: {value}. "
                    f"Expected a single numeric value"
                )
            params[param_name] = float(value)

        print("  Using 'params' format (single values)")

    elif "param_range" in model_data:
        # Format 2: Parameter ranges - extract middle value
        param_ranges = model_data["param_range"]
        param_names = model_data.get("param_name", param_ranges.keys())

        for param_name in param_names:
            if param_name not in param_ranges:
                raise ValueError(f"Parameter '{param_name}' not found in param_range")

            range_vals = param_ranges[param_name]
            if not isinstance(range_vals, list) or len(range_vals) != 2:
                raise ValueError(
                    f"Invalid range for {param_name}: {range_vals}. "
                    f"Expected [min, max] format"
                )

            # Use middle value of the range
            params[param_name] = (range_vals[0] + range_vals[1]) / 2.0

        print("  Using 'param_range' format (taking middle values)")

    else:
        raise ValueError(
            f"Neither 'params' nor 'param_range' found in parameter file. "
            f"Expected format: {model_name}: {{param_name: [...], params: {{...}}}} "
            f"or {model_name}: {{param_name: [...], param_range: {{...}}}}"
        )

    return params


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simple XAJ model simulation using UnifiedSimulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use single parameter values (recommended for simulation)
  python run_xaj_simulate.py --config configs/example_simulate_config.yaml \\
         --param-file configs/example_xaj_params_values.yaml

  # Use calibration results (param_range format - will use middle values)
  python run_xaj_simulate.py --config results/xaj_SCE_UA/calibration_config.yaml \\
         --param-file results/xaj_SCE_UA/param_range.yaml

  # Use example parameters with ranges (for demonstration)
  python run_xaj_simulate.py --config configs/example_simulate_config.yaml \\
         --param-file configs/example_xaj_params.yaml

  # Use SCE-UA CSV results (legacy format)
  python run_xaj_simulate.py --config results/xaj_SCE_UA/calibration_config.yaml \\
         --param-file results/xaj_SCE_UA/01013500_sceua.csv

  # Override warmup manually and show plot
  python run_xaj_simulate.py --config configs/example_simulate_config.yaml \\
         --param-file configs/example_xaj_params.yaml --warmup 30 --plot

Parameter File Formats:
  YAML Format 1 (recommended): Single parameter values
    xaj:
      param_name: [K, B, IM, ...]
      params:
        K: 0.75
        B: 0.25
        IM: 0.06
        ...

  YAML Format 2: Parameter ranges (from calibration)
    xaj:
      param_name: [K, B, IM, ...]
      param_range:
        K: [0.1, 1.0]
        B: [0.1, 0.4]
        ...
    The script takes the middle value of each range.

  CSV (legacy): SCE-UA specific format (*.csv)
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/example_simulate_config.yaml",
        help="Configuration file (default: configs/example_simulate_config.yaml)",
    )

    parser.add_argument(
        "--param-file",
        type=str,
        required=True,
        help="Parameter file: YAML (param_range format) or CSV (SCE-UA legacy)",
    )

    parser.add_argument(
        "--basin-id",
        type=str,
        help="Basin ID to simulate (default: first basin in config)",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file for results (optional)",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plot of simulation vs observation",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Warmup period (default: from config or auto-detected based on data type)",
    )

    return parser.parse_args()


def main():
    """Main simulation function."""
    args = parse_arguments()

    print("=" * 80)
    print("XAJ Model Simulation using UnifiedSimulator")
    print("=" * 80)

    # ========================================================================
    # Step 1: Load configuration
    # ========================================================================
    print(f"\n[1/4] Loading configuration from: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_config = config["data_cfgs"]
    model_cfgs = config["model_cfgs"]
    model_name = model_cfgs.get("model_name")

    basin_ids = data_config.get("basin_ids", [])
    basin_id = args.basin_id or basin_ids[0]
    basin_index = basin_ids.index(basin_id)

    # Determine warmup period and time unit
    data_source_type = data_config.get("data_source_type", "")
    time_unit_config = data_config.get("time_unit", ["1D"])

    # Check if this is hourly/sub-daily data (event data typically uses hourly)
    is_hourly_data = any(unit in ["1h", "3h", "6h", "12h"] for unit in time_unit_config)
    is_event_data = data_source_type == "floodevent" or is_hourly_data

    if args.warmup is not None:
        # User specified warmup via command line (highest priority)
        warmup_length = args.warmup
    elif "warmup_length" in data_config:
        # Use warmup from config (second priority)
        warmup_length = data_config["warmup_length"]
    else:
        # Auto-detect based on data type (last resort)
        if is_event_data:
            warmup_length = 15  # hours for event/hourly data
        else:
            warmup_length = 365  # days for continuous daily data

    # Determine time unit for display
    time_unit = "hours" if is_event_data else "days"

    print(f"  Model: {model_name}")
    print(f"  Basin: {basin_id} (index {basin_index})")
    print(f"  Period: {data_config.get('test_period')}")
    print(f"  Data type: {data_source_type}")
    print(f"  Warmup: {warmup_length} {time_unit}")

    # ========================================================================
    # Step 2: Load parameters
    # ========================================================================
    print(f"\n[2/4] Loading parameters from: {args.param_file}")

    # Get parameter names from MODEL_PARAM_DICT
    param_range = read_model_param_dict(None)  # Use default
    param_names = param_range[model_name]["param_name"]

    # Load parameters based on file type
    if args.param_file.endswith(".csv"):
        parameters = load_parameters_from_csv(args.param_file, param_names)
    elif args.param_file.endswith((".yaml", ".yml")):
        parameters = load_parameters_from_yaml(args.param_file, model_name)
    else:
        raise ValueError("Parameter file must be .csv or .yaml")

    print("  Parameters:")
    for name, value in parameters.items():
        print(f"    {name:8s} = {value:.6f}")

    # ========================================================================
    # Step 3: Load data and create UnifiedSimulator
    # ========================================================================
    print(f"\n[3/4] Loading data and initializing simulator")

    # Load data using UnifiedDataLoader
    data_loader = UnifiedDataLoader(data_config, is_train_val_test="test")
    p_and_e, qobs = data_loader.load_data()

    print(f"  Input shape: {p_and_e.shape}")
    print(f"  Qobs shape: {qobs.shape}")

    # Extract data for target basin
    basin_inputs = p_and_e[:, basin_index : basin_index + 1, :]
    basin_qobs = qobs[:, basin_index : basin_index + 1, :]

    # Create model configuration for UnifiedSimulator
    model_config = {
        "model_name": model_name,
        "model_params": model_cfgs,
        "parameters": parameters,
    }

    # Get basin configuration
    basin_configs = data_loader.get_basin_configs()
    basin_config = basin_configs.get(basin_id)

    # Create UnifiedSimulator instance
    simulator = UnifiedSimulator(model_config, basin_config)
    print(" UnifiedSimulator initialized")

    # ========================================================================
    # Step 4: Run simulation using UnifiedSimulator
    # ========================================================================
    print(f"\n[4/4] Running simulation (warmup={warmup_length} {time_unit})")

    # Call the unified simulate interface
    sim_results = simulator.simulate(
        inputs=basin_inputs,
        qobs=basin_qobs,
        warmup_length=warmup_length,
        return_intermediate=False,
    )

    # Extract results (UnifiedSimulator returns model-specific output names)
    # For XAJ models, the main output is "qsim"
    qsim = sim_results["qsim"]  # [time, 1, 1]
    qobs_out = basin_qobs  # Use original qobs

    # Remove warmup period
    qsim_eval = qsim[warmup_length :, 0, 0]
    qobs_eval = qobs_out[warmup_length :, 0, 0]

    # Get time array - handle both continuous and flood event data
    if data_loader.ds is not None:
        # Continuous data: use xarray dataset
        times = data_loader.ds["time"].data[warmup_length :]
    elif hasattr(data_loader, "time_array"):
        # Flood event data: use time_array
        times = data_loader.time_array[warmup_length :]
    else:
        # Fallback: generate time index
        times = np.arange(len(qsim_eval))
        print("  Warning: No time information available, using index instead")

    print(f" Simulation completed ({len(qsim_eval)} time steps)")

    # Calculate basic statistics
    from hydroutils import hydro_stat

    # Ensure both arrays have same shape and are 2D [1, time]
    qsim_2d = qsim_eval.flatten().reshape(1, -1)
    qobs_2d = qobs_eval.flatten().reshape(1, -1)

    # Debug: print shapes
    print(f"  qsim shape: {qsim_2d.shape}, qobs shape: {qobs_2d.shape}")

    # Ensure same length
    min_len = min(qsim_2d.shape[1], qobs_2d.shape[1])
    qsim_2d = qsim_2d[:, :min_len]
    qobs_2d = qobs_2d[:, :min_len]

    metrics = hydro_stat.stat_error(qobs_2d, qsim_2d)

    print("\n" + "=" * 80)
    print("Simulation Results")
    print("=" * 80)
    print(f"Basin: {basin_id}")
    print(f"Time steps: {len(qsim_eval)}")
    print("\nPerformance Metrics:")
    for name, value in metrics.items():
        if isinstance(value, np.ndarray):
            value = value.flatten()[0]
        print(f"  {name:10s} = {value:8.4f}")
    print("=" * 80)

    # ========================================================================
    # Optional: Save results to CSV
    # ========================================================================
    if args.output:
        results_df = pd.DataFrame(
            {
                "time": times,
                "qsim": qsim_eval,
                "qobs": qobs_eval,
                "prcp": basin_inputs[warmup_length :, 0, 0],
                "pet": basin_inputs[warmup_length :, 0, 1],
            }
        )
        results_df.to_csv(args.output, index=False)
        print(f"\n  Results saved to: {args.output}")

    # ========================================================================
    # Optional: Plot results
    # ========================================================================
    if args.plot:
        plt.figure(figsize=(12, 6))
        plt.plot(times, qobs_eval, label="Observed", linewidth=1.5, alpha=0.7)
        plt.plot(times, qsim_eval, label="Simulated", linewidth=1.5, alpha=0.7)
        plt.xlabel("Time")
        plt.ylabel("Streamflow (mm/day)")
        plt.title(f"Basin {basin_id} - {model_name} Simulation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    print("\n Simulation completed successfully!")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
