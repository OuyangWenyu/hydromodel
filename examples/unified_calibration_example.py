"""
Example script demonstrating the new unified calibration interface.

This example shows how to use the refactored UnifiedModelSetup with data_config
instead of the old p_and_e and qobs interface.

Author: Wenyu Ouyang
Date: 2025-01-22
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydromodel.configs.unified_config import UnifiedConfig
from hydromodel.trainers.unified_calibrate import calibrate


def example_flood_event_calibration():
    """Example: XAJ model calibration with flood event data."""

    # Load configuration for flood event data
    config_file = "configs/examples/floodevent_xaj_example.yaml"
    config = UnifiedConfig(config_file=config_file)

    # Update data path to actual data location
    config.update_config(
        {
            "data_cfgs": {
                "data_path": "path/to/your/flood/event/data",
                "basin_ids": ["your_basin_id"],
            }
        }
    )

    print("Configuration loaded:")
    print(config)

    try:
        # Run calibration with the unified interface
        results = calibrate(config)

        print("Calibration completed successfully!")
        print(f"Results for basins: {list(results.keys())}")

        # Print best parameters for each basin
        for basin_id, result in results.items():
            print(f"\nBasin {basin_id}:")
            print(
                f"Best objective value: {result.get('best_objective', 'N/A')}"
            )
            print(f"Best parameters: {result.get('best_params', 'N/A')}")

    except Exception as e:
        print(f"Calibration failed with error: {e}")
        print(
            "This is expected if hydrodatasource is not available or data path is not valid."
        )


def example_continuous_calibration():
    """Example: XAJ model calibration with continuous time series data."""

    # Create configuration programmatically
    config_dict = {
        "data_cfgs": {
            "data_type": "camels",
            "data_path": "path/to/camels/data",
            "basin_ids": ["01013500"],
            "time_periods": {
                "calibration": ["2010-01-01", "2015-12-31"],
                "testing": ["2016-01-01", "2020-12-31"],
            },
            "variables": ["prcp", "PET", "streamflow"],
            "warmup_length": 365,
        },
        "model_cfgs": {
            "model_name": "xaj",
            "model_params": {
                "source_type": "sources",
                "time_interval_hours": 24,
            },
        },
        "training_cfgs": {
            "algorithm_name": "SCE_UA",
            "algorithm_params": {
                "rep": 100,  # Reduced for quick testing
                "ngs": 100,
            },
            "output_dir": "results/example_continuous",
            "experiment_name": "continuous_example",
        },
        "evaluation_cfgs": {
            "loss_type": "time_series",
            "objective_function": "RMSE",
        },
    }

    config = UnifiedConfig(config_dict=config_dict)

    print("Configuration created:")
    print(config)

    try:
        # Run calibration
        results = calibrate(config)
        print("Continuous calibration completed successfully!")

    except Exception as e:
        print(f"Continuous calibration failed with error: {e}")
        print("This is expected if data is not available.")


def demonstrate_unified_interface_advantages():
    """Demonstrate the advantages of the new unified interface approach."""

    print("=== New Unified Interface Advantages ===")
    print()

    print("1. Clean Three-Config Interface:")
    print("   - data_config: All data-related settings")
    print("   - model_config: All model-related settings")
    print(
        "   - training_config: All training-related settings (algorithm, loss, params)"
    )
    print()

    print("2. Simplified Function Signature:")
    print(
        "   Before: calibrate(data_config, model_config, algorithm_config, loss_config, output_dir, ...)"
    )
    print("   After:  calibrate(data_config, model_config, training_config)")
    print()

    print("3. Logical Configuration Grouping:")
    print("   - Loss function → training_config (it's the training objective)")
    print("   - Parameter ranges → training_config (training-related)")
    print(
        "   - Algorithm settings → training_config (obviously training-related)"
    )
    print(
        "   - Evaluation metrics → evaluation_config (post-training assessment)"
    )
    print()

    print(
        "4. Unified data loading through hydrodatasource.read_ts_xrdataset():"
    )
    print("   - Supports flood events (FloodEventDatasource)")
    print("   - Supports continuous data (Camels, SelfMadeHydroDataset)")
    print("   - Same interface for all data types")
    print()

    print("5. Event data handling for traditional models:")
    print("   - Automatically detects event segments")
    print("   - Runs model on each event separately")
    print("   - Combines results while maintaining timeline")
    print("   - Handles gaps between events properly")


if __name__ == "__main__":
    print("Unified Calibration Interface Examples")
    print("=====================================")

    # Demonstrate the conceptual advantages
    demonstrate_unified_interface_advantages()
    print()

    # Try flood event example
    print("--- Flood Event Calibration Example ---")
    example_flood_event_calibration()
    print()

    # Try continuous data example
    print("--- Continuous Data Calibration Example ---")
    example_continuous_calibration()
    print()

    print("Examples completed!")
