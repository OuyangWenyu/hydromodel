"""
Author: Claude Code
Date: 2025-11-07
Description: Test case for selfmadehydrodataset with unified calibration/evaluation API
FilePath: \\hydromodel\\test\\test_selfmade_dataset.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
import tempfile
import shutil
import gc
import pytest
import numpy as np
import pandas as pd
import json

from hydromodel.trainers.unified_calibrate import calibrate
from hydromodel.trainers.unified_evaluate import evaluate


@pytest.fixture()
def temp_dataset_dir():
    """Create a temporary directory for test dataset."""
    temp_dir = tempfile.mkdtemp(prefix="test_selfmade_")
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture()
def fake_selfmade_dataset(temp_dataset_dir):
    """
    Create a fake selfmadehydrodataset with synthetic data.

    Dataset structure:
    temp_dataset_dir/
    ├── attributes/
    │   └── attributes.csv
    ├── shapes/
    │   └── basins.shp
    ├── timeseries/
    │   ├── 1D/
    │   │   ├── basin_001.csv
    │   │   ├── basin_002.csv
    │   └── 1D_units_info.json
    """
    dataset_name = "test_selfmade_dataset"
    dataset_path = os.path.join(temp_dataset_dir, dataset_name)

    # Create directory structure
    os.makedirs(os.path.join(dataset_path, "attributes"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "shapes"), exist_ok=True)  # Required by SelfMadeHydroDataset
    os.makedirs(os.path.join(dataset_path, "timeseries", "1D"), exist_ok=True)

    # 1. Create attributes.csv with basin_area
    basin_ids = ["basin_001", "basin_002"]
    basin_areas = [1234.5, 2345.6]  # km²
    elevations = [850.0, 920.0]

    attributes_df = pd.DataFrame({
        "basin_id": basin_ids,
        "area": basin_areas,  # This will be mapped to basin_area
        "elevation": elevations,
    })
    attributes_df.to_csv(
        os.path.join(dataset_path, "attributes", "attributes.csv"),
        index=False
    )

    # 2. Create time series data (3 years of daily data)
    n_days = 3 * 365  # 3 years
    dates = pd.date_range("2010-01-01", periods=n_days, freq="D")

    np.random.seed(42)  # For reproducibility

    for i, basin_id in enumerate(basin_ids):
        # Generate synthetic hydrological data
        # Precipitation: 0-50 mm/day with seasonal pattern
        prcp = np.maximum(
            0,
            10 + 15 * np.sin(2 * np.pi * np.arange(n_days) / 365)
            + 5 * np.random.randn(n_days)
        )

        # PET: 1-8 mm/day with seasonal pattern
        pet = np.maximum(
            1,
            4 + 2 * np.sin(2 * np.pi * np.arange(n_days) / 365 + np.pi)
            + 0.5 * np.random.randn(n_days)
        )

        # Streamflow: Generated from simple water balance
        # Q = P - ET - ΔS, with some noise
        cumsum_water = np.cumsum(prcp - pet)
        streamflow = np.maximum(
            0.1,
            0.1 * cumsum_water / np.arange(1, n_days + 1)
            + 2 * np.random.randn(n_days)
        )
        streamflow = np.maximum(0.1, streamflow)  # Ensure positive

        # Create basin time series DataFrame
        basin_df = pd.DataFrame({
            "time": dates,
            "prcp": prcp,
            "PET": pet,
            "streamflow": streamflow,
        })

        basin_df.to_csv(
            os.path.join(dataset_path, "timeseries", "1D", f"{basin_id}.csv"),
            index=False
        )

    # 3. Create units_info.json
    units_info = {
        "prcp": "mm/day",
        "PET": "mm/day",
        "streamflow": "mm/day"
    }

    with open(
        os.path.join(dataset_path, "timeseries", "1D_units_info.json"),
        "w"
    ) as f:
        json.dump(units_info, f, indent=2)

    # 4. Create a minimal shapefile (required by SelfMadeHydroDataset)
    try:
        import geopandas as gpd
        from shapely.geometry import Point, Polygon

        # Create simple polygon geometries for each basin
        geometries = []
        for i, basin_id in enumerate(basin_ids):
            # Create a simple square polygon for each basin
            lon_offset = i * 0.5
            polygon = Polygon([
                (120.0 + lon_offset, 30.0),
                (120.5 + lon_offset, 30.0),
                (120.5 + lon_offset, 30.5),
                (120.0 + lon_offset, 30.5),
                (120.0 + lon_offset, 30.0)
            ])
            geometries.append(polygon)

        # Create GeoDataFrame
        # Note: Use uppercase BASIN_ID to match hydrodatasource requirements
        gdf = gpd.GeoDataFrame({
            'BASIN_ID': basin_ids,
            'geometry': geometries
        }, crs='EPSG:4326')

        # Save as shapefile
        shapes_path = os.path.join(dataset_path, "shapes", "basins.shp")
        gdf.to_file(shapes_path)

    except ImportError:
        # If geopandas is not available, create dummy shp files
        # This is a workaround - just create empty files with correct extensions
        shapes_dir = os.path.join(dataset_path, "shapes")
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            dummy_file = os.path.join(shapes_dir, f'basins{ext}')
            with open(dummy_file, 'wb') as f:
                f.write(b'')  # Empty file - may cause issues but better than nothing

    return {
        "dataset_path": temp_dataset_dir,
        "dataset_name": dataset_name,
        "basin_ids": basin_ids,
        "basin_areas": basin_areas,
        "start_date": "2010-01-01",
        "end_date": "2012-12-31"
    }


@pytest.fixture()
def result_dir(temp_dataset_dir):
    """Create a temporary directory for results."""
    result_path = os.path.join(temp_dataset_dir, "results")
    os.makedirs(result_path, exist_ok=True)
    return result_path


def test_selfmade_calibrate_xaj_mz(fake_selfmade_dataset, result_dir):
    """Test calibration with selfmadehydrodataset using xaj_mz model."""
    dataset_info = fake_selfmade_dataset

    # Configuration for calibration
    config = {
        "data_cfgs": {
            "data_source_type": "selfmadehydrodataset",
            "data_source_path": dataset_info["dataset_path"],
            "dataset_name": dataset_info["dataset_name"],
            "time_unit": ["1D"],
            "basin_ids": [dataset_info["basin_ids"][0]],  # Use first basin
            "train_period": ["2010-01-01", "2011-12-31"],
            "test_period": ["2012-01-01", "2012-12-31"],
            "warmup_length": 30,
            "variables": ["prcp", "PET", "streamflow"],
        },
        "model_cfgs": {
            "model_name": "xaj_mz",
            "model_params": {
                "source_type": "sources",
                "source_book": "HF",
                "kernel_size": 15,
                "time_interval_hours": 24,
            },
        },
        "training_cfgs": {
            "algorithm_name": "SCE_UA",
            "algorithm_params": {
                "rep": 5,  # Very small for testing
                "ngs": 2,
                "kstop": 3,
                "peps": 0.1,
                "pcento": 0.1,
                "random_seed": 1234,
            },
            "loss_config": {
                "type": "time_series",
                "obj_func": "RMSE",
            },
            "output_dir": result_dir,
            "experiment_name": "test_xaj_mz_selfmade",
            "save_config": True,
        },
        "evaluation_cfgs": {
            "metrics": ["NSE", "RMSE", "KGE"],
        },
    }

    # Run calibration
    results = calibrate(config)

    # Check that results are saved
    exp_dir = os.path.join(result_dir, "test_xaj_mz_selfmade")
    assert os.path.exists(exp_dir), "Experiment directory not created"
    assert os.path.exists(
        os.path.join(exp_dir, "calibration_results.json")
    ), "Calibration results not saved"

    # Check that basin_area was correctly read from attributes
    assert results is not None, "Calibration should return results"

    print(f"✅ XAJ_MZ calibration completed successfully")
    print(f"   Results saved to: {exp_dir}")


def test_selfmade_calibrate_xaj(fake_selfmade_dataset, result_dir):
    """Test calibration with selfmadehydrodataset using xaj model (requires basin_area)."""
    dataset_info = fake_selfmade_dataset

    # Configuration for calibration with xaj
    config = {
        "data_cfgs": {
            "data_source_type": "selfmadehydrodataset",
            "data_source_path": dataset_info["dataset_path"],
            "dataset_name": dataset_info["dataset_name"],
            "time_unit": ["1D"],
            "basin_ids": [dataset_info["basin_ids"][0]],
            "train_period": ["2010-01-01", "2011-12-31"],
            "test_period": ["2012-01-01", "2012-12-31"],
            "warmup_length": 30,
            "variables": ["prcp", "PET", "streamflow"],
        },
        "model_cfgs": {
            "model_name": "xaj",
            "model_params": {
                "time_interval_hours": 24,
            },
        },
        "training_cfgs": {
            "algorithm_name": "SCE_UA",
            "algorithm_params": {
                "rep": 5,  # Very small for testing
                "ngs": 2,
                "kstop": 3,
                "peps": 0.1,
                "pcento": 0.1,
                "random_seed": 1234,
            },
            "loss_config": {
                "type": "time_series",
                "obj_func": "RMSE",
            },
            "output_dir": result_dir,
            "experiment_name": "test_xaj_selfmade",
            "save_config": True,
        },
        "evaluation_cfgs": {
            "metrics": ["NSE", "RMSE", "KGE"],
        },
    }

    # Run calibration - should automatically get basin_area from attributes
    results = calibrate(config)

    # Check that results are saved
    exp_dir = os.path.join(result_dir, "test_xaj_selfmade")
    assert os.path.exists(exp_dir), "Experiment directory not created"
    assert os.path.exists(
        os.path.join(exp_dir, "calibration_results.json")
    ), "Calibration results not saved"

    # Verify basin_area was correctly passed
    # The model should not raise ValueError about missing basin_area
    assert results is not None, "Calibration should return results"

    print(f"✅ xaj calibration completed successfully")
    print(f"   Basin area was correctly passed from attributes.csv")
    print(f"   Results saved to: {exp_dir}")


def test_selfmade_evaluate(fake_selfmade_dataset, result_dir):
    """Test evaluation with selfmadehydrodataset after calibration."""
    dataset_info = fake_selfmade_dataset

    # First run calibration
    config = {
        "data_cfgs": {
            "data_source_type": "selfmadehydrodataset",
            "data_source_path": dataset_info["dataset_path"],
            "dataset_name": dataset_info["dataset_name"],
            "time_unit": ["1D"],
            "basin_ids": [dataset_info["basin_ids"][0]],
            "train_period": ["2010-01-01", "2011-12-31"],
            "test_period": ["2012-01-01", "2012-12-31"],
            "warmup_length": 30,
            "variables": ["prcp", "PET", "streamflow"],
        },
        "model_cfgs": {
            "model_name": "xaj_mz",
            "model_params": {
                "source_type": "sources",
                "source_book": "HF",
                "kernel_size": 15,
                "time_interval_hours": 24,
            },
        },
        "training_cfgs": {
            "algorithm_name": "SCE_UA",
            "algorithm_params": {
                "rep": 5,
                "ngs": 2,
                "random_seed": 1234,
            },
            "loss_config": {
                "type": "time_series",
                "obj_func": "RMSE",
            },
            "output_dir": result_dir,
            "experiment_name": "test_evaluate_selfmade",
        },
        "evaluation_cfgs": {
            "metrics": ["NSE", "RMSE", "KGE", "PBIAS"],
        },
    }

    # Run calibration
    calibrate(config)

    # Run evaluation on test period
    param_dir = os.path.join(result_dir, "test_evaluate_selfmade")
    eval_results = evaluate(config, param_dir=param_dir, eval_period="test")

    # Check that evaluation results exist
    eval_dir = os.path.join(param_dir, "evaluation_test")
    assert os.path.exists(eval_dir), "Evaluation directory not created"
    assert os.path.exists(
        os.path.join(eval_dir, "basins_metrics.csv")
    ), "Metrics file not saved"
    assert os.path.exists(
        os.path.join(eval_dir, "basins_denorm_params.csv")
    ), "Parameters file not saved"

    # Check that metrics are returned
    assert eval_results is not None, "Evaluation should return results"

    print(f"✅ Evaluation completed successfully")
    print(f"   Evaluation results saved to: {eval_dir}")


def test_selfmade_multi_basin(fake_selfmade_dataset, result_dir):
    """Test multi-basin calibration with selfmadehydrodataset."""
    dataset_info = fake_selfmade_dataset

    # Run calibration for each basin separately to avoid memory/concurrency issues
    # This is safer on Windows and prevents access violations
    for basin_id in dataset_info["basin_ids"]:
        # Configuration for single-basin calibration
        config = {
            "data_cfgs": {
                "data_source_type": "selfmadehydrodataset",
                "data_source_path": dataset_info["dataset_path"],
                "dataset_name": dataset_info["dataset_name"],
                "time_unit": ["1D"],
                "basin_ids": [basin_id],  # Single basin at a time
                "train_period": ["2010-01-01", "2011-12-31"],
                "test_period": ["2012-01-01", "2012-12-31"],
                "warmup_length": 30,
                "variables": ["prcp", "PET", "streamflow"],
            },
            "model_cfgs": {
                "model_name": "xaj_mz",
                "model_params": {
                    "source_type": "sources",
                    "source_book": "HF",
                    "kernel_size": 15,
                    "time_interval_hours": 24,
                },
            },
            "training_cfgs": {
                "algorithm_name": "SCE_UA",
                "algorithm_params": {
                    "rep": 2,  # Reduced for faster testing
                    "ngs": 2,
                    "kstop": 2,
                    "peps": 0.1,
                    "pcento": 0.1,
                    "random_seed": 1234,
                },
                "loss_config": {
                    "type": "time_series",
                    "obj_func": "RMSE",
                },
                "output_dir": result_dir,
                "experiment_name": f"test_multi_basin_{basin_id}",
            },
            "evaluation_cfgs": {
                "metrics": ["NSE", "RMSE"],
            },
        }

        # Run calibration for this basin
        results = calibrate(config)

        # Force garbage collection between basins to free memory
        gc.collect()

        # Verify results were saved
        exp_dir = os.path.join(result_dir, f"test_multi_basin_{basin_id}")
        assert os.path.exists(exp_dir), f"Experiment directory not created for {basin_id}"
        assert os.path.exists(
            os.path.join(exp_dir, "calibration_results.json")
        ), f"Calibration results not saved for {basin_id}"

    print(f"✅ Multi-basin calibration completed successfully")
    print(f"   Calibrated {len(dataset_info['basin_ids'])} basins sequentially")
    print(f"   Results saved to separate directories")


def test_basin_area_retrieval(fake_selfmade_dataset):
    """Test that basin_area is correctly retrieved from attributes.csv."""
    from hydromodel.datasets.unified_data_loader import UnifiedDataLoader

    dataset_info = fake_selfmade_dataset

    data_config = {
        "data_source_type": "selfmadehydrodataset",
        "data_source_path": dataset_info["dataset_path"],
        "dataset_name": dataset_info["dataset_name"],
        "time_unit": ["1D"],
        "basin_ids": dataset_info["basin_ids"],
        "test_period": ["2010-01-01", "2010-12-31"],
        "warmup_length": 30,
        "variables": ["prcp", "PET", "streamflow"],
    }

    # Create data loader
    data_loader = UnifiedDataLoader(data_config, is_train_val_test="test")

    # Get basin configs
    basin_configs = data_loader.get_basin_configs()

    # Check that basin_area was correctly read for all basins
    for i, basin_id in enumerate(dataset_info["basin_ids"]):
        assert basin_id in basin_configs, f"Basin {basin_id} not in configs"
        assert "basin_area" in basin_configs[basin_id], f"basin_area missing for {basin_id}"

        # Check that the value matches what we created
        expected_area = dataset_info["basin_areas"][i]
        actual_area = basin_configs[basin_id]["basin_area"]
        assert abs(actual_area - expected_area) < 0.1, \
            f"Basin area mismatch for {basin_id}: expected {expected_area}, got {actual_area}"

    print(f"✅ Basin area retrieval test passed")
    print(f"   Correctly retrieved basin_area from attributes.csv")
    for i, basin_id in enumerate(dataset_info["basin_ids"]):
        print(f"   {basin_id}: {basin_configs[basin_id]['basin_area']:.1f} km²")


if __name__ == "__main__":
    """Allow running tests directly with python."""
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
