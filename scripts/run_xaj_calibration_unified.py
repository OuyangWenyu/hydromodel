"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-07 11:46:00
LastEditors: Wenyu Ouyang
Description: Unified interface script for XAJ model calibration using different algorithms
FilePath: \hydromodel\scripts\run_xaj_calibration_unified.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import json
import argparse
import shutil
import sys
import os
from pathlib import Path
import yaml
import numpy as np

# Add hydromodel to path
repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)

from hydromodel.datasets.data_preprocess import (
    _get_pe_q_from_ts,
    cross_val_split_tsdata,
)
from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.trainers.unified_calibrate import calibrate, DEAP_AVAILABLE


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="XAJ Model Calibration using Unified Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Types Supported:
  - xaj: XinAnJiang model (original version with routing)
  - xaj_mz: XinAnJiang model (with mizuroute routing)

Algorithm Types Supported:
  - SCE_UA: Shuffled Complex Evolution via spotpy  
  - genetic_algorithm: Genetic algorithm via DEAP (if installed)
  - scipy_minimize: SciPy optimization methods

Usage Examples:
  # XAJ model with SCE-UA (default)
  python run_xaj_calibration_unified.py --model-type xaj_mz --algorithm SCE_UA
  
  # XAJ model with Genetic Algorithm
  python run_xaj_calibration_unified.py --model-type xaj_mz --algorithm genetic_algorithm
  
  # XAJ model with scipy optimization
  python run_xaj_calibration_unified.py --model-type xaj_mz --algorithm scipy_minimize
        """,
    )

    # Data configuration
    parser.add_argument(
        "--data-type",
        type=str,
        default="selfmadehydrodataset",
        choices=["camels", "selfmadehydrodataset", "owndata"],
        help="Dataset type (default: selfmadehydrodataset)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="C:\\Users\\wenyu\\OneDrive\\data\\FD_sources",
        help="Data directory path",
    )

    parser.add_argument(
        "--result-dir",
        type=str,
        default=os.path.join(repo_path, "result"),
        help="Results directory path",
    )

    parser.add_argument(
        "--exp",
        type=str,
        default="exp_xaj_unified_001",
        help="Experiment name for result organization",
    )

    # Model configuration
    parser.add_argument(
        "--model-type",
        type=str,
        default="xaj_mz",
        choices=["xaj", "xaj_mz"],
        help="XAJ model variant (default: xaj_mz)",
    )

    parser.add_argument(
        "--source-type",
        type=str,
        default="sources",
        choices=["sources", "sources5mm"],
        help="Source data type (default: sources)",
    )

    parser.add_argument(
        "--source-book",
        type=str,
        default="HF",
        choices=["HF", "EH"],
        help="Source computation method: HF=Hydrological Forecast, EH=Engineering Hydrology (default: HF)",
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=15,
        help="Convolutional kernel size (default: 15)",
    )

    parser.add_argument(
        "--time-interval-hours",
        type=int,
        default=24,
        help="Time interval in hours (default: 24)",
    )

    # Algorithm configuration
    parser.add_argument(
        "--algorithm",
        type=str,
        default="SCE_UA",
        choices=["SCE_UA", "genetic_algorithm", "scipy_minimize"],
        help="Optimization algorithm (default: SCE_UA)",
    )

    # SCE-UA parameters
    parser.add_argument(
        "--rep",
        type=int,
        default=1000,
        help="SCE-UA repetitions (default: 1000)",
    )

    parser.add_argument(
        "--ngs",
        type=int,
        default=1000,
        help="SCE-UA number of complexes (default: 1000)",
    )

    parser.add_argument(
        "--kstop",
        type=int,
        default=50,
        help="SCE-UA evolution loops (default: 50)",
    )

    parser.add_argument(
        "--peps",
        type=float,
        default=0.1,
        help="SCE-UA convergence criterion (default: 0.1)",
    )

    parser.add_argument(
        "--pcento",
        type=float,
        default=0.1,
        help="SCE-UA convergence criterion (default: 0.1)",
    )

    # Genetic Algorithm parameters
    parser.add_argument(
        "--pop-size",
        type=int,
        default=80,
        help="GA population size (default: 80)",
    )

    parser.add_argument(
        "--n-generations",
        type=int,
        default=50,
        help="GA number of generations (default: 50)",
    )

    parser.add_argument(
        "--cx-prob",
        type=float,
        default=0.7,
        help="GA crossover probability (default: 0.7)",
    )

    parser.add_argument(
        "--mut-prob",
        type=float,
        default=0.2,
        help="GA mutation probability (default: 0.2)",
    )

    # Scipy parameters
    parser.add_argument(
        "--method",
        type=str,
        default="L-BFGS-B",
        help="Scipy optimization method (default: L-BFGS-B)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Scipy maximum iterations (default: 1000)",
    )

    # Time period configuration
    parser.add_argument(
        "--cv-fold",
        type=int,
        default=1,
        help="Cross-validation fold number (default: 1)",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=365,
        help="Warmup period length in days (default: 365)",
    )

    parser.add_argument(
        "--period",
        nargs="+",
        default=["2014-10-01", "2021-09-30"],
        help="Overall time period (default: 2014-10-01 2021-09-30)",
    )

    parser.add_argument(
        "--calibrate-period",
        nargs="+",
        default=["2014-10-01", "2019-09-30"],
        help="Calibration period (default: 2014-10-01 2019-09-30)",
    )

    parser.add_argument(
        "--test-period",
        nargs="+",
        default=["2019-10-01", "2021-09-30"],
        help="Testing period (default: 2019-10-01 2021-09-30)",
    )

    parser.add_argument(
        "--basin-id",
        nargs="+",
        default=["changdian_61561", "changdian_62618"],
        help="Basin IDs to calibrate",
    )

    # Parameter range file
    parser.add_argument(
        "--param-range-file",
        type=str,
        default=None,
        help="Parameter range file path (uses default if not specified)",
    )

    # Loss function configuration
    parser.add_argument(
        "--obj-func",
        type=str,
        default="RMSE",
        choices=["RMSE", "NSE", "KGE", "MAE"],
        help="Objective function (default: RMSE)",
    )

    # Other options
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility (default: 1234)",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode"
    )

    return parser.parse_args()


def create_model_config(args):
    """Create model configuration dictionary"""
    return {
        "name": args.model_type,
        "source_type": args.source_type,
        "source_book": args.source_book,
        "kernel_size": args.kernel_size,
        "time_interval_hours": args.time_interval_hours,
    }


def create_algorithm_config(args):
    """Create algorithm configuration dictionary"""
    if args.algorithm == "SCE_UA":
        return {
            "name": "SCE_UA",
            "random_seed": args.random_seed,
            "rep": args.rep,
            "ngs": args.ngs,
            "kstop": args.kstop,
            "peps": args.peps,
            "pcento": args.pcento,
        }
    elif args.algorithm == "genetic_algorithm":
        return {
            "name": "genetic_algorithm",
            "random_seed": args.random_seed,
            "pop_size": args.pop_size,
            "n_generations": args.n_generations,
            "cx_prob": args.cx_prob,
            "mut_prob": args.mut_prob,
        }
    elif args.algorithm == "scipy_minimize":
        return {
            "name": "scipy_minimize",
            "method": args.method,
            "max_iterations": args.max_iterations,
        }
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")


def create_loss_config(args):
    """Create loss function configuration dictionary"""
    return {
        "type": "time_series",
        "obj_func": args.obj_func,
        "events": None,
    }


def calibrate_xaj_unified(args):
    """Main calibration function using unified interface"""
    verbose = not args.quiet

    # Setup output directory
    where_save = Path(os.path.join(args.result_dir, args.exp))
    if not os.path.exists(where_save):
        os.makedirs(where_save)

    if verbose:
        print("=" * 80)
        print("🚀 XAJ Model Calibration - Unified Interface")
        print("=" * 80)
        print(f"📁 Data path: {args.data_dir}")
        print(f"💾 Results: {where_save}")
        print(f"🤖 Model: {args.model_type}")
        print(f"🔧 Algorithm: {args.algorithm}")
        print(f"📊 Objective function: {args.obj_func}")
        print(f"🏭 Basins: {', '.join(args.basin_id)}")
        print(f"⏱️ Warmup: {args.warmup} days")
        if args.algorithm == "genetic_algorithm":
            print(f"🧬 DEAP available: {DEAP_AVAILABLE}")
        print("-" * 80)

    # Check GA availability
    if args.algorithm == "genetic_algorithm" and not DEAP_AVAILABLE:
        print("❌ Genetic algorithm requested but DEAP is not available")
        print("💡 Install DEAP with: pip install deap")
        return

    # Prepare data
    if verbose:
        print("🔄 Preparing training and testing data...")

    train_and_test_data = cross_val_split_tsdata(
        args.data_type,
        args.data_dir,
        args.cv_fold,
        args.calibrate_period,
        args.test_period,
        args.period,
        args.warmup,
        args.basin_id,
    )

    # Create configurations
    model_config = create_model_config(args)
    algorithm_config = create_algorithm_config(args)
    loss_config = create_loss_config(args)

    if verbose:
        print("✅ Configuration created:")
        print(f"   📊 Model: {model_config}")
        print(f"   🔧 Algorithm: {algorithm_config['name']}")
        print(f"   📉 Loss: {loss_config['obj_func']}")

    # Handle parameter range file
    param_range_file = args.param_range_file
    if param_range_file is None:
        param_range_file = os.path.join(where_save, "param_range.yaml")
        if verbose:
            print(f"   📋 Creating default parameter file: {param_range_file}")
        yaml.dump(MODEL_PARAM_DICT, open(param_range_file, "w"))
    else:
        # Copy user-provided parameter file to results directory
        dest_param_file = os.path.join(
            where_save, os.path.basename(param_range_file)
        )
        shutil.copy(param_range_file, dest_param_file)
        param_range_file = dest_param_file
        if verbose:
            print(f"   📋 Using parameter file: {param_range_file}")

    print(
        f"\n🚀 Starting {args.algorithm} calibration with unified interface..."
    )

    # Calibration
    if args.cv_fold <= 1:
        # Single fold calibration
        p_and_e, qobs = _get_pe_q_from_ts(train_and_test_data[0])

        results = calibrate(
            data=(p_and_e, qobs),
            model_config=model_config,
            algorithm_config=algorithm_config,
            loss_config=loss_config,
            output_dir=os.path.join(
                where_save, f"{args.algorithm.lower()}_xaj"
            ),
            warmup_length=args.warmup,
            param_file=param_range_file,
            basin_ids=args.basin_id,
        )

        # Process results
        process_calibration_results(results, args, verbose)

    else:
        # Cross-validation calibration
        cv_results = {}
        for i in range(args.cv_fold):
            if verbose:
                print(f"\n📊 Cross-validation fold {i+1}/{args.cv_fold}")

            train_data, _ = train_and_test_data[i]
            p_and_e_cv, qobs_cv = _get_pe_q_from_ts(train_data)

            fold_results = calibrate(
                data=(p_and_e_cv, qobs_cv),
                model_config=model_config,
                algorithm_config=algorithm_config,
                loss_config=loss_config,
                output_dir=os.path.join(
                    where_save, f"{args.algorithm.lower()}_xaj_cv{i+1}"
                ),
                warmup_length=args.warmup,
                param_file=param_range_file,
                basin_ids=args.basin_id,
            )

            cv_results[f"cv_{i+1}"] = fold_results

        # Process cross-validation results
        process_cv_results(cv_results, args, verbose)

    # Save configuration
    save_experiment_config(args, where_save, param_range_file)

    print(f"\n🎉 XAJ calibration completed using unified interface!")
    print(f"✨ Used unified calibrate() function")
    print(f"🔧 Model: {args.model_type} | Algorithm: {args.algorithm}")
    print(f"💾 Results saved to: {where_save}")


def process_calibration_results(results, args, verbose=True):
    """Process and display calibration results"""
    if verbose:
        print(f"\n📈 Calibration Results Summary:")

    for basin_id, basin_result in results.items():
        if verbose:
            print(f"\n🏭 Basin: {basin_id}")

        convergence = basin_result.get("convergence", "unknown")
        objective_value = basin_result.get("objective_value", float("inf"))
        best_params = basin_result.get("best_params", {})

        if verbose:
            print(f"   ✅ Convergence: {convergence}")
            print(f"   🎯 Best {args.obj_func}: {objective_value:.6f}")
            print(
                f"   📋 Parameters: {len(best_params.get(basin_id, {}))} optimized"
            )

        # Display parameter values if convergence was successful
        if convergence == "success" and basin_id in best_params:
            if verbose:
                print(f"   📊 Optimized Parameters:")
            for param_name, param_value in best_params[basin_id].items():
                if verbose:
                    print(f"      {param_name}: {param_value:.6f}")


def process_cv_results(cv_results, args, verbose=True):
    """Process cross-validation results"""
    if verbose:
        print(f"\n📈 Cross-Validation Results Summary:")

    all_objectives = []
    successful_folds = 0

    for fold_name, fold_results in cv_results.items():
        if verbose:
            print(f"\n📊 {fold_name.upper()}:")

        for basin_id, basin_result in fold_results.items():
            convergence = basin_result.get("convergence", "unknown")
            objective_value = basin_result.get("objective_value", float("inf"))

            if verbose:
                print(
                    f"   🏭 {basin_id}: {convergence}, {args.obj_func}={objective_value:.6f}"
                )

            if convergence == "success":
                all_objectives.append(objective_value)
                successful_folds += 1

    if all_objectives and verbose:
        print(f"\n📊 Cross-Validation Statistics:")
        print(
            f"   ✅ Successful folds: {successful_folds}/{len(cv_results) * len(args.basin_id)}"
        )
        print(f"   🎯 Mean {args.obj_func}: {np.mean(all_objectives):.6f}")
        print(f"   📊 Std {args.obj_func}: {np.std(all_objectives):.6f}")
        print(f"   📈 Best {args.obj_func}: {np.min(all_objectives):.6f}")
        print(f"   📉 Worst {args.obj_func}: {np.max(all_objectives):.6f}")


def save_experiment_config(args, where_save, param_range_file):
    """Save experiment configuration for reproducibility"""
    config = {
        "experiment_name": args.exp,
        "model_config": create_model_config(args),
        "algorithm_config": create_algorithm_config(args),
        "loss_config": create_loss_config(args),
        "data_config": {
            "data_type": args.data_type,
            "data_dir": args.data_dir,
            "basin_ids": args.basin_id,
            "cv_fold": args.cv_fold,
            "warmup": args.warmup,
            "period": args.period,
            "calibrate_period": args.calibrate_period,
            "test_period": args.test_period,
        },
        "param_range_file": param_range_file,
        "random_seed": args.random_seed,
    }

    config_file = os.path.join(where_save, "unified_calibration_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"💾 Configuration saved: {config_file}")


if __name__ == "__main__":
    args = parse_arguments()
    calibrate_xaj_unified(args)
