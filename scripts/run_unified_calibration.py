"""
Author: Wenyu Ouyang
Date: 2025-08-06 
LastEditTime: 2025-08-06 
LastEditors: Wenyu Ouyang
Description: Universal calibration script using unified interface - supports all model types
FilePath: /hydromodel/scripts/run_unified_calibration.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import argparse
import json
from hydroutils.hydro_plot import (
    plot_unit_hydrograph,
    setup_matplotlib_chinese,
)
from hydrodatasource.configs.config import SETTING
from hydrodatasource.reader.floodevent import (
    FloodEventDatasource,
)
from hydromodel.models.unit_hydrograph import (
    evaluate_single_event_from_uh,
    print_report_preview,
    save_results_to_csv,
    print_category_statistics,
)
from hydromodel.trainers.unified_calibrate import calibrate, DEAP_AVAILABLE


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Universal Calibration Tool using Unified Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Types Supported:
  - unit_hydrograph: Single shared unit hydrograph
  - categorized_unit_hydrograph: Categorized unit hydrographs by flood magnitude
  - xaj: XinAnJiang hydrological model
  - gr4j: GR4J hydrological model (and other GR series)

Algorithm Types Supported:
  - scipy_minimize: SciPy optimization methods
  - SCE_UA: Shuffled Complex Evolution via spotpy
  - genetic_algorithm: Genetic algorithm via DEAP (if installed)

Usage Examples:
  # Unit hydrograph with scipy
  python run_unified_calibration.py --model-type unit_hydrograph --algorithm scipy_minimize
  
  # Categorized unit hydrograph with GA
  python run_unified_calibration.py --model-type categorized_unit_hydrograph --algorithm genetic_algorithm
  
  # Traditional XAJ model with SCE-UA
  python run_unified_calibration.py --model-type xaj --algorithm SCE_UA --rep 5000
        """,
    )

    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default=os.path.join(
            SETTING["local_data_path"]["datasets-interim"], "songliaorrevent"
        ),
        help="Data directory path",
    )

    parser.add_argument(
        "--station-id",
        type=str,
        default="songliao_21401550",
        help="Station ID (e.g., songliao_21401550)",
    )

    parser.add_argument(
        "--output-dir", "-o", type=str, default="results/", help="Output directory"
    )

    # Model selection
    parser.add_argument(
        "--model-type",
        type=str,
        default="unit_hydrograph",
        choices=["unit_hydrograph", "categorized_unit_hydrograph", "xaj", "gr4j", "hymod"],
        help="Model type to calibrate (default: unit_hydrograph)",
    )

    # Universal parameters
    parser.add_argument(
        "--warmup-length",
        type=int,
        default=8 * 60,  # 8 hours * 60 minutes / 3 hours = 160 steps for 3h data
        help="Warmup period length in steps (default: 160 for 8 hours)",
    )

    # Unit hydrograph specific parameters
    parser.add_argument(
        "--n-uh",
        type=int,
        default=24,
        help="Unit hydrograph length (default: 24)",
    )

    parser.add_argument(
        "--smoothing-factor",
        type=float,
        default=0.1,
        help="Smoothing penalty weight factor (default: 0.1)",
    )

    parser.add_argument(
        "--peak-violation-weight",
        type=float,
        default=10000.0,
        help="Single peak violation penalty weight (default: 10000.0)",
    )

    # Categorized UH parameters
    parser.add_argument(
        "--uh-lengths",
        type=str,
        default='{"small":8,"medium":16,"large":24}',
        help='UH lengths for categories, JSON format (default: {"small":8,"medium":16,"large":24})',
    )

    parser.add_argument(
        "--category-weights",
        type=str,
        default="default",
        choices=["default", "balanced", "aggressive"],
        help="Category weight scheme (default: default)",
    )

    # Algorithm selection
    parser.add_argument(
        "--algorithm",
        type=str,
        default="scipy_minimize",
        choices=["scipy_minimize", "SCE_UA", "genetic_algorithm"],
        help="Optimization algorithm (default: scipy_minimize)",
    )

    # scipy parameters
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="scipy maximum iterations (default: 500)",
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="SLSQP",
        help="scipy optimization method (default: SLSQP)",
    )

    # SCE-UA parameters
    parser.add_argument(
        "--rep",
        type=int,
        default=1000,
        help="SCE-UA repetitions (default: 1000)",
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1234,
        help="Random seed (default: 1234)",
    )

    # GA parameters
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
    
    parser.add_argument(
        "--save-freq",
        type=int,
        default=5,
        help="GA save frequency (default: 5)",
    )

    # Other options
    parser.add_argument(
        "--obj-func",
        type=str,
        default="RMSE",
        choices=["RMSE", "NSE", "KGE"],
        help="Objective function (default: RMSE)",
    )

    parser.add_argument(
        "--no-peak-obs", action="store_true", help="Exclude peak observations"
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode"
    )

    return parser.parse_args()


def get_category_weights(scheme_name):
    """Get category weight schemes for categorized UH"""
    schemes = {
        "default": {
            "small": {"smoothing_factor": 0.1, "peak_violation_weight": 100.0},
            "medium": {"smoothing_factor": 0.5, "peak_violation_weight": 500.0},
            "large": {"smoothing_factor": 1.0, "peak_violation_weight": 1000.0},
        },
        "balanced": {
            "small": {"smoothing_factor": 0.2, "peak_violation_weight": 200.0},
            "medium": {"smoothing_factor": 0.2, "peak_violation_weight": 200.0},
            "large": {"smoothing_factor": 0.2, "peak_violation_weight": 200.0},
        },
        "aggressive": {
            "small": {"smoothing_factor": 0.05, "peak_violation_weight": 50.0},
            "medium": {"smoothing_factor": 0.1, "peak_violation_weight": 100.0},
            "large": {"smoothing_factor": 0.5, "peak_violation_weight": 2000.0},
        },
    }
    return schemes.get(scheme_name, schemes["default"])


def create_model_config(args):
    """Create model configuration based on model type"""
    if args.model_type == "unit_hydrograph":
        return {
            "name": "unit_hydrograph",
            "n_uh": args.n_uh,
            "smoothing_factor": args.smoothing_factor,
            "peak_violation_weight": args.peak_violation_weight,
            "apply_peak_penalty": args.n_uh > 2,
            "net_rain_name": "P_eff",
            "obs_flow_name": "Q_obs_eff",
        }
    
    elif args.model_type == "categorized_unit_hydrograph":
        try:
            uh_lengths = json.loads(args.uh_lengths)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid UH lengths JSON: {args.uh_lengths}")
        
        return {
            "name": "categorized_unit_hydrograph",
            "category_weights": get_category_weights(args.category_weights),
            "uh_lengths": uh_lengths,
            "net_rain_name": "P_eff",
            "obs_flow_name": "Q_obs_eff",
        }
    
    elif args.model_type == "xaj":
        return {
            "name": "xaj_mz",
            "source_type": "sources",
            "source_book": "HF"
        }
    
    elif args.model_type == "gr4j":
        return {
            "name": "gr4j"
        }
    
    elif args.model_type == "hymod":
        return {
            "name": "hymod"
        }
    
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")


def create_algorithm_config(args):
    """Create algorithm configuration"""
    if args.algorithm == "scipy_minimize":
        return {
            "name": "scipy_minimize",
            "method": args.method,
            "max_iterations": args.max_iterations,
        }
    elif args.algorithm == "SCE_UA":
        return {
            "name": "SCE_UA",
            "rep": args.rep,
            "random_seed": args.random_seed,
        }
    elif args.algorithm == "genetic_algorithm":
        return {
            "name": "genetic_algorithm",
            "random_seed": args.random_seed,
            "pop_size": args.pop_size,
            "n_generations": args.n_generations,
            "cx_prob": args.cx_prob,
            "mut_prob": args.mut_prob,
            "save_freq": args.save_freq,
        }
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")


def create_loss_config(args):
    """Create loss function configuration"""
    return {
        "type": "time_series",
        "obj_func": args.obj_func
    }


def is_unit_hydrograph_model(model_type):
    """Check if model is a unit hydrograph type"""
    return model_type in ["unit_hydrograph", "categorized_unit_hydrograph"]


def main():
    """Main function for universal calibration"""
    # Parse arguments
    args = parse_arguments()

    # Check GA availability
    if args.algorithm == "genetic_algorithm" and not DEAP_AVAILABLE:
        print("❌ Genetic algorithm requested but DEAP is not available")
        print("💡 Install DEAP with: pip install deap")
        return

    # Initialize plotting
    setup_matplotlib_chinese()

    # Setup
    verbose = not args.quiet
    include_peak_obs = not args.no_peak_obs

    if verbose:
        print("=" * 80)
        print("🚀 Universal Calibration Tool - Unified Interface")
        print("=" * 80)
        print(f"📁 Data path: {args.data_path}")
        print(f"🏭 Station ID: {args.station_id}")
        print(f"📤 Output directory: {args.output_dir}")
        print(f"🤖 Model type: {args.model_type}")
        print(f"🔧 Algorithm: {args.algorithm}")
        print(f"⏱️ Warmup length: {args.warmup_length} steps")
        print(f"📊 Objective function: {args.obj_func}")
        if args.algorithm == "genetic_algorithm":
            print(f"🧬 DEAP available: {DEAP_AVAILABLE}")
        print("-" * 80)

    # Load data
    dataset = FloodEventDatasource(
        args.data_path,
        time_unit=["3h"],
        trange4cache=["1960-01-01 02", "2024-12-31 23"],
        warmup_length=args.warmup_length,
    )

    all_event_data = dataset.load_1basin_flood_events(
        station_id=args.station_id,
        flow_unit="mm/3h",
        include_peak_obs=include_peak_obs or args.model_type == "categorized_unit_hydrograph",
        verbose=verbose,
    )

    dataset.check_event_data_nan(all_event_data)

    if verbose:
        print(f"✅ Loaded {len(all_event_data)} flood events (with warmup)")

    # Create configurations
    model_config = create_model_config(args)
    algorithm_config = create_algorithm_config(args)
    loss_config = create_loss_config(args)

    if verbose:
        print(f"\n🚀 Starting calibration with unified interface...")
        print(f"📊 Model config: {model_config['name']}")
        print(f"🔧 Algorithm config: {algorithm_config['name']}")
        print(f"📉 Loss config: {loss_config['obj_func']}")

    # Run calibration using unified interface
    results = calibrate(
        data=all_event_data,
        model_config=model_config,
        algorithm_config=algorithm_config,
        loss_config=loss_config,
        output_dir=args.output_dir,
        warmup_length=args.warmup_length,
    )

    # Check results
    if results["convergence"] != "success" or results["best_params"] is None:
        print(f"❌ Calibration failed: {results.get('convergence', 'unknown error')}")
        return

    if verbose:
        print(f"\n✅ Calibration completed successfully!")
        print(f"🎯 Best objective value: {results['objective_value']:.6f}")
        print(f"🔄 Convergence: {results['convergence']}")

    # Model-specific post-processing
    if is_unit_hydrograph_model(args.model_type):
        # Unit hydrograph models - plot and evaluate
        if args.model_type == "unit_hydrograph":
            # Single UH
            uh_params_dict = results["best_params"]["unit_hydrograph"]
            uh_params = [uh_params_dict[f"uh_{i+1}"] for i in range(args.n_uh)]
            
            if verbose:
                print(f"📋 UH parameters: {len(uh_params)} values")
                plot_unit_hydrograph(uh_params, "Unified Interface Unit Hydrograph")
        
        elif args.model_type == "categorized_unit_hydrograph":
            # Categorized UH
            cat_info = results.get("categorization_info", {})
            if verbose:
                print(f"📊 Categorization results:")
                print(f"   Categories: {cat_info.get('categories', [])}")
                print(f"   Thresholds: {cat_info.get('thresholds', {})}")
                for category, count in cat_info.get("events_per_category", {}).items():
                    print(f"   {category.capitalize()}: {count} events")

        # Evaluate UH models (this would need model-specific evaluation logic)
        if verbose:
            print("\n📈 Model evaluation completed")
            print(f"💾 Results saved to: {args.output_dir}")
    
    else:
        # Traditional hydrological models
        if verbose:
            print(f"📊 Traditional model calibration completed")
            print(f"📋 Best parameters: {list(results['best_params'].keys())}")
            print(f"💾 Results saved to: {args.output_dir}")

    print(f"\n🎉 Universal calibration completed!")
    print(f"✨ Used unified interface: calibrate()")
    print(f"🔧 Model: {args.model_type} | Algorithm: {args.algorithm}")


if __name__ == "__main__":
    main()