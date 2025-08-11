"""
Author: Wenyu Ouyang
Date: 2025-08-08
LastEditTime: 2025-08-08 20:23:22
LastEditors: Wenyu Ouyang
Description: Unified script utilities for all hydromodel scripts
FilePath: \hydromodel\hydromodel\configs\script_utils.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import ast
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional


from .config_manager import ConfigManager
from .unified_config import UnifiedConfig


class ScriptUtils:
    """Unified utilities for hydromodel scripts"""

    @staticmethod
    def apply_overrides(
        config: Dict[str, Any], overrides: Optional[List[str]]
    ) -> None:
        """
        Apply command line overrides to configuration

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to update
        overrides : Optional[List[str]]
            List of override strings in format "key.path=value"
        """
        if not overrides:
            return

        print("🔧 Applying configuration overrides:")

        for override in overrides:
            if "=" not in override:
                print(f"❌ Invalid override format: {override}")
                continue

            key_path, value = override.split("=", 1)
            keys = key_path.split(".")

            # Navigate to the nested key and set the value
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Convert value to appropriate type
            final_key = keys[-1]
            try:
                current[final_key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                current[final_key] = value

            print(f"   ✅ {key_path} = {value}")

    @staticmethod
    def validate_and_show_config(
        config: Dict[str, Any],
        verbose: bool = True,
        model_type: str = "General",
    ) -> bool:
        """
        Validate configuration and show summary

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to validate
        verbose : bool
            Whether to show detailed output
        model_type : str
            Type of model for display purposes

        Returns
        -------
        bool
            True if validation passed
        """
        if not verbose:
            return True

        print(f"🔍 {model_type} Model Configuration Summary:")
        print("=" * 60)

        data_cfgs = config.get("data_cfgs", {})
        model_cfgs = config.get("model_cfgs", {})
        training_cfgs = config.get("training_cfgs", {})
        eval_cfgs = config.get("evaluation_cfgs", {})

        print("📊 Data Configuration:")
        data_dir = data_cfgs.get("data_source_path") or data_cfgs.get(
            "data_path", "default"
        )
        print(f"   📂 Data directory: {data_dir}")
        print(
            f"   🏭 Station/Basin IDs: {', '.join(data_cfgs.get('basin_ids', []))}"
        )
        print(
            f"   ⏱️ Warmup length: {data_cfgs.get('warmup_length', 365)} steps"
        )
        print(f"   📋 Variables: {data_cfgs.get('variables', [])}")

        print("\n🔧 Model Configuration:")
        model_params = model_cfgs.get("model_params", {})
        print(f"   🏷️ Model name: {model_cfgs.get('model_name')}")

        # Model-specific parameter display
        if model_type == "Unit Hydrograph":
            print(f"   📏 Unit hydrograph length: {model_params.get('n_uh')}")
            print(
                f"   🔀 Smoothing factor: {model_params.get('smoothing_factor')}"
            )
        elif model_type == "Categorized Unit Hydrograph":
            uh_lengths = model_params.get("uh_lengths", {})
            print(f"   📏 UH lengths: {uh_lengths}")
            print(
                f"   🏷️ Category weights available: {list(model_params.get('category_weights', {}).keys())}"
            )
        elif model_type == "XAJ Model":
            print(f"   📋 Source type: {model_params.get('source_type')}")
            print(f"   📖 Source book: {model_params.get('source_book')}")
            print(f"   🔧 Kernel size: {model_params.get('kernel_size')}")

        print("\n🎯 Training Configuration:")
        print(f"   🔬 Algorithm: {training_cfgs.get('algorithm_name')}")
        print(
            f"   📊 Objective: {training_cfgs.get('loss_config', {}).get('obj_func')}"
        )
        print(
            f"   📁 Output: {training_cfgs.get('output_dir')}/{training_cfgs.get('experiment_name')}"
        )

        # Check algorithm availability
        algorithm_name = training_cfgs.get("algorithm_name")
        if algorithm_name == "genetic_algorithm":
            import importlib.util as _importlib_util

            if _importlib_util.find_spec("deap") is not None:
                print("   🧬 DEAP Available: Yes")
            else:
                print(
                    f"   ❌ ERROR: Algorithm '{algorithm_name}' requires DEAP package"
                )
                print("   💡 Install with: pip install deap")
                return False

        # Show algorithm-specific parameters preview
        algo_params = training_cfgs.get("algorithm_params", {})
        if algo_params:
            print("   ⚙️ Algorithm Parameters:")
            for key, value in list(algo_params.items())[:3]:  # Show first 3
                print(f"      {key}: {value}")
            if len(algo_params) > 3:
                print(f"      ... ({len(algo_params)-3} more parameters)")

        print("\n✅ Configuration validation passed")
        return True

    @staticmethod
    def load_flood_events_data(config: Dict[str, Any], verbose: bool = True):
        """
        Load flood events data based on configuration

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration containing data settings
        verbose : bool
            Whether to show loading progress

        Returns
        -------
        List
            Loaded flood events data
        """
        try:
            from hydrodatasource.reader.floodevent import FloodEventDatasource
        except ImportError:
            raise ImportError(
                "FloodEventDatasource not available - install hydrodatasource package"
            )

        data_cfgs = config.get("data_cfgs", {})

        if verbose:
            print("\n🔄 Loading flood events data...")

        # Load flood events
        dataset = FloodEventDatasource(
            data_cfgs.get("data_source_path"),
            time_unit=["3h"],
            trange4cache=["1960-01-01 02", "2024-12-31 23"],
            warmup_length=data_cfgs.get("warmup_length", 480),
        )

        basin_ids = data_cfgs.get("basin_ids", [])
        if not basin_ids:
            raise ValueError("Basin IDs must be specified")

        all_event_data = dataset.load_1basin_flood_events(
            station_id=basin_ids[0],
            flow_unit="mm/3h",
            include_peak_obs=True,
            verbose=verbose,
        )

        if all_event_data is None:
            raise ValueError(f"No flood events found for basin {basin_ids[0]}")

        # Check for NaN values (excluding warmup period)
        dataset.check_event_data_nan(all_event_data, exclude_warmup=True)

        if verbose:
            print(f"   ✅ Loaded {len(all_event_data)} flood events")

        return all_event_data

    @staticmethod
    def show_help_information(
        script_name: str, examples_pattern: Optional[str] = None
    ):
        """
        Show helpful information when no config is provided

        Parameters
        ----------
        script_name : str
            Name of the script
        examples_pattern : Optional[str], optional
            Pattern to search for example configs
        """
        print(f"🔍 Model Calibration with Unified Config")
        print("=" * 60)
        print()
        print("📋 Available options:")
        print()
        print("1️⃣ Use existing configuration file:")
        print(f"   python {script_name} --config my_config.yaml")
        print()
        print("2️⃣ Quick setup mode (no config file needed):")
        print(f"   python {script_name} --quick-setup")
        print(
            f"   python {script_name} --quick-setup --station-id basin_001 --algorithm SCE_UA"
        )
        print()
        print("3️⃣ Create a configuration template:")
        print(f"   python {script_name} --create-template my_config.yaml")
        print()
        print("4️⃣ Use example configurations:")

        # Check for example configs
        repo_path = Path(__file__).parent.parent.parent
        examples_dir = repo_path / "configs" / "examples"

        if examples_dir.exists() and examples_pattern:
            example_files = list(examples_dir.glob(examples_pattern))
            if example_files:
                print("   Available examples:")
                for example_file in example_files[:3]:  # Show first 3
                    print(f"   - python {script_name} --config {example_file}")
                if len(example_files) > 3:
                    print(f"   ... and {len(example_files)-3} more examples")
            else:
                print(
                    f"   - python {script_name} --config configs/examples/example_config.yaml"
                )
        else:
            print(
                f"   - python {script_name} --config configs/examples/example_config.yaml"
            )

        print()
        print(f"💡 For more options, run: python {script_name} --help")
        print()

    @staticmethod
    def prompt_quick_setup(args, create_config_func):
        """
        Prompt user for quick setup and create config

        Parameters
        ----------
        args : argparse.Namespace
            Command line arguments
        create_config_func : callable
            Function to create config from args

        Returns
        -------
        Dict[str, Any] or None
            Configuration if user agrees to quick setup, None otherwise
        """
        try:
            response = (
                input(
                    "Would you like to run with default quick setup? (y/n): "
                )
                .lower()
                .strip()
            )
            if response in ["y", "yes", ""]:
                print("🚀 Using default quick setup configuration...")
                args.quick_setup = True
                return create_config_func(args)
            else:
                return None
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return None

    @staticmethod
    def save_config_file(config: Dict[str, Any], output_path: str) -> None:
        """
        Save configuration to file

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to save
        output_path : str
            Path where to save the configuration
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            ConfigManager.save_config_to_file(config, output_path)
            print(f"💾 Configuration saved: {output_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save config file: {e}")

    @staticmethod
    def print_completion_message(
        config: Dict[str, Any], task_name: str = "operation"
    ):
        """
        Print completion message with summary

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration used
        task_name : str
            Name of the completed task
        """
        training_cfgs = config.get("training_cfgs", {})
        model_name = config.get("model_cfgs", {}).get("model_name", "unknown")
        algorithm = training_cfgs.get("algorithm_name", "unknown")

        output_path = os.path.join(
            training_cfgs.get("output_dir", "results"),
            training_cfgs.get("experiment_name", "experiment"),
        )

        print(f"\n🎉 {task_name.capitalize()} completed successfully!")
        print(f"✨ Used latest unified architecture")
        print(f"🔧 Model: {model_name} | Algorithm: {algorithm}")
        print(f"💾 Results saved to: {output_path}")

    @staticmethod
    def add_common_arguments(parser: argparse.ArgumentParser) -> None:
        """
        Add common arguments that are shared across scripts

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Parser to add arguments to
        """
        # Configuration file mode
        parser.add_argument(
            "--config",
            "-c",
            type=str,
            help="Path to configuration file (YAML or JSON)",
        )

        parser.add_argument(
            "--create-template",
            type=str,
            help="Create configuration template and exit",
        )

        # Quick setup mode
        parser.add_argument(
            "--quick-setup",
            action="store_true",
            help="Quick setup mode using command line arguments",
        )

        # Data configuration
        parser.add_argument(
            "--data-path",
            type=str,
            help="Data directory path (quick setup mode)",
        )

        parser.add_argument(
            "--station-id",
            type=str,
            default="basin_001",
            help="Station/Basin ID for calibration (quick setup mode)",
        )

        parser.add_argument(
            "--warmup-length",
            type=int,
            default=365,
            help="Warmup length in time steps (quick setup mode)",
        )

        # Training configuration
        parser.add_argument(
            "--algorithm",
            type=str,
            default="SCE_UA",
            choices=["scipy_minimize", "SCE_UA", "genetic_algorithm"],
            help="Optimization algorithm (quick setup mode)",
        )

        # Common options
        parser.add_argument(
            "--override",
            "-o",
            action="append",
            help="Override config values (e.g., -o model_cfgs.model_params.n_uh=32)",
        )

        parser.add_argument(
            "--output-dir", type=str, help="Override output directory"
        )

        parser.add_argument(
            "--experiment-name", type=str, help="Override experiment name"
        )

        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Verbose output"
        )

        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Dry run - validate config and show what would be done",
        )

        parser.add_argument(
            "--plot-results",
            action="store_true",
            help="Generate plots of results",
        )

        parser.add_argument(
            "--save-evaluation",
            action="store_true",
            help="Save detailed evaluation results to CSV",
        )

    @staticmethod
    def setup_configuration(
        args,
    ):
        """
        Setup configuration from arguments with unified workflow

        Parameters
        ----------
        args : argparse.Namespace
            Command line arguments

                    Returns
        -------
        Dict[str, Any] or None
            Configuration if successful, None to exit
        """
        if args.config:
            # Configuration file mode
            try:
                config = ConfigManager.load_config_from_file(args.config)
                print(f"✅ Loaded configuration: {args.config}")
                return config
            except Exception as e:
                print(f"❌ Failed to load configuration: {e}")
                return None

        else:
            # Always create config from unified defaults + args (no prompts)
            return ScriptUtils.create_config_from_unified_defaults(args)

    @staticmethod
    def create_config_from_unified_defaults(
        args: argparse.Namespace,
    ) -> UnifiedConfig:
        """
        Build configuration from UnifiedConfig defaults, then update with args.

        Returns
        -------
        UnifiedConfig
            Unified configuration instance updated by args
        """
        config = UnifiedConfig()

        updates: Dict[str, Any] = {
            "data_cfgs": {},
            "model_cfgs": {"model_params": {}},
            "training_cfgs": {},
        }

        # Data configuration mapping
        if (
            hasattr(args, "data_source_type")
            and args.data_source_type is not None
        ):
            updates["data_cfgs"]["data_type"] = args.data_source_type
        if (
            hasattr(args, "data_source_path")
            and args.data_source_path is not None
        ):
            updates["data_cfgs"]["data_path"] = args.data_source_path
        if hasattr(args, "data_path") and args.data_path is not None:
            updates["data_cfgs"]["data_path"] = args.data_path
        if hasattr(args, "basin_ids") and args.basin_ids is not None:
            updates["data_cfgs"]["basin_ids"] = args.basin_ids
        if hasattr(args, "warmup_length") and args.warmup_length is not None:
            updates["data_cfgs"]["warmup_length"] = args.warmup_length
        if hasattr(args, "variables") and args.variables is not None:
            updates["data_cfgs"]["variables"] = args.variables

        # Model configuration mapping
        model_name = None
        if hasattr(args, "model_type") and args.model_type is not None:
            model_name = args.model_type
        elif hasattr(args, "model") and args.model is not None:
            model_name = args.model
        if model_name is not None:
            updates["model_cfgs"]["model_name"] = model_name

        # Model parameters (optional overrides)
        for key in [
            "source_type",
            "source_book",
            "kernel_size",
            "n_uh",
            "smoothing_factor",
            "peak_violation_weight",
            "apply_peak_penalty",
            "net_rain_name",
            "obs_flow_name",
        ]:
            if hasattr(args, key) and getattr(args, key) is not None:
                updates["model_cfgs"]["model_params"][key] = getattr(args, key)

        # Categorized UH specific (JSON/text mapping should be done by caller via --override)
        if (
            hasattr(args, "uh_lengths")
            and getattr(args, "uh_lengths") is not None
        ):
            updates["model_cfgs"]["model_params"]["uh_lengths"] = getattr(
                args, "uh_lengths"
            )

        # Training configuration
        if hasattr(args, "algorithm") and args.algorithm is not None:
            updates.setdefault("training_cfgs", {})[
                "algorithm_name"
            ] = args.algorithm

        algo_params: Dict[str, Any] = {}
        # SCE-UA
        for key in ["rep", "ngs", "kstop", "peps", "pcento"]:
            if hasattr(args, key) and getattr(args, key) is not None:
                algo_params[key] = getattr(args, key)
        # SciPy
        if hasattr(args, "scipy_method") and args.scipy_method is not None:
            algo_params["method"] = args.scipy_method
        if hasattr(args, "max_iterations") and args.max_iterations is not None:
            algo_params["max_iterations"] = args.max_iterations
        # GA
        for key in [
            "pop_size",
            "n_generations",
            "cx_prob",
            "mut_prob",
            "random_seed",
        ]:
            if hasattr(args, key) and getattr(args, key) is not None:
                algo_params[key] = getattr(args, key)
        if algo_params:
            updates.setdefault("training_cfgs", {}).setdefault(
                "algorithm_params", {}
            ).update(algo_params)

        # Loss / output settings
        if hasattr(args, "obj_func") and args.obj_func is not None:
            updates.setdefault("training_cfgs", {}).setdefault(
                "loss_config", {}
            )["obj_func"] = args.obj_func
        if hasattr(args, "output_dir") and args.output_dir is not None:
            updates.setdefault("training_cfgs", {})[
                "output_dir"
            ] = args.output_dir
        if (
            hasattr(args, "experiment_name")
            and args.experiment_name is not None
        ):
            updates.setdefault("training_cfgs", {})[
                "experiment_name"
            ] = args.experiment_name
        if hasattr(args, "random_seed") and args.random_seed is not None:
            updates.setdefault("training_cfgs", {})["algorithm_params"] = (
                updates.get("training_cfgs", {}).get("algorithm_params", {})
            )
            updates["training_cfgs"]["algorithm_params"][
                "random_seed"
            ] = args.random_seed

        # Apply updates
        config.update_config(updates)
        return config.config
