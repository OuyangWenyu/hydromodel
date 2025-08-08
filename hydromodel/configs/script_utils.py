"""
Author: Wenyu Ouyang
Date: 2025-08-08
LastEditTime: 2025-08-08 11:00:00
LastEditors: Wenyu Ouyang
Description: Unified script utilities for all hydromodel scripts
FilePath: /hydromodel/hydromodel/configs/script_utils.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import ast
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .config_manager import ConfigManager


class ScriptUtils:
    """Unified utilities for hydromodel scripts"""
    
    @staticmethod
    def apply_overrides(config: Dict[str, Any], overrides: Optional[List[str]]) -> None:
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
    def validate_and_show_config(config: Dict[str, Any], verbose: bool = True, 
                                model_type: str = "General") -> bool:
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
        print(f"   📂 Data directory: {data_cfgs.get('data_source_path', 'default')}")
        print(f"   🏭 Station/Basin IDs: {', '.join(data_cfgs.get('basin_ids', []))}")
        print(f"   ⏱️ Warmup length: {data_cfgs.get('warmup_length', 365)} steps")
        print(f"   📋 Variables: {data_cfgs.get('variables', [])}")

        print("\n🔧 Model Configuration:")
        model_params = model_cfgs.get("model_params", {})
        print(f"   🏷️ Model name: {model_cfgs.get('model_name')}")
        
        # Model-specific parameter display
        if model_type == "Unit Hydrograph":
            print(f"   📏 Unit hydrograph length: {model_params.get('n_uh')}")
            print(f"   🔀 Smoothing factor: {model_params.get('smoothing_factor')}")
        elif model_type == "Categorized Unit Hydrograph":
            uh_lengths = model_params.get("uh_lengths", {})
            print(f"   📏 UH lengths: {uh_lengths}")
            print(f"   🏷️ Category weights available: {list(model_params.get('category_weights', {}).keys())}")
        elif model_type == "XAJ":
            print(f"   📋 Source type: {model_params.get('source_type')}")
            print(f"   📖 Source book: {model_params.get('source_book')}")
            print(f"   🔧 Kernel size: {model_params.get('kernel_size')}")

        print("\n🎯 Training Configuration:")
        print(f"   🔬 Algorithm: {training_cfgs.get('algorithm_name')}")
        print(f"   📊 Objective: {training_cfgs.get('loss_config', {}).get('obj_func')}")
        print(f"   📁 Output: {training_cfgs.get('output_dir')}/{training_cfgs.get('experiment_name')}")

        # Algorithm parameters
        algo_params = training_cfgs.get("algorithm_params", {})
        if algo_params:
            print(f"   ⚙️ Algorithm Parameters:")
            for key, value in list(algo_params.items())[:3]:  # Show first 3
                print(f"      {key}: {value}")
            if len(algo_params) > 3:
                print(f"      ... ({len(algo_params)-3} more parameters)")

        # Check algorithm availability
        algorithm_name = training_cfgs.get("algorithm_name")
        if algorithm_name == "genetic_algorithm":
            try:
                import deap
                print(f"   🧬 DEAP Available: Yes")
            except ImportError:
                print(f"   ❌ ERROR: Algorithm '{algorithm_name}' requires DEAP package")
                print("   💡 Install with: pip install deap")
                return False

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
            print(f"\n🔄 Loading flood events data...")

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
    def show_help_information(script_name: str, model_type: str, examples_pattern: str = None):
        """
        Show helpful information when no config is provided
        
        Parameters
        ----------
        script_name : str
            Name of the script
        model_type : str
            Type of model for examples
        examples_pattern : str, optional
            Pattern to search for example configs
        """
        print(f"🔍 {model_type} Model Calibration with Unified Config")
        print("=" * 60)
        print()
        print("📋 Available options:")
        print()
        print("1️⃣ Use existing configuration file:")
        print(f"   python {script_name} --config my_config.yaml")
        print()
        print("2️⃣ Quick setup mode (no config file needed):")
        print(f"   python {script_name} --quick-setup")
        print(f"   python {script_name} --quick-setup --station-id basin_001 --algorithm SCE_UA")
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
                print(f"   - python {script_name} --config configs/examples/example_config.yaml")
        else:
            print(f"   - python {script_name} --config configs/examples/example_config.yaml")

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
                input("Would you like to run with default quick setup? (y/n): ")
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
    def print_completion_message(config: Dict[str, Any], task_name: str = "operation"):
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
            "--config", "-c", type=str,
            help="Path to configuration file (YAML or JSON)"
        )

        parser.add_argument(
            "--create-template", type=str,
            help="Create configuration template and exit"
        )

        # Quick setup mode
        parser.add_argument(
            "--quick-setup", action="store_true",
            help="Quick setup mode using command line arguments"
        )

        # Data configuration
        parser.add_argument(
            "--data-path", type=str,
            help="Data directory path (quick setup mode)"
        )

        parser.add_argument(
            "--station-id", type=str, default="basin_001",
            help="Station/Basin ID for calibration (quick setup mode)"
        )

        parser.add_argument(
            "--warmup-length", type=int, default=365,
            help="Warmup length in time steps (quick setup mode)"
        )

        # Training configuration
        parser.add_argument(
            "--algorithm", type=str, default="SCE_UA",
            choices=["scipy_minimize", "SCE_UA", "genetic_algorithm"],
            help="Optimization algorithm (quick setup mode)"
        )

        # Common options
        parser.add_argument(
            "--override", "-o", action="append",
            help="Override config values (e.g., -o model_cfgs.model_params.n_uh=32)"
        )

        parser.add_argument(
            "--output-dir", type=str,
            help="Override output directory"
        )

        parser.add_argument(
            "--experiment-name", type=str,
            help="Override experiment name"
        )

        parser.add_argument(
            "--verbose", "-v", action="store_true", 
            help="Verbose output"
        )

        parser.add_argument(
            "--dry-run", action="store_true",
            help="Dry run - validate config and show what would be done"
        )

        parser.add_argument(
            "--plot-results", action="store_true",
            help="Generate plots of results"
        )

        parser.add_argument(
            "--save-evaluation", action="store_true",
            help="Save detailed evaluation results to CSV"
        )

    @staticmethod
    def handle_template_creation(args, template_creation_func, model_type: str) -> bool:
        """
        Handle template creation request
        
        Parameters
        ----------
        args : argparse.Namespace
            Command line arguments
        template_creation_func : callable
            Function to create template
        model_type : str
            Type of model for display
            
        Returns
        -------
        bool
            True if template was created (script should exit)
        """
        if args.create_template:
            print(f"🔧 Creating {model_type.lower()} configuration template: {args.create_template}")
            config = template_creation_func(args.create_template)
            print(f"✅ Template saved to: {args.create_template}")
            print("\n📋 Configuration template preview:")
            
            # Show compact preview
            preview = {
                "data_cfgs": {k: v for k, v in config.get("data_cfgs", {}).items() if k in ["data_source_type", "model_name"]},
                "model_cfgs": {"model_name": config.get("model_cfgs", {}).get("model_name")},
                "training_cfgs": {k: v for k, v in config.get("training_cfgs", {}).items() if k in ["algorithm_name", "experiment_name"]},
            }
            print(yaml.dump(preview, default_flow_style=False, indent=2))
            return True
        return False

    @staticmethod
    def setup_configuration(args, create_config_func, script_name: str, 
                          model_type: str, examples_pattern: str = None):
        """
        Setup configuration from arguments with unified workflow
        
        Parameters
        ----------
        args : argparse.Namespace
            Command line arguments
        create_config_func : callable
            Function to create config from args
        script_name : str
            Name of the script
        model_type : str
            Type of model
        examples_pattern : str, optional
            Pattern for example configs
            
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
                
        elif args.quick_setup:
            # Quick setup mode
            print("🚀 Quick setup mode - creating configuration from command line arguments")
            return create_config_func(args)
            
        else:
            # No config provided - show help and prompt
            ScriptUtils.show_help_information(script_name, model_type, examples_pattern)
            return ScriptUtils.prompt_quick_setup(args, create_config_func)