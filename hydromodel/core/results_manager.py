"""
Author: Wenyu Ouyang
Date: 2025-08-08
LastEditTime: 2025-08-08 10:00:00
LastEditors: Wenyu Ouyang
Description: Unified results management system for all hydrological models
FilePath: /hydromodel/hydromodel/core/results_manager.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path

# Optional imports - handle missing dependencies gracefully
try:
    from hydroutils.hydro_plot import (
        plot_unit_hydrograph,
        setup_matplotlib_chinese,
    )

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from hydromodel.trainers.unit_hydrograph_trainer import (
        evaluate_single_event_from_uh,
        print_report_preview,
        save_results_to_csv,
        print_category_statistics,
        categorize_floods_by_peak,
    )

    UH_TRAINER_AVAILABLE = True
except ImportError:
    UH_TRAINER_AVAILABLE = False

try:
    from hydrodatasource.reader.floodevent import FloodEventDatasource

    FLOODEVENT_AVAILABLE = True
except ImportError:
    FLOODEVENT_AVAILABLE = False


class ModelResultsProcessor(ABC):
    """Abstract base class for model-specific results processing"""

    @abstractmethod
    def extract_parameters(
        self, results: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract model parameters from calibration results"""
        pass

    @abstractmethod
    def display_parameters(
        self, parameters: Dict[str, Any], config: Dict[str, Any]
    ) -> None:
        """Display model parameters in a readable format"""
        pass

    @abstractmethod
    def visualize_model(
        self,
        parameters: Dict[str, Any],
        config: Dict[str, Any],
        output_dir: str,
    ) -> List[str]:
        """Generate model-specific visualizations"""
        pass

    @abstractmethod
    def evaluate_performance(
        self, parameters: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate model performance with specific parameters"""
        pass


class XAJResultsProcessor(ModelResultsProcessor):
    """Results processor for XAJ models"""

    def extract_parameters(
        self, results: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract XAJ parameters from calibration results"""
        basin_id = config.get("data_cfgs", {}).get("basin_ids", [""])[0]
        best_params = self._get_best_params(results)

        if basin_id in best_params:
            return best_params[basin_id].get("xaj", {}) or best_params[
                basin_id
            ].get("xaj_mz", {})
        return {}

    def display_parameters(
        self, parameters: Dict[str, Any], config: Dict[str, Any]
    ) -> None:
        """Display XAJ parameters"""
        if not parameters:
            print("⚠️ No XAJ parameters found")
            return

        print(f"📊 XAJ Model Parameters ({len(parameters)} parameters):")

        # Group parameters by category
        evaporation_params = ["K", "B", "IM", "UM", "LM", "DM", "C"]
        runoff_params = ["SM", "EX", "KI", "KG"]
        routing_params = ["A", "THETA", "CI", "CG"]

        for group, group_params in [
            ("Evaporation", evaporation_params),
            ("Runoff", runoff_params),
            ("Routing", routing_params),
        ]:
            group_values = {
                k: v for k, v in parameters.items() if k in group_params
            }
            if group_values:
                print(f"  {group} Parameters:")
                for param, value in group_values.items():
                    print(f"    {param}: {value:.6f}")

    def visualize_model(
        self,
        parameters: Dict[str, Any],
        config: Dict[str, Any],
        output_dir: str,
    ) -> List[str]:
        """Generate XAJ model visualizations"""
        plots = []

        # For XAJ, we can create parameter distribution plots, sensitivity plots, etc.
        # This is a placeholder for future XAJ-specific visualizations
        if PLOTTING_AVAILABLE:
            # Could add parameter correlation plots, sensitivity analysis, etc.
            pass
        else:
            print("⚠️ Plotting not available for XAJ visualizations")

        return plots

    def evaluate_performance(
        self, parameters: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate XAJ model performance"""
        # This would integrate with simulate interface to get predictions
        # and calculate performance metrics
        return {"status": "XAJ performance evaluation not implemented yet"}

    def _get_best_params(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract best parameters from results"""
        if isinstance(results, dict) and len(results) == 1:
            basin_result = list(results.values())[0]
            return basin_result.get("best_params", {})
        return results.get("best_params", {})


class UnitHydrographResultsProcessor(ModelResultsProcessor):
    """Results processor for unit hydrograph models"""

    def extract_parameters(
        self, results: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract unit hydrograph parameters from calibration results"""
        basin_id = config.get("data_cfgs", {}).get("basin_ids", [""])[0]
        best_params = self._get_best_params(results)

        if (
            basin_id in best_params
            and "unit_hydrograph" in best_params[basin_id]
        ):
            uh_params_dict = best_params[basin_id]["unit_hydrograph"]
            model_cfgs = config.get("model_cfgs", {})
            n_uh = model_cfgs.get("model_params", {}).get("n_uh", 24)

            # Extract unit hydrograph parameters
            if isinstance(uh_params_dict, dict):
                uh_params = [
                    uh_params_dict.get(f"uh_{i+1}", 0.0) for i in range(n_uh)
                ]
            else:
                uh_params = (
                    list(uh_params_dict)
                    if hasattr(uh_params_dict, "__iter__")
                    else []
                )

            return {"uh_values": uh_params, "n_uh": len(uh_params)}
        return {}

    def display_parameters(
        self, parameters: Dict[str, Any], config: Dict[str, Any]
    ) -> None:
        """Display unit hydrograph parameters"""
        uh_params = parameters.get("uh_values", [])

        if not uh_params:
            print("⚠️ No unit hydrograph parameters found")
            return

        print(f"📊 Unit Hydrograph Parameters ({len(uh_params)} values):")
        for i, param in enumerate(uh_params[:5]):  # Show first 5 values
            print(f"   uh_{i+1}: {param:.6f}")
        if len(uh_params) > 5:
            print(f"   ... ({len(uh_params)-5} more parameters)")

        # Show statistical summary
        uh_array = np.array(uh_params)
        print(f"   Sum: {uh_array.sum():.6f} (should be ~1.0)")
        print(
            f"   Peak: {uh_array.max():.6f} at time step {np.argmax(uh_array)+1}"
        )

    def visualize_model(
        self,
        parameters: Dict[str, Any],
        config: Dict[str, Any],
        output_dir: str,
    ) -> List[str]:
        """Generate unit hydrograph visualizations"""
        plots = []
        uh_params = parameters.get("uh_values", [])

        if not uh_params or not PLOTTING_AVAILABLE:
            if not PLOTTING_AVAILABLE:
                print("⚠️ Plotting not available - install hydroutils package")
            return plots

        try:
            setup_matplotlib_chinese()

            # Generate unit hydrograph plot
            plot_path = os.path.join(output_dir, "unit_hydrograph.png")
            plot_unit_hydrograph(
                uh_params, "Calibrated Unit Hydrograph", save_path=plot_path
            )
            plots.append(plot_path)

            print(f"📈 Unit hydrograph plot saved to: {plot_path}")

        except Exception as e:
            print(f"⚠️ Failed to generate unit hydrograph plot: {e}")

        return plots

    def evaluate_performance(
        self, parameters: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate unit hydrograph performance"""
        if not UH_TRAINER_AVAILABLE or not FLOODEVENT_AVAILABLE:
            print(
                "⚠️ Performance evaluation not available - missing dependencies"
            )
            return {"status": "evaluation_unavailable"}

        try:
            # Load flood events data
            data_cfgs = config.get("data_cfgs", {})
            # Get time unit from config, default to ["3h"] if not specified
            time_unit = data_cfgs.get("time_unit", ["3h"])
            # Ensure time_unit is a list
            if isinstance(time_unit, str):
                time_unit = [time_unit]

            dataset = FloodEventDatasource(
                data_cfgs.get("data_source_path"),
                time_unit=time_unit,
                trange4cache=["1960-01-01 02", "2024-12-31 23"],
                warmup_length=data_cfgs.get("warmup_length", 480),
            )

            basin_ids = data_cfgs.get("basin_ids", [])
            all_event_data = dataset.load_1basin_flood_events(
                station_id=basin_ids[0],
                flow_unit="mm/3h",
                include_peak_obs=True,
                verbose=False,
            )

            if not all_event_data:
                return {"status": "no_data"}

            # Evaluate each event
            uh_params = parameters.get("uh_values", [])
            evaluation_results = []

            for event in all_event_data:
                result = evaluate_single_event_from_uh(
                    event,
                    uh_params,
                    net_rain_key=data_cfgs.get("net_rain_key"),
                    obs_flow_key=data_cfgs.get("obs_flow_key"),
                )
                if result:
                    evaluation_results.append(result)

            if evaluation_results:
                df = pd.DataFrame(evaluation_results)
                df_sorted = df.sort_values("NSE", ascending=False)

                # Calculate summary statistics
                summary = {
                    "n_events": len(evaluation_results),
                    "mean_nse": df["NSE"].mean(),
                    "median_nse": df["NSE"].median(),
                    "mean_rmse": df["RMSE"].mean(),
                    "good_events": len(df[df["NSE"] > 0.5]),
                }

                return {
                    "status": "success",
                    "summary": summary,
                    "detailed_results": df_sorted,
                }

            return {"status": "no_valid_results"}

        except Exception as e:
            print(f"❌ Performance evaluation failed: {e}")
            return {"status": "evaluation_failed", "error": str(e)}

    def _get_best_params(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract best parameters from results"""
        if isinstance(results, dict) and len(results) == 1:
            basin_result = list(results.values())[0]
            return basin_result.get("best_params", {})
        return results.get("best_params", {})


class CategorizedUHResultsProcessor(ModelResultsProcessor):
    """Results processor for categorized unit hydrograph models"""

    def extract_parameters(
        self, results: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract categorized unit hydrograph parameters"""
        basin_id = config.get("data_cfgs", {}).get("basin_ids", [""])[0]
        best_params = self._get_best_params(results)

        cat_uh_params = None
        if (
            basin_id in best_params
            and "categorized_unit_hydrograph" in best_params[basin_id]
        ):
            cat_uh_params = best_params[basin_id][
                "categorized_unit_hydrograph"
            ]
        elif "categorized_unit_hydrograph" in best_params:
            cat_uh_params = best_params["categorized_unit_hydrograph"]

        if cat_uh_params:
            # Extract UH parameters for each category
            extracted_params = {}
            for category, params_dict in cat_uh_params.items():
                if isinstance(params_dict, dict):
                    extracted_params[category] = list(params_dict.values())
            return extracted_params
        return {}

    def display_parameters(
        self, parameters: Dict[str, Any], config: Dict[str, Any]
    ) -> None:
        """Display categorized unit hydrograph parameters"""
        if not parameters:
            print("⚠️ No categorized unit hydrograph parameters found")
            return

        print(f"📏 Unit Hydrograph Parameters by Category:")
        for category, uh_params in parameters.items():
            print(
                f"   📈 {category.capitalize()}: {len(uh_params)} parameters"
            )
            print(f"      First 3 values: {uh_params[:3]}")

            # Show statistical summary for each category
            uh_array = np.array(uh_params)
            print(
                f"      Sum: {uh_array.sum():.6f}, Peak: {uh_array.max():.6f}"
            )

    def visualize_model(
        self,
        parameters: Dict[str, Any],
        config: Dict[str, Any],
        output_dir: str,
    ) -> List[str]:
        """Generate categorized unit hydrograph visualizations"""
        plots = []

        if not parameters or not PLOTTING_AVAILABLE:
            if not PLOTTING_AVAILABLE:
                print("⚠️ Plotting not available - install hydroutils package")
            return plots

        try:
            setup_matplotlib_chinese()

            for category, uh_params in parameters.items():
                plot_path = os.path.join(
                    output_dir, f"categorized_uh_{category}.png"
                )
                plot_unit_hydrograph(
                    uh_params,
                    f"Categorized Unit Hydrograph - {category.capitalize()}",
                    save_path=plot_path,
                )
                plots.append(plot_path)

            print(
                f"📈 Generated {len(plots)} categorized unit hydrograph plots"
            )

        except Exception as e:
            print(f"⚠️ Failed to generate categorized UH plots: {e}")

        return plots

    def evaluate_performance(
        self, parameters: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate categorized unit hydrograph performance"""
        if not UH_TRAINER_AVAILABLE or not FLOODEVENT_AVAILABLE:
            print(
                "⚠️ Performance evaluation not available - missing dependencies"
            )
            return {"status": "evaluation_unavailable"}

        # Implementation similar to unit hydrograph but with categorization
        # This would be more complex as it needs to categorize events first
        return {"status": "categorized_uh_evaluation_not_fully_implemented"}

    def _get_best_params(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract best parameters from results"""
        if isinstance(results, dict) and len(results) == 1:
            basin_result = list(results.values())[0]
            return basin_result.get("best_params", {})
        return results.get("best_params", {})


class ResultsManager:
    """Unified results management system for all hydrological models"""

    def __init__(self):
        self.processors = {
            "xaj": XAJResultsProcessor(),
            "xaj_mz": XAJResultsProcessor(),
            "unit_hydrograph": UnitHydrographResultsProcessor(),
            "categorized_unit_hydrograph": CategorizedUHResultsProcessor(),
            # Add more processors as needed
            "gr4j": XAJResultsProcessor(),  # Placeholder - would need specific GR processor
            "gr6j": XAJResultsProcessor(),  # Placeholder - would need specific GR processor
        }

    def process_results(
        self, results: Dict[str, Any], config: Dict[str, Any], args: Any = None
    ) -> Dict[str, Any]:
        """
        Unified results processing for all model types

        Parameters
        ----------
        results : Dict[str, Any]
            Calibration results from unified calibrate interface
        config : Dict[str, Any]
            Configuration used for calibration
        args : Any, optional
            Command line arguments (for backward compatibility)

        Returns
        -------
        Dict[str, Any]
            Processed results with extracted parameters, evaluations, and file paths
        """
        print(f"\n📈 Calibration Results Processing")
        print("=" * 60)

        # Extract basic information
        convergence, objective_value = self._extract_basic_info(results)
        model_name = config.get("model_cfgs", {}).get("model_name", "unknown")

        print(f"✅ Convergence: {convergence}")
        print(f"🎯 Best objective value: {objective_value:.6f}")
        print(f"🏷️ Model type: {model_name}")

        # Get model-specific processor
        processor = self.processors.get(model_name)
        if not processor:
            print(
                f"⚠️ No specific processor for model type '{model_name}', using default"
            )
            processor = self.processors["xaj"]  # Use XAJ as default

        # Extract and display parameters
        parameters = processor.extract_parameters(results, config)
        if parameters:
            processor.display_parameters(parameters, config)

        # Set up output directory
        training_cfgs = config.get("training_cfgs", {})
        output_dir = os.path.join(
            training_cfgs.get("output_dir", "results"),
            training_cfgs.get("experiment_name", "experiment"),
        )
        os.makedirs(output_dir, exist_ok=True)

        # Generate visualizations if requested
        plots = []
        if args and getattr(args, "plot_results", False):
            plots = processor.visualize_model(parameters, config, output_dir)

        # Evaluate performance if requested
        performance = {}
        if args and getattr(args, "save_evaluation", False):
            performance = processor.evaluate_performance(parameters, config)
            if performance.get("status") == "success":
                self._save_performance_results(
                    performance, output_dir, model_name
                )

        # Save parameters to file
        params_file = os.path.join(output_dir, f"{model_name}_parameters.json")
        self._save_parameters(parameters, params_file)

        return {
            "status": "success",
            "convergence": convergence,
            "objective_value": objective_value,
            "model_name": model_name,
            "parameters": parameters,
            "performance": performance,
            "output_dir": output_dir,
            "plots": plots,
            "parameters_file": params_file,
        }

    def _extract_basic_info(
        self, results: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Extract basic convergence and objective value information"""
        if isinstance(results, dict) and len(results) == 1:
            basin_result = list(results.values())[0]
            convergence = basin_result.get("convergence", "unknown")
            objective_value = basin_result.get("objective_value", float("inf"))
        else:
            convergence = results.get("convergence", "unknown")
            objective_value = results.get("objective_value", float("inf"))

        return convergence, objective_value

    def _save_parameters(
        self, parameters: Dict[str, Any], filepath: str
    ) -> None:
        """Save parameters to JSON file"""
        try:
            with open(filepath, "w") as f:
                json.dump(parameters, f, indent=2, default=str)
            print(f"💾 Parameters saved to: {filepath}")
        except Exception as e:
            print(f"⚠️ Failed to save parameters: {e}")

    def _save_performance_results(
        self, performance: Dict[str, Any], output_dir: str, model_name: str
    ) -> None:
        """Save performance evaluation results"""
        try:
            if "detailed_results" in performance:
                df = performance["detailed_results"]
                csv_file = os.path.join(
                    output_dir, f"{model_name}_evaluation.csv"
                )
                df.to_csv(csv_file, index=False)

                # Show preview
                if UH_TRAINER_AVAILABLE:
                    print_report_preview(
                        df,
                        f"{model_name.replace('_', ' ').title()} Evaluation",
                        top_n=5,
                    )

                print(f"💾 Detailed evaluation saved to: {csv_file}")

            # Save summary
            summary_file = os.path.join(
                output_dir, f"{model_name}_summary.json"
            )
            with open(summary_file, "w") as f:
                json.dump(performance.get("summary", {}), f, indent=2)

        except Exception as e:
            print(f"⚠️ Failed to save performance results: {e}")


# Global instance for easy access
results_manager = ResultsManager()
