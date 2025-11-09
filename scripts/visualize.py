r"""
Author: zhuanglaihong
Date: 2025-10-30
LastEditTime: 2025-11-03
LastEditors: zhuanglaihong
Description: Command-line interface for visualization of unified evaluation results
FilePath: \hydromodel\scripts\visualize.py
Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.

This script provides a command-line interface for visualizing model evaluation results.
All visualization logic is implemented in hydromodel.datasets.data_visualize module.
"""

import argparse
import sys
from hydromodel.datasets.data_visualize import visualize_evaluation


def main():
    """Main entry point for visualization CLI."""
    parser = argparse.ArgumentParser(
        description="Visualize model evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
  # Visualize all plot types for all basins (default)
  python visualize.py --eval-dir results/my_exp/evaluation_test

  # Visualize specific plot types
  python visualize.py --eval-dir results/my_exp/evaluation_test --plot-types timeseries scatter

  # Visualize specific basins
  python visualize.py --eval-dir results/my_exp/evaluation_test --basins 01013500 01022500

  # Custom output directory
  python visualize.py --eval-dir results/my_exp/evaluation_test --output-dir my_figures

Available plot types:
  - timeseries: Time series with precipitation and streamflow
  - scatter: Observed vs simulated scatter plot
  - fdc: Flow duration curve
  - monthly: Monthly average comparison
  - all: Generate all plot types (default)
        """
    )

    parser.add_argument(
        "--eval-dir",
        type=str,
        required=True,
        help="Path to evaluation directory (e.g., results/exp_name/evaluation_test/)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: eval_dir/figures)"
    )

    parser.add_argument(
        "--basins",
        type=str,
        nargs='+',
        default=None,
        help="Basin IDs to plot (default: all basins)"
    )

    parser.add_argument(
        "--plot-types",
        type=str,
        nargs='+',
        default='all',
        choices=['all', 'timeseries', 'scatter', 'fdc', 'monthly'],
        help="Types of plots to generate (default: all)"
    )

    args = parser.parse_args()

    try:
        # Call the main visualization function from data_visualize module
        visualize_evaluation(
            eval_dir=args.eval_dir,
            output_dir=args.output_dir,
            plot_types=args.plot_types if isinstance(args.plot_types, list) else [args.plot_types],
            basins=args.basins
        )
        return 0
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
