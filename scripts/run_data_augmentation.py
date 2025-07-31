"""
Author: Wenyu Ouyang
Date: 2025-07-16 17:14:24
LastEditTime: 2025-07-31 16:25:03
LastEditors: Wenyu Ouyang
Description: Real Data Augmentation Script for Hydrological Data
FilePath: \hydromodel\scripts\run_data_augmentation.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np

from hydromodel.models.consts import OBS_FLOW, NET_RAIN
from hydromodel.models.data_augment import (
    create_real_data_augmenter,
    load_real_hydrological_data,
)
from hydromodel.models.data_augment import HydrologicalDataAugmenter


def create_sample_data():
    """
    Create sample data for demonstration purposes

    Returns:
        Dict with sample optimal events and unit hydrographs
    """
    print("📊 Creating sample data for demonstration...")

    # Sample optimal events (based on the data format shown in the conversation)
    optimal_events = [
        {
            NET_RAIN: np.array(
                [15.2, 8.5, 3.2, 1.1, 0.0]
            ),  # Net rainfall in mm
            OBS_FLOW: np.array(
                [2.5, 12.8, 18.3, 15.6, 8.9, 4.2, 1.8]
            ),  # Direct runoff
            "filepath": "event_19940816.csv",
            "peak_obs": 18.3,
            "m_eff": 4,
            "n_specific": 7,
        },
        {
            NET_RAIN: np.array([8.7, 12.4, 6.8, 2.3, 0.5]),
            OBS_FLOW: np.array([1.8, 8.9, 15.2, 14.8, 10.3, 6.1, 2.9]),
            "filepath": "event_20120804.csv",
            "peak_obs": 15.2,
            "m_eff": 5,
            "n_specific": 7,
        },
        {
            NET_RAIN: np.array([22.1, 18.6, 9.4, 4.2, 1.8]),
            OBS_FLOW: np.array([3.2, 15.8, 25.4, 22.9, 16.7, 9.8, 5.1]),
            "filepath": "event_19970821.csv",
            "peak_obs": 25.4,
            "m_eff": 5,
            "n_specific": 7,
        },
    ]

    # Sample unit hydrographs (corresponding to the events)
    unit_hydrographs = {
        "event_19940816.csv": np.array([0.0, 1.2, 3.8, 5.2, 4.1, 2.3, 0.8]),
        "event_20120804.csv": np.array([0.0, 0.9, 3.2, 4.8, 4.5, 2.8, 1.1]),
        "event_19970821.csv": np.array([0.0, 1.5, 4.2, 6.1, 5.3, 3.1, 1.2]),
    }

    # Watershed information (optional)
    watershed_info = {
        "name": "Sample Watershed",
        "area_km2": 1250.0,
        "description": "Demonstration watershed for data augmentation",
    }

    return {
        "optimal_events": optimal_events,
        "unit_hydrographs": unit_hydrographs,
        "watershed_info": watershed_info,
    }


def real_data_shared_optimization():
    """Demonstrate real data augmentation with shared optimization"""
    print("\n" + "=" * 60)
    print("🚀 Real Data Augmentation Demo - Shared Optimization")
    print("=" * 60)

    try:
        # Create augmenter with real data using shared optimization
        augmenter = create_real_data_augmenter(
            station_id="songliao_21401550",
            optimization_mode="shared",
            top_n_events=5,
            min_nse_threshold=0.7,
            uh_length=24,
            scaling_factors=[0.8, 1.0, 1.2, 1.5],
            verbose=True,
        )

        # Generate augmented events (data already loaded and fitted)
        print("\n🔄 Generating augmented events...")
        augmented_events = augmenter.transform()

        # Show results
        print(f"\n✅ Generated {len(augmented_events)} augmented events")

        # Generate and save summary
        summary_df = augmenter.get_augmentation_summary(augmented_events)
        print(f"\n📊 Summary of first 5 events:")
        print(summary_df.head())

        # Save results
        output_dir = "results/real_data_augmentation_shared"
        augmenter.save_augmented_events(augmented_events, output_dir)

        summary_file = os.path.join(output_dir, "augmentation_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"📊 Summary saved to: {summary_file}")

        return augmented_events, summary_df

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print(
            "💡 This may be because real data is not available or paths are incorrect"
        )
        return None, None


def real_data_categorized_optimization():
    """Demonstrate real data augmentation with categorized optimization"""
    print("\n" + "=" * 60)
    print("🚀 Real Data Augmentation Demo - Categorized Optimization")
    print("=" * 60)

    try:
        # Create augmenter with real data using categorized optimization
        augmenter = create_real_data_augmenter(
            station_id="songliao_21401550",
            optimization_mode="categorized",
            top_n_events=6,  # 2 per category
            min_nse_threshold=0.6,
            uh_length=24,
            scaling_factors=[0.8, 1.2, 1.5],
            verbose=True,
        )

        # Generate augmented events
        print("\n🔄 Generating augmented events...")
        augmented_events = augmenter.transform()

        # Show results
        print(f"\n✅ Generated {len(augmented_events)} augmented events")

        # Generate and save summary
        summary_df = augmenter.get_augmentation_summary(augmented_events)
        print(f"\n📊 Summary by source event:")
        source_summary = (
            summary_df.groupby("source_event")
            .agg(
                {
                    "scale_factor": "count",
                    "peak_flow": ["min", "max", "mean"],
                    "total_rainfall": ["min", "max", "mean"],
                }
            )
            .round(2)
        )
        print(source_summary)

        # Save results
        output_dir = "results/real_data_augmentation_categorized"
        augmenter.save_augmented_events(augmented_events, output_dir)

        summary_file = os.path.join(output_dir, "augmentation_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"📊 Summary saved to: {summary_file}")

        return augmented_events, summary_df

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print(
            "💡 This may be because real data is not available or paths are incorrect"
        )
        return None, None


def demo_load_from_results_file():
    """Demonstrate loading optimal events from existing results file"""
    print("\n" + "=" * 60)
    print("🚀 Load from Results File Demo")
    print("=" * 60)

    # Check for existing results files
    possible_results = [
        "results/UH_shared_eva_output_songliao_songliao_21401550.csv",
        "results/UH_categorized_eva_output_songliao_songliao_21401550.csv",
        "results/UH_shared_eva_output.csv",
        "results/UH_categorized_eva_output.csv",
    ]

    results_file = None
    for file_path in possible_results:
        if os.path.exists(file_path):
            results_file = file_path
            break

    if not results_file:
        print("❌ No existing results file found. Available files:")
        for file_path in possible_results:
            print(
                f"   - {file_path} {'✅' if os.path.exists(file_path) else '❌'}"
            )
        print(
            "💡 Run unit hydrograph optimization scripts first to generate results"
        )
        return None, None

    try:
        print(f"📂 Using results file: {results_file}")

        # Create augmenter from results file
        augmenter = create_real_data_augmenter(
            results_file=results_file,
            top_n_events=5,
            min_nse_threshold=0.8,
            scaling_factors=[0.8, 1.2, 1.5],
            verbose=True,
        )

        # Generate augmented events
        print("\n🔄 Generating augmented events...")
        augmented_events = augmenter.transform()

        # Show results
        print(
            f"\n✅ Generated {len(augmented_events)} augmented events from results file"
        )

        # Generate and save summary
        summary_df = augmenter.get_augmentation_summary(augmented_events)
        print(f"\n📊 Summary:")
        print(
            summary_df[
                [
                    "event_name",
                    "source_event",
                    "scale_factor",
                    "peak_flow",
                    "year",
                ]
            ].head()
        )

        # Save results
        output_dir = "results/real_data_augmentation_from_results"
        augmenter.save_augmented_events(augmented_events, output_dir)

        summary_file = os.path.join(output_dir, "augmentation_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"📊 Summary saved to: {summary_file}")

        return augmented_events, summary_df

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def demo_manual_data_loading():
    """Demonstrate manual data loading and augmentation"""
    print("\n" + "=" * 60)
    print("🚀 Manual Data Loading Demo")
    print("=" * 60)

    try:
        # Load real data manually
        print("🔄 Loading real data manually...")
        real_data = load_real_hydrological_data(
            station_id="songliao_21401550",
            optimization_mode="shared",
            top_n_events=3,
            min_nse_threshold=0.7,
            uh_length=20,
            verbose=True,
        )

        augmenter = HydrologicalDataAugmenter(
            scaling_factors=[0.8, 1.2], verbose=True
        )

        # Fit with real data
        augmenter.fit(real_data)

        # Generate augmented events
        augmented_events = augmenter.transform()

        print(
            f"\n✅ Manual loading successful! Generated {len(augmented_events)} events"
        )

        # Show first few events
        summary_df = augmenter.get_augmentation_summary(augmented_events)
        print("\n📊 First 3 augmented events:")
        print(
            summary_df[
                [
                    "event_name",
                    "source_event",
                    "scale_factor",
                    "periods_used",
                    "peak_flow",
                ]
            ].head(3)
        )

        return augmented_events, summary_df

    except Exception as e:
        print(f"❌ Manual loading demo failed: {e}")
        return None, None


def main():
    """Main demonstration function"""
    print("🚀 Real Data Hydrological Data Augmentation Demonstration")
    print("=" * 70)

    success_count = 0
    total_demos = 4

    # Demo 1: Real data with shared optimization
    result = real_data_shared_optimization()
    if result[0] is not None:
        success_count += 1

    # Demo 2: Real data with categorized optimization
    result = real_data_categorized_optimization()
    if result[0] is not None:
        success_count += 1

    # Demo 3: Load from existing results file
    result = demo_load_from_results_file()
    if result[0] is not None:
        success_count += 1

    # Demo 4: Manual data loading
    result = demo_manual_data_loading()
    if result[0] is not None:
        success_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("📊 Demonstration Summary")
    print("=" * 70)
    print(f"✅ Successful demos: {success_count}/{total_demos}")

    if success_count > 0:
        print(
            f"📁 Check the 'results/' directory for generated augmented events"
        )
        print(f"🔍 Each demo creates its own subdirectory with:")
        print(f"   - Individual CSV files for each augmented event")
        print(f"   - Summary CSV file with statistics")

    if success_count < total_demos:
        print(
            f"💡 Some demos failed - this is normal if real data is not set up"
        )
        print(f"   To use real data:")
        print(f"   1. Ensure hydrodatasource is properly configured")
        print(f"   2. Run unit hydrograph optimization scripts first")
        print(f"   3. Check data paths in SETTING configuration")


if __name__ == "__main__":
    main()
