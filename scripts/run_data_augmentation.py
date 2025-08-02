"""
Author: Wenyu Ouyang
Date: 2025-07-16 17:14:24
LastEditTime: 2025-08-01 19:31:41
LastEditors: Wenyu Ouyang
Description: Real Data Augmentation Script for Hydrological Data
FilePath: /hydromodel/scripts/run_data_augmentation.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os

from hydromodel.models.data_augment import HydrologicalDataAugmenter


def real_data_shared_optimization():
    """Demonstrate real data augmentation with shared optimization"""
    print("\n" + "=" * 60)
    print("🚀 Real Data Augmentation Demo - Shared Optimization")
    print("=" * 60)

    try:
        # Create augmenter with real data using shared optimization
        augmenter = HydrologicalDataAugmenter(
            station_id="songliao_21401550",
            optimization_mode="shared",
            top_n_events=5,
            min_nse_threshold=0.7,
            uh_length=24,
            scaling_factors=[0.8, 1.0, 1.2, 1.5],
            verbose=True,
        )

        # Generate augmented events (data already loaded and initialized)
        print("\n🔄 Generating augmented events...")
        augmented_events = augmenter.augment_data({})

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
        augmenter = HydrologicalDataAugmenter(
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
        augmented_events = augmenter.augment_data({})

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


def main():
    """Main demonstration function"""
    print("🚀 Real Data Hydrological Data Augmentation Demonstration")
    print("=" * 70)

    success_count = 0
    total_demos = 2

    # Demo 1: Real data with shared optimization
    result = real_data_shared_optimization()
    if result[0] is not None:
        success_count += 1

    # Demo 2: Real data with categorized optimization
    result = real_data_categorized_optimization()
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
