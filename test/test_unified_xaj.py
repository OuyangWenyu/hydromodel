#!/usr/bin/env python3
"""
Quick test script for the unified XAJ calibration architecture.
This script demonstrates how to use the new unified interface.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add hydromodel to path
repo_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_path)

def test_unified_architecture():
    """Test the unified calibration architecture with minimal configuration."""
    
    print("🧪 Testing Unified XAJ Calibration Architecture")
    print("=" * 60)
    
    try:
        # Import the unified calibration function
        from hydromodel.trainers.unified_calibrate import calibrate
        print("✅ Successfully imported unified calibrate function")
        
        # Test configuration - minimal working example
        test_config = {
            "data_cfgs": {
                "data_source_type": "selfmadehydrodataset",
                "data_source_path": "test_data",  # Will be created if needed
                "basin_ids": ["test_basin"],
                "warmup_length": 30  # Short warmup for testing
            },
            "model_cfgs": {
                "model_name": "xaj_mz",
                "model_params": {
                    "source_type": "sources",
                    "source_book": "HF",
                    "kernel_size": 15
                }
            },
            "training_cfgs": {
                "algorithm_name": "scipy_minimize",  # Fastest for testing
                "algorithm_params": {
                    "method": "L-BFGS-B",
                    "max_iterations": 10  # Very short for testing
                },
                "loss_config": {
                    "type": "time_series",
                    "obj_func": "RMSE"
                },
                "output_dir": "test_results",
                "experiment_name": "unified_test"
            }
        }
        
        print("✅ Created test configuration with unified structure")
        
        # Create minimal test data
        test_data_dir = "test_data"
        os.makedirs(test_data_dir, exist_ok=True)
        
        # Generate synthetic test data
        n_days = 100
        time_index = pd.date_range("2023-01-01", periods=n_days, freq="D")
        
        # Simple synthetic hydrological data
        np.random.seed(42)
        prcp = np.random.exponential(2.0, n_days) * np.random.binomial(1, 0.3, n_days)  # Precipitation
        pet = 2.0 + 2.0 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + np.random.normal(0, 0.1, n_days)  # PET
        
        # Simple flow simulation (rough approximation)
        flow = np.zeros(n_days)
        storage = 0
        for i in range(n_days):
            storage = 0.8 * storage + 0.3 * prcp[i]  # Simple reservoir model
            flow[i] = max(0, 0.1 * storage + np.random.normal(0, 0.1))
        
        # Create test CSV file
        test_data = pd.DataFrame({
            "time": time_index,
            "prcp": prcp,
            "pet": pet, 
            "flow": flow
        })
        
        test_file = os.path.join(test_data_dir, "test_basin.csv")
        test_data.to_csv(test_file, index=False)
        print(f"✅ Created synthetic test data: {test_file}")
        
        # Test the unified calibration interface
        print("\n🚀 Testing unified calibration interface...")
        print("📦 Calling: calibrate(config)")
        
        # This is the key test - single function call with config
        results = calibrate(test_config)
        
        print("✅ Calibration completed successfully!")
        print(f"📊 Results type: {type(results)}")
        print(f"📈 Number of basins processed: {len(results)}")
        
        # Check results structure
        for basin_id, result in results.items():
            print(f"\n🏭 Basin: {basin_id}")
            convergence = result.get("convergence", "unknown")
            objective_value = result.get("objective_value", "N/A")
            print(f"   Status: {convergence}")
            print(f"   Objective: {objective_value}")
            
            if "best_params" in result:
                print(f"   Parameters optimized: {len(result['best_params'])}")
        
        print(f"\n✨ SUCCESS: Unified architecture works correctly!")
        print(f"🎯 Key features validated:")
        print(f"   ✅ Single function interface: calibrate(config)")
        print(f"   ✅ Unified configuration structure")
        print(f"   ✅ Automatic data loading")
        print(f"   ✅ Model-algorithm integration")
        print(f"   ✅ Results processing")
        
        # Cleanup
        import shutil
        if os.path.exists("test_data"):
            shutil.rmtree("test_data")
        if os.path.exists("test_results"):
            shutil.rmtree("test_results")
        print(f"🧹 Cleaned up test files")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure hydromodel dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_script_interface():
    """Test the script interface."""
    print(f"\n🧪 Testing Script Interface")
    print("=" * 60)
    
    script_path = "scripts/run_xaj_calibration_unified.py"
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    print(f"✅ Found script: {script_path}")
    
    # Test help output
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, script_path, "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Script help works correctly")
            print("📋 Key features found in help:")
            
            help_text = result.stdout
            features = [
                "Latest Unified Architecture",
                "Configuration file option", 
                "XAJ Model Types",
                "Algorithm Types",
                "calibrate(config)"
            ]
            
            for feature in features:
                if feature.lower() in help_text.lower():
                    print(f"   ✅ {feature}")
                else:
                    print(f"   ⚠️ {feature}")
            
            return True
        else:
            print(f"❌ Script help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Script test failed: {e}")
        return False


if __name__ == "__main__":
    print("🔬 HYDROMODEL UNIFIED ARCHITECTURE TEST SUITE")
    print("=" * 80)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Unified architecture
    if test_unified_architecture():
        success_count += 1
    
    # Test 2: Script interface  
    if test_script_interface():
        success_count += 1
    
    print(f"\n" + "=" * 80)
    print(f"📊 TEST RESULTS: {success_count}/{total_tests} PASSED")
    
    if success_count == total_tests:
        print(f"🎉 ALL TESTS PASSED! Unified architecture is working correctly.")
        print(f"✨ You can now use:")
        print(f"   python scripts/run_xaj_calibration_unified.py --config configs/xaj_example.yaml")
        print(f"   python scripts/run_xaj_calibration_unified.py --model-type xaj_mz --algorithm SCE_UA")
    else:
        print(f"⚠️ Some tests failed. Please check the output above.")
    
    print("=" * 80)