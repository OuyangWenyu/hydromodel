"""
Comprehensive test runner for unified calibration system.

This module runs all unified calibration tests in sequence to verify
the complete system functionality.
"""

import pytest
import sys
from pathlib import Path

def run_unified_calibration_tests():
    """
    Run all unified calibration tests.
    
    This function can be used to run all unified calibration tests
    in a specific order to verify system functionality.
    """
    
    # Get the test directory
    test_dir = Path(__file__).parent
    
    # Define test modules in dependency order
    test_modules = [
        "test_unified_calibration.py",
        "test_categorized_unit_hydrograph.py", 
        "test_usage_examples.py"
    ]
    
    print("Running unified calibration test suite...")
    print("=" * 60)
    
    all_passed = True
    
    for test_module in test_modules:
        test_path = test_dir / test_module
        if test_path.exists():
            print(f"\nRunning {test_module}...")
            result = pytest.main([
                str(test_path), 
                "-v",  # verbose
                "--tb=short",  # short traceback
                "-x"  # stop on first failure
            ])
            if result != 0:
                print(f"FAILED: {test_module}")
                all_passed = False
            else:
                print(f"PASSED: {test_module}")
        else:
            print(f"WARNING: {test_module} not found")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("SUCCESS: All unified calibration tests passed!")
        print("✅ Unified calibration system is working correctly")
    else:
        print("FAILED: Some unified calibration tests failed")
        print("❌ Check test output for details")
    
    return all_passed


def test_import_all_unified_modules():
    """Test that all unified calibration modules can be imported."""
    
    # Test core unified calibration imports
    try:
        from hydromodel.trainers.unified_calibrate import (
            calibrate,
            calibrate_unit_hydrograph,
            calibrate_categorized_unit_hydrograph,
            ModelSetupBase,
            UnitHydrographSetup,
            CategorizedUnitHydrographSetup,
            TraditionalModelSetup,
            DEAP_AVAILABLE
        )
        print("SUCCESS: All core unified calibration imports work")
    except ImportError as e:
        pytest.fail(f"Core import failed: {e}")
    
    # Test trainer module imports
    try:
        from hydromodel.trainers import (
            calibrate,
            calibrate_by_sceua,
            calibrate_unit_hydrograph,
            calibrate_categorized_unit_hydrograph
        )
        print("SUCCESS: All trainer module imports work")
    except ImportError as e:
        pytest.fail(f"Trainer import failed: {e}")


def test_deap_availability():
    """Test DEAP availability for GA functionality."""
    from hydromodel.trainers.unified_calibrate import DEAP_AVAILABLE
    
    if DEAP_AVAILABLE:
        print("INFO: DEAP is available - GA tests will run")
        try:
            from deap import base, creator, tools
            print("SUCCESS: DEAP imports work correctly")
        except ImportError:
            pytest.fail("DEAP_AVAILABLE is True but DEAP cannot be imported")
    else:
        print("WARNING: DEAP is not available - GA tests will be skipped")


if __name__ == "__main__":
    run_unified_calibration_tests()