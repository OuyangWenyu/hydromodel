"""Regression test for GitHub issue #39: ArithmeticError with w0>wm.

When w0 (initial water storage) exceeds wm (max water storage), or b < -1,
the calculation produces NaN or complex numbers. The original
ArithmeticError check was silently bypassed by numpy's complex-number
fallback. This test ensures the guard fires correctly.
"""
import numpy as np
import pytest
from hydromodel.models.xaj import calculate_prcp_runoff


class TestCalculatePrcpRunoffGuard:
    """Guard conditions for calculate_prcp_runoff edge cases."""

    PE = np.array([[5.0], [10.0], [3.0]])

    def test_w0_greater_than_wm_raises(self):
        """w0 > wm must raise ArithmeticError, not silently return complex."""
        with pytest.raises(ArithmeticError, match="w0>wm"):
            calculate_prcp_runoff(b=0.2, im=0.01, wm=100.0, w0=150.0, pe=self.PE)

    def test_b_negative_raises(self):
        """b < 0 must raise ArithmeticError."""
        with pytest.raises(ArithmeticError, match="b is a negative"):
            calculate_prcp_runoff(b=-0.5, im=0.01, wm=100.0, w0=50.0, pe=self.PE)

    def test_b_less_than_neg1_raises(self):
        """b < -1 must raise ArithmeticError."""
        with pytest.raises(ArithmeticError, match="b is a negative"):
            calculate_prcp_runoff(b=-1.5, im=0.01, wm=100.0, w0=50.0, pe=self.PE)

    def test_normal_case_passes(self):
        """Normal valid parameters should succeed."""
        r, r_im = calculate_prcp_runoff(b=0.2, im=0.01, wm=100.0, w0=50.0, pe=self.PE)
        assert r.shape == (3, 1)
        assert np.isrealobj(r), "result should be real, not complex"

    def test_w0_equal_wm_passes(self):
        """w0 == wm is the boundary case; should succeed."""
        r, r_im = calculate_prcp_runoff(b=0.2, im=0.01, wm=100.0, w0=100.0, pe=self.PE)
        assert np.isrealobj(r)
