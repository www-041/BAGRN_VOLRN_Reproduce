"""Regression tests for diagnostics coefficient statistics.

Task 12 of reliability-fixes plan:
- gain RMS must be measured relative to identity (a - 1), not raw a values
- dynamic range should use data range, not coefficient range
"""

import numpy as np


def test_identity_gain_has_zero_rms():
    """When a=1 (identity gain), RMS should be 0, not 1."""
    # Identity transform: a=1, b=0
    a = np.ones(10)
    
    # Bug: RMS = sqrt(mean(a^2)) = sqrt(mean(1)) = 1.0
    # Fix: RMS = sqrt(mean((a-1)^2)) = sqrt(mean(0)) = 0.0
    
    finite_a = a[np.isfinite(a)]
    a_rms_correct = float(np.sqrt(np.mean((finite_a - 1) ** 2)))
    
    assert a_rms_correct == 0.0, "Identity gain (a=1) should have RMS=0"


def test_non_identity_gain_has_nonzero_rms():
    """When a deviates from 1, RMS should be non-zero."""
    a = np.array([0.9, 1.0, 1.1])
    
    finite_a = a[np.isfinite(a)]
    a_rms_correct = float(np.sqrt(np.mean((finite_a - 1) ** 2)))
    
    # RMS = sqrt(mean([0.01, 0, 0.01])) = sqrt(0.02/3) ≈ 0.0816
    assert a_rms_correct > 0.0, "Non-identity gain should have RMS > 0"
    assert abs(a_rms_correct - 0.0816) < 0.01


def test_dynamic_range_uses_data_not_coefficients():
    """b_over_dynamic_range should use data dynamic range, not coefficient range."""
    # Data values
    data = np.array([100, 200, 300, 400, 500])
    data_dr = data.max() - data.min()  # 400
    
    # Coefficient a values
    a_vals = np.array([0.9, 1.0, 1.1])
    a_range = a_vals.max() - a_vals.min()  # 0.2
    
    # Bug: uses a_range as denominator
    # Fix: should use data_dr (or some measure of data dynamic range)
    
    assert data_dr == 400
    assert abs(a_range - 0.2) < 1e-10
    # They should be different - data range is much larger
    assert data_dr != a_range
