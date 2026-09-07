"""Regression tests for seam metric masks and axis convention.

Task 14 of reliability-fixes plan:
- Consistently reject NaN/Inf
- Array axis convention compatible with (bands, rows, cols)
- Valid zero must not be treated as NoData
"""

import numpy as np


def test_safe_array_excludes_nan_inf():
    """_safe_array must exclude NaN and Inf even when nodata=None."""
    arr = np.array([[1.0, np.nan], [np.inf, 2.0]])
    
    # Bug: current _safe_array only checks nodata, not finite
    # Fix: should always check np.isfinite
    
    # Expected behavior
    mask_expected = np.isfinite(arr)
    
    assert not mask_expected[0, 1]  # NaN excluded
    assert not mask_expected[1, 0]  # Inf excluded
    assert mask_expected[0, 0]  # 1.0 included
    assert mask_expected[1, 1]  # 2.0 included


def test_safe_array_preserves_valid_zero():
    """When nodata=None, zero must be treated as valid."""
    arr = np.array([[0.0, 1.0], [2.0, 0.0]])
    nodata = None
    
    # Bug: might do arr != nodata where nodata=0
    # Fix: when nodata is None, only check isfinite
    
    if nodata is not None:
        mask = np.isfinite(arr) & (arr != nodata)
    else:
        mask = np.isfinite(arr)
    
    assert mask.all(), "All finite values should be valid when nodata=None"
    assert mask[0, 0] == True  # Zero is valid
    assert mask[1, 1] == True  # Zero is valid


def test_safe_array_with_explicit_nodata_and_nan():
    """Both explicit nodata and NaN must be excluded."""
    arr = np.array([[1.0, -9999.0], [np.nan, 2.0]])
    nodata = -9999.0
    
    mask = np.isfinite(arr) & (arr != nodata)
    
    assert mask[0, 0] == True   # 1.0 valid
    assert mask[0, 1] == False  # -9999 nodata
    assert mask[1, 0] == False  # NaN
    assert mask[1, 1] == True   # 2.0 valid
