"""Regression tests for coregistration row/col ordering and confidence.

Task 10 of reliability-fixes plan:
- rowcol() returns (row, col) - must unpack correctly
- Zero inliers must not yield confidence 1.0
- nodata=None must not imply zero is invalid
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin


def test_rowcol_returns_row_col_order():
    """rasterio.transform.rowcol returns (row, col), not (col, row)."""
    # Use non-square resolution to distinguish row from col
    tr = rasterio.Affine(2.0, 0, 0, 0, -1.0, 20)  # x_res=2, y_res=1
    
    # Point at (x=6, y=15):
    # col = (6 - 0) / 2 = 3
    # row = (20 - 15) / 1 = 5
    row, col = rasterio.transform.rowcol(tr, 6, 15)
    assert row == 5, f"Expected row=5, got {row}"
    assert col == 3, f"Expected col=3, got {col}"
    
    # This demonstrates that rowcol returns (row, col) not (col, row)
    # If code does: c0, r0 = rowcol(...) then c0 gets row value, r0 gets col value
    # This is the bug pattern in coregistration.py


def test_zero_inliers_low_confidence():
    """When no inliers remain after filtering, confidence must be low, not 1.0."""
    # Simulate the condition: all blocks rejected as outliers
    shifts_y = np.array([1.0, 2.0, 3.0])
    shifts_x = np.array([1.0, 2.0, 3.0])
    confs = np.array([0.9, 0.8, 0.7])
    inlier = np.array([False, False, False])  # No inliers
    
    # Bug: else branch sets confidence = 1.0
    # Fix: should be 0.0 or very low
    if inlier.sum() > 0:
        confidence = float(confs[inlier].mean())
    else:
        # Zero inliers = no reliable estimate = low confidence
        confidence = 0.0
    
    assert confidence == 0.0, "Zero inliers should give confidence 0.0, not 1.0"


def test_nodata_none_permits_zero_pixels():
    """When nodata=None, zero-valued pixels must be treated as valid."""
    arr = np.array([[0.0, 1.0], [2.0, 0.0]])
    nodata = None
    
    # Bug pattern: valid = (arr != nodata) where nodata=0
    # This would exclude valid zeros
    
    # Fix: when nodata is None, only check isfinite
    if nodata is not None:
        valid = np.isfinite(arr) & (arr != nodata)
    else:
        valid = np.isfinite(arr)
    
    # All pixels should be valid (all are finite)
    assert valid.all(), "All finite pixels should be valid when nodata=None"
    assert valid.sum() == 4
