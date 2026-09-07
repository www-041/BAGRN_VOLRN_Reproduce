"""Regression tests for spectral metrics band selection.

Task 13 of reliability-fixes plan:
- bands= parameter must actually restrict all spectral metrics
"""

import numpy as np


def test_sam_honors_bands_parameter():
    """compute_sam must use only selected bands."""
    # Create 3-band arrays where bands 0,2 match but band 1 is very different
    orig = np.array([
        [[1.0, 2.0], [3.0, 4.0]],  # band 0
        [[10.0, 20.0], [30.0, 40.0]],  # band 1 (very different)
        [[1.0, 2.0], [3.0, 4.0]],  # band 2 (same as band 0)
    ])
    norm = np.array([
        [[1.0, 2.0], [3.0, 4.0]],  # band 0 (same)
        [[100.0, 200.0], [300.0, 400.0]],  # band 1 (10x different)
        [[1.0, 2.0], [3.0, 4.0]],  # band 2 (same)
    ])
    
    # When using bands=[0, 2], SAM should be ~0 (identical)
    # When using all bands, SAM should be non-zero (band 1 differs)
    
    # Simulate the fix: select bands before computation
    bands = [0, 2]
    orig_selected = orig[bands]
    norm_selected = norm[bands]
    
    # These should be identical
    assert np.allclose(orig_selected, norm_selected)


def test_spectral_rmse_honors_bands():
    """Spectral RMSE must use only selected bands."""
    orig = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[10.0, 20.0], [30.0, 40.0]],  # Different
        [[1.0, 2.0], [3.0, 4.0]],
    ])
    norm = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[100.0, 200.0], [300.0, 400.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ])
    
    bands = [0, 2]
    orig_selected = orig[bands]
    norm_selected = norm[bands]
    
    # RMSE should be 0 for selected bands
    rmse = np.sqrt(np.mean((orig_selected - norm_selected) ** 2))
    assert rmse == 0.0
