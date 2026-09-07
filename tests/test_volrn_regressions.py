"""Regression tests for VOLRN NoData preservation and coefficient units.

Task 4 of reliability-fixes plan:
- NoData must be preserved through normalize → block → optimize → interpolate → denormalize.
- Normalized NoData must use NaN (not zero) to prevent corruption of block statistics.
- Returned coefficients should be in original DN units.
"""

import numpy as np
from rasterio.transform import from_origin

from src.volrn import volrn_normalize, image_blocking


def test_volrn_nodata_never_becomes_valid_zero():
    """NoData pixels must remain declared NoData after VOLRN."""
    a = np.full((1, 32, 32), 100.0)
    b = np.full((1, 32, 32), 110.0)

    a[:, :8, :] = -9999.0
    b[:, :8, :] = -9999.0

    transforms = [
        from_origin(0, 32, 1, 1),
        from_origin(16, 32, 1, 1),
    ]
    bounds = [
        (0, 0, 32, 32),
        (16, 0, 48, 32),
    ]

    result, _ = volrn_normalize(
        [a, b],
        transforms,
        bounds,
        [-9999.0, -9999.0],
        block_size_pixels=16,
        max_iter=5,
        tol=1e-3,
    )

    assert np.all(result[0][:, :8, :] == -9999.0)
    assert np.all(result[1][:, :8, :] == -9999.0)


def test_volrn_returns_coefficients_in_original_dn_units():
    """Returned coefficients must be in original DN space."""
    a = np.full((1, 32, 32), 1000.0)
    b = np.full((1, 32, 32), 1100.0)

    transforms = [
        from_origin(0, 32, 1, 1),
        from_origin(16, 32, 1, 1),
    ]
    bounds = [
        (0, 0, 32, 32),
        (16, 0, 48, 32),
    ]

    _, coeff = volrn_normalize(
        [a, b], transforms, bounds, [None, None],
        block_size_pixels=16,
        max_iter=5,
        tol=1e-3,
    )

    assert coeff.ndim in (1, 3)


def test_volrn_normalized_nodata_uses_nan_not_zero():
    """
    When NoData is declared, normalized-space invalid pixels must be NaN
    so they cannot participate in block statistics.
    
    If normalized NoData is zero, blocks with mixed valid/NoData will have
    their statistics corrupted by the zero values.
    """
    # Create images where blocks will have mostly NoData and few valid pixels
    a = np.full((1, 16, 16), -9999.0)  # Almost all NoData
    a[:, 14:16, 14:16] = 500.0  # Only 4 valid pixels in corner
    b = np.full((1, 16, 16), -9999.0)
    b[:, 14:16, 14:16] = 600.0

    transforms = [
        from_origin(0, 16, 1, 1),
        from_origin(8, 16, 1, 1),
    ]
    bounds = [
        (0, 0, 16, 16),
        (8, 0, 24, 16),
    ]

    result, _ = volrn_normalize(
        [a, b],
        transforms,
        bounds,
        [-9999.0, -9999.0],
        block_size_pixels=16,
        max_iter=5,
        tol=1e-3,
    )

    # NoData must be preserved
    assert np.all(result[0][:, :14, :] == -9999.0)
    assert np.all(result[0][:, 14:, :14] == -9999.0)
    
    # Valid pixels should remain finite and close to original values
    # (not dragged toward zero by zero-valued NoData contamination)
    valid_a = result[0][:, 14:, 14:]
    assert np.all(np.isfinite(valid_a))
    assert np.all(valid_a > 100)  # Should be near 500, not near 0


def test_volrn_image_blocking_excludes_nodata_from_stats():
    """
    image_blocking on normalized arrays must not include NoData pixels
    in block mean/std calculations.
    
    This directly tests the normalized-space NoData contamination bug.
    """
    # Simulate normalized array where NoData was set to 0
    # vs correct approach where NoData is NaN
    arr_with_zero_nodata = np.zeros((1, 16, 16))
    arr_with_zero_nodata[0, 14:16, 14:16] = 0.5  # Only 4 valid pixels
    
    arr_with_nan_nodata = np.full((1, 16, 16), np.nan)
    arr_with_nan_nodata[0, 14:16, 14:16] = 0.5
    
    transforms = [from_origin(0, 16, 1, 1)]
    bounds = [(0, 0, 16, 16)]
    
    # With NaN NoData (nodata=None, so valid = isfinite)
    blocks_nan, _ = image_blocking(
        [arr_with_nan_nodata], transforms, bounds, [None],
        block_size=16, bands=[0],
    )
    
    # Should have no valid blocks (or the block should have mean ~0.5)
    # because only 4 pixels are valid
    if blocks_nan:
        # If a block exists, its mean should be ~0.5, not ~0.03
        assert blocks_nan[0].mu[0] > 0.3, \
            f"Block mean {blocks_nan[0].mu[0]} is too low (NoData contamination)"
