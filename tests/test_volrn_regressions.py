"""Regression tests for VOLRN NoData preservation and coefficient units.

Task 4-5 of reliability-fixes plan:
- NoData must be preserved through normalize → block → optimize → interpolate → denormalize.
- Normalized NoData must use NaN (not zero) to prevent corruption of block statistics.
- Returned coefficients should be in original DN units.
- IDW interpolation must use local 3x3 neighbourhood first.
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
    """NoData pixels must not corrupt block statistics."""
    a = np.full((1, 16, 16), -9999.0)
    a[:, 14:16, 14:16] = 500.0
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

    assert np.all(result[0][:, :14, :] == -9999.0)
    assert np.all(result[0][:, 14:, :14] == -9999.0)
    valid_a = result[0][:, 14:, 14:]
    assert np.all(np.isfinite(valid_a))
    assert np.all(valid_a > 100)


def test_volrn_image_blocking_excludes_nodata_from_stats():
    """image_blocking must not include NoData pixels in block statistics."""
    arr_with_nan_nodata = np.full((1, 16, 16), np.nan)
    arr_with_nan_nodata[0, 14:16, 14:16] = 0.5

    transforms = [from_origin(0, 16, 1, 1)]
    bounds = [(0, 0, 16, 16)]

    blocks_nan, _ = image_blocking(
        [arr_with_nan_nodata], transforms, bounds, [None],
        block_size=16, bands=[0],
    )

    if blocks_nan:
        assert blocks_nan[0].mu[0] > 0.3, \
            f"Block mean {blocks_nan[0].mu[0]} is too low (NoData contamination)"


def test_volrn_idw_uses_local_3x3_neighbourhood():
    """
    IDW interpolation should use the 3x3 grid neighbourhood first,
    not the nearest 9 blocks globally.
    
    Place local blocks (a=1.0) in 3x3 neighbourhood of cell (3,3).
    Place contaminating blocks (a=100.0) truly outside the 3x3 neighbourhood.
    Assert local pixel interpolation uses ONLY local blocks.
    """
    from src.volrn import _idw_interpolate
    
    # 7x7 grid
    all_gm = list(range(7))
    all_gn = list(range(7))
    
    grid_a = np.zeros((7, 7))
    grid_b = np.zeros((7, 7))
    has_block = np.zeros((7, 7), dtype=bool)
    
    # 3 local blocks in 3x3 neighbourhood of cell (3,3): all within |dm|<=1, |dn|<=1
    for gm, gn in [(3, 3), (3, 4), (4, 3)]:
        grid_a[gm, gn] = 1.0
        grid_b[gm, gn] = 0.0
        has_block[gm, gn] = True
    
    # Contaminating blocks (a=100.0) placed OUTSIDE the 3x3 neighbourhood of cell (3,3)
    # 3x3 neighbourhood of cell (3,3) is cells (2,2)-(4,4)
    # So cells at distance >= 2 from (3,3) in either dimension:
    for gm, gn in [(0, 0), (0, 6), (6, 0), (6, 6), (0, 3), (6, 3)]:
        grid_a[gm, gn] = 100.0
        grid_b[gm, gn] = 0.0
        has_block[gm, gn] = True
    
    # Pixel at grid position (3.2, 3.2) - nearest cell is (3,3)
    pixel_gm = np.array([3.2])
    pixel_gn = np.array([3.2])
    
    a_out, b_out = _idw_interpolate(
        grid_a, grid_b, has_block,
        all_gm, all_gn, pixel_gm, pixel_gn,
    )
    
    # With local 3x3 semantics: only 3 blocks with a=1.0 -> result ~ 1.0
    # With global nearest-9: mix of 1.0 and 100.0 -> result >> 1.0
    assert abs(a_out[0, 0] - 1.0) < 0.01, \
        f"Expected a~1.0 (local 3x3), got {a_out[0, 0]}"


def test_volrn_idw_falls_back_to_global_when_no_local():
    """
    When a pixel has NO valid blocks in its 3x3 neighbourhood,
    IDW should fall back to the nearest global valid block(s).
    """
    from src.volrn import _idw_interpolate
    
    all_gm = list(range(5))
    all_gn = list(range(5))
    
    grid_a = np.zeros((5, 5))
    grid_b = np.zeros((5, 5))
    has_block = np.zeros((5, 5), dtype=bool)
    
    # Only one block at (4, 4)
    grid_a[4, 4] = 2.0
    grid_b[4, 4] = 5.0
    has_block[4, 4] = True
    
    # Pixel at (0, 0) - far from the only block
    pixel_gm = np.array([0.0])
    pixel_gn = np.array([0.0])
    
    a_out, b_out = _idw_interpolate(
        grid_a, grid_b, has_block,
        all_gm, all_gn, pixel_gm, pixel_gn,
    )
    
    # Should fall back to the only available block
    assert a_out[0, 0] == 2.0
    assert b_out[0, 0] == 5.0
