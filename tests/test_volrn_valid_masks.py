"""Task10C explicit valid-mask and normalized NoData invariants."""

import numpy as np
from rasterio.transform import Affine

from src.volrn import volrn_normalize


def _inputs():
    transform = Affine(1, 0, 0, 0, -1, 4)
    bounds = [(0.0, 0.0, 4.0, 4.0)] * 2
    first = np.array(
        [[0.0, -1.25, 0.5, 1.0], [2.0, 3.0, 4.0, 5.0],
         [6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0]],
        dtype=np.float64,
    )[None, ...]
    second = (first * 1.1 + 2.0).copy()
    return [first, second], [transform, transform], bounds


def test_explicit_valid_mask_keeps_valid_zero_with_source_nodata_zero():
    arrays, transforms, bounds = _inputs()
    valid = [np.ones((4, 4), dtype=bool), np.ones((4, 4), dtype=bool)]

    results, _ = volrn_normalize(
        arrays,
        transforms,
        bounds,
        [0.0, 0.0],
        valid_masks=valid,
        block_size_pixels=4,
        max_iter=1,
    )

    assert np.isfinite(results[0][0, 0, 0])
    assert np.isfinite(results[1][0, 0, 0])


def test_explicit_valid_mask_preserves_negative_fractional_values_and_invalid_zero():
    arrays, transforms, bounds = _inputs()
    valid = [np.ones((4, 4), dtype=bool), np.ones((4, 4), dtype=bool)]
    valid[0][3, 3] = False

    results, _ = volrn_normalize(
        arrays,
        transforms,
        bounds,
        [0.0, 0.0],
        valid_masks=valid,
        block_size_pixels=4,
        max_iter=1,
    )

    assert np.isfinite(results[0][0, 0, 1])
    assert np.isfinite(results[0][0, 0, 2])
    assert results[0][0, 3, 3] == 0.0


def test_explicit_valid_mask_cannot_make_nan_or_inf_valid():
    arrays, transforms, bounds = _inputs()
    arrays[0][0, 1, 1] = np.nan
    arrays[1][0, 2, 2] = np.inf
    valid = [np.ones((4, 4), dtype=bool), np.ones((4, 4), dtype=bool)]

    results, _ = volrn_normalize(
        arrays,
        transforms,
        bounds,
        [None, None],
        valid_masks=valid,
        block_size_pixels=4,
        max_iter=1,
    )

    assert not np.isfinite(results[0][0, 1, 1])
    assert not np.isfinite(results[1][0, 2, 2])
