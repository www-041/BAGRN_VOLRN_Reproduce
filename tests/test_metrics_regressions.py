"""Regression tests for multi-resolution metrics support.

Task 2 of reliability-fixes plan:
- ADM/ADSD/CD must support different sensor resolutions (different pixel counts).
- Explicit NoData must also exclude NaN/Inf.
- GL must divide by actual contributors, not all images.
"""

import numpy as np

from src.metrics import compute_adm, compute_adsd, compute_cd, compute_gl


def _different_resolution_case():
    a = np.full((1, 4, 4), 10.0)
    b = np.full((1, 8, 8), 20.0)

    overlaps = [{
        "idx_i": 0,
        "idx_j": 1,
        "window_i": (0, 4, 0, 4),
        "window_j": (0, 8, 0, 8),
    }]
    return [a, b], [None, None], overlaps


def test_adm_allows_different_pixel_counts():
    arrays, nodata, overlaps = _different_resolution_case()
    assert np.isclose(compute_adm(arrays, nodata, overlaps, [0]), 10.0)


def test_adsd_allows_different_pixel_counts():
    arrays, nodata, overlaps = _different_resolution_case()
    assert np.isclose(compute_adsd(arrays, nodata, overlaps, [0]), 0.0)


def test_cd_allows_different_pixel_counts():
    arrays, nodata, overlaps = _different_resolution_case()
    value = compute_cd(arrays, nodata, overlaps, [0])
    assert np.isfinite(value)


def test_explicit_nodata_still_excludes_nan_inf():
    a = np.array([[[1.0, 2.0], [-9999.0, np.nan]]])
    b = np.array([[[1.0, 4.0], [-9999.0, np.inf]]])

    ov = [{
        "idx_i": 0,
        "idx_j": 1,
        "window_i": (0, 2, 0, 2),
        "window_j": (0, 2, 0, 2),
    }]

    adm = compute_adm([a, b], [-9999.0, -9999.0], ov, [0])
    assert np.isfinite(adm)


def test_gl_averages_only_actual_contributors():
    before = [
        np.arange(16, dtype=float).reshape(1, 4, 4),
        np.full((1, 4, 4), np.nan),
    ]
    after = [
        before[0].copy(),
        before[1].copy(),
    ]

    assert np.isclose(compute_gl(before, after, [None, None], [0]), 0.0)
