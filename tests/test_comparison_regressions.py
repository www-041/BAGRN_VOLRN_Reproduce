"""Regression tests for comparison method masking and Wallis control behavior.

Task 7 of reliability-fixes plan:
- Different-resolution overlap must not broadcast.
- Explicit NoData plus NaN must be excluded.
- Wallis must leave control_idx unchanged.
"""

import numpy as np
import pytest

from src.comparison import (
    wallis_normalize,
    moment_matching_normalize,
    _overlap_valid_pixels,
)


def test_overlap_valid_pixels_independent_filtering():
    """Different-resolution windows must not cause broadcasting error."""
    a = np.full((1, 4, 4), 10.0)
    b = np.full((1, 8, 8), 20.0)

    pi, pj = _overlap_valid_pixels(
        a, b,
        (0, 4, 0, 4), (0, 8, 0, 8),
        None, None, 0,
    )

    assert len(pi) == 16
    assert len(pj) == 64


def test_wallis_leaves_control_unchanged():
    """Wallis normalization must NOT modify the control image."""
    control = np.arange(16, dtype=float).reshape(1, 4, 4) * 10.0
    target = np.arange(16, dtype=float).reshape(1, 4, 4) * 10.0 + 100.0

    ov = [{
        "idx_i": 0, "idx_j": 1,
        "window_i": (0, 4, 0, 4),
        "window_j": (0, 4, 0, 4),
    }]

    result = wallis_normalize(
        [control, target], [None, None], ov,
        control_idx=0, window_size=3,
    )

    # Control image must be unchanged
    assert np.allclose(result[0], control)


def test_comparison_nodata_excludes_nan_inf():
    """Explicit NoData plus NaN/Inf must both be excluded."""
    a = np.array([[[1.0, 2.0], [-9999.0, np.nan]]])
    b = np.array([[[1.0, 4.0], [-9999.0, np.inf]]])

    pi, pj = _overlap_valid_pixels(
        a, b,
        (0, 2, 0, 2), (0, 2, 0, 2),
        -9999.0, -9999.0, 0,
    )

    # Only (0,0)=1.0 and (0,1)=2.0/4.0 should be valid
    # NaN, Inf, and -9999 all excluded
    assert len(pi) == 2
    assert len(pj) == 2
    assert np.all(np.isfinite(pi))
    assert np.all(np.isfinite(pj))
