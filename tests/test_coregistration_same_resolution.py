"""Tests for same-resolution phase_correlation and overlap-aware registration.

Task 1-2 of dz01v mosaic readiness plan:
- phase_correlation must be a pure same-grid array function.
- No undefined transform/CRS variables.
- Registration fallback uses geographic overlap.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as ndi_shift

from src.coregistration import phase_correlation


def test_phase_correlation_same_resolution_known_shift():
    """phase_correlation must recover a known subpixel shift on same-grid arrays."""
    rng = np.random.default_rng(123)
    ref = gaussian_filter(rng.normal(size=(64, 64)), sigma=1.2)

    moving = ndi_shift(
        ref,
        shift=(2.0, -3.0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    valid_ref = np.ones_like(ref, dtype=bool)
    valid_tgt = np.ones_like(moving, dtype=bool)

    sy, sx, confidence = phase_correlation(
        ref,
        moving,
        valid_ref=valid_ref,
        valid_tgt=valid_tgt,
    )

    assert sy == pytest.approx(-2.0, abs=0.15)
    assert sx == pytest.approx(3.0, abs=0.15)
    assert np.isfinite(confidence)
    assert confidence > 0.5


def test_phase_correlation_no_undefined_variables():
    """phase_correlation must not reference tr_ref/tr_tgt/crs_ref/crs_tgt."""
    import inspect
    import re
    from src.coregistration import phase_correlation
    
    source = inspect.getsource(phase_correlation)
    # Remove comments and docstrings for analysis
    source_no_comments = re.sub(r'#.*$', '', source, flags=re.MULTILINE)
    # Should NOT contain variable references to transform/CRS (not in docstrings)
    forbidden_vars = [r'\btr_ref\b', r'\btr_tgt\b', r'\bcrs_ref\b', r'\bcrs_tgt\b']
    found = []
    for pat in forbidden_vars:
        if re.search(pat, source_no_comments):
            found.append(pat)
    assert not found, f"phase_correlation contains forbidden variable references: {found}"


def test_phase_correlation_from_overlap_uses_ground_overlap_only():
    """phase_correlation_from_overlap must use geographic overlap, not full arrays."""
    from rasterio.transform import from_origin
    from src.coregistration import phase_correlation_from_overlap
    
    tr_ref = from_origin(0, 64, 1, 1)
    tr_tgt = from_origin(32, 64, 1, 1)

    ref = np.zeros((64, 64), dtype=np.float64)
    tgt = np.zeros((64, 64), dtype=np.float64)

    rng = np.random.default_rng(456)
    pattern = rng.normal(size=(64, 32))

    # Same ground region x=[32,64].
    ref[:, 32:64] = pattern
    tgt[:, 0:32] = pattern

    sy, sx, confidence = phase_correlation_from_overlap(
        ref, tr_ref,
        tgt, tr_tgt,
        nodata_ref=None,
        nodata_tgt=None,
    )

    assert abs(sy) < 0.15
    assert abs(sx) < 0.15
    assert confidence > 0.5
