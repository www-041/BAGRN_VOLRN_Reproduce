"""Regression tests for overlap geometry semantics.

Task 1 of reliability-fixes plan:
- has_overlap must require positive intersection area (edge-touching = False).
- get_overlap_window must use floor/ceil for conservative coverage.
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.overlap import has_overlap, get_overlap_window


def test_edge_touch_has_zero_usable_overlap():
    """Two rectangles sharing only an edge have zero-area intersection."""
    assert has_overlap((0, 0, 10, 10), (10, 0, 20, 10)) is False


def test_edge_touch_vertical():
    """Vertical edge touch should also be False."""
    assert has_overlap((0, 0, 10, 10), (10, 5, 20, 15)) is False


def test_edge_touch_horizontal():
    """Horizontal edge touch should also be False."""
    assert has_overlap((0, 0, 10, 10), (5, 10, 15, 20)) is False


def test_genuine_overlap_still_true():
    """A genuine overlapping pair must still return True."""
    assert has_overlap((0, 0, 10, 10), (5, 5, 15, 15)) is True


def test_no_overlap_returns_false():
    """Completely separated rectangles."""
    assert has_overlap((0, 0, 10, 10), (20, 20, 30, 30)) is False


def test_overlap_window_conservatively_covers_fractional_edges():
    """get_overlap_window must use floor for starts and ceil for ends."""
    tr = from_origin(0, 10, 1, 1)

    result = get_overlap_window(
        (0, 0, 10, 10), tr,
        (2.2, 1.2, 7.7, 8.8), tr,
    )

    assert result is not None
    wi, _ = result

    # conservative floor/ceil coverage of the geographic intersection
    assert wi[0] <= 2   # row_start floored
    assert wi[1] >= 9   # row_end ceiled
    assert wi[2] <= 2   # col_start floored
    assert wi[3] >= 8   # col_end ceiled


def test_overlap_window_same_image():
    """An image overlapping with itself should cover the full extent."""
    tr = from_origin(0, 10, 1, 1)
    bounds = (0, 0, 10, 10)

    result = get_overlap_window(bounds, tr, bounds, tr)
    assert result is not None
    wi, wj = result
    assert wi == (0, 10, 0, 10)
    assert wj == (0, 10, 0, 10)


def test_no_overlap_returns_none():
    """Non-overlapping images return None."""
    tr = from_origin(0, 10, 1, 1)
    result = get_overlap_window(
        (0, 0, 10, 10), tr,
        (20, 20, 30, 30), tr,
    )
    assert result is None


def test_edge_touch_window_returns_none():
    """Edge-touching images should return None (zero-area intersection)."""
    tr = from_origin(0, 10, 1, 1)
    result = get_overlap_window(
        (0, 0, 10, 10), tr,
        (10, 0, 20, 10), tr,
    )
    assert result is None
