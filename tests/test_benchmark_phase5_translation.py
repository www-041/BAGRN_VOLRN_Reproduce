"""Synthetic checks for the frozen robust 2-D translation fit."""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_v1_phase5.coordinates import NativeMatches
from benchmark_v1_phase5.protocol import load_config
from benchmark_v1_phase5.translation import estimate_translation


CONFIG = load_config(ROOT / "configs/benchmark_v1_phase5.yaml")


def _matches(delta=(2.25, -3.5), *, inliers=24, outliers=5, single_cell=False):
    rng = np.random.default_rng(44)
    if single_cell:
        xy = rng.uniform(2, 10, size=(inliers, 2))
    else:
        cells = np.array([(x, y) for y in (30, 70, 110, 150)
                          for x in (30, 70, 110, 150)], dtype=float)
        xy = cells[np.arange(inliers) % len(cells)] + rng.uniform(0, 5, size=(inliers, 2))
    i_rc = rng.uniform(10, 100, size=(inliers, 2))
    d = np.broadcast_to(np.asarray(delta), (inliers, 2)).copy()
    d += rng.normal(0, .08, size=d.shape)
    j_rc = i_rc + d
    if outliers:
        oi = rng.uniform(10, 100, size=(outliers, 2))
        oj = oi + rng.uniform(100, 300, size=(outliers, 2))
        i_rc = np.vstack((i_rc, oi))
        j_rc = np.vstack((j_rc, oj))
        ox = rng.uniform(0, 160, size=(outliers, 2))
        xy = np.vstack((xy, ox))
    return NativeMatches(i_rc, j_rc, xy.copy(), xy.copy(),
                         np.ones(len(i_rc)), "synthetic", (160, 160))


def test_known_forward_and_reverse_subpixel_translation_with_outliers():
    forward = estimate_translation(_matches(), CONFIG)
    assert forward.status == "SUCCESS"
    np.testing.assert_allclose(forward.delta_row_col, (2.25, -3.5), atol=.25)
    reverse = estimate_translation(_matches(delta=(-2.25, 3.5)), CONFIG)
    assert reverse.status == "SUCCESS"
    np.testing.assert_allclose(reverse.delta_row_col, (-2.25, 3.5), atol=.25)


def test_fixed_seed_is_bit_deterministic_and_confidence_does_not_change_fit():
    matches = _matches()
    first = estimate_translation(matches, CONFIG)
    second = estimate_translation(matches, CONFIG)
    np.testing.assert_array_equal(first.inlier_mask, second.inlier_mask)
    np.testing.assert_array_equal(first.delta_row_col, second.delta_row_col)
    changed = NativeMatches(matches.i_rc, matches.j_rc, matches.i_view_xy,
                            matches.j_view_xy, np.linspace(0, 1, len(matches.i_rc)),
                            matches.method, matches.view_shape)
    third = estimate_translation(changed, CONFIG)
    np.testing.assert_array_equal(first.inlier_mask, third.inlier_mask)
    np.testing.assert_array_equal(first.delta_row_col, third.delta_row_col)


@pytest.mark.parametrize("matches,code", [
    (_matches(inliers=11, outliers=0), "INSUFFICIENT_MATCHES"),
    (_matches(inliers=7, outliers=5), "INSUFFICIENT_INLIERS"),
    (_matches(inliers=8, outliers=80), "LOW_INLIER_FRACTION"),
    (_matches(inliers=24, outliers=0, single_cell=True), "INSUFFICIENT_SPATIAL_COVERAGE"),
])
def test_named_acceptance_gates(matches, code):
    result = estimate_translation(matches, CONFIG)
    assert result.status == "ALGORITHM_FAIL"
    assert result.failure_code == code
    assert result.delta_row_col is None


def test_empty_and_nonfinite_correspondences_are_rejected():
    empty = NativeMatches(np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)),
                          np.empty((0, 2)), np.empty(0), "synthetic", (160, 160))
    assert estimate_translation(empty, CONFIG).failure_code == "INSUFFICIENT_MATCHES"
    broken = NativeMatches(np.array([[np.nan, 0.0]]), np.zeros((1, 2)),
                           np.zeros((1, 2)), np.zeros((1, 2)), np.ones(1),
                           "synthetic", (160, 160))
    with pytest.raises(ValueError, match="finite"):
        estimate_translation(broken, CONFIG)


def test_target_view_coordinates_must_remain_in_the_shared_view():
    matches = _matches()
    bad_target = NativeMatches(matches.i_rc, matches.j_rc, matches.i_view_xy,
                               matches.j_view_xy.copy(), matches.confidence,
                               matches.method, matches.view_shape)
    bad_target.j_view_xy[0, 0] = matches.view_shape[1]
    with pytest.raises(ValueError, match="coordinates"):
        estimate_translation(bad_target, CONFIG)
