"""RED tests for Task10D circular gradient orientation loss."""

import numpy as np
import pytest

from src.multiscene_sift.radiometric_metrics import (
    circular_angle_difference,
    compute_cgl,
)


def _x_ramp(size=9):
    x = np.arange(size, dtype=np.float64)
    return np.broadcast_to(x, (size, size)).copy()


def _y_ramp(size=9):
    y = np.arange(size, dtype=np.float64)[:, None]
    return np.broadcast_to(y, (size, size)).copy()


def test_raw_vs_raw_and_affine_radiometry_preserve_orientation():
    raw = _x_ramp()
    valid = np.ones_like(raw, dtype=bool)

    assert compute_cgl(raw, raw, valid)["cgl_rad"] == pytest.approx(0.0)
    assert compute_cgl(raw, raw + 7.0, valid)["cgl_rad"] == pytest.approx(0.0)
    assert compute_cgl(raw, raw * 3.0 + 2.0, valid)["cgl_rad"] == pytest.approx(0.0)


def test_known_ninety_degree_orientation_change():
    valid = np.ones((9, 9), dtype=bool)
    metrics = compute_cgl(_x_ramp(), _y_ramp(), valid)

    assert metrics["cgl_rad"] == pytest.approx(np.pi / 2, abs=1e-12)
    assert metrics["cgl_deg"] == pytest.approx(90.0, abs=1e-10)


def test_circular_difference_uses_shortest_arc_at_pi_boundary():
    difference = circular_angle_difference(np.deg2rad(179.0), np.deg2rad(-179.0))

    assert difference == pytest.approx(np.deg2rad(2.0))


def test_invalid_three_by_three_stencils_are_excluded():
    raw = _x_ramp()
    valid = np.ones_like(raw, dtype=bool)
    valid[4, 4] = False

    metrics = compute_cgl(raw, raw, valid)

    assert metrics["valid_pixels"] == 40
