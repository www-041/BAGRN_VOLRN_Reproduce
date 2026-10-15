"""Synthetic checks for the frozen Task13A single-band seam cost."""

import numpy as np

from src.seam_local.cost import compute_seam_cost


def test_identical_images_have_zero_cost():
    image = np.arange(25, dtype=np.float64).reshape(5, 5)
    cost = compute_seam_cost(image, image, np.ones_like(image, dtype=bool))
    np.testing.assert_array_equal(cost, np.zeros_like(image))


def test_constant_brightness_mismatch_has_half_unit_cost():
    a = np.full((5, 5), 10.0)
    b = np.full((5, 5), 12.0)
    cost = compute_seam_cost(a, b, np.ones_like(a, dtype=bool))
    np.testing.assert_allclose(cost, 0.5)


def test_displaced_edges_increase_cost_near_the_edges():
    a = np.zeros((9, 9), dtype=float)
    b = np.zeros((9, 9), dtype=float)
    a[:, 4:] = 10.0
    b[:, 5:] = 10.0
    cost = compute_seam_cost(a, b, np.ones_like(a, dtype=bool))
    assert cost[4, 4] > cost[4, 0]
    assert cost[4, 5] > cost[4, 8]


def test_invalid_pixels_are_impassable_and_do_not_propagate_nan():
    a = np.ones((5, 5), dtype=float)
    b = a.copy()
    a[2, 2] = np.nan
    valid = np.ones((5, 5), dtype=bool)
    valid[2, 2] = False
    cost = compute_seam_cost(a, b, valid)
    assert np.isposinf(cost[2, 2])
    assert np.isfinite(cost[valid]).all()


def test_zero_p95_is_finite():
    a = np.full((5, 5), 7.0)
    cost = compute_seam_cost(a, a.copy(), np.ones_like(a, dtype=bool))
    assert np.isfinite(cost).all()
    assert np.count_nonzero(cost) == 0


def test_p95_outlier_is_clipped_to_unit_normalized_cost():
    a = np.zeros((9, 9), dtype=float)
    b = np.ones_like(a)
    b[4, 4] = 1000.0
    cost = compute_seam_cost(a, b, np.ones_like(a, dtype=bool))
    assert np.isfinite(cost).all()
    assert np.max(cost) <= 1.0
