"""Synthetic tests for the frozen Task13A monotonic seam topology."""

import numpy as np

from src.seam_local.seam import find_monotonic_seam


def test_vertical_valley_and_path_statistics():
    cost = np.full((9, 5), 10.0)
    cost[:, 2] = 1.0
    result = find_monotonic_seam(cost, np.ones_like(cost, dtype=bool))
    assert result.status == "OK"
    assert result.orientation == "vertical"
    np.testing.assert_array_equal(result.row_col_path, [(row, 2) for row in range(9)])
    assert result.total_cost == 9.0
    assert result.mean_cost == 1.0
    assert result.p95_cost == 1.0


def test_horizontal_valley():
    cost = np.full((5, 9), 10.0)
    cost[2, :] = 1.0
    result = find_monotonic_seam(cost, np.ones_like(cost, dtype=bool))
    assert result.status == "OK"
    assert result.orientation == "horizontal"
    np.testing.assert_array_equal(result.row_col_path, [(2, col) for col in range(9)])


def test_irregular_connected_mask_never_leaves_joint_valid():
    cost = np.ones((8, 5))
    valid = np.zeros_like(cost, dtype=bool)
    cols = [1, 1, 2, 3, 3, 2, 1, 1]
    for row, col in enumerate(cols):
        valid[row, col] = True
    result = find_monotonic_seam(cost, valid)
    assert result.status == "OK"
    np.testing.assert_array_equal(result.row_col_path, [(row, col) for row, col in enumerate(cols)])
    assert np.all(valid[result.row_col_path[:, 0], result.row_col_path[:, 1]])


def test_disconnected_mask_reports_unsupported_topology():
    cost = np.ones((7, 4))
    valid = np.ones_like(cost, dtype=bool)
    valid[3, :] = False
    result = find_monotonic_seam(cost, valid)
    assert result.status == "UNSUPPORTED_TOPOLOGY"
    assert result.row_col_path.shape == (0, 2)


def test_nonfinite_cost_cannot_be_part_of_path():
    cost = np.ones((8, 4))
    cost[:, 0] = np.nan
    cost[:, 1] = np.inf
    result = find_monotonic_seam(cost, np.ones_like(cost, dtype=bool))
    assert result.status == "OK"
    assert np.all(result.row_col_path[:, 1] >= 2)


def test_coarse_to_fine_tracks_full_resolution_valley():
    height, width = 272, 272
    center = 100 + np.arange(height) // 8
    cost = np.abs(np.arange(width)[None, :] - center[:, None]).astype(float)
    valid = np.ones((height, width), dtype=bool)
    coarse = find_monotonic_seam(cost, valid)
    full = find_monotonic_seam(cost, valid, coarse_factor=1)
    assert coarse.status == full.status == "OK"
    assert np.max(np.abs(coarse.row_col_path[:, 1] - full.row_col_path[:, 1])) <= 4
    assert np.all(valid[coarse.row_col_path[:, 0], coarse.row_col_path[:, 1]])


def test_small_dimension_uses_full_resolution_despite_coarse_factor():
    cost = np.full((270, 20), 5.0)
    cost[:, 3] = 0.0
    result = find_monotonic_seam(cost, np.ones_like(cost, dtype=bool))
    assert result.status == "OK"
    assert np.all(result.row_col_path[:, 1] == 3)


def test_sparse_coarse_stubs_do_not_hide_valid_full_resolution_seam():
    cost = np.ones((272, 272))
    valid = np.zeros_like(cost, dtype=bool)
    valid[:, 20] = True  # The sole full-resolution spanning path.
    valid[::4, 220] = True  # Cheap block minima that cannot connect at full resolution.
    cost[::4, 220] = 0.0
    result = find_monotonic_seam(cost, valid)
    assert result.status == "OK"
    np.testing.assert_array_equal(result.row_col_path[:, 1], np.full(272, 20))
