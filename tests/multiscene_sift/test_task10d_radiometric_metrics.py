"""RED tests for Task10D global shared-valid metrics."""

import numpy as np
import pytest

from src.multiscene_sift.radiometric_metrics import (
    aggregate_weighted_pair_metric,
    compute_pair_mamd,
    compute_pair_msdd,
    compute_pair_rdd,
)


def test_global_metrics_match_analytic_shared_valid_values():
    left = np.array([[1.0, 2.0, 100.0], [3.0, 4.0, 5.0]])
    right = np.array([[3.0, 4.0, -100.0], [5.0, 6.0, 7.0]])
    valid = np.array([[True, True, False], [True, True, True]])

    assert compute_pair_mamd(left, right, valid) == 2.0
    assert compute_pair_msdd(left, right, valid) == 0.0
    assert compute_pair_rdd(left, right, valid) == pytest.approx(2.0)


def test_identical_arrays_are_zero_and_wasserstein_shift_is_exact():
    values = np.array([0.0, 1.0, 4.0, 9.0])
    valid = np.ones(values.shape, dtype=bool)

    assert compute_pair_mamd(values, values, valid) == 0.0
    assert compute_pair_msdd(values, values, valid) == 0.0
    assert compute_pair_rdd(values, values, valid) == 0.0
    assert compute_pair_rdd(values, values + 3.5, valid) == pytest.approx(3.5)


def test_weighted_aggregate_uses_pixel_counts_and_reports_edges():
    rows = [
        {"pair": "a-b", "value": 1.0, "valid_pixels": 10},
        {"pair": "a-c", "value": 5.0, "valid_pixels": 30},
    ]

    summary = aggregate_weighted_pair_metric(rows)

    assert summary["weighted_mean"] == 4.0
    assert summary["edge_median"] == 3.0
    assert summary["worst_pair"] == {"pair": "a-c", "value": 5.0, "valid_pixels": 30}
    assert summary["per_pair"] == rows
