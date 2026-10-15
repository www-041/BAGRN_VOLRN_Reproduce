"""Synthetic end-to-end checks for the frozen Task13A pair pipeline."""

import numpy as np

from src.seam_local.pipeline import _structure_metrics, process_pair


def _synthetic_pair():
    rows, cols = np.indices((320, 500))
    base = 35.0 + 0.09 * rows + 0.14 * cols + 2.0 * np.sin(cols / 9.0)
    a = base.copy()
    b = base + 12.0
    b[:, 255] = base[:, 255]  # Known low-cost, top-to-bottom seam.
    valid_a = cols < 400
    valid_b = cols >= 120
    b[12, 200] = np.nan
    valid_b[12, 200] = False
    return a, b, valid_a, valid_b


def test_pair_pipeline_refines_local_mismatch_and_preserves_outer_sources():
    a, b, valid_a, valid_b = _synthetic_pair()
    result = process_pair(a, b, valid_a, valid_b)

    assert result.status == "PASS"
    assert result.v0_status == result.v1_status == result.v2_status == "PASS"
    assert np.isfinite(result.v0_mosaic[result.valid_union]).all()
    assert result.initial_seam.orientation == "vertical"
    assert np.median(result.initial_seam.row_col_path[:, 1]) == 255
    assert np.max(np.abs(result.refined_seam.row_col_path[:, 1] - 255)) <= 64
    assert result.metrics["fixed_corridor"]["local"]["mae"] < result.metrics["fixed_corridor"]["bagrn"]["mae"]
    assert np.isfinite(result.v1_mosaic[result.valid_union]).all()
    assert np.isfinite(result.v2_mosaic[result.valid_union]).all()
    assert np.array_equal(result.corrected_a[:, 0], a[:, 0])
    assert np.array_equal(result.corrected_b[:, 499], b[:, 499])
    assert result.diagnostics["initial_search_mode"] == "coarse_refined"


def test_unsupported_pair_returns_status_without_mosaic():
    a = np.ones((32, 20), dtype=float)
    b = a + 2
    valid_a = np.ones_like(a, dtype=bool)
    valid_b = np.zeros_like(a, dtype=bool)
    result = process_pair(a, b, valid_a, valid_b)
    assert result.status == "UNSUPPORTED_TOPOLOGY"
    assert result.v1_mosaic is None
    assert result.v2_mosaic is None


def test_degenerate_local_moments_retain_successful_v1():
    a = np.full((320, 500), 20.0)
    b = np.full((320, 500), 30.0)
    cols = np.indices(a.shape)[1]
    result = process_pair(a, b, cols < 400, cols >= 120)
    assert result.status == "LOCAL_MOMENT_UNAVAILABLE"
    assert result.v1_status == "PASS"
    assert result.v2_status == "LOCAL_MOMENT_UNAVAILABLE"
    assert result.v1_mosaic is not None
    assert result.v2_mosaic is None


def test_v0_uses_existing_distance_weighted_feather_semantics():
    a = np.full((5, 9), 10.0)
    b = np.full((5, 9), 40.0)
    cols = np.indices(a.shape)[1]
    result = process_pair(a, b, cols < 7, cols >= 2)
    assert result.v0_status == "PASS"
    assert result.v0_mosaic[2, 0] == 10.0
    assert result.v0_mosaic[2, 8] == 40.0
    assert np.isclose(result.v0_mosaic[2, 3], 20.0, atol=1e-5)


def test_disconnected_joint_overlap_is_unsupported_even_if_each_island_spans():
    a = np.ones((320, 500))
    b = a + 1
    valid = np.zeros_like(a, dtype=bool)
    valid[:, 50:80] = True
    valid[:, 200:240] = True
    result = process_pair(a, b, valid, valid)
    assert result.status == "UNSUPPORTED_TOPOLOGY"
    assert result.initial_seam is None


def test_gradient_ncc_excludes_invalid_hole_stencil():
    rows, cols = np.indices((48, 48))
    source = np.sin(cols / 4.0) + np.cos(rows / 6.0)
    corrected = source.copy()
    corrected[24, 24] = 1e9
    valid = np.ones(source.shape, dtype=bool)
    valid[24, 24] = False
    metrics = _structure_metrics(source, corrected, valid)
    # The existing NCC uses sample std with population covariance, so an
    # identical signal scores (n-1)/n rather than exactly 1.
    assert metrics["gradient_magnitude_ncc"] > 0.999


