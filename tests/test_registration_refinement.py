import numpy as np
import pytest
from rasterio.transform import from_origin

from src.coregistration import (
    analyze_displacement_spikes,
    build_robust_pair_measurement,
    compute_local_shift_field,
    warp_multiband_with_displacement_field,
)


def test_analyze_displacement_spikes_handles_distance_field(tmp_path):
    field = np.zeros((12, 12), dtype=float)
    overlap = np.ones_like(field, dtype=bool)
    hull = np.ones_like(field, dtype=bool)
    nodata_boundary = np.zeros_like(field, dtype=bool)
    result = analyze_displacement_spikes(field, field, overlap, hull, nodata_boundary, str(tmp_path))
    assert result["quality_warning"] is False
    assert result["n_spike_pixels"] == 0


def test_compute_local_shift_field_accepts_valid_confidence(monkeypatch):
    from src import coregistration
    arr = np.arange(64, dtype=float).reshape(8, 8)
    monkeypatch.setattr(coregistration, "phase_correlation", lambda *a, **k: (0.0, 0.0, 0.5))
    result = compute_local_shift_field(
        arr, from_origin(0, 8, 1, 1), arr, from_origin(0, 8, 1, 1),
        nodata=None, block_size=4, confidence_threshold=0.5,
    )
    assert result[0] is not None
    assert len(result[0]) >= 4


def test_warp_with_nodata_none_preserves_valid_zero_pixels():
    arr = np.zeros((1, 4, 4), dtype=float)
    arr[0, 1, 1] = 5.0
    local = np.zeros((4, 4), dtype=float)
    result = warp_multiband_with_displacement_field(arr, 0.0, 0.0, local, local, None)
    assert np.array_equal(result, arr)


def test_robust_pair_measurement_rejects_joint_xy_outliers():
    matches = [{"shift_dx": 3.0 + dx, "shift_dy": -2.0 + dy, "confidence": 0.9}
               for dx, dy in [(0.0, 0.0), (0.1, -0.1), (-0.1, 0.1),
                              (0.0, 0.1), (0.1, 0.0), (18.0, 15.0), (-20.0, 12.0)]]
    result = build_robust_pair_measurement(
        matches, {"global_confidence_threshold": 0.5, "robust_mad_scale": 3.0,
                  "robust_residual_floor": 0.75, "robust_min_inliers": 5,
                  "robust_min_inlier_ratio": 0.35})
    assert result["status"] == "pass"
    assert result["n_blocks_total"] == 7
    assert result["n_blocks_inlier"] == 5
    assert np.isclose(result["shift_dx"], 3.0, atol=0.1)
    assert np.isclose(result["shift_dy"], -2.0, atol=0.1)
    assert result["rmse"] < 0.2


def test_n2_pair_uses_block_samples_not_one_pair_sample():
    matches = [{"shift_dx": 4.0 + dx, "shift_dy": 1.5 + dy, "confidence": 0.8}
               for dx, dy in [(0.0, 0.0), (0.1, 0.0), (-0.1, 0.0),
                              (0.0, 0.1), (0.0, -0.1)]]
    result = build_robust_pair_measurement(
        matches, {"global_confidence_threshold": 0.5, "robust_min_inliers": 5,
                  "robust_min_inlier_ratio": 0.35})
    assert result["n_blocks_total"] == 5
    assert result["n_blocks_inlier"] == 5
    assert result["shift_dx"] != matches[0]["shift_dx"] or result["shift_dy"] != matches[0]["shift_dy"]


def test_post_global_refinement_reduces_known_translation_residual(monkeypatch):
    from src import coregistration
    monkeypatch.setattr(coregistration, "rematch_pair_on_registered",
        lambda *a, **k: {"shift_dx": 1.0, "shift_dy": -0.5, "confidence": 0.9,
                         "n_blocks": 8, "rmse": 0.1, "p95": 0.2,
                         "matches": [], "available": True})
    arrays = [np.ones((8, 8)), np.ones((8, 8))]
    shifts = np.array([[0.0, 0.0], [2.0, -1.0]])
    known_global_shifts = np.array([[0.0, 0.0], [3.0, -1.5]])
    pre_residual = np.linalg.norm(shifts[1] - known_global_shifts[1])
    result = coregistration.refine_global_residual_shifts_from_original(
        arrays, shifts, [None, None], [None, None], [(0, 1)],
        {"global_refine_max_iterations": 1, "global_refine_stop_magnitude": 0.15,
         "global_refine_max_correction": 5.0})
    post_residual = np.linalg.norm(result["global_shifts"][1] - known_global_shifts[1])
    assert post_residual < pre_residual
    assert result["global_shifts"][1] == pytest.approx(known_global_shifts[1])
    assert result["history"]


def test_global_refinement_rewarps_from_original_not_previous_warp(monkeypatch):
    from src import coregistration
    seen = []
    def fake_warp(arr, *args):
        seen.append(arr.copy())
        return arr + len(seen)
    monkeypatch.setattr(coregistration, "warp_multiband_with_displacement_field", fake_warp)
    monkeypatch.setattr(coregistration, "rematch_pair_on_registered",
        lambda *a, **k: {"shift_dx": 0.2, "shift_dy": 0.0, "confidence": 0.9,
                         "n_blocks": 8, "rmse": 0.1, "p95": 0.2,
                         "matches": [], "available": True})
    original = [np.zeros((8, 8)), np.ones((8, 8))]
    coregistration.refine_global_residual_shifts_from_original(
        original, np.array([[0.0, 0.0], [0.3, -0.2]]), [None, None], [None, None], [(0, 1)],
        {"global_refine_max_iterations": 2, "global_refine_stop_magnitude": 0.01,
         "global_refine_max_correction": 5.0})
    assert seen
    assert all(np.array_equal(arr, original[0]) or np.array_equal(arr, original[1]) for arr in seen)


def test_global_refinement_rejects_over_limit_before_stop(monkeypatch):
    from src import coregistration
    monkeypatch.setattr(coregistration, "rematch_pair_on_registered",
        lambda *a, **k: {"shift_dx": 1.0, "shift_dy": 0.0, "confidence": 0.9,
                         "n_blocks": 8, "rmse": 0.1, "p95": 0.2,
                         "matches": [], "available": True})
    original = [np.zeros((8, 8)), np.ones((8, 8))]
    result = coregistration.refine_global_residual_shifts_from_original(
        original, np.zeros((2, 2)), [None, None], [None, None], [(0, 1)],
        {"global_refine_max_iterations": 1, "global_refine_stop_magnitude": 5.0,
         "global_refine_max_correction": 0.5})
    assert np.array_equal(result["global_shifts"], np.zeros((2, 2)))
    assert any("max_correction" in warning for warning in result["warnings"])
    assert result["history"][0]["accepted"] is False


def test_local_rbf_reduces_smooth_spatial_residual(monkeypatch):
    from src import multiband_pipeline
    controls = {
        "points_xy": np.array([[1, 1], [6, 1], [1, 6], [6, 6], [3, 3], [4, 4]], float),
        "residual_dx": np.array([0.0, 0.5, 0.0, 0.5, 0.25, 0.33]),
        "residual_dy": np.zeros(6), "confidence": np.ones(6), "n_valid": 6,
    }
    result = multiband_pipeline._accept_local_rbf_candidate(
        controls,
        {"local_min_controls": 5, "local_min_spatial_groups": 3,
         "local_cv_min_rmse_improvement": 0.01, "local_cv_min_p95_improvement": 0.01},
        cv_result={"baseline_rmse": 1.0, "candidate_rmse": 0.5,
                   "baseline_p95": 1.2, "candidate_p95": 0.8},
    )
    assert result["accepted"] is True


def test_local_rbf_rejected_when_cv_does_not_improve():
    from src import multiband_pipeline
    controls = {"points_xy": np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]], float),
                "residual_dx": np.zeros(5), "residual_dy": np.zeros(5),
                "confidence": np.ones(5), "n_valid": 5}
    result = multiband_pipeline._accept_local_rbf_candidate(
        controls,
        {"local_min_controls": 5, "local_min_spatial_groups": 3,
         "local_cv_min_rmse_improvement": 0.1, "local_cv_min_p95_improvement": 0.1},
        cv_result={"baseline_rmse": 1.0, "candidate_rmse": 0.99,
                   "baseline_p95": 1.2, "candidate_p95": 1.2},
    )
    assert result["accepted"] is False
