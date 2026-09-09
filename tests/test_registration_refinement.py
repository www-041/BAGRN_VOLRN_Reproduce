import numpy as np
import pytest
from types import SimpleNamespace
from rasterio.transform import from_origin

from src.coregistration import (
    analyze_displacement_spikes,
    build_robust_pair_measurement,
    compute_local_shift_field,
    warp_multiband_with_displacement_field,
)


def test_independent_validation_uses_configured_thresholds(monkeypatch):
    from src import coregistration
    monkeypatch.setattr(coregistration, "phase_correlation",
                        lambda *a, **k: (0.0, 0.0, 0.7))
    arr = np.arange(64 * 64, dtype=float).reshape(64, 64)
    result = coregistration.validate_registration_independent_grid(
        arr, from_origin(0, 64, 1, 1), arr, from_origin(0, 64, 1, 1),
        None, None, np.empty((0, 2)), block_size=16, step=16,
        min_distance_from_training=0, confidence_threshold=0.7,
        max_residual_shift=1.0, min_accepted=1)
    assert result["stats"] is not None or result["failure_reason"] is not None


def test_register_scenes_validates_final_arrays_and_all_training_controls(monkeypatch):
    from src import coregistration, multiband_pipeline
    from src.multiband_pipeline import MultibandPipeline

    params = {
        "enable_local_refinement": True,
        "local_min_controls": 1,
        "local_min_spatial_groups": 1,
        "local_cv_min_rmse_improvement": 0.01,
        "local_cv_min_p95_improvement": 0.01,
        "local_max_controls": 10,
        "local_block_size": 4,
        "local_confidence_threshold": 0.6,
        "validation_block_size": 4,
        "validation_step": 4,
        "validation_offset_row": 0,
        "validation_offset_col": 0,
        "validation_min_distance_from_training": 0,
        "validation_confidence_threshold": 0.6,
        "validation_max_residual_shift": 1.0,
        "final_min_blocks": 5,
        "pass_min_mean_confidence": 0.5,
        "pass_max_median": 0.35,
        "pass_max_rmse": 0.6,
        "pass_max_p95": 1.0,
        "warn_min_mean_confidence": 0.45,
        "warn_max_median": 0.5,
        "warn_max_rmse": 0.75,
        "warn_max_p95": 1.25,
    }
    original_match = {
        "ref_x": 2.0, "ref_y": 2.0, "tgt_x": 2.0, "tgt_y": 2.0,
        "shift_dx": 1.0, "shift_dy": 0.0, "confidence": 0.9,
    }
    post_global_match = {
        "ref_x": 10.0, "ref_y": 10.0, "tgt_x": 11.0, "tgt_y": 10.0,
        "shift_dx": 0.0, "shift_dy": 0.0, "confidence": 0.9,
    }
    monkeypatch.setattr(coregistration, "collect_block_matches",
                        lambda *a, **k: ([original_match], {}))
    monkeypatch.setattr(coregistration, "build_robust_pair_measurement",
                        lambda *a, **k: {
                            "status": "pass", "shift_dx": 1.0, "shift_dy": 0.0,
                            "confidence": 0.9, "n_blocks_inlier": 1,
                            "n_blocks_total": 1, "rmse": 0.0, "p95": 0.0,
                            "matches": [original_match], "screening": {},
                        })
    monkeypatch.setattr(coregistration, "multi_image_network_adjustment",
                        lambda *a, **k: {
                            "global_shifts": np.array([[0.0, 0.0], [1.0, 0.0]]),
                            "loop_errors": [],
                        })
    monkeypatch.setattr(coregistration, "refine_global_residual_shifts_from_original",
                        lambda *a, **k: {
                            "global_shifts": np.array([[0.0, 0.0], [1.0, 0.0]]),
                            "history": [], "warnings": [],
                        })
    warp_calls = []
    def fake_warp(array, *args, **kwargs):
        warp_calls.append(array.copy())
        return array.astype(float) + len(warp_calls)
    monkeypatch.setattr(coregistration, "warp_multiband_with_displacement_field", fake_warp)
    monkeypatch.setattr(coregistration, "rematch_pair_on_registered",
                        lambda *a, **k: {
                            "available": True, "shift_dx": 0.0, "shift_dy": 0.0,
                            "confidence": 0.9, "n_blocks": 1, "rmse": 0.0,
                            "p95": 0.0, "matches": [post_global_match],
                        })
    monkeypatch.setattr(coregistration, "build_parent_based_local_controls",
                        lambda *a, **k: {
                            "points_xy": np.array([[11.0, 10.0]]),
                            "residual_dx": np.array([0.0]),
                            "residual_dy": np.array([0.0]),
                            "confidence": np.array([0.9]), "n_valid": 1,
                        })
    monkeypatch.setattr(coregistration, "balance_edge_controls",
                        lambda controls, **kwargs: controls)
    monkeypatch.setattr(multiband_pipeline, "_local_holdout_cv",
                        lambda *a, **k: {
                            "available": True, "baseline_rmse": 1.0,
                            "candidate_rmse": 0.5, "baseline_p95": 1.2,
                            "candidate_p95": 0.8,
                        })
    monkeypatch.setattr(multiband_pipeline, "_fit_local_rbf_field",
                        lambda controls, shape, params, **kwargs: (
                            np.zeros(shape), np.zeros(shape), {},
                        ))
    validation_calls = []
    def fake_validate(arr_ref, tr_ref, arr_registered, tr_registered,
                      nodata_ref, nodata_tgt, training_points, **kwargs):
        validation_calls.append((arr_ref.copy(), arr_registered.copy(),
                                 nodata_ref, nodata_tgt,
                                 np.asarray(training_points).copy(), kwargs))
        return {
            "blocks": [],
            "stats": {"median": 0.1, "rmse": 0.2, "p95": 0.3,
                      "mean_confidence": 0.8, "n_accepted": 5},
            "coverage": {}, "failure_reason": None,
        }
    monkeypatch.setattr(coregistration,
                        "validate_registration_independent_grid", fake_validate)

    pipeline = MultibandPipeline.__new__(MultibandPipeline)
    pipeline.common_bands = ["B14"]
    pipeline.registration_band_idx = 0
    pipeline.control_idx = 0
    pipeline.smoke = False
    pipeline.config = type("Config", (), {"registration_params": params})()
    arrays = [np.ones((1, 8, 8)), np.ones((1, 8, 8)) * 2]
    transforms = [from_origin(0, 8, 1, 1), from_origin(0, 8, 1, 1)]
    result = pipeline.register_scenes(
        {"arrays": arrays, "transforms": transforms,
         "nodata_values": [None, None]},
        [{"idx_i": 0, "idx_j": 1}],
    )

    assert result["spanning_tree"] == [(0, 1)]
    assert len(validation_calls) == 2
    before_ref, before_target, nd_ref, nd_tgt, training_points, kwargs = validation_calls[0]
    after_ref, after_target, *_ = validation_calls[1]
    assert np.array_equal(before_ref, result["global_only_arrays"][0][0])
    assert np.array_equal(after_ref, result["registered_arrays"][0][0])
    assert not np.array_equal(before_target, arrays[1][0])
    assert not np.array_equal(after_target, arrays[1][0])
    assert nd_ref is None and nd_tgt is None
    assert {tuple(point) for point in training_points} == {(2.0, 2.0)}
    after_training_points = np.asarray(validation_calls[1][4])
    assert {tuple(point) for point in after_training_points} == {
        (2.0, 2.0), (10.0, 10.0)
    }
    assert kwargs["confidence_threshold"] == 0.6
    assert kwargs["max_residual_shift"] == 1.0
    assert kwargs["min_accepted"] == 5
    assert result["quality"]["quality"] == "pass"


def test_register_scenes_returns_stable_final_registration_schema(monkeypatch):
    from src import coregistration
    from src.multiband_pipeline import MultibandPipeline

    params = {
        "enable_local_refinement": False,
        "validation_block_size": 4,
        "validation_step": 4,
        "validation_offset_row": 0,
        "validation_offset_col": 0,
        "validation_min_distance_from_training": 0,
        "validation_confidence_threshold": 0.6,
        "validation_max_residual_shift": 1.0,
        "final_min_blocks": 5,
        "pass_min_mean_confidence": 0.5,
        "pass_max_median": 0.35,
        "pass_max_rmse": 0.6,
        "pass_max_p95": 1.0,
        "warn_min_mean_confidence": 0.45,
        "warn_max_median": 0.5,
        "warn_max_rmse": 0.75,
        "warn_max_p95": 1.25,
    }
    matches = [
        {"ref_x": 2.0, "ref_y": 2.0, "tgt_x": 2.0, "tgt_y": 2.0,
         "shift_dx": 1.0, "shift_dy": 0.0, "confidence": 0.9},
    ]
    monkeypatch.setattr(
        coregistration, "collect_block_matches",
        lambda *args, **kwargs: (matches, {}),
    )
    monkeypatch.setattr(
        coregistration, "build_robust_pair_measurement",
        lambda *args, **kwargs: {
            "status": "pass", "shift_dx": 1.0, "shift_dy": 0.0,
            "confidence": 0.9, "n_blocks_inlier": 1,
            "n_blocks_total": 1, "rmse": 0.0, "p95": 0.0,
            "matches": matches, "screening": {},
        },
    )
    monkeypatch.setattr(
        coregistration, "multi_image_network_adjustment",
        lambda *args, **kwargs: {
            "global_shifts": np.array([[0.0, 0.0], [1.0, 0.0]]),
            "loop_errors": [],
        },
    )
    monkeypatch.setattr(
        coregistration, "refine_global_residual_shifts_from_original",
        lambda *args, **kwargs: {
            "global_shifts": np.array([[0.0, 0.0], [1.0, 0.0]]),
            "history": [], "warnings": [],
        },
    )
    monkeypatch.setattr(
        coregistration, "warp_multiband_with_displacement_field",
        lambda array, *args, **kwargs: array.astype(float).copy(),
    )
    validation = {
        "blocks": [],
        "stats": {"median": 0.1, "rmse": 0.2, "p95": 0.3,
                  "mean_confidence": 0.8, "n_accepted": 5},
        "coverage": {}, "failure_reason": None,
    }
    monkeypatch.setattr(
        coregistration, "validate_registration_independent_grid",
        lambda *args, **kwargs: validation.copy(),
    )

    pipeline = MultibandPipeline.__new__(MultibandPipeline)
    pipeline.common_bands = ["B14"]
    pipeline.registration_band_idx = 0
    pipeline.control_idx = 0
    pipeline.smoke = False
    pipeline.config = type("Config", (), {"registration_params": params})()
    arrays = [np.ones((1, 8, 8)), np.ones((1, 8, 8)) * 2]
    result = pipeline.register_scenes(
        {"arrays": arrays,
         "transforms": [from_origin(0, 8, 1, 1)] * 2,
         "nodata_values": [None, None]},
        [{"idx_i": 0, "idx_j": 1}],
    )

    assert result["connected"] is True
    assert result["pair_matches"]
    assert {"spanning_tree", "geometric_edges", "matching_edges",
            "rejected_edges", "connected_components", "unreachable_scenes"} <= result.keys()
    assert {"enabled", "used_for_scenes", "fallback_scenes", "cv_results"} <= result["local_refinement"].keys()
    assert {"edges", "overall"} == set(result["final_validation"])
    assert set(result["quality"]) == {
        "quality", "rmse", "p95", "median", "confidence", "n_blocks",
    }
    assert result["quality"]["quality"] == "pass"
    assert result["quality"]["confidence"] == pytest.approx(0.8)
    assert result["quality"]["n_blocks"] == 5


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


def test_robust_pair_measurement_repeats_mad_filter_until_stable():
    shifts = [
        (0.7562135272, -2.0909937659),
        (-1.6522541736, -9.7658695306),
        (7.1988295309, 4.5766634881),
        (-1.3016913475, 3.0952263469),
        (1.1248426792, -2.2152913457),
        (3.9102698045, -1.2422261866),
        (-1.3152956162, -3.1685870214),
        (1.8198322850, -0.3967922069),
    ]
    result = build_robust_pair_measurement(
        [{"shift_dx": dx, "shift_dy": dy, "confidence": 0.9}
         for dx, dy in shifts],
        {"global_confidence_threshold": 0.5, "robust_mad_scale": 3.0,
         "robust_residual_floor": 0.75, "robust_min_inliers": 5,
         "robust_min_inlier_ratio": 0.35},
    )

    assert result["status"] == "pass"
    assert result["n_blocks_inlier"] == 6
    assert all(match["shift_dy"] > -9.0 for match in result["matches"])


def test_robust_pair_measurement_fails_when_mad_filter_drops_below_minimum():
    matches = [
        {"shift_dx": 2.0, "shift_dy": -1.0, "confidence": 0.9},
        {"shift_dx": 2.1, "shift_dy": -1.0, "confidence": 0.9},
        {"shift_dx": 1.9, "shift_dy": -1.0, "confidence": 0.9},
        {"shift_dx": 2.0, "shift_dy": -0.9, "confidence": 0.9},
        {"shift_dx": 15.0, "shift_dy": 12.0, "confidence": 0.9},
    ]

    result = build_robust_pair_measurement(
        matches,
        {
            "global_confidence_threshold": 0.5,
            "robust_mad_scale": 3.0,
            "robust_residual_floor": 0.75,
            "robust_min_inliers": 5,
            "robust_min_inlier_ratio": 0.35,
        },
    )

    assert result["status"] == "fail"
    assert result["n_blocks_inlier"] == 4


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


def _four_corner_local_controls():
    points = np.array([
        [1, 1], [2, 1], [1, 2],
        [8, 1], [9, 1], [8, 2],
        [1, 8], [2, 8], [1, 9],
        [8, 8], [9, 8], [8, 9],
    ], dtype=float)
    return {
        "points_xy": points,
        "residual_dx": points[:, 0] / 20.0,
        "residual_dy": points[:, 1] / 20.0,
        "confidence": np.ones(len(points)),
        "n_valid": len(points),
    }


def test_local_holdout_cv_rejects_candidate_when_any_planned_fold_fails(monkeypatch):
    from src import coregistration, multiband_pipeline
    original_fit = coregistration.fit_local_rbf
    calls = {"count": 0}

    def fail_first_fit(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("synthetic fold failure")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(coregistration, "fit_local_rbf", fail_first_fit)
    result = multiband_pipeline._local_holdout_cv(
        _four_corner_local_controls(),
        {"local_min_controls": 3, "local_smoothing_candidates": [0.1],
         "local_cv_min_validation_controls": 6},
    )
    assert result["available"] is False
    assert result["n_folds_attempted"] == 4
    assert result["candidate_results"][0]["available"] is False
    assert result["candidate_results"][0]["failed_group_ids"]


def test_local_holdout_cv_rejects_too_few_successful_folds(monkeypatch):
    from src import coregistration, multiband_pipeline
    original_fit = coregistration.fit_local_rbf
    calls = {"count": 0}

    def fail_first_two_fits(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] <= 2:
            raise ValueError("synthetic fold failure")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(coregistration, "fit_local_rbf", fail_first_two_fits)
    result = multiband_pipeline._local_holdout_cv(
        _four_corner_local_controls(),
        {"local_min_controls": 3, "local_smoothing_candidates": [0.1]}
    )
    assert result["available"] is False
    assert result["n_folds"] == 4
    assert result["n_folds_attempted"] == 4
    assert result["candidate_results"][0]["available"] is False


def test_local_holdout_cv_scores_clipped_candidate_field(monkeypatch):
    from src import coregistration, multiband_pipeline

    class ConstantRBF:
        def __call__(self, points):
            return np.full(len(points), 10.0)

    monkeypatch.setattr(
        coregistration, "fit_local_rbf",
        lambda *args, **kwargs: (ConstantRBF(), ConstantRBF(), (0.0, 0.0), (10.0, 10.0)),
    )
    controls = _four_corner_local_controls()
    controls["residual_dx"] = np.zeros(12)
    controls["residual_dy"] = np.zeros(12)
    result = multiband_pipeline._local_holdout_cv(
        controls, {"local_min_controls": 3, "local_smoothing_candidates": [0.1],
                   "local_max_component": 2.5}
    )
    assert result["candidate_rmse"] == pytest.approx(np.hypot(2.5, 2.5))


def test_post_global_rematch_exception_is_recorded_as_fallback():
    from src import multiband_pipeline

    def fail_rematch(*args, **kwargs):
        raise RuntimeError("synthetic rematch failure")

    result = multiband_pipeline._collect_post_global_residual_pairs(
        [np.zeros((1, 8, 8)), np.zeros((1, 8, 8))],
        0,
        [from_origin(0, 8, 1, 1), from_origin(0, 8, 1, 1)],
        [None, None],
        [(0, 1)],
        {"local_block_size": 4, "local_confidence_threshold": 0.6,
         "local_max_residual_shift": 3.0},
        fail_rematch,
    )
    assert result["pairs"] == []
    assert result["failures"][0]["idx_i"] == 0
    assert "synthetic rematch failure" in result["failures"][0]["reason"]


def test_global_residual_rematch_exception_is_recorded_per_edge(monkeypatch):
    from src import coregistration

    calls = {"count": 0}

    def fail_rematch(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("synthetic global rematch failure")
        return {
            "available": True,
            "shift_dx": 0.0,
            "shift_dy": 0.0,
            "confidence": 0.9,
            "n_blocks": 1,
            "rmse": 0.0,
            "p95": 0.0,
            "matches": [],
        }

    monkeypatch.setattr(coregistration, "rematch_pair_on_registered", fail_rematch)
    monkeypatch.setattr(
        coregistration,
        "multi_image_network_adjustment",
        lambda *args, **kwargs: {
            "global_shifts": np.zeros((3, 2)), "loop_errors": []
        },
    )
    result = coregistration.refine_global_residual_shifts_from_original(
        [np.zeros((8, 8)) for _ in range(3)],
        np.zeros((3, 2)),
        [from_origin(0, 8, 1, 1) for _ in range(3)],
        [None, None, None],
        [(0, 1), (1, 2)],
        {"global_refine_max_iterations": 1},
    )

    assert np.array_equal(result["global_shifts"], np.zeros((3, 2)))
    assert result["history"][0]["edges"][0]["available"] is False
    assert result["history"][0]["edges"][1]["available"] is True
    assert "synthetic global rematch failure" in result["history"][0]["edges"][0]["warning"]
    assert result["history"][0]["accepted"] is False


def test_single_scene_registration_has_local_refinement_diagnostics():
    from src.multiband_pipeline import MultibandPipeline

    pipeline = MultibandPipeline.__new__(MultibandPipeline)
    pipeline.common_bands = ["B14"]
    pipeline.registration_band_idx = 0
    result = pipeline.register_scenes({
        "arrays": [np.ones((1, 4, 4))],
        "transforms": [from_origin(0, 4, 1, 1)],
        "nodata_values": [None],
    }, [])
    diagnostics = result["diagnostics"]["local_refinement"]
    assert set(("enabled", "used_for_scenes", "fallback_scenes", "cv_results")) <= set(diagnostics)


def test_local_rbf_forces_global_only_when_scene_rematch_fails():
    from src import multiband_pipeline

    result = multiband_pipeline._accept_local_rbf_candidate(
        _four_corner_local_controls(),
        {"local_min_controls": 5, "local_min_spatial_groups": 3,
         "local_cv_min_rmse_improvement": 0.01,
         "local_cv_min_p95_improvement": 0.01},
        cv_result={"baseline_rmse": 1.0, "candidate_rmse": 0.5,
                   "baseline_p95": 1.2, "candidate_p95": 0.8},
        rematch_failures=[{"idx_i": 0, "idx_j": 1,
                           "reason": "post-global rematch unavailable"}],
    )
    assert result["accepted"] is False
    assert result["reason"] == "global-only fallback: required post-global rematch failed"


def test_register_scenes_excludes_affected_scene_but_processes_surviving_edge(monkeypatch):
    from src import coregistration, multiband_pipeline
    from src.multiband_pipeline import MultibandPipeline

    params = {
        "enable_local_refinement": True,
        "local_min_controls": 5,
        "local_min_spatial_groups": 3,
        "local_cv_min_rmse_improvement": 0.01,
        "local_cv_min_p95_improvement": 0.01,
        "local_max_controls": 60,
        "local_block_size": 4,
        "local_confidence_threshold": 0.6,
    }
    controls = _four_corner_local_controls()
    monkeypatch.setattr(
        coregistration, "collect_block_matches",
        lambda *args, **kwargs: ([{}], {}),
    )
    monkeypatch.setattr(
        coregistration, "build_robust_pair_measurement",
        lambda *args, **kwargs: {
            "status": "pass", "shift_dx": 0.0, "shift_dy": 0.0,
            "confidence": 0.9, "n_blocks_inlier": 1, "n_blocks_total": 1,
            "rmse": 0.0, "p95": 0.0, "matches": [], "screening": {},
        },
    )
    monkeypatch.setattr(
        coregistration, "multi_image_network_adjustment",
        lambda *args, **kwargs: {"global_shifts": np.zeros((3, 2)), "loop_errors": []},
    )
    monkeypatch.setattr(
        coregistration, "refine_global_residual_shifts_from_original",
        lambda *args, **kwargs: {
            "global_shifts": np.zeros((3, 2)), "history": [], "warnings": [],
        },
    )
    monkeypatch.setattr(
        coregistration, "warp_multiband_with_displacement_field",
        lambda array, *args, **kwargs: array.copy(),
    )
    rematch_calls = {"count": 0}

    def rematch_with_one_unavailable_edge(*args, **kwargs):
        rematch_calls["count"] += 1
        if rematch_calls["count"] == 1:
            return {"available": False}
        return {
            "available": True, "shift_dx": 0.0, "shift_dy": 0.0,
            "confidence": 0.9, "n_blocks": 8, "rmse": 0.1, "p95": 0.2,
            "matches": [],
        }

    monkeypatch.setattr(coregistration, "rematch_pair_on_registered", rematch_with_one_unavailable_edge)
    monkeypatch.setattr(
        coregistration, "build_parent_based_local_controls",
        lambda *args, **kwargs: controls.copy(),
    )
    monkeypatch.setattr(
        coregistration, "build_local_residual_controls",
        lambda *args, **kwargs: controls.copy(),
    )
    monkeypatch.setattr(
        coregistration, "balance_edge_controls",
        lambda edge_controls, **kwargs: edge_controls,
    )
    monkeypatch.setattr(
        multiband_pipeline, "_local_holdout_cv",
        lambda *args, **kwargs: {
            "available": True, "baseline_rmse": 1.0, "candidate_rmse": 0.5,
            "baseline_p95": 1.2, "candidate_p95": 0.8,
            "selected_smoothing": 0.5, "has_passing_candidate": True,
            "cv_strategy": "buffered_spatial_group", "buffer_pixels": 256.0,
            "fold_diagnostics": [], "dropped_folds": [],
        },
    )
    fitted_smoothing = []
    monkeypatch.setattr(
        multiband_pipeline, "_fit_local_rbf_field",
        lambda controls, shape, params, **kwargs: (
            fitted_smoothing.append(kwargs.get("smoothing"))
            or (np.zeros(shape), np.zeros(shape), {"dx": {}, "dy": {}, "clipping": {}})
        ),
    )

    pipeline = MultibandPipeline.__new__(MultibandPipeline)
    pipeline.common_bands = ["B14"]
    pipeline.registration_band_idx = 0
    pipeline.control_idx = 0
    pipeline.smoke = False
    pipeline.config = type("Config", (), {"registration_params": params})()
    arrays = [np.ones((1, 8, 8)) for _ in range(3)]
    transforms = [from_origin(0, 8, 1, 1) for _ in range(3)]
    result = pipeline.register_scenes(
        {"arrays": arrays, "transforms": transforms, "nodata_values": [None] * 3},
        [{"idx_i": 0, "idx_j": 1}, {"idx_i": 1, "idx_j": 2}],
    )

    diagnostics = result["diagnostics"]["local_refinement"]
    assert diagnostics["used_for_scenes"] == [2]
    assert fitted_smoothing == [0.5]
    assert diagnostics["fallback_scenes"] == [1]
    assert diagnostics["cv_results"]["1"]["cv_strategy"] == "buffered_spatial_group"
    assert diagnostics["cv_results"]["1"]["buffer_pixels"] == 256.0
    assert diagnostics["scenes"]["1"]["reason"] == (
        "global-only fallback: required post-global rematch failed"
    )
    assert np.allclose(result["registered_arrays"][1], result["global_only_arrays"][1])


def test_register_scenes_returns_actual_registration_schema(monkeypatch):
    from src.experiment_config import ExperimentConfig
    from src import multiband_pipeline
    from src.multiband_pipeline import MultibandPipeline

    params = {
        "global_block_size": 16,
        "global_confidence_threshold": 0.5,
        "max_global_shift": 40.0,
        "robust_min_inliers": 1,
        "robust_min_inlier_ratio": 0.1,
        "enable_local_refinement": False,
        "global_refine_max_iterations": 0,
        "validation_block_size": 4,
        "validation_step": 4,
        "validation_offset_row": 0,
        "validation_offset_col": 0,
        "validation_min_distance_from_training": 0,
        "validation_confidence_threshold": 0.5,
        "validation_max_residual_shift": 1.0,
        "final_min_blocks": 1,
        "pass_min_mean_confidence": 0.5,
        "pass_max_median": 0.35,
        "pass_max_rmse": 0.6,
        "pass_max_p95": 1.0,
        "warn_min_mean_confidence": 0.45,
        "warn_max_median": 0.5,
        "warn_max_rmse": 0.75,
        "warn_max_p95": 1.25,
        "required_quality": "warn",
    }
    pipeline = object.__new__(MultibandPipeline)
    pipeline.registration_band_idx = 0
    pipeline.control_idx = 0
    pipeline.common_bands = ["B14"]
    pipeline.n_bands = 1
    pipeline.smoke = False
    pipeline.config = ExperimentConfig(
        selected_bands=["B14"], registration_band="B14",
        registration_params=params,
    )
    arrays = [
        np.arange(64, dtype=float).reshape(1, 8, 8),
        np.arange(64, dtype=float).reshape(1, 8, 8),
    ]
    scene_data = {
        "arrays": arrays,
        "transforms": [from_origin(0, 8, 1, 1)] * 2,
        "nodata_values": [None, None],
        "scene_ids": ["a", "b"],
    }
    monkeypatch.setattr(
        "src.coregistration.collect_block_matches",
        lambda *a, **k: ([{
            "shift_dx": 0.0, "shift_dy": 0.0, "confidence": 0.9,
            "ref_x": 4.0, "ref_y": 4.0, "tgt_x": 4.0, "tgt_y": 4.0,
        }], {}),
    )
    monkeypatch.setattr(
        "src.coregistration.multi_image_network_adjustment",
        lambda *a, **k: {"global_shifts": np.zeros((2, 2)), "loop_errors": []},
    )
    warp_inputs = []
    def fake_warp(array, *args, **kwargs):
        warp_inputs.append(array)
        return array.copy()
    monkeypatch.setattr(
        "src.coregistration.warp_multiband_with_displacement_field", fake_warp,
    )
    validation_calls = []
    def fake_validate(*args, **kwargs):
        validation_calls.append((args, kwargs))
        magnitude = 1.0 if len(validation_calls) == 1 else 0.5
        return {
            "stats": {
                "median": magnitude, "rmse": magnitude, "p95": magnitude,
                "mean_confidence": 0.9, "n_accepted": 1,
            },
            "blocks": [{
                "validation_row": 0, "validation_col": 0,
                "center_x": 2.0, "center_y": 2.0,
                "residual_dx": magnitude, "residual_dy": 0.0,
                "residual_magnitude": magnitude,
                "accepted": True, "reject_reason": None,
            }],
            "failure_reason": None,
        }
    monkeypatch.setattr(
        "src.coregistration.validate_registration_independent_grid", fake_validate,
    )
    sampler_calls = []
    monkeypatch.setattr(
        multiband_pipeline, "_sample_local_field_at_validation_blocks",
        lambda *args: sampler_calls.append(args) or {},
    )

    result = pipeline.register_scenes(scene_data, [{"idx_i": 0, "idx_j": 1}])

    assert result["connected"] is True
    assert result["quality"]["quality"] in {"pass", "warn", "fail"}
    assert set(result["quality"]) >= {
        "quality", "rmse", "p95", "median", "confidence", "n_blocks",
    }
    assert set(result["local_refinement"]) >= {
        "enabled", "used_for_scenes", "fallback_scenes", "cv_results",
    }
    assert set(result["final_validation"]) >= {"edges", "overall"}
    assert result["spanning_tree"] == [(0, 1)]
    assert "geometric_edges" in result
    assert "matching_edges" in result
    assert "rejected_edges" in result
    assert "connected_components" in result
    assert "unreachable_scenes" in result
    assert len(validation_calls) == 2
    assert "global_only_validation" in result
    assert "global_only_quality" in result
    assert result["stage_validation_comparison"]["n_blocks_common"] == 1
    assert result["stage_validation_comparison"]["rmse_improvement"] == pytest.approx(0.5)
    assert sampler_calls
    assert sampler_calls[0][4] is scene_data["transforms"]
    assert sampler_calls[0][5] is params
    assert len(warp_inputs) == 1
    assert warp_inputs[0] is arrays[1]


def test_register_scenes_forwards_configured_initial_matching_controls(monkeypatch):
    from src import coregistration, multiband_pipeline
    from src.multiband_pipeline import MultibandPipeline

    params = {
        "global_block_size": 17,
        "max_global_shift": 7.5,
        "global_confidence_threshold": 0.8,
        "enable_local_refinement": False,
        "global_refine_max_iterations": 0,
    }
    seen = {}

    def fake_collect(*args, **kwargs):
        seen.update(kwargs)
        return [], {}

    monkeypatch.setattr(coregistration, "collect_block_matches", fake_collect)
    monkeypatch.setattr(
        coregistration,
        "phase_correlation_from_overlap",
        lambda *args, **kwargs: (1.0, 2.0, 0.9),
    )
    monkeypatch.setattr(
        coregistration,
        "multi_image_network_adjustment",
        lambda *args, **kwargs: {
            "global_shifts": np.zeros((2, 2)), "loop_errors": []
        },
    )
    monkeypatch.setattr(
        multiband_pipeline,
        "_validate_final_registration_arrays",
        lambda *args, **kwargs: (
            {"quality": "pass"}, {"edges": [], "overall": {"quality": "pass"}}
        ),
    )

    pipeline = MultibandPipeline.__new__(MultibandPipeline)
    pipeline.registration_band_idx = 0
    pipeline.control_idx = 0
    pipeline.common_bands = ["B14"]
    pipeline.smoke = False
    pipeline.config = SimpleNamespace(registration_params=params)
    scene_data = {
        "arrays": [np.ones((1, 8, 8)), np.ones((1, 8, 8))],
        "transforms": [from_origin(0, 8, 1, 1)] * 2,
        "nodata_values": [None, None],
        "scene_ids": ["a", "b"],
    }

    result = pipeline.register_scenes(
        scene_data, [{"idx_i": 0, "idx_j": 1}],
    )

    assert seen["block_size"] == 17
    assert seen["max_global_shift"] == 7.5
    assert seen["confidence_threshold"] == 0.8
    assert result["pair_matches"][0]["method"] == "overlap_phase_correlation"


def test_register_scenes_applies_configured_phase_fallback_limits(monkeypatch):
    from src import coregistration
    from src.multiband_pipeline import MultibandPipeline

    params = {
        "global_block_size": 17,
        "max_global_shift": 7.5,
        "global_confidence_threshold": 0.95,
        "enable_local_refinement": False,
    }
    monkeypatch.setattr(
        coregistration,
        "collect_block_matches",
        lambda *args, **kwargs: ([], {}),
    )
    monkeypatch.setattr(
        coregistration,
        "phase_correlation_from_overlap",
        lambda *args, **kwargs: (8.0, 0.0, 0.9),
    )

    pipeline = MultibandPipeline.__new__(MultibandPipeline)
    pipeline.registration_band_idx = 0
    pipeline.control_idx = 0
    pipeline.common_bands = ["B14"]
    pipeline.smoke = False
    pipeline.config = SimpleNamespace(registration_params=params)
    scene_data = {
        "arrays": [np.ones((1, 8, 8)), np.ones((1, 8, 8))],
        "transforms": [from_origin(0, 8, 1, 1)] * 2,
        "nodata_values": [None, None],
        "scene_ids": ["a", "b"],
    }

    result = pipeline.register_scenes(
        scene_data, [{"idx_i": 0, "idx_j": 1}],
    )

    assert result["pair_matches"] == []
    assert result["rejected_edges"]


def test_registration_quality_meets_requirement_uses_final_quality_order():
    from src.multiband_pipeline import registration_quality_meets_requirement

    assert registration_quality_meets_requirement("fail", "pass") is False
    assert registration_quality_meets_requirement("warn", "pass") is False
    assert registration_quality_meets_requirement("pass", "pass") is True
    assert registration_quality_meets_requirement("pass", "warn") is True
    assert registration_quality_meets_requirement("warn", "warn") is True


def test_final_quality_fails_when_required_validation_edge_is_unavailable():
    from src.coregistration import aggregate_final_validation_quality

    params = {
        "final_min_blocks": 1,
        "pass_min_mean_confidence": 0.5,
        "pass_max_median": 0.35,
        "pass_max_rmse": 0.6,
        "pass_max_p95": 1.0,
        "warn_min_mean_confidence": 0.45,
        "warn_max_median": 0.5,
        "warn_max_rmse": 0.75,
        "warn_max_p95": 1.25,
    }
    result = aggregate_final_validation_quality(
        [
            {"idx_i": 0, "idx_j": 1,
             "stats": {"median": 0.1, "rmse": 0.2, "p95": 0.3,
                       "mean_confidence": 0.9, "n_accepted": 1}},
            {"idx_i": 1, "idx_j": 2, "stats": None,
             "failure_reason": "no independent blocks"},
        ],
        params,
        required_edges=[(0, 1), (1, 2)],
    )

    assert result["quality"] == "fail"
    assert result["unavailable_edges"] == [(1, 2)]


def test_register_scenes_returns_stable_failure_schema_when_no_pair_is_usable(monkeypatch):
    from src import coregistration
    from src.multiband_pipeline import MultibandPipeline

    monkeypatch.setattr(
        coregistration, "collect_block_matches",
        lambda *args, **kwargs: ([], {"low_valid": 1}),
    )
    monkeypatch.setattr(
        coregistration, "phase_correlation_from_overlap",
        lambda *args, **kwargs: (0.0, 0.0, 0.0),
    )
    pipeline = MultibandPipeline.__new__(MultibandPipeline)
    pipeline.registration_band_idx = 0
    pipeline.control_idx = 0
    pipeline.common_bands = ["B14"]
    pipeline.smoke = False
    pipeline.config = SimpleNamespace(registration_params={"required_quality": "pass"})
    scene_data = {
        "arrays": [np.ones((1, 8, 8)), np.ones((1, 8, 8))],
        "transforms": [from_origin(0, 8, 1, 1)] * 2,
        "nodata_values": [None, None],
        "scene_ids": ["a", "b"],
    }

    result = pipeline.register_scenes(
        scene_data, [{"idx_i": 0, "idx_j": 1}],
    )

    assert result["connected"] is False
    assert result["quality"]["quality"] == "fail"
    assert result["registered_arrays"]
    assert result["pair_matches"] == []
    assert result["rejected_edges"]
    assert result["final_validation"]["overall"]["quality"] == "fail"


def test_register_scenes_returns_stable_failure_schema_when_graph_is_disconnected(monkeypatch):
    from src import coregistration
    from src.multiband_pipeline import MultibandPipeline

    match = {"shift_dx": 0.0, "shift_dy": 0.0, "confidence": 0.9,
             "ref_x": 2.0, "ref_y": 2.0, "tgt_x": 2.0, "tgt_y": 2.0}
    monkeypatch.setattr(
        coregistration, "collect_block_matches",
        lambda *args, **kwargs: ([match], {}),
    )
    monkeypatch.setattr(
        coregistration, "build_robust_pair_measurement",
        lambda *args, **kwargs: {
            "status": "pass", "shift_dx": 0.0, "shift_dy": 0.0,
            "confidence": 0.9, "n_blocks_inlier": 1,
            "n_blocks_total": 1, "rmse": 0.0, "p95": 0.0,
            "matches": [match], "screening": {},
        },
    )
    pipeline = MultibandPipeline.__new__(MultibandPipeline)
    pipeline.registration_band_idx = 0
    pipeline.control_idx = 0
    pipeline.common_bands = ["B14"]
    pipeline.smoke = False
    pipeline.config = SimpleNamespace(registration_params={"required_quality": "pass"})
    scene_data = {
        "arrays": [np.ones((1, 8, 8)) for _ in range(3)],
        "transforms": [from_origin(0, 8, 1, 1)] * 3,
        "nodata_values": [None] * 3,
        "scene_ids": ["a", "b", "c"],
    }

    result = pipeline.register_scenes(
        scene_data, [{"idx_i": 0, "idx_j": 1}],
    )

    assert result["connected"] is False
    assert result["unreachable_scenes"] == ["c"]
    assert result["quality"]["quality"] == "fail"
    assert result["spanning_tree"] == [(0, 1)]
    assert result["final_validation"]["overall"]["quality"] == "fail"


def test_smoke_registration_quality_is_recomputed_for_recropped_arrays(monkeypatch):
    from src import coregistration, multiband_pipeline
    from src.multiband_pipeline import MultibandPipeline

    params = {
        "required_quality": "pass",
        "final_min_blocks": 1,
        "validation_block_size": 2,
        "validation_step": 2,
        "validation_offset_row": 0,
        "validation_offset_col": 0,
        "validation_min_distance_from_training": 0,
        "validation_confidence_threshold": 0.5,
        "validation_max_residual_shift": 1.0,
        "pass_min_mean_confidence": 0.5,
        "pass_max_median": 0.35,
        "pass_max_rmse": 0.6,
        "pass_max_p95": 1.0,
        "warn_min_mean_confidence": 0.45,
        "warn_max_median": 0.5,
        "warn_max_rmse": 0.75,
        "warn_max_p95": 1.25,
    }
    config = SimpleNamespace(
        experiment_name="smoke-quality-test",
        scenes=[{"id": "a"}, {"id": "b"}],
        selected_bands=["B14"], registration_band="B14",
        registration_params=params, ablation_methods=[],
        mosaic_modes=[], feather_widths=[], enable_spectral_metrics=False,
        output_root="data/test-output",
    )
    full_arrays = [np.ones((1, 4, 4)), np.ones((1, 4, 4))]
    cropped_arrays = [np.ones((1, 2, 2)), np.ones((1, 2, 2))]
    transform = from_origin(0, 4, 1, 1)
    full_scene_data = {
        "arrays": full_arrays,
        "transforms": [transform, transform],
        "nodata_values": [None, None],
        "bounds": [(0, 0, 4, 4), (0, 0, 4, 4)],
        "crs": None, "scene_ids": ["a", "b"],
    }
    cropped_scene_data = {
        **full_scene_data,
        "arrays": cropped_arrays,
        "transforms": [from_origin(2, 2, 1, 1)] * 2,
        "bounds": [(2, 0, 4, 2), (2, 0, 4, 2)],
    }
    full_quality = {
        "quality": "pass", "rmse": 0.1, "p95": 0.2,
        "median": 0.1, "confidence": 0.9, "n_blocks": 5,
    }
    full_registration = {
        "registered_arrays": full_arrays,
        "global_shifts": np.zeros((2, 2)),
        "local_dx_fields": [np.zeros((4, 4)) for _ in range(2)],
        "local_dy_fields": [np.zeros((4, 4)) for _ in range(2)],
        "local_refinement": {"enabled": False, "used_for_scenes": [],
                              "fallback_scenes": [], "cv_results": {}},
        "pair_matches": [{
            "idx_i": 0,
            "idx_j": 1,
            "matches": [{
                "ref_x": 3.0, "ref_y": 3.0,
                "tgt_x": 3.5, "tgt_y": 3.5,
            }],
        }],
        "connected": True, "spanning_tree": [(0, 1)],
        "geometric_edges": [(0, 1)], "matching_edges": [(0, 1)],
        "rejected_edges": [], "connected_components": [[0, 1]],
        "unreachable_scenes": [], "quality": full_quality,
        "final_validation": {"edges": [], "overall": full_quality},
        "diagnostics": {},
    }
    pipeline = MultibandPipeline.__new__(MultibandPipeline)
    pipeline.config = config
    pipeline.dry_run = False
    pipeline.smoke = True
    pipeline.smoke_crop_size = 2
    pipeline.output_root = config.output_root
    pipeline.control_idx = 0
    pipeline.registration_band_idx = 0
    pipeline.common_bands = ["B14"]
    pipeline.load_scenes = lambda: full_scene_data
    overlap = [{"idx_i": 0, "idx_j": 1}]
    pipeline.detect_overlaps = lambda scene_data: overlap
    pipeline.register_scenes = lambda scene_data, overlaps: full_registration
    pipeline._smoke_recrop_by_spanning_tree = lambda scene_data, tree: cropped_scene_data
    normalization_calls = []
    pipeline.apply_radiometric_normalization = lambda *args, **kwargs: (
        normalization_calls.append(True) or {}
    )
    pipeline.compute_mosaics = lambda *args, **kwargs: {}
    pipeline.evaluate_metrics = lambda *args, **kwargs: {}
    pipeline.check_data_quality = lambda *args, **kwargs: {}
    monkeypatch.setattr(multiband_pipeline, "validate_config", lambda config: [])
    monkeypatch.setattr(multiband_pipeline, "validate_band_consistency", lambda *args, **kwargs: [])
    validation_calls = []

    def validate_cropped_only(arr_ref, tr_ref, arr_tgt, tr_tgt,
                              nodata_ref, nodata_tgt, training_points, **kwargs):
        validation_calls.append(
            (arr_ref.shape, arr_tgt.shape, np.asarray(training_points).copy())
        )
        return {"blocks": [], "stats": None, "coverage": {},
                "failure_reason": "cropped validation unavailable"}

    monkeypatch.setattr(
        coregistration, "validate_registration_independent_grid",
        validate_cropped_only,
    )

    result = pipeline.run()

    assert validation_calls[0][0:2] == ((2, 2), (2, 2))
    assert {tuple(point) for point in validation_calls[0][2]} == {(1.0, 1.0)}
    assert result["registration"]["quality"]["quality"] == "fail"
    assert normalization_calls == []
    assert result["pipeline_status"] == "failed"
