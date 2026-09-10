"""Regression tests for the diagnostic-only HULL-C1 RBF counterfactual."""

import numpy as np
import pytest
from rasterio.transform import from_origin


def _controls(points=None):
    points = np.asarray(points if points is not None else [
        [1.0, 1.0], [6.0, 1.0], [1.0, 6.0], [6.0, 6.0],
    ])
    return {
        "points_xy": points,
        "residual_dx": np.linspace(1.0, 4.0, len(points)),
        "residual_dy": np.linspace(-1.0, -4.0, len(points)),
        "confidence": np.ones(len(points)),
        "n_valid": len(points),
    }


def test_raw_rbf_field_has_no_support_or_clipping_side_effect(monkeypatch):
    from src import coregistration, multiband_pipeline

    calls = {"fit": 0}

    def fake_fit(*args, **kwargs):
        calls["fit"] += 1
        return (
            lambda xy: np.full(len(xy), 7.0),
            lambda xy: np.full(len(xy), -9.0),
            np.array([1.0, 1.0]),
            np.array([6.0, 6.0]),
        )

    monkeypatch.setattr(coregistration, "fit_local_rbf", fake_fit)
    monkeypatch.setattr(
        coregistration,
        "compute_hull_fade_support",
        lambda *args, **kwargs: pytest.fail("raw prediction must not compute support"),
    )

    dx, dy, stats = multiband_pipeline._predict_local_rbf_raw_field(
        _controls(), (4, 5), smoothing=0.1,
    )

    assert calls["fit"] == 1
    assert dx.shape == (4, 5)
    assert dy.shape == (4, 5)
    assert np.all(dx == 7.0)
    assert np.all(dy == -9.0)
    assert stats["smoothing"] == 0.1


def test_apply_weight_then_clip_matches_bf477_order():
    from src.multiband_pipeline import _apply_local_rbf_weight_and_clip

    raw_dx = np.array([[4.0, -4.0], [1.0, -1.0]])
    raw_dy = np.array([[2.0, -2.0], [3.0, -3.0]])
    weight = np.array([[0.5, 0.5], [2.0, 0.25]])
    params = {"local_hard_max_component": 1.5, "local_max_component": 99.0}

    dx, dy, stats = _apply_local_rbf_weight_and_clip(
        raw_dx, raw_dy, weight, params,
    )

    assert np.array_equal(dx, np.clip(raw_dx * weight, -1.5, 1.5))
    assert np.array_equal(dy, np.clip(raw_dy * weight, -1.5, 1.5))
    assert stats["clipping"] == {"dx": 3, "dy": 1, "total": 4}


def test_fit_local_rbf_field_legacy_wrapper_matches_expected_fade_and_clip(monkeypatch):
    from src import coregistration, multiband_pipeline

    raw_dx_value = 4.0
    raw_dy_value = -3.0

    def fake_fit(*args, **kwargs):
        return (
            lambda xy: np.full(len(xy), raw_dx_value),
            lambda xy: np.full(len(xy), raw_dy_value),
            np.array([1.0, 1.0]),
            np.array([6.0, 6.0]),
        )

    monkeypatch.setattr(coregistration, "fit_local_rbf", fake_fit)
    controls = _controls()
    params = {
        "local_hull_buffer": 2,
        "local_hard_max_component": 1.5,
    }
    dx, dy, stats = multiband_pipeline._fit_local_rbf_field(
        controls, (8, 8), params, smoothing=0.1,
    )
    fade = coregistration.compute_hull_fade_support(
        controls["points_xy"], 8, 8, buffer=2,
    )["fade_mask"]
    expected_dx = np.clip(raw_dx_value * fade, -1.5, 1.5)
    expected_dy = np.clip(raw_dy_value * fade, -1.5, 1.5)

    assert np.array_equal(dx, expected_dx)
    assert np.array_equal(dy, expected_dy)
    assert stats["fade"]["max"] == 1.0
    assert stats["clipping"]["total"] > 0


def test_fit_local_rbf_field_return_schema_preserves_existing_keys(monkeypatch):
    from src import coregistration, multiband_pipeline

    monkeypatch.setattr(
        coregistration,
        "fit_local_rbf",
        lambda *args, **kwargs: (
            lambda xy: np.ones(len(xy)),
            lambda xy: np.ones(len(xy)),
            np.array([1.0, 1.0]),
            np.array([6.0, 6.0]),
        ),
    )
    _, _, stats = multiband_pipeline._fit_local_rbf_field(
        _controls(), (8, 8), {"local_hull_buffer": 2}, smoothing=0.1,
    )

    assert {
        "smoothing", "clipping", "fade", "local_dx", "local_dy",
        "max_dx", "max_dy", "p95_magnitude", "max_spatial_gradient",
    } <= set(stats)


def test_hull_causal_variants_fit_raw_rbf_exactly_once(monkeypatch):
    from src import multiband_pipeline

    calls = {"raw": 0}

    def fake_raw(*args, **kwargs):
        calls["raw"] += 1
        return (
            np.full((8, 8), 4.0), np.full((8, 8), -4.0),
            {"smoothing": kwargs["smoothing"]},
        )

    monkeypatch.setattr(multiband_pipeline, "_predict_local_rbf_raw_field", fake_raw)
    variants = multiband_pipeline._fit_hull_causal_rbf_variants(
        _controls(), (8, 8), {"local_hull_buffer": 2}, smoothing=0.1,
    )

    assert calls["raw"] == 1
    assert variants["integrity"]["same_raw_field"] is True
    assert variants["raw_dx"].shape == (8, 8)


def test_strict_weight_is_exactly_zero_outside_hull(monkeypatch):
    from src import multiband_pipeline

    monkeypatch.setattr(
        multiband_pipeline,
        "_predict_local_rbf_raw_field",
        lambda *args, **kwargs: (
            np.ones((8, 8)), np.ones((8, 8)), {"smoothing": 0.1}
        ),
    )
    variants = multiband_pipeline._fit_hull_causal_rbf_variants(
        _controls(), (8, 8), {"local_hull_buffer": 2}, smoothing=0.1,
    )
    outside = ~variants["support"]["inside_hull_mask"]

    assert np.all(variants["strict_weight"][outside] == 0.0)
    assert np.count_nonzero(variants["strict_dx"][outside]) == 0
    assert variants["integrity"]["strict_outside_nonzero_count"] == 0


def test_legacy_and_strict_weights_are_identical_inside_hull(monkeypatch):
    from src import multiband_pipeline

    monkeypatch.setattr(
        multiband_pipeline,
        "_predict_local_rbf_raw_field",
        lambda *args, **kwargs: (
            np.ones((8, 8)), np.ones((8, 8)), {"smoothing": 0.1}
        ),
    )
    variants = multiband_pipeline._fit_hull_causal_rbf_variants(
        _controls(), (8, 8), {"local_hull_buffer": 2}, smoothing=0.1,
    )
    inside = variants["support"]["inside_hull_mask"]

    assert np.array_equal(
        variants["legacy_weight"][inside], variants["strict_weight"][inside]
    )
    assert variants["integrity"]["legacy_strict_equal_inside_hull"] is True


def test_legacy_branch_keeps_buffered_hull_fade_outside(monkeypatch):
    from src import multiband_pipeline

    monkeypatch.setattr(
        multiband_pipeline,
        "_predict_local_rbf_raw_field",
        lambda *args, **kwargs: (
            np.full((8, 8), 2.0), np.full((8, 8), -2.0), {"smoothing": 0.1}
        ),
    )
    variants = multiband_pipeline._fit_hull_causal_rbf_variants(
        _controls(), (8, 8), {"local_hull_buffer": 3}, smoothing=0.1,
    )
    outside = ~variants["support"]["inside_hull_mask"]

    assert np.any(variants["legacy_weight"][outside] > 0.0)
    assert np.all(variants["strict_weight"][outside] == 0.0)
    assert np.any(np.abs(variants["legacy_dx"][outside]) > 0.0)


def test_both_branches_use_identical_component_cap(monkeypatch):
    from src import multiband_pipeline

    monkeypatch.setattr(
        multiband_pipeline,
        "_predict_local_rbf_raw_field",
        lambda *args, **kwargs: (
            np.full((8, 8), 9.0), np.full((8, 8), -9.0), {"smoothing": 0.1}
        ),
    )
    variants = multiband_pipeline._fit_hull_causal_rbf_variants(
        _controls(), (8, 8),
        {"local_hull_buffer": 2, "local_hard_max_component": 1.25},
        smoothing=0.1,
    )

    assert np.max(np.abs(variants["legacy_dx"])) <= 1.25
    assert np.max(np.abs(variants["legacy_dy"])) <= 1.25
    assert np.max(np.abs(variants["strict_dx"])) <= 1.25
    assert np.max(np.abs(variants["strict_dy"])) <= 1.25
    assert variants["legacy_stats"]["clipping"] == variants["strict_stats"]["clipping"]


def _causal_validation(inside_center=False):
    from src.coregistration import compute_hull_fade_support
    points = np.asarray([[2.0, 2.0], [9.0, 2.0], [2.0, 9.0], [9.0, 9.0]])
    support = compute_hull_fade_support(points, 12, 12, buffer=2)
    return {
        "validation_block_size_selected": 4,
        "edges": [{
            "idx_i": 0,
            "idx_j": 1,
            "blocks": [{
                "validation_row": 1 if inside_center else 0,
                "validation_col": 1 if inside_center else 0,
                "center_x": 5.0 if inside_center else 1.0,
                "center_y": 5.0 if inside_center else 1.0,
                "accepted": True,
                "residual_magnitude": 0.5,
            }],
        }],
        "_variants": {
            "support": support,
            "raw_dx": np.ones((12, 12)),
            "raw_dy": np.ones((12, 12)),
            "legacy_weight": np.asarray(support["fade_mask"]),
            "strict_weight": np.asarray(support["inside_hull_mask"], dtype=float),
            "legacy_dx": np.asarray(support["fade_mask"]),
            "legacy_dy": np.asarray(support["fade_mask"]),
            "strict_dx": np.asarray(support["inside_hull_mask"], dtype=float),
            "strict_dy": np.asarray(support["inside_hull_mask"], dtype=float),
        },
    }


def test_window_stats_uses_full_window_not_only_center():
    from rasterio.transform import from_origin
    from src import multiband_pipeline

    validation = _causal_validation(inside_center=True)
    variants = validation.pop("_variants")
    stats = multiband_pipeline._hull_causal_window_stats(
        validation, [from_origin(0, 12, 1, 1), from_origin(0, 12, 1, 1)],
        {1: variants}, {},
    )

    row = stats["blocks"][0]
    assert row["window_valid_sample_count"] == 16
    assert 0.0 < row["window_inside_hull_fraction"] < 1.0
    assert row["window_support_difference_fraction"] > 0.0


def test_window_stats_detects_partial_hull_overlap():
    from rasterio.transform import from_origin
    from src import multiband_pipeline

    validation = _causal_validation(inside_center=False)
    variants = validation.pop("_variants")
    stats = multiband_pipeline._hull_causal_window_stats(
        validation, [from_origin(0, 12, 1, 1), from_origin(0, 12, 1, 1)],
        {1: variants}, {},
    )

    row = stats["blocks"][0]
    assert 0.0 < row["window_inside_hull_fraction"] < 1.0
    assert row["center_inside_hull"] is False


def test_window_stats_marks_negative_control_candidate_when_supports_equal():
    from rasterio.transform import from_origin
    from src import multiband_pipeline

    validation = _causal_validation(inside_center=True)
    variants = validation.pop("_variants")
    variants["legacy_weight"] = variants["strict_weight"] = np.ones((12, 12))
    variants["legacy_dx"] = variants["strict_dx"] = np.ones((12, 12))
    variants["legacy_dy"] = variants["strict_dy"] = np.ones((12, 12))
    variants["raw_dx"] = variants["raw_dy"] = np.ones((12, 12))
    stats = multiband_pipeline._hull_causal_window_stats(
        validation, [from_origin(0, 12, 1, 1), from_origin(0, 12, 1, 1)],
        {1: variants}, {},
    )

    assert stats["blocks"][0]["negative_control_candidate"] is True


def test_window_stats_never_serializes_pixel_arrays():
    from rasterio.transform import from_origin
    from src import multiband_pipeline

    validation = _causal_validation(inside_center=True)
    variants = validation.pop("_variants")
    stats = multiband_pipeline._hull_causal_window_stats(
        validation, [from_origin(0, 12, 1, 1), from_origin(0, 12, 1, 1)],
        {1: variants}, {},
    )

    assert not any(isinstance(value, np.ndarray)
                   for value in stats["blocks"][0].values())


def _run_minimal_causal_register(monkeypatch, *, enabled=True,
                                 use_holdout=False, strict_quality=None):
    from types import SimpleNamespace
    from src import coregistration, multiband_pipeline
    from src.multiband_pipeline import MultibandPipeline

    params = {
        "enable_local_refinement": True,
        "enable_spatial_holdout": use_holdout,
        "local_min_controls": 1,
        "local_min_spatial_groups": 1,
        "local_cv_min_rmse_improvement": 0.01,
        "local_cv_min_p95_improvement": 0.01,
        "local_max_controls": 10,
        "local_block_size": 4,
        "local_confidence_threshold": 0.6,
        "local_hull_buffer": 2,
        "local_hard_max_component": 8.0,
        "validation_block_size": 4,
        "validation_step": 4,
        "validation_offset_row": 0,
        "validation_offset_col": 0,
        "validation_min_distance_from_training": 0,
        "validation_confidence_threshold": 0.6,
        "validation_max_residual_shift": 1.0,
        "final_min_blocks": 5,
        "required_quality": "pass",
    }
    match = {
        "ref_x": 2.0, "ref_y": 2.0, "tgt_x": 2.0, "tgt_y": 2.0,
        "shift_dx": 0.0, "shift_dy": 0.0, "confidence": 0.9,
    }
    controls = _controls()
    monkeypatch.setattr(
        coregistration, "collect_block_matches",
        lambda *args, **kwargs: ([match], {}),
    )
    monkeypatch.setattr(
        coregistration, "build_robust_pair_measurement",
        lambda *args, **kwargs: {
            "status": "pass", "shift_dx": 0.0, "shift_dy": 0.0,
            "confidence": 0.9, "n_blocks_inlier": 5,
            "n_blocks_total": 5, "rmse": 0.0, "p95": 0.0,
            "matches": [match], "screening": {},
        },
    )
    monkeypatch.setattr(
        coregistration, "multi_image_network_adjustment",
        lambda *args, **kwargs: {
            "global_shifts": np.zeros((2, 2)), "loop_errors": [],
        },
    )
    monkeypatch.setattr(
        coregistration, "refine_global_residual_shifts_from_original",
        lambda *args, **kwargs: {
            "global_shifts": np.zeros((2, 2)),
            "history": [], "warnings": [],
        },
    )
    monkeypatch.setattr(
        coregistration, "rematch_pair_on_registered",
        lambda *args, **kwargs: {
            "available": True, "shift_dx": 0.0, "shift_dy": 0.0,
            "confidence": 0.9, "n_blocks": 5, "rmse": 0.0,
            "p95": 0.0, "matches": [match],
        },
    )
    monkeypatch.setattr(
        coregistration, "build_parent_based_local_controls",
        lambda *args, **kwargs: controls,
    )
    monkeypatch.setattr(
        coregistration, "balance_edge_controls",
        lambda edge_controls, **kwargs: edge_controls,
    )
    cv_calls = {"count": 0}

    def fake_cv(*args, **kwargs):
        cv_calls["count"] += 1
        return {
            "available": True, "baseline_rmse": 1.0,
            "candidate_rmse": 0.5, "baseline_p95": 1.2,
            "candidate_p95": 0.8, "selected_smoothing": 0.1,
            "has_passing_candidate": True,
        }

    monkeypatch.setattr(multiband_pipeline, "_local_holdout_cv", fake_cv)
    variants = {
        "smoothing": 0.1,
        "raw_dx": np.full((8, 8), 3.0),
        "raw_dy": np.full((8, 8), -3.0),
        "legacy_weight": np.ones((8, 8)),
        "strict_weight": np.ones((8, 8)),
        "legacy_dx": np.ones((8, 8)),
        "legacy_dy": np.ones((8, 8)),
        "strict_dx": np.full((8, 8), 2.0),
        "strict_dy": np.full((8, 8), -2.0),
        "raw_stats": {}, "legacy_stats": {}, "strict_stats": {},
        "support": {
            "inside_hull_mask": np.ones((8, 8), dtype=bool),
            "fade_mask": np.ones((8, 8)),
            "distance_outside_hull": np.zeros((8, 8)),
        },
        "integrity": {
            "same_raw_field": True,
            "legacy_buffer_pixels": 2,
            "strict_outside_nonzero_count": 0,
            "legacy_strict_equal_inside_hull": True,
        },
    }
    monkeypatch.setattr(
        multiband_pipeline, "_fit_hull_causal_rbf_variants",
        lambda *args, **kwargs: variants,
    )
    monkeypatch.setattr(
        multiband_pipeline, "_fit_local_rbf_field",
        lambda *args, **kwargs: (
            np.zeros((8, 8)), np.zeros((8, 8)), {}
        ),
    )
    warp_calls = []

    def fake_warp(array, global_dx, global_dy, dx_field, dy_field, nodata):
        warp_calls.append({"array": array.copy(), "dx": np.asarray(dx_field).copy()})
        return array.astype(float) + float(np.mean(dx_field))

    monkeypatch.setattr(coregistration, "warp_multiband_with_displacement_field", fake_warp)
    validation_calls = []

    def fake_validation(*args, **kwargs):
        validation_calls.append(kwargs.get("holdout_contexts"))
        quality = {"quality": "pass", "median": 0.1, "rmse": 0.2,
                   "p95": 0.3, "confidence": 0.9, "n_blocks": 5}
        if strict_quality is not None and len(validation_calls) == 3:
            quality = strict_quality
        return quality, {"edges": [], "overall": quality}

    monkeypatch.setattr(multiband_pipeline, "_validate_final_registration_arrays", fake_validation)
    monkeypatch.setattr(
        multiband_pipeline, "_sample_local_field_at_validation_blocks",
        lambda *args, **kwargs: {},
    )
    if use_holdout:
        context = {
            "available": True,
            "patch_window_ref": (0, 8, 0, 8),
            "holdout_region_mask": np.ones((8, 8), dtype=bool),
            "holdout_exclusion_mask": np.zeros((8, 8), dtype=bool),
            "reserved_count": 5,
            "validation_reservation": {
                "reserved_windows": [(0, 0, 4, 4)] * 5,
            },
        }
        monkeypatch.setattr(
            multiband_pipeline, "_build_pair_holdout_context",
            lambda *args, **kwargs: context,
        )
    pipeline = MultibandPipeline.__new__(MultibandPipeline)
    pipeline.common_bands = ["B14"]
    pipeline.registration_band_idx = 0
    pipeline.control_idx = 0
    pipeline.smoke = False
    pipeline.config = SimpleNamespace(registration_params=params)
    arrays = [np.ones((1, 8, 8)), np.ones((1, 8, 8)) * 2.0]
    scene_data = {
        "arrays": arrays,
        "transforms": [from_origin(0, 8, 1, 1), from_origin(0, 8, 1, 1)],
        "nodata_values": [None, None],
    }
    result = pipeline.register_scenes(
        scene_data, [{"idx_i": 0, "idx_j": 1}],
        hull_causal_diagnostic=enabled,
    )
    return result, cv_calls, warp_calls, validation_calls


def test_hull_causal_flag_defaults_off_and_preserves_normal_register_schema(monkeypatch):
    import inspect
    from src.multiband_pipeline import MultibandPipeline

    parameter = inspect.signature(MultibandPipeline.register_scenes).parameters[
        "hull_causal_diagnostic"
    ]
    assert parameter.default is False
    result, _, _, _ = _run_minimal_causal_register(monkeypatch, enabled=False)
    assert "hull_causal" not in result


def test_hull_causal_path_uses_legacy_field_for_production_final(monkeypatch):
    result, _, warp_calls, _ = _run_minimal_causal_register(monkeypatch)

    production_warp = [call for call in warp_calls if np.mean(call["dx"]) == 1.0]
    assert production_warp
    assert np.array_equal(result["registered_arrays"][1], np.full((1, 8, 8), 3.0))
    assert result["hull_causal"]["available"] is True


def test_strict_counterfactual_warps_from_original_not_legacy(monkeypatch):
    result, _, warp_calls, _ = _run_minimal_causal_register(monkeypatch)

    strict_warp = [call for call in warp_calls if np.mean(call["dx"]) == 2.0]
    assert strict_warp
    assert np.array_equal(strict_warp[-1]["array"], np.full((1, 8, 8), 2.0))
    assert np.array_equal(result["strict_counterfactual_arrays"][1],
                          np.full((1, 8, 8), 4.0))


def test_all_three_validations_receive_same_reserved_holdout_context(monkeypatch):
    result, _, _, validation_calls = _run_minimal_causal_register(
        monkeypatch, use_holdout=True,
    )

    assert result["hull_causal"]["available"] is True
    assert len(validation_calls) == 3
    assert validation_calls[0] is validation_calls[1] is validation_calls[2]


def test_hull_causal_does_not_call_local_cv_twice(monkeypatch):
    _, cv_calls, _, _ = _run_minimal_causal_register(monkeypatch)

    assert cv_calls["count"] == 1


def test_strict_holdout_never_changes_main_quality_or_status(monkeypatch):
    strict_quality = {
        "quality": "fail", "median": 9.0, "rmse": 9.0,
        "p95": 9.0, "confidence": 0.1, "n_blocks": 5,
    }
    result, _, _, _ = _run_minimal_causal_register(
        monkeypatch, strict_quality=strict_quality,
    )

    assert result["quality"]["quality"] == "pass"
    assert result["status"] == "pass"
    assert result["hull_causal"]["strict_counterfactual_quality"]["quality"] == "fail"
