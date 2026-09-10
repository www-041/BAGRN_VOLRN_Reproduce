"""Regression tests for the diagnostic-only HULL-C1 RBF counterfactual."""

import numpy as np
import pytest


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
