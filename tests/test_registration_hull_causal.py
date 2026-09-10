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
