"""Fixed-fold CV tests for the MODEL-C1 affine residual candidate."""

import numpy as np


def _controls():
    points = np.asarray([
        [x, y] for y in (0, 100, 200, 300)
        for x in (0, 100, 200, 300)
    ], dtype=float)
    dx = 0.004 * points[:, 0] + 0.001 * points[:, 1] + 0.25
    dy = -0.001 * points[:, 0] + 0.003 * points[:, 1] - 0.15
    return {
        "points_xy": points,
        "residual_dx": dx,
        "residual_dy": dy,
        "n_valid": len(points),
        "confidence": np.ones(len(points)),
    }


def _params(**overrides):
    params = {
        "local_min_controls": 12,
        "local_cv_min_folds": 3,
        "local_cv_min_validation_controls": 3,
        "local_cv_buffer_pixels": 0,
        "affine_min_controls": 8,
        "affine_min_spatial_groups": 3,
        "affine_min_inliers": 8,
        "affine_min_inlier_ratio": 0.8,
        "affine_ransac_residual_threshold": 0.2,
        "affine_ransac_max_trials": 200,
        "affine_ransac_seed": 42,
        "affine_max_scale_delta": 0.02,
        "affine_max_rotation_deg": 1.0,
        "affine_max_shear_deg": 1.0,
        "affine_max_component": 8.0,
        "affine_cv_min_rmse_improvement": 0.10,
        "affine_cv_min_p95_improvement": 0.15,
    }
    params.update(overrides)
    return params


def test_affine_candidate_cv_uses_fixed_spatial_folds_and_both_gates():
    from src.multiband_pipeline import (
        _build_local_cv_fold_plan,
        _evaluate_affine_residual_candidate,
    )

    controls = _controls()
    plan = _build_local_cv_fold_plan(controls["points_xy"], _params())
    result = _evaluate_affine_residual_candidate(controls, plan, _params())

    assert result["available"] is True
    assert result["passes_rmse_gate"] is True
    assert result["passes_p95_gate"] is True
    assert result["passes_gate"] is True
    assert result["candidate_rmse"] < result["baseline_rmse"]
    assert result["candidate_p95"] < result["baseline_p95"]
    assert result["n_validation_controls"] == len(plan["validation_indices"])


def test_affine_candidate_cv_fails_closed_when_any_fold_cannot_fit():
    from src.multiband_pipeline import (
        _build_local_cv_fold_plan,
        _evaluate_affine_residual_candidate,
    )

    controls = _controls()
    controls["residual_dx"] = controls["residual_dx"].copy()
    controls["residual_dx"][0] = np.nan
    params = _params(affine_min_controls=20)
    plan = _build_local_cv_fold_plan(controls["points_xy"], params)
    result = _evaluate_affine_residual_candidate(controls, plan, params)

    assert result["available"] is False
    assert result["passes_gate"] is False
    assert result["failure_reason"]


def test_affine_causal_candidate_returns_safe_field_only_after_all_gates():
    from src.multiband_pipeline import _run_affine_causal_candidate

    result = _run_affine_causal_candidate(_controls(), (80, 80), _params())

    assert result["accepted"] is True
    assert result["dx_field"].shape == (80, 80)
    assert result["dy_field"].shape == (80, 80)
    assert result["field_stats"]["safe"] is True


def test_affine_field_is_not_clipped_when_extrapolation_exceeds_cap():
    from skimage.transform import AffineTransform
    from src.coregistration import build_affine_residual_field

    dx, dy, stats = build_affine_residual_field(
        AffineTransform(matrix=[[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0, 0, 1]]),
        (3, 4), max_component=8.0,
    )

    assert np.all(dx == 10.0)
    assert np.all(dy == 0.0)
    assert stats["safe"] is False
