"""TDD coverage for the diagnostic affine residual model."""

import numpy as np
import pytest

from src.experiment_config import _dict_to_config, validate_config


def _controls(points, matrix):
    points = np.asarray(points, dtype=float)
    target = np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix).T
    return {
        "points_xy": points,
        "residual_dx": target[:, 0] - points[:, 0],
        "residual_dy": target[:, 1] - points[:, 1],
        "confidence": np.ones(len(points)),
        "n_valid": len(points),
    }


def _params(**overrides):
    params = {
        "affine_min_controls": 6,
        "affine_min_spatial_groups": 3,
        "affine_min_inliers": 6,
        "affine_min_inlier_ratio": 0.8,
        "affine_ransac_residual_threshold": 0.2,
        "affine_ransac_max_trials": 200,
        "affine_ransac_seed": 42,
        "affine_max_scale_delta": 0.02,
        "affine_max_rotation_deg": 1.0,
        "affine_max_shear_deg": 1.0,
        "affine_max_component": 8.0,
    }
    params.update(overrides)
    return params


def test_affine_residual_plane_recovery_uses_target_pixel_convention():
    from src.coregistration import (
        fit_affine_residual_model,
        predict_affine_residual,
    )

    points = np.asarray([
        [0, 0], [10, 0], [20, 0], [0, 10], [10, 10], [20, 10],
        [0, 20], [10, 20], [20, 20],
    ], dtype=float)
    matrix = [[1.005, 0.004, 0.7], [-0.003, 0.998, -0.4], [0, 0, 1]]
    controls = _controls(points, matrix)

    result = fit_affine_residual_model(controls, _params())

    assert result["accepted"] is True
    probe = np.asarray([[5.0, 7.0], [15.0, 18.0]])
    expected = np.column_stack([probe, np.ones(len(probe))]) @ np.asarray(matrix).T
    assert np.allclose(
        predict_affine_residual(result["model"], probe),
        expected[:, :2] - probe,
        atol=1e-6,
    )


def test_affine_ransac_seed_is_deterministic():
    from src.coregistration import fit_affine_residual_model

    points = np.asarray([
        [0, 0], [10, 0], [20, 0], [0, 10], [10, 10], [20, 10],
        [0, 20], [10, 20], [20, 20], [40, 40],
    ], dtype=float)
    controls = _controls(points, [[1.002, 0.001, 0.3], [0.0, 0.999, -0.2], [0, 0, 1]])
    controls["residual_dx"][-1] += 5.0

    first = fit_affine_residual_model(controls, _params(affine_min_inlier_ratio=0.7))
    second = fit_affine_residual_model(controls, _params(affine_min_inlier_ratio=0.7))

    assert first["accepted"] is True
    assert second["accepted"] is True
    assert np.array_equal(first["inlier_mask"], second["inlier_mask"])
    assert np.allclose(first["model"].params, second["model"].params)


def test_affine_residual_model_fails_closed_for_too_few_and_collinear_controls():
    from src.coregistration import fit_affine_residual_model

    few = _controls([[0, 0], [1, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    collinear = _controls(
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )

    assert fit_affine_residual_model(few, _params())["accepted"] is False
    result = fit_affine_residual_model(collinear, _params())
    assert result["accepted"] is False
    assert "rank" in result["reason"] or "collinear" in result["reason"]


def test_affine_model_rejects_unsafe_transform_limits():
    from src.coregistration import fit_affine_residual_model

    points = np.asarray([
        [0, 0], [10, 0], [20, 0], [0, 10], [10, 10], [20, 10],
        [0, 20], [10, 20], [20, 20],
    ], dtype=float)
    controls = _controls(points, [[1.04, 0, 0], [0, 1.04, 0], [0, 0, 1]])
    result = fit_affine_residual_model(controls, _params())
    assert result["accepted"] is False
    assert "scale" in result["reason"]


def test_affine_protocol_defaults_are_schema_backed():
    cfg = _dict_to_config({
        "experiment_name": "affine-test",
        "selected_bands": ["B14"],
        "registration_band": "B14",
        "scenes": [{"id": "a", "bands": {"B14": "a.tif"}},
                   {"id": "b", "bands": {"B14": "b.tif"}}],
    })
    assert cfg.registration_params["affine_min_controls"] == 12
    assert cfg.registration_params["affine_ransac_seed"] == 42
    assert validate_config(cfg, skip_file_check=True) == []


@pytest.mark.parametrize("key,value", [
    ("affine_min_controls", 2),
    ("affine_min_inlier_ratio", 0.0),
    ("affine_ransac_residual_threshold", 0.0),
    ("affine_cv_min_p95_improvement", -0.1),
])
def test_affine_protocol_invalid_ranges_are_reported(key, value):
    cfg = _dict_to_config({
        "experiment_name": "affine-test",
        "selected_bands": ["B14"],
        "registration_band": "B14",
        "scenes": [{"id": "a", "bands": {"B14": "a.tif"}},
                   {"id": "b", "bands": {"B14": "b.tif"}}],
        "registration_params": {key: value},
    })
    assert any(key in error for error in validate_config(cfg, skip_file_check=True))
