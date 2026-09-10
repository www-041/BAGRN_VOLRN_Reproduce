"""Synthetic diagonal-overlap integration coverage for MODEL-C1 contracts."""

import numpy as np


def _affine_params():
    return {
        "global_block_size": 512,
        "global_refine_block_size": 384,
        "local_block_size": 256,
        "robust_min_inliers": 5,
        "local_min_controls": 12,
        "final_min_blocks": 5,
        "validation_block_size_candidates": [384, 256, 192],
        "validation_reservation_step": 64,
        "validation_min_common_valid_ratio": 0.30,
        "validation_reservation_margin": 2,
        "affine_min_controls": 12,
        "affine_min_spatial_groups": 3,
        "affine_min_inliers": 8,
        "affine_min_inlier_ratio": 0.50,
        "affine_ransac_residual_threshold": 0.75,
        "affine_ransac_max_trials": 2000,
        "affine_ransac_seed": 42,
        "affine_max_scale_delta": 0.02,
        "affine_max_rotation_deg": 1.0,
        "affine_max_shear_deg": 1.0,
        "affine_max_component": 8.0,
    }


def test_model_c1_synthetic_diagonal_overlap_preserves_validation_and_training_contract():
    from src.coregistration import (
        derive_training_window_requirements,
        reserve_validation_windows,
    )

    yy, xx = np.mgrid[:2644, :1104]
    common = np.abs(xx - (0.20 * yy + 100.0)) <= 180.0
    params = _affine_params()
    requirements = derive_training_window_requirements(params)
    reservation = reserve_validation_windows(
        common,
        params["validation_block_size_candidates"],
        params["final_min_blocks"],
        step=params["validation_reservation_step"],
        min_common_valid_ratio=params["validation_min_common_valid_ratio"],
        reservation_margin=params["validation_reservation_margin"],
        training_block_sizes=list(requirements),
        min_training_windows=requirements,
    )

    assert reservation["available"] is True
    assert reservation["reserved_count"] >= 5
    assert reservation["training_feasibility"][512]["available_windows"] >= 5
    assert reservation["training_feasibility"][384]["available_windows"] >= 5
    assert reservation["training_feasibility"][256]["available_windows"] >= 12
    assert len(reservation["candidate_cells"]) == (
        reservation["reserved_count"] + len(reservation["unused_cells"])
    )


def test_model_c1_affine_candidate_is_one_pass_candidate_without_rbf_component():
    from src.multiband_pipeline import _run_affine_causal_candidate

    points = np.asarray([
        [x, y] for y in (0, 100, 200, 300)
        for x in (0, 100, 200, 300)
    ], dtype=float)
    controls = {
        "points_xy": points,
        "residual_dx": 0.002 * points[:, 0] + 0.001 * points[:, 1],
        "residual_dy": -0.001 * points[:, 0] + 0.002 * points[:, 1],
        "confidence": np.ones(len(points)),
        "n_valid": len(points),
    }
    result = _run_affine_causal_candidate(controls, (64, 64), _affine_params())

    assert result["accepted"] is True
    assert np.count_nonzero(result["dx_field"]) > 0
    assert np.count_nonzero(result["dy_field"]) > 0
