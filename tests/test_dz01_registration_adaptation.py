import numpy as np


def _dz01_like_controls():
    yy, xx = np.mgrid[0:5, 0:5]
    points = np.column_stack([xx.ravel() * 64.0, yy.ravel() * 64.0])
    residual_dx = -1.0 * np.ones(len(points))
    residual_dx += -5.0 * np.exp(-((points[:, 1] - 128.0) / 128.0) ** 2)
    residual_dy = np.zeros(len(points))
    confidence = np.full(len(points), 0.9)
    return {
        "points_xy": points,
        "residual_dx": residual_dx,
        "residual_dy": residual_dy,
        "confidence": confidence,
    }


def test_spatially_varying_dz01_like_shift_is_not_treated_as_global_outlier():
    from src.coregistration import filter_local_residual_controls

    result = filter_local_residual_controls(
        _dz01_like_controls(), confidence_threshold=0.6,
        mad_scale=3.0, hard_max_shift=8.0, neighbor_radius=5,
    )

    assert result["n_valid"] >= 20
    assert np.min(result["residual_dx"]) < -5.5


def test_random_large_phase_peak_is_rejected():
    from src.coregistration import filter_local_residual_controls

    controls = _dz01_like_controls()
    controls["residual_dx"][12] = 172.0
    result = filter_local_residual_controls(
        controls, confidence_threshold=0.6,
        mad_scale=3.0, hard_max_shift=8.0, neighbor_radius=5,
    )

    assert result["reject_reasons"][12] == "gross_residual"
    assert result["n_rejected_gross"] == 1


def test_dz01_b14_config_uses_256px_buffered_local_cv():
    from src.experiment_config import load_config

    config = load_config("configs/dz01_mosaic_series_b14.yaml")

    assert config.registration_params["local_cv_buffer_pixels"] == 256
