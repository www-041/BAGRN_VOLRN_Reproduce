import numpy as np


def test_local_rbf_improves_smooth_dz01_like_deformation():
    from src.multiband_pipeline import _fit_local_rbf_field

    yy, xx = np.mgrid[8:56:8j, 8:56:8j]
    points = np.column_stack([xx.ravel(), yy.ravel()])
    residual_dx = -4.0 - 4.0 * (points[:, 1] / 63.0)
    controls = {
        "points_xy": points,
        "residual_dx": residual_dx,
        "residual_dy": np.zeros(len(points)),
        "confidence": np.full(len(points), 0.9),
    }

    dx, dy, stats = _fit_local_rbf_field(
        controls, (64, 64),
        {"local_smoothing_candidates": [0.1], "local_hull_buffer": 8,
         "local_hard_max_component": 8.0},
    )

    assert np.isfinite(dx).all()
    assert np.isfinite(dy).all()
    assert np.max(np.abs(dx)) <= 8.0
    assert stats["fade"]["min"] == 0.0
    assert np.max(np.abs(dx[24:40, 24:40])) > 3.0


def test_local_field_fades_outside_control_support():
    from src.multiband_pipeline import _fit_local_rbf_field

    points = np.array([[20.0, 20.0], [44.0, 20.0], [20.0, 44.0], [44.0, 44.0]])
    controls = {
        "points_xy": points,
        "residual_dx": np.full(4, -6.0),
        "residual_dy": np.zeros(4),
    }

    dx, _, _ = _fit_local_rbf_field(
        controls, (64, 64),
        {"local_smoothing_candidates": [0.1], "local_hull_buffer": 0,
         "local_hard_max_component": 8.0},
    )

    assert dx[0, 0] == 0.0
    assert dx[32, 32] < -3.0
