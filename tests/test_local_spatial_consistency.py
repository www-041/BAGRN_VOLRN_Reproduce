import numpy as np


def _grid_controls(values):
    values = np.asarray(values, dtype=float)
    yy, xx = np.mgrid[:4, :4]
    return {
        "points_xy": np.column_stack([xx.ravel(), yy.ravel()]).astype(float),
        "residual_dx": values.reshape(-1),
        "residual_dy": np.zeros(values.size),
        "confidence": np.full(values.size, 0.9),
    }


def test_spatially_coherent_large_residuals_are_retained():
    from src.coregistration import score_spatial_residual_consistency

    result = score_spatial_residual_consistency(
        _grid_controls(np.full((4, 4), -6.0)), k_neighbors=4
    )

    assert np.all(result["scores"] > 0.5)


def test_isolated_large_residual_is_rejected():
    from src.coregistration import score_spatial_residual_consistency

    values = np.full((4, 4), -6.0)
    values[1, 1] = 7.0
    result = score_spatial_residual_consistency(
        _grid_controls(values), k_neighbors=4
    )

    assert result["scores"].reshape(4, 4)[1, 1] < 0.5


def test_consistency_filter_preserves_gradual_y_dependent_shift():
    from src.coregistration import filter_local_residual_controls

    values = np.repeat(np.array([-4.0, -5.5, -7.0, -8.0]), 4)
    result = filter_local_residual_controls(
        _grid_controls(values), confidence_threshold=0.6,
        mad_scale=3.0, hard_max_shift=8.0, neighbor_radius=2.0,
    )

    assert result["n_valid"] >= 14
