import numpy as np

from scripts import two_image_pipeline as tip


def test_summarize_residuals_uses_vector_magnitude():
    dx = np.array([3.0, 0.0])
    dy = np.array([4.0, 0.0])
    result = tip._summarize_residuals(dx, dy)

    assert result["count"] == 2
    assert result["mean_magnitude_pixels"] == 2.5
    assert result["median_magnitude_pixels"] == 2.5
    assert np.isclose(result["rmse_pixels"], np.sqrt((25.0 + 0.0) / 2.0))
    assert result["max_magnitude_pixels"] == 5.0
    assert "mae_pixels" not in result
    assert "mean_error_pixels" not in result


def test_summarize_residuals_empty_returns_none_metrics():
    result = tip._summarize_residuals(np.array([]), np.array([]))

    assert result["count"] == 0
    assert result["rmse_pixels"] is None
    assert result["p95_magnitude_pixels"] is None


def test_spatial_coverage_is_relative_to_overlap_not_full_image():
    matches = []
    for y in [125, 375, 625, 875]:
        for x in [562, 687, 812, 937]:
            matches.append({"ref_x": x, "ref_y": y})

    result = tip._spatial_coverage(
        matches,
        reference_shape=(1000, 1000),
        overlap_window=(0, 1000, 500, 1000),
        grid_rows=4,
        grid_cols=4,
    )

    assert result["coverage_domain"] == "overlap"
    assert result["grid_cells_covered"] >= 12
    assert result["grid_coverage_ratio"] >= 0.75
