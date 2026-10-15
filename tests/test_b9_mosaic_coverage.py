import numpy as np

from src.multiscene_sift.mosaic_diagnostics import compute_contributor_metrics


def test_contributor_count_audit_uses_exact_valid_pixel_counts():
    valid = np.array([[1, 1, 0], [1, 1, 1]], dtype=bool)
    contributors = np.array([[1, 2, 0], [3, 1, 2]], dtype=np.uint8)
    weight_sum = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 4.0]], dtype=np.float32)

    result = compute_contributor_metrics(valid, contributors, weight_sum)

    assert result == {
        "valid_mosaic_pixels": 5,
        "nodata_pixels": 1,
        "one_contributor_pixels": 2,
        "two_contributor_pixels": 2,
        "three_plus_contributor_pixels": 1,
        "max_contributor_count": 3,
        "mean_contributor_count_valid": 1.8,
        "invalid_weight_sum_pixels": 1,
        "total_pixels": 6,
    }
