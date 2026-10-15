"""RED tests for Task10D seam-zone metrics."""

import numpy as np
import pytest

from src.multiscene_sift.radiometric_metrics import compute_seam_zone_metrics
from src.registration_benchmark.mosaic_diagnostics import build_seam_zone_mask


def test_task10d_uses_the_exact_task9_seam_zone_rule():
    valid_a = np.ones((4, 4), dtype=bool)
    valid_b = np.ones((4, 4), dtype=bool)
    weight_a = np.array([[0.8, 0.6, 0.2, 0.5]] * 4)
    weight_b = np.array([[0.2, 0.4, 0.8, 0.5]] * 4)

    seam_mask = build_seam_zone_mask(
        valid_a, valid_b, weight_a, weight_b, balance_threshold=0.25
    )

    assert seam_mask.tolist() == [[False, True, False, True]] * 4


def test_identical_seam_images_have_zero_mae_rmse_and_rdd():
    image = np.arange(16, dtype=np.float64).reshape(4, 4)
    seam_mask = np.ones_like(image, dtype=bool)

    metrics = compute_seam_zone_metrics(image, image, seam_mask)

    assert metrics["seam_mae"] == 0.0
    assert metrics["seam_rmse"] == 0.0
    assert metrics["seam_rdd"] == 0.0


def test_known_seam_offset_is_reported_without_histogram_binning():
    left = np.arange(16, dtype=np.float64).reshape(4, 4)
    right = left + 2.5
    seam_mask = np.ones_like(left, dtype=bool)

    metrics = compute_seam_zone_metrics(left, right, seam_mask)

    assert metrics["seam_mae"] == pytest.approx(2.5)
    assert metrics["seam_rmse"] == pytest.approx(2.5)
    assert metrics["seam_rdd"] == pytest.approx(2.5)
    assert metrics["valid_pixels"] == 16
