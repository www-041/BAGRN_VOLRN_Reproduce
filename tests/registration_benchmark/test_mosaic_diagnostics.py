import numpy as np
import pytest

from src.registration_benchmark.mosaic_diagnostics import compute_overlap_metrics, write_overlap_metrics_csv


def _textured_image(size=64):
    yy, xx = np.mgrid[:size, :size]
    return (xx * 0.7 + yy * 1.3 + 8.0 * np.sin(xx / 4.0) + 5.0 * np.cos(yy / 7.0)).astype(np.float32)


def test_shifted_image_has_lower_structural_scores_than_aligned_image():
    image = _textured_image()
    shifted = np.roll(image, shift=(2, 1), axis=(0, 1))
    valid = np.zeros_like(image, dtype=bool)
    valid[2:-2, 2:-2] = True

    aligned = compute_overlap_metrics(image, image, valid)
    displaced = compute_overlap_metrics(image, shifted, valid)

    assert aligned["status"] == "PASS"
    assert aligned["ZNCC_intensity"] > displaced["ZNCC_intensity"]
    assert aligned["gradient_magnitude_NCC"] > displaced["gradient_magnitude_NCC"]
    assert aligned["gradient_orientation_cosine"] > displaced["gradient_orientation_cosine"]


def test_constant_brightness_offset_preserves_zncc_but_changes_auxiliary_mae():
    image = _textured_image()
    offset = image + 25.0
    valid = np.ones_like(image, dtype=bool)

    result = compute_overlap_metrics(image, offset, valid)

    assert result["ZNCC_intensity"] > 0.999
    assert result["intensity_MAE"] == pytest.approx(25.0)
    assert result["auxiliary_metric_label"] == "RADIOMETRY_SENSITIVE"


def test_overlap_metrics_use_only_shared_valid_pixels():
    image_a = np.ones((4, 4), dtype=np.float32)
    image_b = image_a.copy()
    image_b[~np.eye(4, dtype=bool)] = 99.0
    valid = np.eye(4, dtype=bool)

    result = compute_overlap_metrics(image_a, image_b, valid, min_valid_pixels=4)

    assert result["valid_overlap_pixels"] == 4
    assert result["intensity_MAE"] == 0.0


def test_overlap_metrics_ignore_nonfinite_gradient_values_outside_valid_overlap():
    image_a = _textured_image(16)
    image_b = image_a.copy()
    image_a[0, :] = np.nan
    image_b[0, :] = np.nan
    valid = np.zeros_like(image_a, dtype=bool)
    valid[1:, :] = True

    result = compute_overlap_metrics(image_a, image_b, valid, min_valid_pixels=10)

    assert result["status"] == "PASS"
    assert np.isfinite(result["gradient_magnitude_NCC"])
    assert np.isfinite(result["gradient_orientation_cosine"])


def test_overlap_metrics_report_insufficient_valid_overlap_without_scores():
    result = compute_overlap_metrics(
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=bool),
        min_valid_pixels=5,
    )

    assert result["status"] == "INSUFFICIENT_VALID_OVERLAP"
    assert result["valid_overlap_pixels"] == 4
    assert result["ZNCC_intensity"] is None
    assert result["intensity_MAE"] is None


def test_overlap_csv_labels_auxiliary_radiometry_metrics(tmp_path):
    path = tmp_path / "metrics.csv"
    write_overlap_metrics_csv(
        [{"matcher": "sift", "global_method": "MST", "pair": "0-1", "auxiliary_metric_label": "RADIOMETRY_SENSITIVE"}],
        path,
    )

    assert "RADIOMETRY_SENSITIVE" in path.read_text(encoding="utf-8")
