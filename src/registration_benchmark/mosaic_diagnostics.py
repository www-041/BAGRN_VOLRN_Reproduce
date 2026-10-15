"""Structural metrics for separately warped overlap imagery."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


_PRIMARY_FIELDS = (
    "ZNCC_intensity",
    "gradient_magnitude_NCC",
    "gradient_orientation_cosine",
)
_AUXILIARY_FIELDS = ("intensity_MAE", "intensity_RMSE", "mean_bias")


def _zncc(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64, copy=False)
    right = right.astype(np.float64, copy=False)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-12:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def compute_overlap_metrics(
    image_a: np.ndarray,
    image_b: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_valid_pixels: int = 32,
) -> dict:
    """Compute geometry-sensitive and explicitly radiometry-sensitive overlap metrics."""
    image_a = np.asarray(image_a, dtype=np.float32)
    image_b = np.asarray(image_b, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if image_a.shape != image_b.shape or image_a.shape != valid_mask.shape:
        raise ValueError("overlap metric arrays must share one shape")
    shared = valid_mask & np.isfinite(image_a) & np.isfinite(image_b)
    count = int(shared.sum())
    base = {
        "status": "PASS" if count >= min_valid_pixels else "INSUFFICIENT_VALID_OVERLAP",
        "valid_overlap_pixels": count,
        "auxiliary_metric_label": "RADIOMETRY_SENSITIVE",
    }
    if count < min_valid_pixels:
        return {**base, **{field: None for field in _PRIMARY_FIELDS + _AUXILIARY_FIELDS}}

    a = image_a[shared].astype(np.float64)
    b = image_b[shared].astype(np.float64)
    grad_a_y, grad_a_x = np.gradient(image_a.astype(np.float64))
    grad_b_y, grad_b_x = np.gradient(image_b.astype(np.float64))
    full_mag_a = np.hypot(grad_a_x, grad_a_y)
    full_mag_b = np.hypot(grad_b_x, grad_b_y)
    gradient_shared = shared & np.isfinite(full_mag_a) & np.isfinite(full_mag_b)
    mag_a = full_mag_a[gradient_shared]
    mag_b = full_mag_b[gradient_shared]
    if mag_a.size == 0:
        gradient_ncc = None
        orientation_value = None
    else:
        denominator = np.maximum(mag_a * mag_b, 1e-12)
        informative = (mag_a > 1e-12) | (mag_b > 1e-12)
        gradient_ncc = _zncc(mag_a, mag_b)
        orientation_terms = (
            grad_a_x[gradient_shared] * grad_b_x[gradient_shared]
            + grad_a_y[gradient_shared] * grad_b_y[gradient_shared]
        ) / denominator
        orientation_value = float(np.mean(orientation_terms[informative])) if informative.any() else None
    return {
        **base,
        "ZNCC_intensity": _zncc(a, b),
        "gradient_magnitude_NCC": gradient_ncc,
        "gradient_orientation_cosine": orientation_value,
        "intensity_MAE": float(np.mean(np.abs(a - b))),
        "intensity_RMSE": float(np.sqrt(np.mean((a - b) ** 2))),
        "mean_bias": float(np.mean(b - a)),
    }


def write_overlap_metrics_csv(rows: list[dict], path: str | Path) -> Path:
    if not rows:
        raise ValueError("cannot write an empty overlap metrics table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_seam_zone_mask(
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    *,
    balance_threshold: float = 0.25,
) -> np.ndarray:
    """Return the fixed pairwise feather-balance seam diagnostic zone."""
    valid_a = np.asarray(valid_a, dtype=bool)
    valid_b = np.asarray(valid_b, dtype=bool)
    weight_a = np.asarray(weight_a, dtype=np.float64)
    weight_b = np.asarray(weight_b, dtype=np.float64)
    if not (valid_a.shape == valid_b.shape == weight_a.shape == weight_b.shape):
        raise ValueError("seam-zone inputs must share one shape")
    if not 0.0 < balance_threshold <= 0.5:
        raise ValueError("balance_threshold must be in (0, 0.5]")
    overlap = valid_a & valid_b
    pair_weight = weight_a + weight_b
    normalized_a = np.divide(weight_a, pair_weight, out=np.zeros_like(weight_a), where=pair_weight > 0)
    normalized_b = np.divide(weight_b, pair_weight, out=np.zeros_like(weight_b), where=pair_weight > 0)
    return overlap & (np.minimum(normalized_a, normalized_b) >= balance_threshold)
