"""Independent Task10D radiometric metrics.

The functions in this module operate only on shared-valid samples supplied by
the caller. They intentionally do not implement the unresolved paper CD/GL
definitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance


def compute_pair_mamd(
    image_a: np.ndarray, image_b: np.ndarray, valid_mask: np.ndarray
) -> float:
    """Return ``abs(mean(a) - mean(b))`` on shared-valid finite pixels."""

    values_a, values_b = _shared_values(image_a, image_b, valid_mask)
    _require_samples(values_a.size)
    return float(abs(np.mean(values_a) - np.mean(values_b)))


def compute_pair_msdd(
    image_a: np.ndarray, image_b: np.ndarray, valid_mask: np.ndarray
) -> float:
    """Return ``abs(std(a) - std(b))`` using population standard deviation."""

    values_a, values_b = _shared_values(image_a, image_b, valid_mask)
    _require_samples(values_a.size)
    return float(abs(np.std(values_a) - np.std(values_b)))


def compute_pair_rdd(
    image_a: np.ndarray, image_b: np.ndarray, valid_mask: np.ndarray
) -> float:
    """Return exact one-dimensional Wasserstein-1 on all shared-valid samples."""

    values_a, values_b = _shared_values(image_a, image_b, valid_mask)
    _require_samples(values_a.size)
    return float(wasserstein_distance(values_a, values_b))


def aggregate_weighted_pair_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str = "value",
    weight_key: str = "valid_pixels",
    pair_key: str = "pair",
) -> dict[str, Any]:
    """Aggregate pair values with shared-pixel weights and retain edge detail."""

    normalized = [
        {
            pair_key: row[pair_key],
            value_key: float(row[value_key]),
            weight_key: int(row[weight_key]),
        }
        for row in rows
    ]
    if not normalized:
        return {
            "weighted_mean": None,
            "edge_median": None,
            "worst_pair": None,
            "per_pair": [],
            "valid_pair_count": 0,
        }
    weights = np.asarray([row[weight_key] for row in normalized], dtype=np.float64)
    values = np.asarray([row[value_key] for row in normalized], dtype=np.float64)
    if np.any(weights <= 0):
        raise ValueError("pair weights must be positive shared-pixel counts")
    worst_index = int(np.argmax(values))
    return {
        "weighted_mean": float(np.average(values, weights=weights)),
        "edge_median": float(np.median(values)),
        "worst_pair": dict(normalized[worst_index]),
        "per_pair": normalized,
        "valid_pair_count": len(normalized),
    }


def _shared_values(
    image_a: np.ndarray, image_b: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    array_a = np.asarray(image_a, dtype=np.float64)
    array_b = np.asarray(image_b, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if array_a.shape != array_b.shape or array_a.shape != mask.shape:
        raise ValueError("image arrays and valid_mask must have identical shapes")
    shared = mask & np.isfinite(array_a) & np.isfinite(array_b)
    return array_a[shared], array_b[shared]


def _require_samples(count: int) -> None:
    if count == 0:
        raise ValueError("shared-valid metric requires at least one finite pixel")
