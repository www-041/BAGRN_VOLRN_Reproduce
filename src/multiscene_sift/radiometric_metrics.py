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


TASK10D_LOCAL_TILE_SIZE = 256
TASK10D_LOCAL_MIN_SHARED_VALID = 4096


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


def build_fixed_grid_tiles(
    shape: tuple[int, int],
    *,
    tile_size: int = TASK10D_LOCAL_TILE_SIZE,
    origin: tuple[int, int] = (0, 0),
) -> list[dict[str, Any]]:
    """Return canonical-origin evaluation tiles independent of image values."""

    if tile_size != TASK10D_LOCAL_TILE_SIZE:
        raise ValueError("Task10D local tile_size is frozen at 256 pixels")
    if origin != (0, 0):
        raise ValueError("Task10D local grid is frozen to canonical origin (0, 0)")
    if len(shape) != 2 or any(int(size) <= 0 for size in shape):
        raise ValueError("shape must contain two positive spatial dimensions")
    height, width = (int(shape[0]), int(shape[1]))
    tiles: list[dict[str, Any]] = []
    for row_start in range(0, height, tile_size):
        for col_start in range(0, width, tile_size):
            row_stop = min(row_start + tile_size, height)
            col_stop = min(col_start + tile_size, width)
            row_index = row_start // tile_size
            col_index = col_start // tile_size
            tiles.append(
                {
                    "tile_id": f"r{row_index:03d}_c{col_index:03d}",
                    "row_start": row_start,
                    "row_stop": row_stop,
                    "col_start": col_start,
                    "col_stop": col_stop,
                    "support": (row_start, row_stop, col_start, col_stop),
                }
            )
    return tiles


def compute_local_pair_metrics(
    image_a: np.ndarray,
    image_b: np.ndarray,
    valid_mask: np.ndarray,
    *,
    tile_size: int = TASK10D_LOCAL_TILE_SIZE,
    min_shared_valid_pixels: int = TASK10D_LOCAL_MIN_SHARED_VALID,
) -> dict[str, Any]:
    """Compute Local MAMD and Local RDD on one immutable evaluation grid."""

    if min_shared_valid_pixels != TASK10D_LOCAL_MIN_SHARED_VALID:
        raise ValueError("Task10D local minimum shared-valid pixels is frozen at 4096")
    array_a = np.asarray(image_a)
    array_b = np.asarray(image_b)
    mask = np.asarray(valid_mask, dtype=bool)
    if array_a.shape != array_b.shape or array_a.shape != mask.shape:
        raise ValueError("image arrays and valid_mask must have identical shapes")
    if array_a.ndim != 2:
        raise ValueError("local metrics require two-dimensional arrays")
    rows: list[dict[str, Any]] = []
    for tile in build_fixed_grid_tiles(array_a.shape, tile_size=tile_size):
        row_slice = slice(tile["row_start"], tile["row_stop"])
        col_slice = slice(tile["col_start"], tile["col_stop"])
        tile_a = array_a[row_slice, col_slice]
        tile_b = array_b[row_slice, col_slice]
        tile_mask = (
            mask[row_slice, col_slice]
            & np.isfinite(tile_a)
            & np.isfinite(tile_b)
        )
        valid_pixels = int(np.count_nonzero(tile_mask))
        if valid_pixels < min_shared_valid_pixels:
            continue
        rows.append(
            {
                "tile_id": tile["tile_id"],
                "support": tile["support"],
                "valid_pixels": valid_pixels,
                "mamd": compute_pair_mamd(tile_a, tile_b, tile_mask),
                "rdd": compute_pair_rdd(tile_a, tile_b, tile_mask),
            }
        )
    return {
        "tiles": rows,
        "valid_tile_count": len(rows),
        "mamd": _summarize_local_rows(rows, "mamd"),
        "rdd": _summarize_local_rows(rows, "rdd"),
    }


def _summarize_local_rows(rows: Sequence[Mapping[str, Any]], value_key: str) -> dict[str, Any]:
    if not rows:
        return {"median": None, "mean": None, "p95": None, "worst_tile": None}
    values = np.asarray([float(row[value_key]) for row in rows], dtype=np.float64)
    worst_index = int(np.argmax(values))
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "worst_tile": dict(rows[worst_index]),
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
