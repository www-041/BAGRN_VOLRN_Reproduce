"""B14 white-artifact diagnosis utilities (diagnostic only, no algorithm edits).

These functions answer: where do white pixels first appear, what are
their real DN values, and do they overlap NoData holes or extreme-high
DN regions. Nothing here modifies the BAGRN/VOLRN math.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_PERCENTILE_KEYS = ["p0_1", "p1", "p50", "p99", "p99_9"]
_PERCENTILES = [0.1, 1.0, 50.0, 99.0, 99.9]


# ---------------------------------------------------------------------------
# Stage statistics
# ---------------------------------------------------------------------------


def _is_nodata(data: np.ndarray, nodata) -> np.ndarray:
    """Boolean mask of pixels equal to the NoData value (or all-False)."""
    if nodata is None:
        return np.zeros(data.shape, dtype=bool)
    return data == nodata


def _is_valid(data: np.ndarray, nodata) -> np.ndarray:
    """Valid = finite and not equal to the NoData value."""
    valid = np.isfinite(data)
    if nodata is not None:
        valid &= (data != nodata)
    return valid


def stage_statistics(data: np.ndarray, nodata) -> dict:
    """Compute one-stage numeric statistics.

    Args:
        data: 2-D image array (single band).
        nodata: NoData value or ``None``.

    Returns:
        Dict with pixel counts, min/max/mean/std and percentiles over
        valid pixels. NaN/Inf are all converted to ``None`` so the dict
        is JSON-serialisable with ``allow_nan=False``.
    """
    data = np.asarray(data)
    valid = _is_valid(data, nodata)
    n_total = int(data.size)
    n_valid = int(valid.sum())
    n_nodata = int(_is_nodata(data, nodata).sum())
    n_nan = int(np.isnan(data).sum())
    n_inf = int(np.isinf(data).sum())

    stats = {
        "total_pixels": n_total,
        "valid_pixels": n_valid,
        "nodata_pixels": n_nodata,
        "nan_pixels": n_nan,
        "inf_pixels": n_inf,
        "valid_min": None,
        "valid_max": None,
        "mean": None,
        "std": None,
        "p0_1": None,
        "p1": None,
        "p50": None,
        "p99": None,
        "p99_9": None,
    }

    if n_valid == 0:
        return stats

    vals = data[valid].astype(np.float64)
    stats["valid_min"] = float(vals.min())
    stats["valid_max"] = float(vals.max())
    stats["mean"] = float(vals.mean())
    stats["std"] = float(vals.std() if vals.size > 1 else 0.0)
    pcts = np.percentile(vals, _PERCENTILES)
    for key, val in zip(_PERCENTILE_KEYS, pcts):
        stats[key] = float(val)

    return stats


def count_new_invalid(
    before: np.ndarray,
    before_nodata,
    after: np.ndarray,
    after_nodata,
) -> dict:
    """Count pixels valid in ``before`` that become nodata/NaN/Inf in ``after``.

    Args:
        before: Input-stage array.
        before_nodata: NoData of the input stage.
        after: Output-stage array.
        after_nodata: NoData of the output stage.

    Returns:
        ``{"new_nodata": int, "new_nan": int, "new_inf": int}`` counts
        over pixels that were valid in ``before``.
    """
    before_valid = _is_valid(before, before_nodata)

    new_nodata = int((before_valid & _is_nodata(after, after_nodata)).sum())
    new_nan = int((before_valid & np.isnan(after)).sum())
    new_inf = int((before_valid & np.isinf(after)).sum())
    return {"new_nodata": new_nodata, "new_nan": new_nan, "new_inf": new_inf}


# ---------------------------------------------------------------------------
# Extreme-high masks
# ---------------------------------------------------------------------------


def extreme_high_threshold(data: np.ndarray, nodata, percentile: float = 99.9) -> float:
    """Return the *percentile* quantile of valid pixels (default P99.9)."""
    data = np.asarray(data)
    valid = _is_valid(data, nodata)
    if not valid.any():
        return float("nan")
    return float(np.percentile(data[valid].astype(np.float64), percentile))


def make_invalid_mask(data: np.ndarray, nodata) -> np.ndarray:
    """Invalid = not finite or equal to NoData."""
    data = np.asarray(data)
    invalid = ~np.isfinite(data)
    if nodata is not None:
        invalid |= (data == nodata)
    return invalid


def make_extreme_high_mask(
    data: np.ndarray, nodata, threshold: float,
) -> np.ndarray:
    """Mask of valid pixels above *threshold*."""
    data = np.asarray(data)
    valid = _is_valid(data, nodata)
    high = np.zeros(data.shape, dtype=bool)
    high[valid] = data[valid] > threshold
    return high


# ---------------------------------------------------------------------------
# TIFF helpers
# ---------------------------------------------------------------------------


def save_stage_tiff(
    data: np.ndarray,
    transform,
    crs,
    nodata,
    path: str | Path,
) -> str:
    """Write a single-band diagnostic GeoTIFF preserving transform/CRS/nodata."""
    import rasterio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data)
    dtype = "float64" if arr.dtype == np.float64 else arr.dtype.name
    with rasterio.open(
        path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=dtype,
        crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(arr, 1)
    logger.info("Saved %s (%dx%d)", path, arr.shape[1], arr.shape[0])
    return str(path)


# ---------------------------------------------------------------------------
# VOLRN block diagnostics
# ---------------------------------------------------------------------------


def block_valid_fraction_stats(
    blocks,
    n_bands: int = 1,
) -> dict:
    """Compute per-block valid-fraction statistics from image_blocking blocks.

    Uses only ``BlockInfo.valid_mask``; does not change block selection.

    Returns:
        Dict with ``n_blocks`` and block-counts below <1%/<5%/<10%/<30%.
    """
    fracs = []
    for blk in blocks:
        vm = blk.valid_mask  # (n_bands, rows, cols)
        total = blk.window[1] - blk.window[0], blk.window[3] - blk.window[2]
        n_total = max(total[0], 1) * max(total[1], 1)
        valid_count = int(vm[n_bands - 1].sum()) if vm.ndim > 1 else int(vm.sum())
        fracs.append(valid_count / n_total if n_total > 0 else 0.0)

    fracs = np.asarray(fracs, dtype=np.float64)
    counts = {
        "n_blocks": int(len(fracs)),
        "valid_fraction_lt_1pct": int((fracs < 0.01).sum()),
        "valid_fraction_lt_5pct": int((fracs < 0.05).sum()),
        "valid_fraction_lt_10pct": int((fracs < 0.10).sum()),
        "valid_fraction_lt_30pct": int((fracs < 0.30).sum()),
    }
    if len(fracs) > 0:
        counts["valid_fraction_mean"] = float(fracs.mean())
        counts["valid_fraction_min"] = float(fracs.min())
    else:
        counts["valid_fraction_mean"] = None
        counts["valid_fraction_min"] = None
    return counts


def coefficient_outlier_rows(
    block_details: list[dict],
    valid_fractions: list[float] | None = None,
) -> dict:
    """Identify extreme a/b blocks using median +/- 5*MAD (robust).

    Args:
        block_details: list from VOLRN diagnostics ``block_details``.
        valid_fractions: optional per-block valid fraction in the same
            block order.

    Returns:
        Dict with per-scene stats and a list of flagged block ids.
    """
    per_scene: dict[int, list[dict]] = {}
    for blk in block_details:
        per_scene.setdefault(blk["image_idx"], []).append(blk)

    scene_stats = {}
    flagged = []
    for scene_idx in sorted(per_scene):
        a_vals = np.array([b["a"] for b in per_scene[scene_idx]], dtype=np.float64)
        b_vals = np.array([b["b"] for b in per_scene[scene_idx]], dtype=np.float64)
        stats = _robust_stats(a_vals, b_vals)
        scene_stats[scene_idx] = {
            "a_min": stats["a_min"], "a_p1": stats["a_p1"],
            "a_median": stats["a_median"], "a_p99": stats["a_p99"],
            "a_max": stats["a_max"],
            "b_min": stats["b_min"], "b_p1": stats["b_p1"],
            "b_median": stats["b_median"], "b_p99": stats["b_p99"],
            "b_max": stats["b_max"],
            "a_lower_fence": stats["a_lower"], "a_upper_fence": stats["a_upper"],
            "b_lower_fence": stats["b_lower"], "b_upper_fence": stats["b_upper"],
        }
        for i, blk in enumerate(per_scene[scene_idx]):
            a_ok = stats["a_lower"] <= blk["a"] <= stats["a_upper"]
            b_ok = stats["b_lower"] <= blk["b"] <= stats["b_upper"]
            if not (a_ok and b_ok):
                row = {
                    "block_id": blk["block_id"], "image_idx": blk["image_idx"],
                    "grid_m": blk["grid_m"], "grid_n": blk["grid_n"],
                    "a": blk["a"], "b": blk["b"],
                    "is_a_outlier": not a_ok, "is_b_outlier": not b_ok,
                }
                if valid_fractions is not None and i < len(valid_fractions):
                    row["valid_fraction"] = valid_fractions[i]
                flagged.append(row)

    return {"per_scene": scene_stats, "n_flagged": len(flagged), "flagged": flagged}


def _robust_stats(a: np.ndarray, b: np.ndarray) -> dict:
    def fence(vals):
        if vals.size == 0:
            return None, None, None, None
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        # 1.4826 scale for a normal approx; guard zero MAD
        scaled = 1.4826 * mad
        if scaled < 1e-12:
            scaled = 1e-12
        return (med, med - 5 * scaled, med + 5 * scaled, scaled)

    a_med, a_lo, a_up, _ = fence(a)
    b_med, b_lo, b_up, _ = fence(b)

    def pct(vals, q):
        return float(np.percentile(vals, q)) if vals.size > 0 else None

    return {
        "a_min": pct(a, 0), "a_p1": pct(a, 1), "a_median": a_med,
        "a_p99": pct(a, 99), "a_max": pct(a, 100),
        "b_min": pct(b, 0), "b_p1": pct(b, 1), "b_median": b_med,
        "b_p99": pct(b, 99), "b_max": pct(b, 100),
        "a_lower": a_lo, "a_upper": a_up,
        "b_lower": b_lo, "b_upper": b_up,
    }


# ---------------------------------------------------------------------------
# Affine / IDW coordinate diagnostics (Task 10)
# ---------------------------------------------------------------------------


def affine_coordinate_diagnostic(
    transforms,
    image_heights: list[int],
    image_widths: list[int],
    res_x: float,
    res_y: float,
    block_size: int,
) -> dict:
    """Quantify the cross-term error of the affine-IDW pixel mapping.

    ``_interpolate_and_apply`` computes:
        geo_y = transforms[i] * (0, row)   # ignores the col term d*col
        geo_x = transforms[i] * (col, 0)   # ignores the row term b*row

    For a general affine with non-zero ``b`` / ``d`` this introduces a
    position-dependent error that grows across the image.

    Returns:
        Dict with per-scene a/b/d/e, rotation/shear magnitude, and the
        maximum cross-term shift in pixels and blocks.
    """
    scenes_diag = []
    worst_px = 0.0
    worst_blocks = 0.0
    for i, tf in enumerate(transforms):
        a = tf.a
        b = tf.b
        c = tf.c
        d = tf.d
        e = tf.e
        f = tf.f
        height = image_heights[i]
        width = image_widths[i]

        rotation_deg = math.degrees(math.atan2(d, a)) if a != 0 else 90.0
        shear_deg = math.degrees(math.atan2(b, e)) if e != 0 else 90.0

        # Worst cross-term error is at the far edge of the image.
        max_dx_world = abs(b) * height   # geo_x = c + a*col + b*row, row in [0,h]
        max_dy_world = abs(d) * width    # geo_y = f + d*col + e*row, col in [0,w]
        cross_px = max(max_dx_world / res_x, max_dy_world / res_y)
        cross_blocks = cross_px / block_size
        worst_px = max(worst_px, cross_px)
        worst_blocks = max(worst_blocks, cross_blocks)

        scenes_diag.append({
            "scene": i,
            "a": float(a), "b": float(b), "c": float(c),
            "d": float(d), "e": float(e), "f": float(f),
            "rotation_deg": float(rotation_deg),
            "shear_deg": float(shear_deg),
            "max_cross_term_shift_pixels": float(cross_px),
            "max_cross_term_shift_blocks": float(cross_blocks),
        })

    return {
        "res_x": res_x,
        "res_y": res_y,
        "block_size": block_size,
        "max_cross_term_shift_pixels": float(worst_px),
        "max_cross_term_shift_blocks": float(worst_blocks),
        "scenes": scenes_diag,
    }


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def write_json(path: str | Path, obj) -> None:
    """Write JSON with NaN/Inf converted to ``None`` and ``allow_nan=False``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(obj), f, indent=2, allow_nan=False)
    logger.info("Saved JSON: %s", path)


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return str(obj)