"""Canonical structure-preservation metrics for Task15/Task16.

All statistics use finite, valid support eroded by one pixel.  The optional
halo documents the tile contract: callers must provide a halo around a tile
before Sobel evaluation and retain only its core support.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, sobel


def _gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe = np.where(np.isfinite(image), image, 0.0)
    gx = sobel(safe, axis=1, mode="nearest")
    gy = sobel(safe, axis=0, mode="nearest")
    return gx, gy, np.hypot(gx, gy)


def gradient_magnitude_ncc(
    baseline: np.ndarray, candidate: np.ndarray, valid: np.ndarray, *, halo: int = 1,
) -> float | None:
    """Centered NCC of Sobel magnitudes on eroded finite support."""
    del halo  # halo is consumed by the tile provider; erosion defines the core.
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    support = np.asarray(valid, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    support = binary_erosion(support, structure=np.ones((3, 3), dtype=bool))
    _, _, ga = _gradient(a)
    _, _, gb = _gradient(b)
    if not np.any(support):
        return None
    x, y = ga[support], gb[support]
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denominator <= 0:
        return 1.0 if np.array_equal(ga[support], gb[support]) else None
    return float(np.sum(x * y) / denominator)


def structure_metrics(
    baseline: np.ndarray, candidate: np.ndarray, valid: np.ndarray, *, halo: int = 1,
) -> dict[str, Any]:
    """Return CGL, centered gradient NCC, orientation cosine and support count."""
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or np.asarray(valid).shape != a.shape:
        raise ValueError("baseline, candidate, and valid must be same-shape 2D arrays")
    support = np.asarray(valid, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    support = binary_erosion(support, structure=np.ones((3, 3), dtype=bool))
    ax, ay, amag = _gradient(a)
    bx, by, bmag = _gradient(b)
    orient = support & (amag > 1e-12) & (bmag > 1e-12)
    if np.any(orient):
        cosine = float(np.mean((ax[orient] * bx[orient] + ay[orient] * by[orient]) / (amag[orient] * bmag[orient])))
        delta = np.arctan2(by[orient], bx[orient]) - np.arctan2(ay[orient], ax[orient])
        cgl = np.abs(np.arctan2(np.sin(delta), np.cos(delta)))
        cgl_values = cgl
    else:
        cosine = None
        cgl_values = np.empty(0, dtype=np.float64)
    ncc = gradient_magnitude_ncc(a, b, valid, halo=halo)
    return {
        "cgl_rad": float(np.mean(cgl_values)) if cgl_values.size else 0.0 if np.array_equal(a[support], b[support]) else None,
        "cgl_deg": float(np.rad2deg(np.mean(cgl_values))) if cgl_values.size else 0.0 if np.array_equal(a[support], b[support]) else None,
        "gradient_magnitude_ncc": ncc,
        "gradient_orientation_cosine": cosine if cosine is not None else (1.0 if np.array_equal(a[support], b[support]) else None),
        "eligible_pixels": int(np.count_nonzero(orient)),
        "support_pixels": int(np.count_nonzero(support)),
    }


def stream_structure_metrics(
    baseline_path: str | Any,
    candidate_path: str | Any,
    valid: np.ndarray,
    *,
    tile_size: int = 512,
    halo: int = 1,
) -> dict[str, Any]:
    """Compute the canonical structure metrics from haloed raster windows.

    The valid mask is kept un-eroded while constructing the safe image.  The
    one-pixel erosion is applied only to the metric support after Sobel has
    seen the original valid-valued neighbourhood.  This prevents NoData
    edges from becoming artificial gradients and makes the result invariant
    to the tile partition.
    """
    import rasterio

    mask = np.asarray(valid, dtype=bool)
    sums = np.zeros(5, dtype=np.float64)  # n, sum(x), sum(y), sum(x^2), sum(y^2)
    cross = 0.0
    cgl_sum = 0.0
    cosine_sum = 0.0
    orientation_count = 0
    support_count = 0
    with rasterio.open(baseline_path) as left, rasterio.open(candidate_path) as right:
        if (left.height, left.width) != mask.shape or (right.height, right.width) != mask.shape:
            raise ValueError("raster and valid mask shapes differ")
        height, width = mask.shape
        for r0 in range(0, height, tile_size):
            for c0 in range(0, width, tile_size):
                r1, c1 = min(height, r0 + tile_size), min(width, c0 + tile_size)
                rs, cs = max(0, r0 - halo), max(0, c0 - halo)
                re, ce = min(height, r1 + halo), min(width, c1 + halo)
                window = rasterio.windows.Window(cs, rs, ce - cs, re - rs)
                a = left.read(1, window=window).astype(np.float64)
                b = right.read(1, window=window).astype(np.float64)
                original_valid = mask[rs:re, cs:ce] & np.isfinite(a) & np.isfinite(b)
                safe_a = np.where(original_valid, a, 0.0)
                safe_b = np.where(original_valid, b, 0.0)
                ax = sobel(safe_a, axis=1, mode="nearest")
                ay = sobel(safe_a, axis=0, mode="nearest")
                bx = sobel(safe_b, axis=1, mode="nearest")
                by = sobel(safe_b, axis=0, mode="nearest")
                amag = np.hypot(ax, ay)
                bmag = np.hypot(bx, by)
                support = binary_erosion(original_valid, structure=np.ones((3, 3), dtype=bool))
                core = np.zeros_like(support, dtype=bool)
                core[r0 - rs:r1 - rs, c0 - cs:c1 - cs] = True
                support &= core
                support_count += int(support.sum())
                orient = support & (amag > 1e-12) & (bmag > 1e-12)
                if not orient.any():
                    continue
                x, y = amag[orient], bmag[orient]
                n = float(x.size)
                sums += np.asarray([n, x.sum(), y.sum(), np.square(x).sum(), np.square(y).sum()])
                cross += float(np.dot(x, y))
                delta = np.arctan2(by[orient], bx[orient]) - np.arctan2(ay[orient], ax[orient])
                cgl_sum += float(np.abs(np.arctan2(np.sin(delta), np.cos(delta))).sum())
                cosine_sum += float(((ax[orient] * bx[orient] + ay[orient] * by[orient]) / (amag[orient] * bmag[orient])).sum())
                orientation_count += int(x.size)
    n, sx, sy, sx2, sy2 = sums
    if n <= 0:
        ncc = 1.0 if np.array_equal(mask, mask) and support_count > 0 else None
    else:
        numerator = cross - sx * sy / n
        denominator = np.sqrt(max(sx2 - sx * sx / n, 0.0) * max(sy2 - sy * sy / n, 0.0))
        ncc = float(numerator / denominator) if denominator > 0 else None
    return {
        "cgl_rad": float(cgl_sum / orientation_count) if orientation_count else None,
        "cgl_deg": float(np.rad2deg(cgl_sum / orientation_count)) if orientation_count else None,
        "gradient_magnitude_ncc": ncc,
        "gradient_orientation_cosine": float(cosine_sum / orientation_count) if orientation_count else None,
        "eligible_pixels": orientation_count,
        "support_pixels": support_count,
        "finite": bool(ncc is not None and all(np.isfinite(value) for value in (ncc, cgl_sum, cosine_sum))),
    }


__all__ = ["gradient_magnitude_ncc", "structure_metrics", "stream_structure_metrics"]
