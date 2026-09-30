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


__all__ = ["gradient_magnitude_ncc", "structure_metrics"]
