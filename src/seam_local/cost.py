"""Frozen single-band seam cost for Task13A."""

import numpy as np
from scipy.ndimage import sobel


def compute_seam_cost(
    a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Return the P95-normalized intensity/gradient cost on joint valid pixels.

    Invalid pixels receive positive infinity and therefore cannot be DP nodes.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if a.ndim != 2 or a.shape != b.shape or a.shape != valid_mask.shape:
        raise ValueError("a, b, and valid_mask must be same-shape 2D arrays")

    valid = valid_mask & np.isfinite(a) & np.isfinite(b)
    cost = np.full(a.shape, np.inf, dtype=np.float64)
    if not np.any(valid):
        return cost

    # Invalid samples must not introduce NaNs into nearby Sobel stencils.
    safe_a = np.where(valid, a, 0.0)
    safe_b = np.where(valid, b, 0.0)
    grad_a = np.hypot(sobel(safe_a, axis=0), sobel(safe_a, axis=1))
    grad_b = np.hypot(sobel(safe_b, axis=0), sobel(safe_b, axis=1))

    # D_I = |A-B|; D_G = ||grad A|-|grad B||.
    intensity_difference = np.abs(safe_a - safe_b)
    gradient_difference = np.abs(grad_a - grad_b)

    # Normalize each term by its valid P95, then clip to [0, 1] before
    # combining C = 0.5 D_I_norm + 0.5 D_G_norm.
    intensity_scale = max(float(np.percentile(intensity_difference[valid], 95)), 1e-6)
    gradient_scale = max(float(np.percentile(gradient_difference[valid], 95)), 1e-6)
    cost[valid] = 0.5 * (
        np.clip(intensity_difference[valid] / intensity_scale, 0.0, 1.0)
        + np.clip(gradient_difference[valid] / gradient_scale, 0.0, 1.0)
    )
    return cost
