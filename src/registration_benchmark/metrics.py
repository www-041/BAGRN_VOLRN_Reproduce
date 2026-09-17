"""Registration quality metrics.

Computes coverage, gradient NCC, and independent phase verification
for every method under identical conditions.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import sobel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spatial coverage
# ---------------------------------------------------------------------------


def spatial_coverage_ratio(
    points_xy: np.ndarray,
    overlap_window: tuple[int, int, int, int],
    grid_rows: int = 8,
    grid_cols: int = 8,
) -> float:
    """Fraction of grid cells that contain at least one point.

    Divides the overlap bounding rectangle into a ``grid_rows × grid_cols``
    regular grid and counts cells with ≥ 1 inlier.

    Args:
        points_xy: Array of shape ``(N, 2)`` (column-major: x, y) of
            RANSAC inlier positions in common-grid coordinates.
        overlap_window: ``(row_start, row_end, col_start, col_end)`` of the
            valid overlap region.
        grid_rows: Number of subdivision rows.
        grid_cols: Number of subdivision columns.

    Returns:
        Coverage ratio ``∈ [0, 1]``.
    """
    if len(points_xy) == 0:
        return 0.0

    row_start, row_end, col_start, col_end = overlap_window
    height = row_end - row_start
    width = col_end - col_start

    if height <= 0 or width <= 0:
        return 0.0

    cell_h = height / grid_rows
    cell_w = width / grid_cols

    # Map points to grid cell indices
    row_idx = ((points_xy[:, 1] - row_start) / cell_h).astype(int)
    col_idx = ((points_xy[:, 0] - col_start) / cell_w).astype(int)

    # Clip to valid range
    row_idx = np.clip(row_idx, 0, grid_rows - 1)
    col_idx = np.clip(col_idx, 0, grid_cols - 1)

    occupied = np.zeros((grid_rows, grid_cols), dtype=bool)
    occupied[row_idx, col_idx] = True

    return float(occupied.sum()) / (grid_rows * grid_cols)


# ---------------------------------------------------------------------------
# Gradient NCC
# ---------------------------------------------------------------------------


def gradient_ncc(
    ref: np.ndarray,
    tgt: np.ndarray,
    valid: np.ndarray | None = None,
) -> float:
    """Normalized cross-correlation of gradient magnitudes.

    Computes Sobel gradient magnitude for both images, then NCC over
    their joint valid region.

    Args:
        ref: Reference image (2-D).
        tgt: Target image (2-D), same shape as *ref*.
        valid: Optional joint valid mask.  If ``None``, uses all finite pixels.

    Returns:
        NCC value ``∈ [-1, 1]``.  Returns ``float('nan')`` if the valid
        region has zero variance.
    """
    # Ensure float
    ref_f = ref.astype(np.float64)
    tgt_f = tgt.astype(np.float64)

    # Replace NaN with 0 for gradient computation
    ref_f = np.nan_to_num(ref_f, nan=0.0)
    tgt_f = np.nan_to_num(tgt_f, nan=0.0)

    # Sobel gradients
    gx_ref = sobel(ref_f, axis=1)
    gy_ref = sobel(ref_f, axis=0)
    mag_ref = np.hypot(gx_ref, gy_ref)

    gx_tgt = sobel(tgt_f, axis=1)
    gy_tgt = sobel(tgt_f, axis=0)
    mag_tgt = np.hypot(gx_tgt, gy_tgt)

    # Build joint valid mask
    if valid is None:
        joint = np.isfinite(ref) & np.isfinite(tgt)
    else:
        joint = valid & np.isfinite(ref) & np.isfinite(tgt)

    # Exclude borders (sobel has edge artefacts)
    joint[:1, :] = False
    joint[-1:, :] = False
    joint[:, :1] = False
    joint[:, -1:] = False

    if not joint.any():
        return float("nan")

    a = mag_ref[joint]
    b = mag_tgt[joint]

    a_mean = a.mean()
    b_mean = b.mean()
    a_std = a.std(ddof=1)
    b_std = b.std(ddof=1)

    if a_std < 1e-10 or b_std < 1e-10:
        return float("nan")

    ncc = float(((a - a_mean) * (b - b_mean)).mean() / (a_std * b_std))
    return ncc


# ---------------------------------------------------------------------------
# Independent phase verification
# ---------------------------------------------------------------------------


def phase_verification(
    ref: np.ndarray,
    registered_tgt: np.ndarray,
    valid_ref: np.ndarray,
    valid_tgt: np.ndarray,
) -> dict:
    """Verify registration quality via an independent phase-correlation check.

    This is used **only** for evaluation, not for fitting.

    Args:
        ref: Reference image (2-D, original DN).
        registered_tgt: Already-warped target image (2-D).
        valid_ref: Boolean mask for valid reference pixels.
        valid_tgt: Boolean mask for valid target pixels.

    Returns:
        Dict with keys ``dx``, ``dy``, ``magnitude``, ``confidence``,
        and ``status``.
    """
    from skimage.registration import phase_cross_correlation

    joint = valid_ref & valid_tgt & np.isfinite(ref) & np.isfinite(registered_tgt)

    if joint.sum() < 100:
        logger.warning("Phase verification: too few valid pixels (%d)", joint.sum())
        return {
            "dx": float("nan"),
            "dy": float("nan"),
            "magnitude": float("nan"),
            "confidence": float("nan"),
            "status": "TOO_FEW_VALID_PIXELS",
        }

    # Use gradient (structural) representation for robustness
    ref_f = np.nan_to_num(ref.astype(np.float64), nan=0.0)
    tgt_f = np.nan_to_num(registered_tgt.astype(np.float64), nan=0.0)

    gx_ref = sobel(ref_f, axis=1)
    gy_ref = sobel(ref_f, axis=0)
    mag_ref = np.hypot(gx_ref, gy_ref)

    gx_tgt = sobel(tgt_f, axis=1)
    gy_tgt = sobel(tgt_f, axis=0)
    mag_tgt = np.hypot(gx_tgt, gy_tgt)

    # Clip joint mask to match
    j = joint & np.isfinite(mag_ref) & np.isfinite(mag_tgt)
    if j.sum() < 100:
        return {
            "dx": float("nan"),
            "dy": float("nan"),
            "magnitude": float("nan"),
            "confidence": float("nan"),
            "status": "TOO_FEW_VALID_PIXELS_GRAD",
        }

    try:
        result = phase_cross_correlation(
            mag_ref, mag_tgt,
            reference_mask=j,
            moving_mask=j,
            normalization="phase",
        )
        # skimage ≥ 0.20 returns (shift, error, phasediff); older returns (shift,)
        if isinstance(result, tuple):
            shift = result[0]
            shift_y, shift_x = float(shift[0]), float(shift[1])
        else:
            shift_y, shift_x = float(result[0]), float(result[1])
        # Confidence: NCC of gradient magnitudes after shift
        from scipy.ndimage import shift as ndimage_shift
        shifted = ndimage_shift(mag_tgt, (shift_y, shift_x), order=1)
        a = mag_ref[j]
        b = shifted[j]
        a_mean, b_mean = a.mean(), b.mean()
        a_std, b_std = a.std(ddof=1), b.std(ddof=1)
        if a_std > 1e-10 and b_std > 1e-10:
            conf = float(((a - a_mean) * (b - b_mean)).mean() / (a_std * b_std))
        else:
            conf = float("nan")
    except Exception as e:
        logger.warning("Phase verification failed: %s", e)
        return {
            "dx": float("nan"),
            "dy": float("nan"),
            "magnitude": float("nan"),
            "confidence": float("nan"),
            "status": f"ERROR: {e}",
        }

    magnitude = float(np.sqrt(shift_x ** 2 + shift_y ** 2))

    return {
        "dx": float(shift_x),
        "dy": float(shift_y),
        "magnitude": magnitude,
        "confidence": conf,
        "status": "OK",
    }