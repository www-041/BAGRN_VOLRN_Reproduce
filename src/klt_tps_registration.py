"""Pure-array KLT/TPS registration primitives.

The implementation preserves the senior-provided KLT matching semantics while
leaving raster I/O and pipeline orchestration to the project modules.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def normalize_klt_image(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Apply the source script's robust local contrast normalization."""
    a = np.asarray(data).astype(np.float32)
    mask = np.asarray(valid, dtype=bool)
    if a.ndim != 2 or mask.shape != a.shape:
        raise ValueError("KLT normalization requires same-shape 2D data/mask")
    mask &= np.isfinite(a)
    if np.count_nonzero(mask) < 100:
        raise ValueError("KLT normalization requires at least 100 valid pixels")
    low, high = np.percentile(a[mask], [1, 99])
    if not np.isfinite(high - low) or high <= low:
        raise ValueError("valid image lacks usable intensity range")
    image = np.clip((a - low) / (high - low), 0, 1)
    image[~mask] = np.median(image[mask])
    mean = gaussian_filter(image, 7)
    std = np.sqrt(
        np.maximum(gaussian_filter(image * image, 7) - mean * mean, 0)
        + 0.0025
    )
    return np.uint8(np.clip((image - mean) / std * 0.18 + 0.5, 0, 1) * 255)


def build_klt_interior_mask(valid: np.ndarray, margin: int) -> np.ndarray:
    """Erode a validity mask with the source square interior kernel."""
    mask = np.asarray(valid, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("KLT interior mask requires a 2D validity mask")
    margin = int(margin)
    if margin < 0:
        raise ValueError("KLT interior margin must be non-negative")
    if margin == 0:
        return (mask != 0).astype(np.uint8)
    kernel = np.ones((2 * margin + 1, 2 * margin + 1), dtype=np.uint8)
    return cv2.erode(
        (mask != 0).astype(np.uint8), kernel,
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )


def _require_same_2d_images(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference)
    mov = np.asarray(moving)
    if ref.ndim != 2 or mov.ndim != 2 or ref.shape != mov.shape:
        raise ValueError("KLT matching requires same-shape 2D overlap images")
    if min(ref.shape) < 32 or max(ref.shape) >= 32767:
        raise ValueError("insufficient KLT overlap dimensions")
    return ref, mov


def match_bidirectional_klt(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_valid: np.ndarray,
    moving_valid: np.ndarray,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Find source-to-target controls with forward/backward Pyramid-LK."""
    params = params or {}
    ref, mov = _require_same_2d_images(reference, moving)
    ref_valid = np.asarray(reference_valid, dtype=bool)
    mov_valid = np.asarray(moving_valid, dtype=bool)
    if ref_valid.shape != ref.shape or mov_valid.shape != mov.shape:
        raise ValueError("KLT matching requires same-shape validity masks")

    cv2.setNumThreads(int(params.get("klt_tps_threads", 4)))
    try:
        ref_norm = normalize_klt_image(ref, ref_valid)
        mov_norm = normalize_klt_image(mov, mov_valid)
    except ValueError as exc:
        raise ValueError(f"insufficient KLT image support: {exc}") from exc

    window = int(params.get("klt_tps_window", 9))
    pyramid_level = int(params.get("klt_tps_pyramid_level", 3))
    ref_interior = build_klt_interior_mask(
        ref_valid, max(20, window // 2 + 2),
    )
    corners = cv2.goodFeaturesToTrack(
        ref_norm,
        maxCorners=int(params.get("klt_tps_max_corners", 4000)),
        qualityLevel=0.003,
        minDistance=float(params.get("klt_tps_min_corner_distance", 5.0)),
        mask=ref_interior,
        blockSize=5,
    )
    initial_count = 0 if corners is None else int(len(corners))
    if corners is None or initial_count < int(params.get("klt_tps_min_points", 30)):
        raise ValueError(
            f"insufficient KLT corners: {initial_count} < "
            f"{int(params.get('klt_tps_min_points', 30))}"
        )
    p = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 0.001)
    lk_kwargs = {
        "winSize": (window, window),
        "maxLevel": pyramid_level,
        "criteria": criteria,
    }
    q, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        ref_norm, mov_norm, p.reshape(-1, 1, 2), None, **lk_kwargs
    )
    if q is None or status_forward is None:
        raise ValueError("insufficient valid forward KLT tracks")
    q = np.asarray(q, dtype=np.float32).reshape(-1, 2)
    sf = np.asarray(status_forward).reshape(-1).astype(bool)
    finite = sf & np.isfinite(q).all(axis=1)
    p_forward = p[finite]
    q_forward = q[finite]
    if len(q_forward) < int(params.get("klt_tps_min_points", 30)):
        raise ValueError("insufficient valid forward KLT tracks")

    p_back, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        mov_norm, ref_norm, q_forward.reshape(-1, 1, 2), None, **lk_kwargs
    )
    if p_back is None or status_backward is None:
        raise ValueError("insufficient valid backward KLT tracks")
    p_back = np.asarray(p_back, dtype=np.float32).reshape(-1, 2)
    sb = np.asarray(status_backward).reshape(-1).astype(bool)
    if len(p_back) != len(p_forward):
        raise ValueError("insufficient valid backward KLT tracks")
    height, width = mov.shape
    q_rows = np.rint(q_forward[:, 1]).astype(int)
    q_cols = np.rint(q_forward[:, 0]).astype(int)
    in_bounds = (
        (q_forward[:, 0] > 10) & (q_forward[:, 0] < width - 10)
        & (q_forward[:, 1] > 10) & (q_forward[:, 1] < height - 10)
    )
    safe_rows = np.clip(q_rows, 0, height - 1)
    safe_cols = np.clip(q_cols, 0, width - 1)
    moving_interior = build_klt_interior_mask(
        mov_valid, max(10, window // 2 + 2),
    )
    interior = moving_interior[safe_rows, safe_cols] != 0
    fb = np.linalg.norm(p_forward - p_back, axis=1)
    accepted = sb & np.isfinite(fb) & in_bounds & interior
    accepted &= fb < float(params.get("klt_tps_fb_threshold", 0.5))
    p_acc = p_forward[accepted].astype(np.float64)
    q_acc = q_forward[accepted].astype(np.float64)
    fb_acc = fb[accepted].astype(np.float64)
    minimum = int(params.get("klt_tps_min_points", 30))
    if len(p_acc) < minimum:
        raise ValueError(f"insufficient accepted KLT matches: {len(p_acc)} < {minimum}")
    centered = p_acc - p_acc.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError("collinear KLT controls")
    return {
        "initial_corner_count": initial_count,
        "accepted_point_count": int(len(p_acc)),
        "reference_points_xy": p_acc,
        "moving_points_xy": q_acc,
        "displacement_xy": q_acc - p_acc,
        "forward_backward_error": fb_acc,
    }
