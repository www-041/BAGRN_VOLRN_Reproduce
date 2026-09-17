"""SIFT tie-point matcher.

Uses OpenCV SIFT with Lowe ratio test and mutual-consistency check.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from src.registration_benchmark.models import MatchSet, MatchView

logger = logging.getLogger(__name__)


def match_sift(
    view: MatchView,
    nfeatures: int = 8000,
    ratio_threshold: float = 0.75,
) -> MatchSet:
    """Extract SIFT keypoints and match them with mutual-consistency check.

    Pipeline::

        float32 [0,1] → uint8 → SIFT → BF+KNN → Lowe ratio →
        symmetric check → valid-mask filter → :meth:`MatchView.to_canvas`

    Args:
        view: Preprocessed overlap view.
        nfeatures: Maximum number of SIFT keypoints to retain.
        ratio_threshold: Lowe's ratio threshold for KNN filtering.

    Returns:
        A :class:`MatchSet` with coordinates in common-grid space.
    """
    t0 = time.perf_counter()

    # --- Convert [0, 1] float32 → uint8 ---------------------------------------
    ref_u8 = _to_uint8(view.ref)
    tgt_u8 = _to_uint8(view.tgt)

    # --- SIFT detect & compute -------------------------------------------------
    sift = cv2.SIFT_create(nfeatures=nfeatures)

    kp_ref, des_ref = sift.detectAndCompute(ref_u8, None)
    kp_tgt, des_tgt = sift.detectAndCompute(tgt_u8, None)

    if des_ref is None or des_tgt is None or len(kp_ref) < 2 or len(kp_tgt) < 2:
        elapsed = time.perf_counter() - t0
        logger.warning("SIFT: insufficient keypoints (ref=%d, tgt=%d)",
                       len(kp_ref) if kp_ref is not None else 0,
                       len(kp_tgt) if kp_tgt is not None else 0)
        return MatchSet(
            method="sift",
            ref_xy=np.empty((0, 2)),
            tgt_xy=np.empty((0, 2)),
            confidence=np.empty(0),
            runtime_sec=elapsed,
            metadata={"nfeatures": nfeatures, "ratio_threshold": ratio_threshold},
        )

    # --- BFMatcher + KNN k=2 ---------------------------------------------------
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = matcher.knnMatch(des_ref, des_tgt, k=2)

    # Lowe ratio
    good_fwd = []
    for m, n in raw_matches:
        if m.distance < ratio_threshold * n.distance:
            good_fwd.append(m)

    # --- Symmetric mutual-consistency check ------------------------------------
    raw_matches_rev = matcher.knnMatch(des_tgt, des_ref, k=2)
    good_rev = []
    for m, n in raw_matches_rev:
        if m.distance < ratio_threshold * n.distance:
            good_rev.append(m)

    # Build reverse index: tgt_idx → ref_idx
    rev_map = {m.queryIdx: m.trainIdx for m in good_rev}

    # Keep only mutually consistent matches
    mutual = []
    for m in good_fwd:
        if m.trainIdx in rev_map and rev_map[m.trainIdx] == m.queryIdx:
            mutual.append(m)

    if not mutual:
        elapsed = time.perf_counter() - t0
        logger.warning("SIFT: zero mutual matches after symmetric check")
        return MatchSet(
            method="sift",
            ref_xy=np.empty((0, 2)),
            tgt_xy=np.empty((0, 2)),
            confidence=np.empty(0),
            runtime_sec=elapsed,
            metadata={
                "nfeatures": nfeatures,
                "ratio_threshold": ratio_threshold,
                "raw_fwd": len(good_fwd),
                "raw_rev": len(good_rev),
                "mutual": 0,
            },
        )

    # --- Extract coordinates in view pixel space -------------------------------
    n = len(mutual)
    ref_xy_view = np.empty((n, 2), dtype=np.float64)
    tgt_xy_view = np.empty((n, 2), dtype=np.float64)
    distances = np.empty(n, dtype=np.float64)

    for i, m in enumerate(mutual):
        ref_xy_view[i, 0] = kp_ref[m.queryIdx].pt[0]  # x = column
        ref_xy_view[i, 1] = kp_ref[m.queryIdx].pt[1]  # y = row
        tgt_xy_view[i, 0] = kp_tgt[m.trainIdx].pt[0]
        tgt_xy_view[i, 1] = kp_tgt[m.trainIdx].pt[1]
        distances[i] = m.distance

    # --- Confidence (method-internal, NOT cross-method comparable) -------------
    max_dist = distances.max() if len(distances) > 0 else 1.0
    if max_dist > 0:
        confidence = 1.0 - distances / max_dist
    else:
        confidence = np.ones_like(distances)
    confidence = np.clip(confidence, 0.0, 1.0)

    # --- Valid-mask filter (discard matches on invalid pixels) -----------------
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        r = int(round(ref_xy_view[i, 1]))
        c = int(round(ref_xy_view[i, 0]))
        if 0 <= r < view.ref_valid.shape[0] and 0 <= c < view.ref_valid.shape[1]:
            if not view.ref_valid[r, c]:
                keep[i] = False

        tr = int(round(tgt_xy_view[i, 1]))
        tc = int(round(tgt_xy_view[i, 0]))
        if 0 <= tr < view.tgt_valid.shape[0] and 0 <= tc < view.tgt_valid.shape[1]:
            if not view.tgt_valid[tr, tc]:
                keep[i] = False

    ref_xy_view = ref_xy_view[keep]
    tgt_xy_view = tgt_xy_view[keep]
    confidence = confidence[keep]

    # --- Map to common-grid coordinates ----------------------------------------
    if len(ref_xy_view) == 0:
        elapsed = time.perf_counter() - t0
        return MatchSet(
            method="sift",
            ref_xy=np.empty((0, 2)),
            tgt_xy=np.empty((0, 2)),
            confidence=np.empty(0),
            runtime_sec=elapsed,
            metadata={
                "nfeatures": nfeatures,
                "ratio_threshold": ratio_threshold,
                "raw_fwd": len(good_fwd),
                "raw_rev": len(good_rev),
                "mutual": n,
                "valid_mask_filtered": True,
            },
        )

    ref_xy = view.to_canvas(ref_xy_view)
    tgt_xy = view.to_canvas(tgt_xy_view)

    elapsed = time.perf_counter() - t0

    return MatchSet(
        method="sift",
        ref_xy=ref_xy,
        tgt_xy=tgt_xy,
        confidence=confidence,
        runtime_sec=elapsed,
        metadata={
            "nfeatures": nfeatures,
            "ratio_threshold": ratio_threshold,
            "raw_fwd": len(good_fwd),
            "raw_rev": len(good_rev),
            "mutual": n,
            "valid_mask_filtered": int((~keep).sum()) if hasattr(keep, '__len__') else 0,
        },
    )


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float32 [0, 1] image to uint8 [0, 255]."""
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)