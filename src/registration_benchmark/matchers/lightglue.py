"""SuperPoint + LightGlue matcher adapter.

Uses the official ``lightglue`` package.  Falls back gracefully if the
package is not installed.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from src.registration_benchmark.models import MatchView, make_matchset_from_view

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import — allow the module to load without lightglue installed
# ---------------------------------------------------------------------------

_LIGHTGLUE_AVAILABLE = False
_LG_IMPORT_ERROR: str | None = None

try:
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd

    _LIGHTGLUE_AVAILABLE = True
except ImportError as e:
    _LG_IMPORT_ERROR = str(e)
    logger.warning("LightGlue not available: %s", e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_lightglue(
    view: MatchView,
    device: str = "auto",
    max_num_keypoints: int = 4096,
    min_confidence: float = 0.0,
) -> MatchSet:
    """Run SuperPoint + LightGlue matching on a :class:`MatchView`.

    Args:
        view: Preprocessed overlap view.
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        max_num_keypoints: SuperPoint keypoint limit.
        min_confidence: Minimum LightGlue match confidence (per-match score).

    Returns:
        A :class:`MatchSet` with coordinates in common-grid space.

    Raises:
        RuntimeError: If ``lightglue`` is not installed.
    """
    if not _LIGHTGLUE_AVAILABLE:
        raise RuntimeError(
            f"LightGlue is not available. Install with: "
            f"pip install git+https://github.com/cvg/LightGlue.git\n"
            f"Import error: {_LG_IMPORT_ERROR}"
        )

    t0 = time.perf_counter()

    # --- Device -------------------------------------------------------------
    _device = _resolve_device(device)

    # --- Initialise models --------------------------------------------------
    feature_t0 = time.perf_counter()
    extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(_device)

    # --- Prepare input tensors (1×1×H×W, float32 [0,1]) --------------------
    ref_t = torch.from_numpy(view.ref).float().unsqueeze(0).unsqueeze(0).to(_device)
    tgt_t = torch.from_numpy(view.tgt).float().unsqueeze(0).unsqueeze(0).to(_device)

    with torch.inference_mode():
        feats0 = extractor.extract(ref_t)
        feats1 = extractor.extract(tgt_t)
    feature_runtime = time.perf_counter() - feature_t0

    matcher_t0 = time.perf_counter()
    matcher = LightGlue(features="superpoint").eval().to(_device)
    with torch.inference_mode():
        matches01 = matcher({
            "image0": feats0,
            "image1": feats1,
        })
    matcher_runtime = time.perf_counter() - matcher_t0

    # Remove batch dimension — rbd each dict separately
    feats0 = rbd(feats0)
    feats1 = rbd(feats1)
    matches01 = rbd(matches01)

    # --- Extract matched keypoints and scores -------------------------------
    ref_xy_view, tgt_xy_view, conf_np, conf_source = _extract_lightglue_matches(
        feats0, feats1, matches01, min_confidence
    )

    # --- Valid-mask filtering ------------------------------------------------
    if len(ref_xy_view) > 0:
        ref_xy_view, tgt_xy_view, conf_np = _filter_by_valid_mask(
            ref_xy_view, tgt_xy_view, conf_np, view
        )

    # --- Map to common-grid coordinates ------------------------------------
    if len(ref_xy_view) == 0:
        elapsed = time.perf_counter() - t0
        return make_matchset_from_view(
            method="lightglue",
            view=view,
            ref_xy_view=np.empty((0, 2)),
            tgt_xy_view=np.empty((0, 2)),
            confidence=np.empty(0),
            runtime_sec=elapsed,
            metadata={
                "max_num_keypoints": max_num_keypoints,
                "min_confidence": min_confidence,
                "device": device,
                "confidence_source": conf_source,
                "keypoints_ref": len(feats0.get("keypoints", [])),
                "keypoints_tgt": len(feats1.get("keypoints", [])),
            },
            runtime_breakdown={
                "feature_runtime_sec": feature_runtime,
                "matcher_runtime_sec": matcher_runtime,
            },
        )

    elapsed = time.perf_counter() - t0

    return make_matchset_from_view(
        method="lightglue",
        view=view,
        ref_xy_view=ref_xy_view,
        tgt_xy_view=tgt_xy_view,
        confidence=conf_np,
        runtime_sec=elapsed,
        metadata={
            "max_num_keypoints": max_num_keypoints,
            "min_confidence": min_confidence,
            "device": device,
            "confidence_source": conf_source,
            "keypoints_ref": len(feats0.get("keypoints", [])),
            "keypoints_tgt": len(feats1.get("keypoints", [])),
        },
        runtime_breakdown={
            "feature_runtime_sec": feature_runtime,
            "matcher_runtime_sec": matcher_runtime,
        },
    )


def is_lightglue_available() -> bool:
    """Return ``True`` if the LightGlue package is importable."""
    return _LIGHTGLUE_AVAILABLE


# ---------------------------------------------------------------------------
# Match extraction (handles multiple LightGlue output formats)
# ---------------------------------------------------------------------------


def _extract_lightglue_matches(
    feats0: dict,
    feats1: dict,
    matches01: dict,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Parse LightGlue output into ``(ref_xy_view, tgt_xy_view, confidence, source_name)``.

    Supports both the current official ``matches`` shape ``(M, 2)`` and
    the legacy ``matches0`` shape ``(N0,)`` formats.
    """
    kpts0 = feats0["keypoints"]   # (N0, 2)  [x, y]
    kpts1 = feats1["keypoints"]   # (N1, 2)

    # Detect output format ---------------------------------------------------
    if "matches" in matches01 and matches01["matches"].ndim == 2:
        # Official format: matches shaped (M, 2) — index pairs
        pairs = matches01["matches"]  # (M, 2)  [ref_idx, tgt_idx]
        ref_idx = pairs[:, 0].long()
        tgt_idx = pairs[:, 1].long()

    elif "matches0" in matches01:
        # Legacy format: matches0 shaped (N0,) with -1 = unmatched
        matches0 = matches01["matches0"]
        valid = matches0 >= 0
        ref_idx = torch.where(valid)[0]
        tgt_idx = matches0[valid].long()

    else:
        # Fallback: try matches as 1-D indices
        m = matches01.get("matches")
        if m is not None and m.ndim == 1:
            valid = m >= 0
            ref_idx = torch.where(valid)[0]
            tgt_idx = m[valid].long()
        else:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0), "detection_failed"

    # Extract coordinates -----------------------------------------------------
    ref_xy_view = kpts0[ref_idx].cpu().numpy()
    tgt_xy_view = kpts1[tgt_idx].cpu().numpy()

    # Confidence --------------------------------------------------------------
    conf = None
    conf_source = "fallback_ones"

    for key in ("scores", "matching_scores0", "scores0"):
        if key in matches01 :
            s = matches01[key]
            if s is not None:
                conf = s[ref_idx] if s.shape[0] == len(kpts0) else s
                conf_source = key
                break

    if conf is None:
        conf = torch.ones(len(ref_idx))
        conf_source = "fallback_ones"
    else:
        conf = conf.float()

    conf_np = conf.cpu().numpy()

    # Filter by min_confidence ------------------------------------------------
    if min_confidence > 0 and len(ref_xy_view) > 0:
        keep = conf_np >= min_confidence
        ref_xy_view = ref_xy_view[keep]
        tgt_xy_view = tgt_xy_view[keep]
        conf_np = conf_np[keep]

    return ref_xy_view, tgt_xy_view, conf_np, conf_source


# ---------------------------------------------------------------------------
# Valid-mask filter
# ---------------------------------------------------------------------------


def _filter_by_valid_mask(
    ref_xy_view: np.ndarray,
    tgt_xy_view: np.ndarray,
    confidence: np.ndarray,
    view: MatchView,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discard points that fall outside the valid-pixel masks."""
    h, w = view.ref_valid.shape
    keep = np.ones(len(ref_xy_view), dtype=bool)

    for i in range(len(ref_xy_view)):
        rx, ry = int(round(ref_xy_view[i, 0])), int(round(ref_xy_view[i, 1]))
        tx, ty = int(round(tgt_xy_view[i, 0])), int(round(tgt_xy_view[i, 1]))
        if 0 <= ry < h and 0 <= rx < w:
            if not view.ref_valid[ry, rx]:
                keep[i] = False
        if 0 <= ty < h and 0 <= tx < w:
            if not view.tgt_valid[ty, tx]:
                keep[i] = False

    return ref_xy_view[keep], tgt_xy_view[keep], confidence[keep]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_device(request: str) -> str:
    """Resolve a device request string to a concrete torch device."""
    if request == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return request
