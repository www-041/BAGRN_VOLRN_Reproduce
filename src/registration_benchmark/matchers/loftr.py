"""LoFTR semi-dense matcher adapter.

Uses Kornia's implementation with ``pretrained="outdoor"``.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from src.registration_benchmark.models import MatchSet, MatchView

logger = logging.getLogger(__name__)

_LOFTR_AVAILABLE = False
_LOFTR_IMPORT_ERROR: str | None = None

try:
    import torch
    import kornia.feature as KF

    _LOFTR_AVAILABLE = True
except ImportError as e:
    _LOFTR_IMPORT_ERROR = str(e)
    logger.warning("LoFTR not available: %s", e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_loftr(
    view: MatchView,
    device: str = "auto",
    confidence_threshold: float = 0.2,
    max_matches: int = 5000,
) -> MatchSet:
    """Run LoFTR (outdoor) on a :class:`MatchView`.

    Args:
        view: Preprocessed overlap view (float32, [0,1]).
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        confidence_threshold: Minimum per-match confidence.
        max_matches: Cap on the number of returned matches (avoids slow
            RANSAC / plotting).

    Returns:
        A :class:`MatchSet` with coordinates in common-grid space.

    Raises:
        RuntimeError: If kornia/torch is not installed.
    """
    if not _LOFTR_AVAILABLE:
        raise RuntimeError(
            f"LoFTR (kornia) is not available. "
            f"Import error: {_LOFTR_IMPORT_ERROR}"
        )

    t0 = time.perf_counter()

    _device = _resolve_device(device)

    # --- Model ---------------------------------------------------------------
    matcher = KF.LoFTR(pretrained="outdoor").eval().to(_device)

    # --- Prepare input tensors (1×1×H×W, float32) ---------------------------
    ref_t = torch.from_numpy(view.ref).float().unsqueeze(0).unsqueeze(0).to(_device)
    tgt_t = torch.from_numpy(view.tgt).float().unsqueeze(0).unsqueeze(0).to(_device)

    # --- Inference -----------------------------------------------------------
    with torch.inference_mode():
        # Kornia LoFTR input format: {"image0": ..., "image1": ...}
        output = matcher({"image0": ref_t, "image1": tgt_t})

    # --- Parse output ---------------------------------------------------------
    # kornia LoFTR returns:
    #   keypoints0: (N, 2)  [x, y] in image0 pixel space
    #   keypoints1: (N, 2)  [x, y] in image1 pixel space
    #   confidence: (N,)    match confidence
    kpts0 = output["keypoints0"].cpu().numpy()  # ref
    kpts1 = output["keypoints1"].cpu().numpy()  # tgt
    conf = output["confidence"].cpu().numpy()

    # Filter by confidence threshold
    keep = conf >= confidence_threshold
    kpts0 = kpts0[keep]
    kpts1 = kpts1[keep]
    conf = conf[keep]

    # Cap at max_matches (sort descending by confidence)
    if len(conf) > max_matches:
        idx = np.argsort(conf)[::-1][:max_matches]
        kpts0 = kpts0[idx]
        kpts1 = kpts1[idx]
        conf = conf[idx]

    elapsed = time.perf_counter() - t0

    if len(kpts0) == 0:
        return MatchSet(
            method="loftr",
            ref_xy=np.empty((0, 2)),
            tgt_xy=np.empty((0, 2)),
            confidence=np.empty(0),
            runtime_sec=elapsed,
            metadata={
                "confidence_threshold": confidence_threshold,
                "max_matches": max_matches,
                "device": device,
                "pretrained": "outdoor",
            },
        )

    ref_xy = view.to_canvas(kpts0)
    tgt_xy = view.to_canvas(kpts1)

    return MatchSet(
        method="loftr",
        ref_xy=ref_xy,
        tgt_xy=tgt_xy,
        confidence=conf,
        runtime_sec=elapsed,
        metadata={
            "confidence_threshold": confidence_threshold,
            "max_matches": max_matches,
            "device": device,
            "pretrained": "outdoor",
        },
    )


def is_loftr_available() -> bool:
    """Return ``True`` if kornia LoFTR is importable."""
    return _LOFTR_AVAILABLE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_device(request: str) -> str:
    if request == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return request