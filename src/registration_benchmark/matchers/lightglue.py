"""SuperPoint + LightGlue matcher adapter.

Uses the official ``lightglue`` package.  Falls back gracefully if the
package is not installed.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from src.registration_benchmark.models import MatchSet, MatchView

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
    extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(_device)
    matcher = LightGlue(features="superpoint").eval().to(_device)

    # --- Prepare input tensors (1×1×H×W, float32 [0,1]) --------------------
    ref_t = torch.from_numpy(view.ref).float().unsqueeze(0).unsqueeze(0).to(_device)
    tgt_t = torch.from_numpy(view.tgt).float().unsqueeze(0).unsqueeze(0).to(_device)

    # --- Feature extraction & matching --------------------------------------
    with torch.inference_mode():
        feats0 = extractor.extract({"image": ref_t})
        feats1 = extractor.extract({"image": tgt_t})

        matches01 = matcher({
            "image0": feats0,
            "image1": feats1,
        })

    # Remove batch dimension
    feats0, feats1, matches01 = rbd(
        {**feats0, **feats1, **matches01}
    )

    # --- Extract matched keypoints and scores -------------------------------
    kpts0 = feats0["keypoints"]  # (N0, 2)  x, y in view pixel space
    kpts1 = feats1["keypoints"]  # (N1, 2)
    m = matches01["matches"]     # (M,)
    scores = matches01.get("scores", torch.ones(len(m)))

    # Filter by confidence
    valid = m >= 0
    if len(valid) > 0:
        valid = valid & (scores >= min_confidence)

    match_indices = m[valid]
    conf = scores[valid]

    ref_xy_view = kpts0[match_indices].cpu().numpy()  # (M', 2)
    tgt_xy_view = kpts1[valid].cpu().numpy()

    # Apply confidence/reorder
    conf_np = conf.cpu().numpy()

    # --- Map to common-grid coordinates ------------------------------------
    if len(ref_xy_view) == 0:
        elapsed = time.perf_counter() - t0
        return MatchSet(
            method="lightglue",
            ref_xy=np.empty((0, 2)),
            tgt_xy=np.empty((0, 2)),
            confidence=np.empty(0),
            runtime_sec=elapsed,
            metadata={
                "max_num_keypoints": max_num_keypoints,
                "min_confidence": min_confidence,
                "device": device,
                "keypoints_ref": len(kpts0),
                "keypoints_tgt": len(kpts1),
            },
        )

    ref_xy = view.to_canvas(ref_xy_view)
    tgt_xy = view.to_canvas(tgt_xy_view)

    elapsed = time.perf_counter() - t0

    return MatchSet(
        method="lightglue",
        ref_xy=ref_xy,
        tgt_xy=tgt_xy,
        confidence=conf_np,
        runtime_sec=elapsed,
        metadata={
            "max_num_keypoints": max_num_keypoints,
            "min_confidence": min_confidence,
            "device": device,
            "keypoints_ref": len(kpts0),
            "keypoints_tgt": len(kpts1),
        },
    )


def is_lightglue_available() -> bool:
    """Return ``True`` if the LightGlue package is importable."""
    return _LIGHTGLUE_AVAILABLE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_device(request: str) -> str:
    """Resolve a device request string to a concrete torch device."""
    if request == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return request