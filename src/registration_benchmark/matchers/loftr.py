"""LoFTR semi-dense matcher adapter.

Uses Kornia's implementation with ``pretrained="outdoor"``.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from src.registration_benchmark.models import (
    MatchView,
    filter_matches_by_valid_mask,
    make_matchset_from_view,
)

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
    *,
    _matcher=None,
    _model_init_runtime_sec: float = 0.0,
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
    model_init_runtime_sec = float(_model_init_runtime_sec)
    if _matcher is None:
        model_t0 = time.perf_counter()
        _matcher = KF.LoFTR(pretrained="outdoor").eval().to(_device)
        model_init_runtime_sec = time.perf_counter() - model_t0

    pair_inference_t0 = time.perf_counter()

    # --- Prepare input tensors (1×1×H×W, float32) ---------------------------
    ref_t = torch.from_numpy(view.ref).float().unsqueeze(0).unsqueeze(0).to(_device)
    tgt_t = torch.from_numpy(view.tgt).float().unsqueeze(0).unsqueeze(0).to(_device)

    # --- Inference -----------------------------------------------------------
    with torch.inference_mode():
        # Kornia LoFTR input format: {"image0": ..., "image1": ...}
        output = _matcher({"image0": ref_t, "image1": tgt_t})

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

    kpts0, kpts1, conf, valid_keep = filter_matches_by_valid_mask(
        kpts0, kpts1, conf, view
    )
    pair_inference_runtime_sec = time.perf_counter() - pair_inference_t0

    elapsed = time.perf_counter() - t0

    if len(kpts0) == 0:
        return make_matchset_from_view(
            method="loftr",
            view=view,
            ref_xy_view=np.empty((0, 2)),
            tgt_xy_view=np.empty((0, 2)),
            confidence=np.empty(0),
            runtime_sec=elapsed,
            runtime_breakdown={
                "feature_runtime_sec": 0.0,
                "matcher_runtime_sec": pair_inference_runtime_sec,
                "model_init_runtime_sec": model_init_runtime_sec,
                "pair_inference_runtime_sec": pair_inference_runtime_sec,
            },
            metadata={
                "confidence_threshold": confidence_threshold,
                "max_matches": max_matches,
                "valid_mask_filtered": int((~valid_keep).sum()),
                "device": device,
                "pretrained": "outdoor",
            },
        )

    return make_matchset_from_view(
        method="loftr",
        view=view,
        ref_xy_view=kpts0,
        tgt_xy_view=kpts1,
        confidence=conf,
        runtime_sec=elapsed,
        runtime_breakdown={
            "feature_runtime_sec": 0.0,
            "matcher_runtime_sec": pair_inference_runtime_sec,
            "model_init_runtime_sec": model_init_runtime_sec,
            "pair_inference_runtime_sec": pair_inference_runtime_sec,
        },
        metadata={
            "confidence_threshold": confidence_threshold,
            "max_matches": max_matches,
            "valid_mask_filtered": int((~valid_keep).sum()),
            "device": device,
            "pretrained": "outdoor",
        },
    )


def is_loftr_available() -> bool:
    """Return ``True`` if kornia LoFTR is importable."""
    return _LOFTR_AVAILABLE


class LoFTRMatcherSession:
    """Reuse one LoFTR model across all pairs in a matcher run."""

    def __init__(self, device: str = "auto") -> None:
        if not _LOFTR_AVAILABLE:
            raise RuntimeError(
                f"LoFTR (kornia) is not available. Import error: {_LOFTR_IMPORT_ERROR}"
            )
        self.device = _resolve_device(device)
        t0 = time.perf_counter()
        self.matcher = KF.LoFTR(pretrained="outdoor").eval().to(self.device)
        self.model_init_runtime_sec = time.perf_counter() - t0
        self._pending_model_init_runtime_sec = self.model_init_runtime_sec
        self._closed = False

    def match(self, view: MatchView) -> MatchSet:
        if self._closed:
            raise RuntimeError("LoFTR matcher session is closed")
        init_runtime = self._pending_model_init_runtime_sec
        self._pending_model_init_runtime_sec = 0.0
        return match_loftr(
            view,
            device=self.device,
            _matcher=self.matcher,
            _model_init_runtime_sec=init_runtime,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.matcher = None
        try:
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (NameError, RuntimeError):
            pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_device(request: str) -> str:
    if request == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return request
