"""LightGlue + DISK adapter for the frozen fourth comparator.

The official LightGlue DISK extractor internally resizes to an edge length of
1024 by default, then maps its keypoints back to the extractor input frame.
This adapter therefore does not apply a second resize inverse: the keypoints
returned by ``DISK.extract`` are already in ``MatchView`` pixel coordinates.
Only the canonical ``MatchView.to_common_grid`` conversion remains.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from src.registration_benchmark.models import MatchView, make_matchset_from_view

logger = logging.getLogger(__name__)

LIGHTGLUE_DISK_UNAVAILABLE = "LIGHTGLUE_DISK_UNAVAILABLE"
OFFICIAL_REPOSITORY = "https://github.com/cvg/LightGlue"
DISK_INTERNAL_RESIZE = 1024

_LIGHTGLUE_DISK_AVAILABLE = False
_LIGHTGLUE_DISK_IMPORT_ERROR: str | None = None

try:
    import torch
    from lightglue import DISK, LightGlue
    from lightglue.utils import rbd

    _LIGHTGLUE_DISK_AVAILABLE = True
except ImportError as exc:
    _LIGHTGLUE_DISK_IMPORT_ERROR = str(exc)
    logger.warning("LightGlue + DISK not available: %s", exc)


def is_lightglue_disk_available() -> bool:
    """Return whether the official LightGlue + DISK API can be imported."""
    return _LIGHTGLUE_DISK_AVAILABLE


def match_lightglue_disk(
    view: MatchView,
    device: str = "auto",
    max_num_keypoints: int = 2048,
    min_confidence: float = 0.0,
    disk_resize: int = DISK_INTERNAL_RESIZE,
) :
    """Run official DISK feature extraction followed by LightGlue matching."""
    if not _LIGHTGLUE_DISK_AVAILABLE:
        raise RuntimeError(
            f"{LIGHTGLUE_DISK_UNAVAILABLE}: {_LIGHTGLUE_DISK_IMPORT_ERROR}"
        )
    if disk_resize <= 0:
        raise ValueError("disk_resize must be positive")

    t0 = time.perf_counter()
    resolved_device = _resolve_device(device)

    try:
        feature_t0 = time.perf_counter()
        extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(resolved_device)
        ref_t = torch.from_numpy(view.ref).float().unsqueeze(0).unsqueeze(0).to(resolved_device)
        tgt_t = torch.from_numpy(view.tgt).float().unsqueeze(0).unsqueeze(0).to(resolved_device)
        with torch.inference_mode():
            feats0 = extractor.extract(ref_t, resize=disk_resize)
            feats1 = extractor.extract(tgt_t, resize=disk_resize)
        feature_runtime = time.perf_counter() - feature_t0

        matcher_t0 = time.perf_counter()
        matcher = LightGlue(features="disk").eval().to(resolved_device)
        with torch.inference_mode():
            matches01 = matcher({"image0": feats0, "image1": feats1})
        matcher_runtime = time.perf_counter() - matcher_t0

        feats0, feats1, matches01 = [
            rbd(item) for item in (feats0, feats1, matches01)
        ]
        ref_xy_view, tgt_xy_view, confidence = _extract_matches(
            feats0, feats1, matches01, min_confidence
        )
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith(LIGHTGLUE_DISK_UNAVAILABLE):
            raise
        raise RuntimeError(f"{LIGHTGLUE_DISK_UNAVAILABLE}: {exc}") from exc

    keep = _valid_match_mask(ref_xy_view, tgt_xy_view, view)
    ref_xy_view = ref_xy_view[keep]
    tgt_xy_view = tgt_xy_view[keep]
    confidence = confidence[keep]
    elapsed = time.perf_counter() - t0

    return make_matchset_from_view(
        method="lightglue_disk",
        view=view,
        ref_xy_view=ref_xy_view,
        tgt_xy_view=tgt_xy_view,
        confidence=confidence,
        runtime_sec=elapsed,
        metadata={
            "official_repository": OFFICIAL_REPOSITORY,
            "feature_extractor": "DISK",
            "matcher": "LightGlue",
            "max_num_keypoints": max_num_keypoints,
            "min_confidence": min_confidence,
            "confidence_source": "lightglue_matching_scores",
            "confidence_semantics": "method_internal_only",
            "output_coordinate_frame": "disk_extractor_input",
            "disk_internal_resize": disk_resize,
            "disk_extractor_inverse_resize_applied": True,
            "pair_common_grid_mapping_applied": True,
            "device": resolved_device,
        },
        runtime_breakdown={
            "feature_runtime_sec": feature_runtime,
            "matcher_runtime_sec": matcher_runtime,
        },
    )


def _extract_matches(
    feats0: dict,
    feats1: dict,
    matches01: dict,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keypoints0 = _to_numpy(feats0["keypoints"])
    keypoints1 = _to_numpy(feats1["keypoints"])
    matches = matches01.get("matches")
    scores = matches01.get("scores")

    if matches is None:
        matches0 = matches01.get("matches0")
        if matches0 is None:
            raise RuntimeError(f"{LIGHTGLUE_DISK_UNAVAILABLE}: missing LightGlue matches")
        matches0 = _to_numpy(matches0).astype(np.int64)
        ref_idx = np.where(matches0 >= 0)[0]
        tgt_idx = matches0[ref_idx]
    else:
        pairs = _to_numpy(matches).astype(np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise RuntimeError(
                f"{LIGHTGLUE_DISK_UNAVAILABLE}: invalid matches shape {pairs.shape}"
            )
        ref_idx, tgt_idx = pairs[:, 0], pairs[:, 1]

    confidence = _to_numpy(scores).reshape(-1) if scores is not None else np.ones(len(ref_idx))
    if confidence.shape != (len(ref_idx),):
        raise RuntimeError(
            f"{LIGHTGLUE_DISK_UNAVAILABLE}: scores shape {confidence.shape} "
            f"does not match {len(ref_idx)} matches"
        )
    if min_confidence > 0.0:
        selected = confidence >= min_confidence
        ref_idx, tgt_idx, confidence = ref_idx[selected], tgt_idx[selected], confidence[selected]

    return keypoints0[ref_idx], keypoints1[tgt_idx], confidence


def _valid_match_mask(ref_xy: np.ndarray, tgt_xy: np.ndarray, view: MatchView) -> np.ndarray:
    keep = np.ones(len(ref_xy), dtype=bool)
    for i, (ref_point, tgt_point) in enumerate(zip(ref_xy, tgt_xy)):
        for point, valid in ((ref_point, view.ref_valid), (tgt_point, view.tgt_valid)):
            x, y = int(round(point[0])), int(round(point[1]))
            if not (0 <= y < valid.shape[0] and 0 <= x < valid.shape[1] and valid[y, x]):
                keep[i] = False
                break
    return keep


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _resolve_device(request: str) -> str:
    if request == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return request
