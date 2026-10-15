"""Thin, truth-free wrappers for the four in-repository local matchers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from registration_benchmark.models import MatchView

from .coordinates import NativeMatches, to_native_correspondences
from .inputs import PairInput
from .protocol import Phase5Config


_METHODS = {"SIFT", "LoFTR", "EfficientLoFTR", "LightGlue+DISK"}
@dataclass(frozen=True)
class AdapterResult:
    status: str
    method: str
    matches: NativeMatches | None
    matcher_runtime_sec: float
    feature_runtime_sec: float
    model_init_runtime_sec: float
    reason: str | None
    provenance: dict[str, Any]


# Lazy imports let callers receive an explicit ENV_UNAVAILABLE result when an
# optional matcher dependency is missing, instead of failing module import.
def match_sift(*args, **kwargs):
    from registration_benchmark.matchers.sift import match_sift as implementation
    return implementation(*args, **kwargs)


def match_loftr(*args, **kwargs):
    from registration_benchmark.matchers.loftr import match_loftr as implementation
    return implementation(*args, **kwargs)


def match_efficient_loftr(*args, **kwargs):
    from registration_benchmark.matchers.efficient_loftr import match_efficient_loftr as implementation
    return implementation(*args, **kwargs)


def match_lightglue_disk(*args, **kwargs):
    from registration_benchmark.matchers.lightglue_disk import match_lightglue_disk as implementation
    return implementation(*args, **kwargs)


def _method_settings(config: Phase5Config, method: str) -> dict[str, Any]:
    return {key: value for name, items in config.method_settings if name == method
            for key, value in items}


def _legacy_view(pair: PairInput) -> MatchView:
    """Adapt Phase5's one shared grid without an extra crop or resize."""
    view = pair.view
    return MatchView(ref=view.left, tgt=view.right,
                     ref_valid=view.mask_i, tgt_valid=view.mask_j,
                     origin_x=0.0, origin_y=0.0, scale_x=1.0, scale_y=1.0)


def _call(method: str, view: MatchView, settings: dict[str, Any],
          efficient_artifact: dict[str, Any] | None = None):
    if method == "SIFT":
        return match_sift(view, nfeatures=settings["nfeatures"],
                          ratio_threshold=settings["ratio"])
    if method == "LoFTR":
        return match_loftr(view, device="auto", confidence_threshold=settings["confidence"],
                           max_matches=settings["max_matches"])
    if method == "EfficientLoFTR":
        return match_efficient_loftr(view, device="auto",
                                     confidence_threshold=settings["confidence"],
                                     repo_dir=efficient_artifact["repo_dir"],
                                     weights_path=efficient_artifact["weights_path"],
                                     model_type=settings["resolution"],
                                     precision=settings["precision"])
    if method == "LightGlue+DISK":
        return match_lightglue_disk(view, device="auto",
                                    max_num_keypoints=settings["features"],
                                    disk_resize=settings["extractor_resize"],
                                    min_confidence=settings["confidence"])
    raise ValueError(f"unsupported local method: {method}")


def _failure_status(method: str, exc: Exception) -> str:
    message = str(exc)
    lower = message.lower()
    if isinstance(exc, ImportError):
        return "ENV_UNAVAILABLE"
    if method == "LoFTR" and "not available" in lower:
        return "ENV_UNAVAILABLE"
    if method == "LightGlue+DISK" and "lightglue_disk_unavailable" in lower:
        if any(term in lower for term in ("no module named", "missing torch", "not available",
                                          "importerror")):
            return "ENV_UNAVAILABLE"
        return "ALGORITHM_FAIL"
    if method == "EfficientLoFTR" and "efficient_loftr_unavailable" in lower:
        if "overlap view must be at least" in lower:
            return "ALGORITHM_FAIL"
        if any(term in lower for term in ("no module", "not found", "set efficient_loftr")):
            return "ENV_UNAVAILABLE"
    return "ALGORITHM_FAIL"


def _verified_efficient_artifact(artifact: dict[str, Any] | None) -> tuple[dict[str, Any] | None, AdapterResult | None]:
    if (not isinstance(artifact, dict)
            or set(artifact) != {"repo_dir", "weights_path", "sha256"}):
        return None, AdapterResult("WEIGHTS_UNAVAILABLE", "EfficientLoFTR", None,
                                   0.0, 0.0, 0.0, "MISSING_FROZEN_WEIGHTS_MANIFEST",
                                   {"matcher_invoked": False})
    try:
        repo = Path(artifact["repo_dir"]).expanduser().resolve()
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        return None, AdapterResult("ENV_UNAVAILABLE", "EfficientLoFTR", None,
                                   0.0, 0.0, 0.0,
                                   f"INVALID_MODEL_REPOSITORY_PATH: {exc}",
                                   {"matcher_invoked": False})
    try:
        weights = Path(artifact["weights_path"]).expanduser().resolve()
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        return None, AdapterResult("WEIGHTS_UNAVAILABLE", "EfficientLoFTR", None,
                                   0.0, 0.0, 0.0,
                                   f"INVALID_CHECKPOINT_PATH: {exc}",
                                   {"matcher_invoked": False})
    if not repo.is_dir():
        return None, AdapterResult("ENV_UNAVAILABLE", "EfficientLoFTR", None,
                                   0.0, 0.0, 0.0, f"MODEL_REPOSITORY_MISSING: {repo}",
                                   {"matcher_invoked": False})
    if not weights.is_file():
        return None, AdapterResult("WEIGHTS_UNAVAILABLE", "EfficientLoFTR", None,
                                   0.0, 0.0, 0.0, f"CHECKPOINT_MISSING: {weights}",
                                   {"matcher_invoked": False})
    try:
        with weights.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        return None, AdapterResult("WEIGHTS_UNAVAILABLE", "EfficientLoFTR", None,
                                   0.0, 0.0, 0.0,
                                   f"CHECKPOINT_UNREADABLE: {exc}",
                                   {"matcher_invoked": False})
    expected = str(artifact["sha256"]).lower()
    if len(expected) != 64 or actual != expected:
        return None, AdapterResult("WEIGHTS_UNAVAILABLE", "EfficientLoFTR", None,
                                   0.0, 0.0, 0.0, "CHECKPOINT_SHA256_MISMATCH",
                                   {"matcher_invoked": False, "actual_sha256": actual})
    return {"repo_dir": repo, "weights_path": weights, "sha256": actual}, None


def run_local_matcher(method: str, pair: PairInput, config: Phase5Config,
                      *, efficient_artifact: dict[str, Any] | None = None) -> AdapterResult:
    """Run one declared local matcher on the unchanged common Phase5 view.

    The interface has no truth/checkpoint argument. Matchers see only the
    shared B9 pixels and masks. Legacy MatchSet coordinates are tagged with
    the exact identity view mapping used here before conversion to native rc.
    """
    if method not in _METHODS:
        raise ValueError(f"unsupported local method: {method}")
    if pair.status != "READY" or pair.view is None:
        return AdapterResult(pair.status, method, None, 0.0, 0.0, 0.0,
                             pair.reason, {"matcher_invoked": False})
    if method not in config.method_ids:
        raise ValueError(f"method is outside frozen protocol: {method}")
    settings = _method_settings(config, method)
    artifact = None
    if method == "EfficientLoFTR":
        artifact, unavailable = _verified_efficient_artifact(efficient_artifact)
        if unavailable is not None:
            return unavailable
    try:
        result = _call(method, _legacy_view(pair), settings, artifact)
        # Existing adapters use MatchView.to_common_grid(). For this wrapper
        # the legacy view is exactly the Phase5 view (origin 0, scale 1).
        result.metadata["phase5_common_grid"] = "overlap_grid"
        result.metadata["legacy_view_origin_xy"] = [0.0, 0.0]
        result.metadata["legacy_view_scale_xy"] = [1.0, 1.0]
        native = to_native_correspondences(result, pair)
        breakdown = result.metadata.get("runtime_breakdown", {})
        return AdapterResult("SUCCESS", method, native,
                             float(breakdown.get("matcher_runtime_sec", result.runtime_sec)),
                             float(breakdown.get("feature_runtime_sec", 0.0)),
                             float(breakdown.get("model_init_runtime_sec", 0.0)),
                             None,
                             {"settings": settings,
                              "coordinate_frame_in": "pair_common_grid",
                              "coordinate_frame_out": "native_pixel_center_row_col",
                              "legacy_view_origin_xy": [0.0, 0.0],
                              "legacy_view_scale_xy": [1.0, 1.0],
                              "weights_sha256": artifact["sha256"] if artifact else None,
                              "matcher_metadata": dict(result.metadata)})
    except Exception as exc:
        return AdapterResult(_failure_status(method, exc), method, None,
                             0.0, 0.0, 0.0, f"{type(exc).__name__}: {exc}",
                             {"settings": settings, "matcher_invoked": True})
