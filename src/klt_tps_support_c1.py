"""Pure protocol and comparison helpers for the TPS-SUPPORT-C1 diagnostic."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.klt_tps_registration import (
    analyze_tps_dense_flow,
    build_inside_hull_taper_weight,
    build_translation_flow,
    compose_supported_tps_flow,
    compute_klt_translation_continuation,
    inspect_tps_dense_flow,
)


TPS_SUPPORT_C1_TAPER_PIXELS: int = 64


def _value(source: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a mapping-based or object-based config."""
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _scene_ids(config: Any) -> list[str]:
    configured = _value(config, "scene_ids")
    if configured is not None:
        return [str(scene_id) for scene_id in configured]
    scenes = _value(config, "scenes", []) or []
    result = []
    for scene in scenes:
        scene_id = _value(scene, "id")
        if scene_id is not None:
            result.append(str(scene_id))
    return result


def validate_tps_support_c1_protocol(
    config,
    holdout_manifest: dict,
    *,
    validation_band: str = "B14",
) -> dict[str, Any]:
    """Validate TPS-SUPPORT-C1 without opening imagery."""
    errors: list[str] = []
    registration_band = _value(config, "registration_band")
    selected_bands = list(_value(config, "selected_bands", []) or [])
    registration_params = _value(config, "registration_params", {}) or {}
    backend = _value(registration_params, "registration_backend")

    if registration_band != "B12":
        errors.append("TPS-SUPPORT-C1 requires registration band B12")
    if backend != "klt_tps":
        errors.append("TPS-SUPPORT-C1 requires the klt_tps registration backend")
    if selected_bands != ["B12", "B14"]:
        errors.append("TPS-SUPPORT-C1 requires selected_bands ['B12', 'B14']")
    if validation_band != "B14":
        errors.append("TPS-SUPPORT-C1 requires validation band B14")

    manifest = holdout_manifest if isinstance(holdout_manifest, Mapping) else {}
    if manifest.get("validation_band") != "B14":
        errors.append("HOLDOUT manifest validation band must be B14")
    if manifest.get("source_registration_band") != "B14":
        errors.append("HOLDOUT manifest source registration band must be B14")

    manifest_ids = [str(scene_id) for scene_id in (manifest.get("scene_ids") or [])]
    configured_ids = _scene_ids(config)[:2]
    if manifest_ids != configured_ids:
        errors.append("HOLDOUT manifest scene IDs do not match the first two config scenes")

    pairs = manifest.get("pairs") or {}
    pair = pairs.get("0-1")
    if not isinstance(pair, Mapping):
        errors.append("HOLDOUT manifest must contain pair 0-1")
        pair = {}

    selected_block_size = pair.get("selected_block_size")
    if selected_block_size != 384:
        errors.append("TPS-SUPPORT-C1 requires selected HOLDOUT block size 384")
    windows = pair.get("reserved_windows") or []
    if len(windows) != 7:
        errors.append("TPS-SUPPORT-C1 requires exactly seven reserved windows")
    for index, window in enumerate(windows):
        if not isinstance(window, Mapping):
            errors.append(f"reserved window {index} is not an object")
            continue
        if window.get("height") != 384 or window.get("width") != 384:
            errors.append(f"reserved window {index} must be 384 x 384")

    return {
        "valid": not errors,
        "errors": errors,
        "registration_band": registration_band,
        "validation_band": validation_band,
        "scene_ids": manifest_ids,
        "reserved_count": len(windows),
        "selected_block_size": selected_block_size,
        "taper_pixels": TPS_SUPPORT_C1_TAPER_PIXELS,
    }


def array_sha256(array: np.ndarray) -> str:
    """Fingerprint dtype, shape, and raw contiguous array bytes."""
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def validation_block_keys(validation: dict) -> list[tuple[int, int, int, int, int]]:
    """Return sorted fixed-window keys regardless of acceptance status."""
    validation = validation or {}
    top_level_size = validation.get("validation_block_size_selected")
    keys: set[tuple[int, int, int, int, int]] = set()
    for edge in validation.get("edges", []) or []:
        try:
            idx_i = int(edge["idx_i"])
            idx_j = int(edge["idx_j"])
        except (KeyError, TypeError, ValueError):
            continue
        edge_size = edge.get("validation_block_size_selected", top_level_size)
        for block in edge.get("blocks", []) or []:
            try:
                row = int(block["validation_row"])
                col = int(block["validation_col"])
                size = block.get("block_size", edge_size)
                if size is None:
                    size = 192
                keys.add((idx_i, idx_j, row, col, int(size)))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(keys)


def _geometry_summary(flow: np.ndarray, analysis: dict[str, Any], max_shift: float):
    """Return compact field statistics and catch only safety-gate rejection."""
    summary = analysis["summary"]
    try:
        gate = inspect_tps_dense_flow(flow, max_shift)
    except ValueError as exc:
        return {
            "geometry_safe": False,
            "rejection_reason": str(exc),
            "fold_pixels": int(summary["fold_pixels"]),
            "max_displacement_pixels": float(summary["displacement_max"]),
            "jacobian_min": float(summary["jacobian_min"]),
            "jacobian_max": float(summary["jacobian_max"]),
        }
    return {
        "geometry_safe": True,
        "rejection_reason": None,
        "fold_pixels": int(gate["fold_pixels"]),
        "max_displacement_pixels": float(gate["max_displacement_pixels"]),
        "jacobian_min": float(gate["jacobian_min"]),
        "jacobian_max": float(gate["jacobian_max"]),
    }


def summarize_tps_support_field_integrity(
    fields: dict[str, Any],
) -> dict[str, Any]:
    """Check the causal support formula using the private in-process arrays."""
    arrays = fields.get("_arrays", {})
    raw_flow = np.asarray(arrays["raw_flow"])
    translation_flow = np.asarray(arrays["translation_flow"])
    support_weight = np.asarray(arrays["support_weight"])
    supported_flow = np.asarray(arrays["supported_flow"])
    hull_mask = np.asarray(arrays["support_hull_mask"], dtype=bool)
    deep_inside_mask = np.asarray(arrays["deep_inside_mask"], dtype=bool)

    if (
        raw_flow.shape != translation_flow.shape
        or raw_flow.shape != supported_flow.shape
        or support_weight.shape != raw_flow.shape[:2]
        or hull_mask.shape != raw_flow.shape[:2]
        or deep_inside_mask.shape != raw_flow.shape[:2]
    ):
        raise ValueError("TPS support integrity arrays have mismatched shapes")

    outside = ~hull_mask
    nonzero_outside = int(np.count_nonzero(support_weight[outside] != 0.0))
    if np.any(outside):
        outside_diff = np.abs(supported_flow[outside] - translation_flow[outside])
        outside_max = float(np.max(outside_diff))
    else:
        outside_max = 0.0
    deep_count = int(np.count_nonzero(deep_inside_mask))
    if deep_count:
        deep_diff = np.abs(supported_flow[deep_inside_mask] - raw_flow[deep_inside_mask])
        deep_max: float | None = float(np.max(deep_diff))
    else:
        deep_max = None

    weight_min = float(np.min(support_weight))
    weight_max = float(np.max(support_weight))
    formula_pass = bool(
        weight_min >= 0.0
        and weight_max <= 1.0
        and nonzero_outside == 0
        and outside_max <= 1e-6
        and (deep_max is None or deep_max <= 1e-6)
    )
    return {
        "raw_flow_sha256": fields.get("raw_flow_sha256", array_sha256(raw_flow)),
        "controls_sha256": fields.get("controls_sha256"),
        "displacements_sha256": fields.get("displacements_sha256"),
        "weight_min": weight_min,
        "weight_max": weight_max,
        "outside_hull_nonzero_weight_pixels": nonzero_outside,
        "outside_max_abs_supported_minus_translation": outside_max,
        "deep_inside_pixel_count": deep_count,
        "deep_inside_max_abs_supported_minus_raw": deep_max,
        "support_formula_pass": formula_pass,
    }


def build_tps_support_c1_fields(
    raw_flow: np.ndarray,
    control_points_xy: np.ndarray,
    displacement_xy: np.ndarray,
    *,
    max_shift: float,
    taper_pixels: int = TPS_SUPPORT_C1_TAPER_PIXELS,
) -> dict[str, Any]:
    """Build raw, translation, and supported C1 fields from one raw TPS result."""
    raw = np.asarray(raw_flow)
    translation_xy = compute_klt_translation_continuation(displacement_xy)
    translation_flow = build_translation_flow(translation_xy, raw.shape[:2])
    support = build_inside_hull_taper_weight(
        control_points_xy, raw.shape[:2], taper_pixels,
    )
    supported_flow = compose_supported_tps_flow(
        raw, translation_xy, support["weight"],
    )

    raw_analysis = analyze_tps_dense_flow(raw)
    translation_analysis = analyze_tps_dense_flow(translation_flow)
    supported_analysis = analyze_tps_dense_flow(supported_flow)
    fields: dict[str, Any] = {
        "translation_xy": translation_xy.astype(float),
        "taper_pixels": int(taper_pixels),
        "raw_flow_sha256": array_sha256(raw),
        "controls_sha256": array_sha256(control_points_xy),
        "displacements_sha256": array_sha256(displacement_xy),
        "support": {
            "available": bool(np.any(support["hull_mask"])),
            "hull_pixel_count": int(np.count_nonzero(support["hull_mask"])),
            "deep_inside_pixel_count": int(np.count_nonzero(support["deep_inside_mask"])),
        },
        "raw_geometry": _geometry_summary(raw, raw_analysis, max_shift),
        "translation_geometry": _geometry_summary(
            translation_flow, translation_analysis, max_shift,
        ),
        "supported_geometry": _geometry_summary(
            supported_flow, supported_analysis, max_shift,
        ),
        "_arrays": {
            "raw_flow": raw,
            "translation_flow": translation_flow,
            "support_weight": support["weight"],
            "supported_flow": supported_flow,
            "supported_jacobian": supported_analysis["jacobian_determinant"],
            "supported_fold_mask": supported_analysis["fold_mask"],
            "support_hull_mask": support["hull_mask"],
            "deep_inside_mask": support["deep_inside_mask"],
        },
    }
    integrity = summarize_tps_support_field_integrity(fields)
    fields["integrity"] = integrity
    fields["support_formula_pass"] = integrity["support_formula_pass"]
    return fields
