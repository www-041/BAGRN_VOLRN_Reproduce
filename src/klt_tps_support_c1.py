"""Pure protocol and comparison helpers for the TPS-SUPPORT-C1 diagnostic."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from src.klt_tps_registration import (
    analyze_tps_dense_flow,
    build_inside_hull_taper_weight,
    build_translation_flow,
    compose_supported_tps_flow,
    compute_klt_translation_continuation,
    inspect_tps_dense_flow,
)


TPS_SUPPORT_C1_TAPER_PIXELS: int = 64
TPS_FOLD_D2_WEIGHT_EPS: float = 1e-6


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
            "support_distance_inside": support["distance_inside"],
        },
    }
    integrity = summarize_tps_support_field_integrity(fields)
    fields["integrity"] = integrity
    fields["support_formula_pass"] = integrity["support_formula_pass"]
    return fields


def _d2_flow_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError(f"{name} must have shape (H, W, 2)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _d2_spatial_array(
    name: str,
    value: np.ndarray,
    shape: tuple[int, int],
    *,
    finite: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _d2_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def diagnose_supported_fold_pixels(
    *,
    raw_flow: np.ndarray,
    translation_flow: np.ndarray,
    supported_flow: np.ndarray,
    support_weight: np.ndarray,
    support_hull_mask: np.ndarray,
    deep_inside_mask: np.ndarray,
    distance_inside: np.ndarray,
    supported_fold_mask: np.ndarray,
    supported_jacobian: np.ndarray,
    control_points_xy: np.ndarray,
    displacement_xy: np.ndarray,
    neighbor_count: int,
) -> dict[str, Any]:
    """Classify supported TPS folds using only already-produced diagnostics."""
    raw = _d2_flow_array("raw_flow", raw_flow)
    translation = _d2_flow_array("translation_flow", translation_flow)
    supported = _d2_flow_array("supported_flow", supported_flow)
    shape = raw.shape[:2]
    if translation.shape != raw.shape or supported.shape != raw.shape:
        raise ValueError("D2 flow arrays must have identical shapes")
    weight = _d2_spatial_array(
        "support_weight", support_weight, shape, finite=True,
    ).astype(float, copy=False)
    hull_mask = _d2_spatial_array(
        "support_hull_mask", support_hull_mask, shape,
    ).astype(bool, copy=False)
    deep_inside = _d2_spatial_array(
        "deep_inside_mask", deep_inside_mask, shape,
    ).astype(bool, copy=False)
    distance = _d2_spatial_array(
        "distance_inside", distance_inside, shape, finite=True,
    ).astype(float, copy=False)
    fold_mask = _d2_spatial_array(
        "supported_fold_mask", supported_fold_mask, shape,
    ).astype(bool, copy=False)
    supported_jacobian = _d2_spatial_array(
        "supported_jacobian", supported_jacobian, shape, finite=True,
    ).astype(float, copy=False)

    controls = np.asarray(control_points_xy, dtype=float)
    displacements = np.asarray(displacement_xy, dtype=float)
    if controls.ndim != 2 or controls.shape[1] != 2:
        raise ValueError("control_points_xy must have shape (N, 2)")
    if displacements.ndim != 2 or displacements.shape[1] != 2:
        raise ValueError("displacement_xy must have shape (N, 2)")
    if len(controls) != len(displacements):
        raise ValueError(
            "control_points_xy and displacement_xy must have equal counts"
        )
    if len(controls) == 0:
        raise ValueError("control_points_xy must contain at least one control")
    if not np.isfinite(controls).all():
        raise ValueError("control_points_xy must contain only finite values")
    if not np.isfinite(displacements).all():
        raise ValueError("displacement_xy must contain only finite values")
    try:
        neighbor_count = int(neighbor_count)
    except (TypeError, ValueError):
        raise ValueError("neighbor_count must be positive") from None
    if neighbor_count <= 0:
        raise ValueError("neighbor_count must be positive")

    raw_analysis = analyze_tps_dense_flow(raw)
    raw_jacobian = np.asarray(raw_analysis["jacobian_determinant"])
    tree = cKDTree(controls)
    local_count = min(neighbor_count, len(controls))
    fold_positions = np.argwhere(fold_mask)
    pixels = []
    classification_counts = {
        "outside_hull": 0,
        "taper": 0,
        "deep_inside": 0,
    }
    weight_class_counts = {"zero": 0, "partial": 0, "one": 0}
    for row, col in fold_positions:
        row = int(row)
        col = int(col)
        pixel_weight = float(weight[row, col])
        if not hull_mask[row, col]:
            region = "outside_hull"
        elif (
            deep_inside[row, col]
            or pixel_weight >= 1.0 - TPS_FOLD_D2_WEIGHT_EPS
        ):
            region = "deep_inside"
        else:
            region = "taper"
        if pixel_weight <= TPS_FOLD_D2_WEIGHT_EPS:
            weight_class = "zero"
        elif pixel_weight >= 1.0 - TPS_FOLD_D2_WEIGHT_EPS:
            weight_class = "one"
        else:
            weight_class = "partial"
        classification_counts[region] += 1
        weight_class_counts[weight_class] += 1

        query_xy = np.asarray([float(col), float(row)])
        nearest_distances, nearest_indices = tree.query(
            query_xy, k=local_count,
        )
        nearest_distances = np.asarray(nearest_distances, dtype=float).reshape(-1)
        nearest_indices = np.asarray(nearest_indices, dtype=int).reshape(-1)
        local_displacements = displacements[nearest_indices]
        local_median_xy = np.median(local_displacements, axis=0)
        vector_deviation = np.linalg.norm(
            local_displacements - local_median_xy, axis=1,
        )
        distance_summary = _d2_summary(nearest_distances)
        dx_summary = _d2_summary(local_displacements[:, 0])
        dy_summary = _d2_summary(local_displacements[:, 1])
        deviation_summary = _d2_summary(vector_deviation)
        nearest_index = int(nearest_indices[0])
        pixels.append({
            "row": row,
            "col": col,
            "region": region,
            "weight_class": weight_class,
            "support_weight": pixel_weight,
            "inside_hull": bool(hull_mask[row, col]),
            "distance_inside_pixels": float(distance[row, col]),
            "deep_inside": bool(deep_inside[row, col]),
            "raw_jacobian": float(raw_jacobian[row, col]),
            "supported_jacobian": float(supported_jacobian[row, col]),
            "raw_dx": float(raw[row, col, 0]),
            "raw_dy": float(raw[row, col, 1]),
            "translation_dx": float(translation[row, col, 0]),
            "translation_dy": float(translation[row, col, 1]),
            "supported_dx": float(supported[row, col, 0]),
            "supported_dy": float(supported[row, col, 1]),
            "nearest_control_index": nearest_index,
            "nearest_control_distance_pixels": float(nearest_distances[0]),
            "nearest_control_x": float(controls[nearest_index, 0]),
            "nearest_control_y": float(controls[nearest_index, 1]),
            "nearest_control_dx": float(displacements[nearest_index, 0]),
            "nearest_control_dy": float(displacements[nearest_index, 1]),
            "local_neighbor_count": int(local_count),
            "neighbor_distance_min": distance_summary["min"],
            "neighbor_distance_median": distance_summary["median"],
            "neighbor_distance_p95": distance_summary["p95"],
            "neighbor_distance_max": distance_summary["max"],
            "local_dx_min": dx_summary["min"],
            "local_dx_median": dx_summary["median"],
            "local_dx_max": dx_summary["max"],
            "local_dy_min": dy_summary["min"],
            "local_dy_median": dy_summary["median"],
            "local_dy_max": dy_summary["max"],
            "local_median_xy": [
                float(local_median_xy[0]), float(local_median_xy[1]),
            ],
            "local_vector_deviation_median": deviation_summary["median"],
            "local_vector_deviation_p95": deviation_summary["p95"],
            "local_vector_deviation_max": deviation_summary["max"],
        })

    labels, component_count = label(
        fold_mask, structure=np.ones((3, 3), dtype=int),
    )
    components = []
    for component_id in range(1, int(component_count) + 1):
        component_positions = np.argwhere(labels == component_id)
        component_pixels = [
            pixels[index] for index, (row, col) in enumerate(fold_positions)
            if labels[int(row), int(col)] == component_id
        ]
        components.append({
            "component_id": int(component_id),
            "pixel_count": int(len(component_positions)),
            "row_min": int(np.min(component_positions[:, 0])),
            "row_max": int(np.max(component_positions[:, 0])),
            "col_min": int(np.min(component_positions[:, 1])),
            "col_max": int(np.max(component_positions[:, 1])),
            "n_outside_hull": int(sum(
                pixel["region"] == "outside_hull" for pixel in component_pixels
            )),
            "n_taper": int(sum(
                pixel["region"] == "taper" for pixel in component_pixels
            )),
            "n_deep_inside": int(sum(
                pixel["region"] == "deep_inside" for pixel in component_pixels
            )),
        })

    return {
        "available": True,
        "fold_pixel_count": int(len(fold_positions)),
        "classification_counts": classification_counts,
        "weight_class_counts": weight_class_counts,
        "component_count": int(component_count),
        "components": components,
        "pixels": pixels,
    }


def _validation_block_lookup(validation: dict) -> dict[tuple[int, int, int, int, int], dict]:
    """Normalize validation blocks to the fixed C1 five-part key."""
    validation = validation or {}
    top_level_size = validation.get("validation_block_size_selected")
    lookup: dict[tuple[int, int, int, int, int], dict] = {}
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
                key = (idx_i, idx_j, row, col, int(size))
            except (KeyError, TypeError, ValueError):
                continue
            normalized = dict(block)
            normalized.update({
                "idx_i": idx_i,
                "idx_j": idx_j,
                "validation_row": row,
                "validation_col": col,
                "block_size": key[-1],
            })
            lookup[key] = normalized
    return lookup


def _finite_metric(validation: dict, name: str) -> float | None:
    value = (validation or {}).get("overall", {}).get(name)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _residual_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "n": 0,
            "median": None,
            "rmse": None,
            "p95": None,
        }

    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "p95": float(np.percentile(values, 95)),
    }


def _finite_residual(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def compare_tps_support_validations(
    translation_validation: dict,
    supported_validation: dict,
) -> dict[str, Any]:
    """Compare translation and supported stages on exactly the same HOLDOUT."""
    translation_lookup = _validation_block_lookup(translation_validation)
    supported_lookup = _validation_block_lookup(supported_validation)
    translation_keys = set(translation_lookup)
    supported_keys = set(supported_lookup)
    common_keys = sorted(translation_keys & supported_keys)
    paired_blocks = []
    for key in common_keys:
        translation_block = translation_lookup[key]
        supported_block = supported_lookup[key]
        translation_residual = _finite_residual(
            translation_block.get("residual_magnitude")
        )
        supported_residual = _finite_residual(
            supported_block.get("residual_magnitude")
        )
        if translation_residual is None and supported_residual is None:
            continue
        improvement = (
            translation_residual - supported_residual
            if translation_residual is not None and supported_residual is not None
            else None
        )
        paired_blocks.append({
            "key": list(key),
            "translation_residual": translation_residual,
            "supported_residual": supported_residual,
            "translation_accepted": translation_block.get("accepted"),
            "supported_accepted": supported_block.get("accepted"),
            "translation_reject_reason": translation_block.get("reject_reason"),
            "supported_reject_reason": supported_block.get("reject_reason"),
            "improvement": improvement,
        })

    holdout_keys_match = translation_keys == supported_keys
    measurable_paired_blocks = [
        block for block in paired_blocks
        if block["translation_residual"] is not None
        and block["supported_residual"] is not None
    ]
    paired_translation_values = [
        block["translation_residual"] for block in measurable_paired_blocks
    ]
    paired_supported_values = [
        block["supported_residual"] for block in measurable_paired_blocks
    ]
    translation_paired = _residual_summary(paired_translation_values)
    supported_paired = _residual_summary(paired_supported_values)
    available = bool(holdout_keys_match and measurable_paired_blocks)
    if available:
        reason = None
    elif not holdout_keys_match:
        reason = "translation and supported HOLDOUT keys differ"
    else:
        reason = "no measurable paired HOLDOUT blocks"

    def paired_improvement(name: str) -> float | None:
        before = translation_paired[name]
        after = supported_paired[name]
        return before - after if before is not None and after is not None else None

    stage_quality = {
        "translation": {
            "median": _finite_metric(translation_validation, "median"),
            "rmse": _finite_metric(translation_validation, "rmse"),
            "p95": _finite_metric(translation_validation, "p95"),
        },
        "supported": {
            "median": _finite_metric(supported_validation, "median"),
            "rmse": _finite_metric(supported_validation, "rmse"),
            "p95": _finite_metric(supported_validation, "p95"),
        },
    }
    all_measurable = {
        "translation": translation_paired,
        "supported": supported_paired,
        "median_improvement": paired_improvement("median"),
        "rmse_improvement": paired_improvement("rmse"),
        "p95_improvement": paired_improvement("p95"),
    }

    return {
        "available": available,
        "reason": reason,
        "holdout_keys_match": holdout_keys_match,
        # Preserve the historical stage-level fields for compatibility. The
        # causal deltas below are computed only from the same paired blocks.
        "translation_rmse": _finite_metric(translation_validation, "rmse"),
        "supported_rmse": _finite_metric(supported_validation, "rmse"),
        "rmse_improvement": all_measurable["rmse_improvement"],
        "translation_p95": _finite_metric(translation_validation, "p95"),
        "supported_p95": _finite_metric(supported_validation, "p95"),
        "p95_improvement": all_measurable["p95_improvement"],
        "translation_median": _finite_metric(translation_validation, "median"),
        "supported_median": _finite_metric(supported_validation, "median"),
        "median_improvement": all_measurable["median_improvement"],
        "stage_quality": stage_quality,
        "all_measurable": all_measurable,
        "n_paired_blocks": len(measurable_paired_blocks),
        "paired_blocks": paired_blocks,
    }
