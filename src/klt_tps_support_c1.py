"""Pure protocol and comparison helpers for the TPS-SUPPORT-C1 diagnostic."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np


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
