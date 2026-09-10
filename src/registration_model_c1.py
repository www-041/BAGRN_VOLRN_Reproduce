"""Frozen, diagnostic-only protocol helpers for MODEL-C1."""

import copy
import hashlib
import json
from typing import Any, Dict

import numpy as np


AFFINE_CAUSAL_PARAM_KEYS = frozenset({
    "affine_min_controls",
    "affine_min_spatial_groups",
    "affine_min_inliers",
    "affine_min_inlier_ratio",
    "affine_ransac_residual_threshold",
    "affine_ransac_max_trials",
    "affine_ransac_seed",
    "affine_max_scale_delta",
    "affine_max_rotation_deg",
    "affine_max_shear_deg",
    "affine_max_component",
    "affine_cv_min_rmse_improvement",
    "affine_cv_min_p95_improvement",
})

MODEL_C1_BASE_FINGERPRINT = (
    "08864b73e8a53990aa4d74038cdb269d10b0cddd7ebc2df26d5d6654713ccb98"
)


def _registration_params(source):
    if hasattr(source, "registration_params"):
        return getattr(source, "registration_params") or {}
    if isinstance(source, dict):
        return source.get("registration_params", source)
    return {}


def model_c1_base_registration_params(config):
    """Return the B12 base registration parameters with affine keys removed."""
    params = _registration_params(config)
    return copy.deepcopy({
        key: value for key, value in params.items()
        if key not in AFFINE_CAUSAL_PARAM_KEYS
    })


def model_c1_base_registration_params_sha256(config):
    payload = json.dumps(
        model_c1_base_registration_params(config),
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_model_c1_protocol(
    config, holdout_manifest, *, validation_band="B14",
):
    """Validate the frozen diagnostic protocol without opening imagery files."""
    errors = []
    registration_band = getattr(config, "registration_band", None)
    selected_bands = getattr(config, "selected_bands", None)
    scene_configs = getattr(config, "scenes", None)
    if isinstance(config, dict):
        registration_band = config.get("registration_band")
        selected_bands = config.get("selected_bands")
        scene_configs = config.get("scenes")
    if registration_band != "B12":
        errors.append("MODEL-C1 registration band must be B12")
    if list(selected_bands or []) != ["B12", "B14"]:
        errors.append("MODEL-C1 selected bands must be exactly [B12, B14]")
    if validation_band != "B14":
        errors.append("MODEL-C1 validation band must be explicitly B14")
    scene_ids = [item.get("id") for item in (scene_configs or [])]
    manifest_ids = list(holdout_manifest.get("scene_ids", []))
    if scene_ids[:2] != manifest_ids:
        errors.append("MODEL-C1 scene IDs do not match the HOLDOUT manifest")
    if holdout_manifest.get("source_registration_band") != "B14":
        errors.append("HOLDOUT source registration band must be B14")
    if holdout_manifest.get("validation_band") != "B14":
        errors.append("HOLDOUT validation band must be B14")
    expected_hash = holdout_manifest.get("base_registration_params_sha256")
    actual_hash = model_c1_base_registration_params_sha256(config)
    if expected_hash and expected_hash != actual_hash:
        errors.append("MODEL-C1 base registration parameter fingerprint mismatch")
    if expected_hash is None and actual_hash != MODEL_C1_BASE_FINGERPRINT:
        errors.append("MODEL-C1 base registration parameter fingerprint mismatch")

    pairs = holdout_manifest.get("pairs", {})
    pair = pairs.get("0-1", {})
    windows = pair.get("reserved_windows", [])
    if int(pair.get("selected_block_size", 0)) != 384:
        errors.append("MODEL-C1 fixed HOLDOUT block size must be 384")
    if len(windows) != 7:
        errors.append("MODEL-C1 fixed HOLDOUT must contain exactly 7 windows")
    return {
        "valid": not errors,
        "errors": errors,
        "registration_band": registration_band,
        "validation_band": validation_band,
        "scene_ids": manifest_ids,
        "base_registration_params_sha256": actual_hash,
        "reserved_count": len(windows),
        "selected_block_size": pair.get("selected_block_size"),
    }


def _block_key(edge, block):
    return (
        int(edge.get("idx_i", -1)), int(edge.get("idx_j", -1)),
        int(block.get("validation_row", block.get("row", block.get("row_start", -1)))),
        int(block.get("validation_col", block.get("col", block.get("col_start", -1)))),
    )


def _block_residual(block):
    for key in (
        "final_residual_magnitude", "residual_magnitude", "residual",
        "error", "distance",
    ):
        value = block.get(key)
        if isinstance(value, dict):
            dx = value.get("dx")
            dy = value.get("dy")
            try:
                return float(np.hypot(float(dx), float(dy)))
            except (TypeError, ValueError):
                continue
        try:
            value = float(value)
            if np.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    stats = block.get("stats") or {}
    for key in ("residual_magnitude", "median", "rmse", "p95"):
        try:
            value = float(stats.get(key))
            if np.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    return None


def _validation_blocks(validation):
    result = {}
    for edge in (validation or {}).get("edges", []):
        for block in edge.get("blocks", []):
            result[_block_key(edge, block)] = block
    return result


def _summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n": 0, "median": None, "rmse": None, "p95": None}
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "p95": float(np.percentile(values, 95)),
    }


def compare_fixed_holdout_validations(global_validation, affine_validation):
    """Compare global-only and affine results on identical measurable blocks."""
    global_blocks = _validation_blocks(global_validation)
    affine_blocks = _validation_blocks(affine_validation)
    paired = []
    transitions = []
    for key in sorted(set(global_blocks) & set(affine_blocks)):
        global_block = global_blocks[key]
        affine_block = affine_blocks[key]
        global_residual = _block_residual(global_block)
        affine_residual = _block_residual(affine_block)
        if global_residual is None or affine_residual is None:
            continue
        global_accepted = bool(global_block.get("accepted", False))
        affine_accepted = bool(affine_block.get("accepted", False))
        transitions.append({
            "key": list(key),
            "global_status": "accepted" if global_accepted else "rejected",
            "affine_status": "accepted" if affine_accepted else "rejected",
        })
        paired.append({
            "key": list(key),
            "global_residual": global_residual,
            "affine_residual": affine_residual,
            "global_accepted": global_accepted,
            "affine_accepted": affine_accepted,
            "improvement": float(global_residual - affine_residual),
        })
    global_values = [item["global_residual"] for item in paired]
    affine_values = [item["affine_residual"] for item in paired]
    common_accepted = [
        item for item in paired if item["global_accepted"] and item["affine_accepted"]
    ]
    return {
        "available": bool(paired),
        "paired_blocks": paired,
        "n_paired": len(paired),
        "n_common_accepted": len(common_accepted),
        "global": _summary(global_values),
        "affine": _summary(affine_values),
        "all_measurable": {
            "global": _summary(global_values),
            "affine": _summary(affine_values),
            "rmse_improvement": (
                _summary(global_values)["rmse"] - _summary(affine_values)["rmse"]
                if paired else None
            ),
            "p95_improvement": (
                _summary(global_values)["p95"] - _summary(affine_values)["p95"]
                if paired else None
            ),
        },
        "status_transitions": transitions,
    }
