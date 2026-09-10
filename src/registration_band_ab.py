"""Contracts and audit helpers for the fixed-holdout B12/B14 comparison."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


ALLOWED_BAND_AB_CONFIG_DIFF_PATHS = {"experiment_name", "registration_band"}


def _as_dict(value):
    return asdict(value) if is_dataclass(value) else value


def _diff_paths(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        result = []
        for key in sorted(keys, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_diff_paths(left.get(key), right.get(key), path))
        return result
    if isinstance(left, list) and isinstance(right, list):
        result = []
        for index in range(max(len(left), len(right))):
            path = f"{prefix}[{index}]"
            if index >= len(left) or index >= len(right):
                result.append(path)
            else:
                result.extend(_diff_paths(left[index], right[index], path))
        return result
    return [] if left == right else [prefix]


def compare_band_ab_configs(config_b14: dict, config_b12: dict) -> dict:
    """Return recursive config differences and the strict parity result."""
    left, right = _as_dict(config_b14), _as_dict(config_b12)
    differing_paths = _diff_paths(left, right)
    invalid = [
        path for path in differing_paths
        if path.split(".", 1)[0].split("[", 1)[0]
        not in ALLOWED_BAND_AB_CONFIG_DIFF_PATHS
    ]
    return {
        "valid": not invalid and set(differing_paths) <= ALLOWED_BAND_AB_CONFIG_DIFF_PATHS,
        "differing_paths": differing_paths,
        "invalid_paths": invalid,
        "allowed_paths": sorted(ALLOWED_BAND_AB_CONFIG_DIFF_PATHS),
    }


def _raster_metadata(path):
    import rasterio

    if not path or not os.path.isfile(path):
        return None
    with rasterio.open(path) as src:
        return {
            "path": str(path), "crs": str(src.crs) if src.crs else None,
            "width": int(src.width), "height": int(src.height),
            "transform": [float(value) for value in src.transform[:6]],
            "pixel_size": [float(src.transform.a), float(src.transform.e)],
            "nodata": src.nodata, "count": int(src.count), "dtype": str(src.dtypes[0]),
        }


def validate_band_ab_metadata(config_b14: dict, config_b12: dict, *, scene_indices=(0, 1)) -> dict:
    """Validate config parity and B12/B14 raster geometry without registration."""
    config_b14, config_b12 = _as_dict(config_b14), _as_dict(config_b12)
    parity = compare_band_ab_configs(config_b14, config_b12)
    failures = []
    warnings = []
    scene_geometry = []
    scenes14, scenes12 = config_b14.get("scenes", []), config_b12.get("scenes", [])
    if not parity["valid"]:
        failures.append({"code": "config_parity", "details": parity})
    for index in scene_indices:
        if index >= len(scenes14) or index >= len(scenes12):
            failures.append({"code": "scene_missing", "scene_index": index})
            continue
        scene14, scene12 = scenes14[index], scenes12[index]
        scene_id = scene14.get("id", f"scene_{index}")
        if scene_id != scene12.get("id"):
            failures.append({"code": "scene_id_mismatch", "scene_index": index})
            continue
        pair = {"scene_index": index, "scene_id": scene_id, "bands": {}}
        for band in ("B12", "B14"):
            path = (scene12 if band == "B12" else scene14).get("bands", {}).get(band)
            meta = _raster_metadata(path)
            pair["bands"][band] = meta or {"path": str(path), "missing": True}
            if meta is None:
                failures.append({"code": "missing_path", "scene_id": scene_id,
                                 "band": band, "path": str(path)})
        b12, b14 = pair["bands"].get("B12"), pair["bands"].get("B14")
        if not b12 or not b14 or b12.get("missing") or b14.get("missing"):
            scene_geometry.append(pair)
            continue
        for field in ("crs", "width", "height", "transform", "pixel_size"):
            left, right = b12[field], b14[field]
            equal = (left == right) if field not in {"transform", "pixel_size"} else all(
                abs(float(a) - float(b)) <= 1e-9 for a, b in zip(left, right)
            )
            if not equal:
                failures.append({"code": "geometry_mismatch", "scene_id": scene_id,
                                 "band": "B12_vs_B14", "field": field,
                                 "b12": left, "b14": right})
        if b12.get("nodata") != b14.get("nodata"):
            warnings.append({"code": "nodata_mismatch", "scene_id": scene_id,
                             "b12": b12.get("nodata"), "b14": b14.get("nodata")})
        scene_geometry.append(pair)
    return {
        "valid": not failures, "config_parity": parity, "scene_geometry": scene_geometry,
        "failures": failures, "warnings": warnings,
    }


def _canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_holdout_manifest(registration: dict, *, source_commit: str,
                           source_registration_band: str, validation_band: str) -> dict:
    """Build a portable manifest from the exact reservation used by a run."""
    diagnostics = registration.get("diagnostics", {}) or {}
    holdout = registration.get("holdout", diagnostics.get("holdout", {})) or {}
    pairs = {}
    for key, summary in holdout.items():
        reservation = (summary or {}).get("validation_reservation", {}) or {}
        windows = reservation.get("reserved_windows", [])
        normalized = []
        for item in windows:
            normalized.append({k: int(item[k]) for k in ("row", "col", "height", "width")})
        pairs[str(key)] = {
            "selected_block_size": reservation.get("selected_block_size"),
            "reserved_windows": normalized,
        }
    return {
        "schema_version": 1, "source_commit": source_commit,
        "source_registration_band": source_registration_band,
        "validation_band": validation_band,
        "scene_ids": registration.get("scene_ids", []), "pairs": pairs,
        "registration_params_sha256": registration.get("registration_params_sha256"),
    }


def load_holdout_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("pairs"), dict):
        raise ValueError("unsupported or malformed HOLDOUT manifest")
    return manifest


def manifest_to_pair_overrides(manifest: dict, scene_ids) -> dict:
    """Translate manifest pair keys to the diagnostic pair's local indices."""
    manifest_ids = list(manifest.get("scene_ids", []))
    requested = list(scene_ids)
    if manifest_ids and manifest_ids != requested:
        raise ValueError("HOLDOUT manifest scene_ids do not match the requested pair")
    pair = (manifest.get("pairs", {}) or {}).get("0-1")
    if pair is None:
        raise ValueError("HOLDOUT manifest has no 0-1 pair")
    return {(0, 1): list(pair.get("reserved_windows", []))}


def _holdout_keys(diag):
    result = []
    for edge in (diag.get("final_validation", {}) or {}).get("edges", []) or []:
        key = (int(edge.get("idx_i", 0)), int(edge.get("idx_j", 0)))
        for block in edge.get("blocks", []) or []:
            if "validation_row" in block and "validation_col" in block:
                result.append((*key, int(block["validation_row"]), int(block["validation_col"]),
                               int(block.get("block_size", edge.get("validation_block_size_selected", 192)))))
    return sorted(result)


def summarize_registration_band_run(diag: dict) -> dict:
    quality = diag.get("quality", {}) or {}
    validation = diag.get("final_validation", {}) or {}
    return {
        "registration_band_name": diag.get("registration_band_name"),
        "validation_band_name": diag.get("validation_band_name"),
        "scene_ids": list(diag.get("scene_ids", [])),
        "quality": {key: quality.get(key) for key in ("median", "rmse", "p95", "confidence", "n_blocks", "quality")},
        "holdout_keys": _holdout_keys(diag),
        "validation_block_sizes": sorted({item[-1] for item in _holdout_keys(diag)}),
        "registration_params_sha256": diag.get("registration_params_sha256"),
        "config_parity": diag.get("config_parity"),
        "validation": validation,
    }


def compare_registration_band_runs(b14_diag: dict, b12_diag: dict) -> dict:
    """Compare two runs only after the fixed-band/fixed-HOLDOUT integrity gate."""
    b14, b12 = summarize_registration_band_run(b14_diag), summarize_registration_band_run(b12_diag)
    failures = []
    if b14["registration_band_name"] != "B14": failures.append("B14 diagnostic registration band mismatch")
    if b12["registration_band_name"] != "B12": failures.append("B12 diagnostic registration band mismatch")
    if b14["validation_band_name"] != "B14" or b12["validation_band_name"] != "B14":
        failures.append("validation band must be B14 for both runs")
    if b14["scene_ids"] != b12["scene_ids"]: failures.append("scene ids differ")
    if b14["holdout_keys"] != b12["holdout_keys"]: failures.append("reserved HOLDOUT keys differ")
    if b14["validation_block_sizes"] != b12["validation_block_sizes"]: failures.append("reserved block sizes differ")
    if b14["registration_params_sha256"] and b12["registration_params_sha256"] \
            and b14["registration_params_sha256"] != b12["registration_params_sha256"]:
        failures.append("registration parameters fingerprint differs")
    if b14.get("config_parity") is False or b12.get("config_parity") is False:
        failures.append("config parity preflight failed")
    available = not failures
    q14, q12 = b14["quality"], b12["quality"]
    delta = {}
    for metric in ("median", "rmse", "p95"):
        try: delta[metric] = float(q14[metric]) - float(q12[metric])
        except (TypeError, ValueError): delta[metric] = None
    if not available:
        conclusion = None
    elif delta["rmse"] is None or delta["p95"] is None:
        conclusion = "mixed_or_inconclusive"
    elif delta["rmse"] > 0 and delta["p95"] > 0:
        conclusion = "B12_directionally_better"
    elif delta["rmse"] < 0 and delta["p95"] < 0:
        conclusion = "B14_directionally_better"
    else:
        conclusion = "mixed_or_inconclusive"
    return {
        "comparison_available": available, "integrity_failures": failures,
        "b14": b14, "b12": b12,
        "delta_b14_minus_b12": delta, "material": available and any(
            abs(value) >= threshold for value, threshold in (
                (delta.get("rmse") or 0.0, 0.10), (delta.get("p95") or 0.0, 0.15)
            )
        ),
        "conclusion": conclusion,
    }
