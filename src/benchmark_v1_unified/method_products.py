"""Read-only intake for existing method artifacts; never fabricates outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio

from .submission import validate_submission


_FORMAL_SCIENCE_STATES = {"VALID", "STRICT_ADMM_CONVERGED", "NOT_APPLICABLE"}


def _scene_ids(root):
    with (root / "pair_overlap_inventory.csv").open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    scenes = []
    for row in rows:
        for key in ("scene_i", "scene_j"):
            if row[key] not in scenes:
                scenes.append(row[key])
    return scenes


def _check_raster(path, grid):
    if not path.is_file():
        return "MISSING_ARTIFACT"
    try:
        with rasterio.open(path) as dataset:
            if (dataset.count != 1 or dataset.width != int(grid["width"])
                    or dataset.height != int(grid["height"])
                    or dataset.crs is None or dataset.crs.to_string() != str(grid["crs"])
                    or not np.allclose(tuple(dataset.transform)[:6], grid["transform"], rtol=0, atol=1e-9)):
                return "INVALID_GRID"
            if np.dtype(dataset.dtypes[0]).kind not in "uif":
                return "INVALID_DTYPE"
    except (OSError, ValueError):
        return "INVALID_RASTER"
    return "PASS"


def prepare_method_submission(config, freeze_root, *, config_dir=None):
    root = Path(freeze_root).resolve()
    base = Path(config_dir or ".").resolve()
    frozen = json.loads((root / "frozen_manifest.json").read_text(encoding="utf-8"))
    scenes = _scene_ids(root)
    kind = str(config.get("method_kind", "")).upper()
    method_id = str(config.get("method_id", "")).strip()
    run_id = str(config.get("run_id", "")).strip()
    reasons = []
    if not method_id or not run_id:
        reasons.append("METHOD_OR_RUN_ID_MISSING")
    science_state = str(config.get("science_status", "UNVERIFIED")).upper()
    if kind != "PAIR_ONLY" and science_state not in _FORMAL_SCIENCE_STATES:
        reasons.append("SCIENCE_STATUS_UNVERIFIED")
    if kind == "PAIR_ONLY":
        pairs = config.get("pair_specific_outputs") or {}
        if not pairs:
            reasons.append("PAIR_ONLY_OUTPUTS_MISSING")
        for pair_id, value in pairs.items():
            path = Path(value)
            path = path if path.is_absolute() else base / path
            if not path.is_file():
                reasons.append(f"PAIR_OUTPUT_MISSING:{pair_id}")
        if reasons:
            return {"status": "NOT_READY", "reasons": reasons, "submission": None}
        return {"status": "PAIR_ONLY_DIAGNOSTIC", "reasons": [],
                "submission": {
                    "schema_version": 1, "method_id": method_id, "run_id": run_id,
                    "method_kind": "PAIR_ONLY", "input_freeze_sha256": frozen.get("input_freeze_sha256"),
                    "geometry_freeze_sha256": frozen.get("geometry_freeze_sha256"), "band": "B9",
                    "pixel_value_unit": "UNKNOWN", "status": str(config.get("status", "COMPLETED")),
                    "corrected_scene_paths": {}, "pair_specific_outputs": pairs,
                    "output_mosaic_path": None, "output_mosaic_mask_path": None,
                    "source_label_path": None, "seam_type": "NONE", "science_status": "PAIR_ONLY",
                }}
    if kind != "FULL_MOSAIC":
        return {"status": "NOT_READY", "reasons": reasons + ["METHOD_KIND_NOT_SUPPORTED_FOR_FULL_INTAKE"], "submission": None}
    if str(config.get("status", "")).upper() != "COMPLETED":
        reasons.append("METHOD_STATUS_NOT_COMPLETED")

    supplied = config.get("corrected_scene_paths") or {}
    if method_id.upper() == "RAW":
        corrected = {scene: str(root / "aligned_inputs" / f"scene_{index:03d}_B9_input.tif")
                     for index, scene in enumerate(scenes)}
    else:
        corrected = dict(supplied)
    missing_scenes = sorted(set(scenes) - set(corrected))
    extra_scenes = sorted(set(corrected) - set(scenes))
    if missing_scenes:
        reasons.append(f"MISSING_SCENE_OUTPUTS:{','.join(missing_scenes)}")
    if extra_scenes:
        reasons.append(f"EXTRA_SCENE_OUTPUTS:{','.join(extra_scenes)}")
    grid = frozen["grid"]
    for scene in scenes:
        if scene not in corrected:
            continue
        path = Path(corrected[scene])
        path = path if path.is_absolute() else base / path
        state = _check_raster(path, grid)
        if state != "PASS":
            reasons.append(f"SCENE_{state}:{scene}")
        corrected[scene] = str(path.resolve())

    outputs = {}
    for field in ("output_mosaic_path", "output_mosaic_mask_path", "source_label_path"):
        value = config.get(field)
        path = Path(value) if value else None
        if path is not None and not path.is_absolute():
            path = base / path
        if path is None:
            reasons.append(f"{field.upper()}_MISSING")
        else:
            state = _check_raster(path, grid)
            if state != "PASS":
                reasons.append(f"{field.upper()}_{state}")
            outputs[field] = str(path.resolve())
    if config.get("seam_type") != "HARD":
        reasons.append("SHARED_HARD_SOURCE_MOSAIC_REQUIRED")
    if reasons:
        return {"status": "NOT_READY", "reasons": reasons, "submission": None}

    submission = {
        "schema_version": 1, "method_id": method_id, "run_id": run_id,
        "method_kind": "FULL_MOSAIC", "input_freeze_sha256": frozen.get("input_freeze_sha256"),
        "geometry_freeze_sha256": frozen.get("geometry_freeze_sha256"), "band": "B9",
        "pixel_value_unit": "UNKNOWN", "status": "COMPLETED",
        "science_status": science_state, "corrected_scene_paths": corrected,
        "output_mosaic_path": outputs["output_mosaic_path"],
        "output_mosaic_mask_path": outputs["output_mosaic_mask_path"],
        "source_label_path": outputs["source_label_path"], "seam_type": "HARD",
        "runtime": config.get("runtime", {}), "resource_budget": config.get("resource_budget", {}),
    }
    audit = validate_submission(submission, {"input_freeze_sha256": frozen.get("input_freeze_sha256"),
                                             "geometry_freeze_sha256": frozen.get("geometry_freeze_sha256"),
                                             "band": "B9"})
    if not audit["eligible"]:
        return {"status": "NOT_READY", "reasons": audit["errors"], "submission": None}
    return {"status": "READY_FOR_SCORING", "reasons": [], "submission": submission}
