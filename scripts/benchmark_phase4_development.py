"""Four-pair DEVELOPMENT-only registration result protocol (no matcher runs)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import rasterio

from benchmark_phase2_core import pair_metrics
from benchmark_phase4_freeze import FrozenTrial


METHODS = ("SIFT", "KARIOS", "AROSICS_GLOBAL", "LoFTR", "EfficientLoFTR", "LightGlue+DISK")


def validate_manifest_destination(path: Path, root: Path) -> Path:
    path = Path(path).resolve()
    allowed = Path(root).resolve() / "docs/benchmark/v1/phase4"
    if not path.is_relative_to(allowed):
        raise ValueError(f"DEVELOPMENT preparation manifest must stay within Phase4: {path}")
    if path.exists():
        raise FileExistsError(f"Preparation manifest already exists: {path}")
    return path


def load_development_truth(snapshot: FrozenTrial) -> dict[str, list[dict]]:
    """Expose only the frozen 43 DEV checkpoints; no TEST row enters this API."""
    result = {pair: [] for pair in snapshot.trial_pairs}
    for key, frozen in snapshot.assignments.items():
        if frozen["split"] != "DEVELOPMENT":
            continue
        source = snapshot.candidates[key]
        if source["human_review_status"] != "APPROVED":
            raise ValueError(f"DEVELOPMENT point lacks valid first review: {key}")
        prefix = "adjusted_" if source.get("human_adjusted", "").lower() == "true" else ""
        row = {"point_id": key[1], "pair_id": key[0], "split": "DEVELOPMENT"}
        for field in ("row_i", "col_i", "x_i", "y_i", "row_j", "col_j", "x_j", "y_j"):
            row[field] = float(source[prefix + field])
            if not math.isfinite(row[field]):
                raise ValueError(f"Nonfinite DEVELOPMENT coordinate: {key}")
        result[key[0]].append(row)
    if {pair: len(rows) for pair, rows in result.items()} != {
            "00_01": 11, "02_03": 9, "05_07": 9, "08_10": 14}:
        raise ValueError("Frozen DEVELOPMENT distribution differs from Phase3")
    return result


def prepare_trial_inputs(snapshot: FrozenTrial, root: Path) -> list[dict]:
    """Read B9 headers only; do not load images or TEST truth."""
    by_index = {int(item["scene_index"]): item for item in snapshot.audit["b9_verified_inputs"]}
    manifest = []
    for pair in snapshot.trial_pairs:
        i, j = map(int, pair.split("_"))
        left, right = by_index[i], by_index[j]
        input_headers = []
        for item in (left, right):
            with rasterio.open(item["path"]) as dataset:
                if dataset.crs.to_epsg() != 32650 or dataset.count != 1:
                    raise ValueError(f"Unexpected B9 grid for scene {item['scene_index']}")
                input_headers.append({"scene_id": item["scene_id"], "path": item["path"],
                                      "sha256": item["sha256"], "width": dataset.width,
                                      "height": dataset.height, "transform": list(dataset.transform)[:6],
                                      "nodata": dataset.nodata})
        manifest.append({"pair_id": pair, "scene_i": left["scene_id"], "scene_j": right["scene_id"],
                         "input_band": "B9", "crs": "EPSG:32650", "images": input_headers,
                         "mask_rule": "rasterio.read_masks(1)>0 on native pixels; common valid overlap",
                         "native_direction": "scene_i_to_scene_j",
                         "preprocessing_chain_required": True, "frozen_assignment_sha256": snapshot.assignment_sha256})
    return manifest


def normalize_result(method: str, pair_id: str, status: str,
                     native_i_to_j_translation_rc: tuple[float, float] | None,
                     elapsed_s: float, reason: str, provenance: dict) -> dict:
    if method == "AROSICS_LOCAL":
        raise ValueError("AROSICS local model belongs to a separate experiment")
    if method not in METHODS or pair_id not in ("00_01", "02_03", "05_07", "08_10"):
        raise ValueError("Unexpected main-table method or trial pair")
    if status not in {"SUCCESS", "FAIL"} or not math.isfinite(elapsed_s) or elapsed_s < 0:
        raise ValueError("Invalid result status or elapsed time")
    translation = None
    if status == "SUCCESS":
        if native_i_to_j_translation_rc is None or len(native_i_to_j_translation_rc) != 2:
            raise ValueError("SUCCESS requires native two-component translation")
        translation = tuple(float(v) for v in native_i_to_j_translation_rc)
        if not all(math.isfinite(v) for v in translation) or provenance.get("frame") != "native":
            raise ValueError("SUCCESS requires finite native-grid translation and frame provenance")
        input_direction = provenance.get("input_direction", "scene_i_to_scene_j")
        expected_semantics = ("full_native_j_to_i" if input_direction == "scene_j_to_scene_i"
                              else "full_native_i_to_j")
        if provenance.get("translation_semantics") != expected_semantics:
            raise ValueError("SUCCESS requires full native translation, not a georeferenced residual")
        if input_direction == "scene_j_to_scene_i":
            translation = tuple(-v for v in translation)
        elif input_direction != "scene_i_to_scene_j":
            raise ValueError("Unknown transform direction")
    elif native_i_to_j_translation_rc is not None or not reason.strip():
        raise ValueError("FAIL requires null transform and nonempty reason")
    return {"method": method, "pair_id": pair_id, "status": status,
            "translation_row_col": translation, "direction": "scene_i_to_scene_j",
            "elapsed_s": float(elapsed_s), "reason": reason, "provenance": dict(provenance)}


def evaluate_development(results: list[dict], dev_truth: dict[str, list[dict]]) -> dict:
    keys = [(method, pair) for method in METHODS for pair in ("00_01", "02_03", "05_07", "08_10")]
    by_key = {}
    for row in results:
        key = (row["method"], row["pair_id"])
        if key not in keys or key in by_key:
            raise ValueError(f"Duplicate or unexpected DEVELOPMENT result: {key}")
        if row.get("status") not in {"SUCCESS", "FAIL"}:
            raise ValueError(f"Invalid DEVELOPMENT result status: {key}")
        if row["status"] == "SUCCESS":
            provenance = row.get("provenance", {})
            translation = row.get("translation_row_col")
            if (row.get("direction") != "scene_i_to_scene_j" or
                    provenance.get("frame") != "native" or
                    provenance.get("translation_semantics") not in
                    {"full_native_i_to_j", "full_native_j_to_i"} or
                    translation is None or len(translation) != 2 or
                    not all(math.isfinite(float(v)) for v in translation)):
                raise ValueError(f"Unnormalized DEVELOPMENT translation: {key}")
        by_key[key] = row
    output = []
    for method, pair in keys:
        result = by_key.get((method, pair))
        if result is None:
            output.append({"method": method, "pair_id": pair, "status": "NOT_ATTEMPTED", "metrics": None})
            continue
        row = dict(result)
        row["metrics"] = None
        if result["status"] == "SUCCESS":
            points = dev_truth[pair]
            if not points:
                row["status"] = "NO_DEVELOPMENT_TRUTH"
            else:
                dr, dc = result["translation_row_col"]
                errors = []
                for point in points:
                    if point.get("split") != "DEVELOPMENT":
                        raise ValueError("Non-DEVELOPMENT point reached DEVELOPMENT evaluator")
                    errors.append(math.hypot(float(point["row_i"]) + dr - float(point["row_j"]),
                                             float(point["col_i"]) + dc - float(point["col_j"])))
                row["metrics"] = pair_metrics(errors)
        output.append(row)
    summary = {}
    for method in METHODS:
        rows = [row for row in output if row["method"] == method]
        successes = [row for row in rows if row["status"] == "SUCCESS" and row["metrics"] is not None]
        summary[method] = {"eligible_pairs": 4, "attempted_pairs": sum(row["status"] != "NOT_ATTEMPTED" for row in rows),
                           "failed_pairs": sum(row["status"] == "FAIL" for row in rows),
                           "success_pairs": len(successes),
                           "macro_rmse_success_only": float(np.mean([r["metrics"]["rmse_px"] for r in successes]))
                           if successes else None}
    return {"rows": output, "summary": summary, "scope": "FOUR_PAIR_DEVELOPMENT_ONLY"}


def environment_preflight() -> dict:
    """Import/adapter discovery only; never instantiate or load a model."""
    modules = {"SIFT": "cv2", "KARIOS": "karios", "AROSICS_GLOBAL": "arosics",
               "LoFTR": "kornia", "EfficientLoFTR": "torch", "LightGlue+DISK": "lightglue"}
    return {method: {"dependency_detected": importlib.util.find_spec(module) is not None,
                     "end_to_end": "NOT_VERIFIED_END_TO_END", "model": "global_translation"}
            for method, module in modules.items()}
