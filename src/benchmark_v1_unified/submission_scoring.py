"""Read-only GeoTIFF submission scorer for the frozen Unified Benchmark V1.

This is an artifact adapter around the frozen metric primitives. It does not
change support, windows, formulas, or ranking rules in the freeze.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import rasterio

from .aggregation import aggregate_pair_rows, aggregate_windows
from .freeze import verify_lock
from .mosaic import compute_mosaic_coverage, compute_single_source_structure
from .radiometric import (
    compute_pair_direct_metrics,
    compute_pair_distribution_metrics,
)
from .seams import compute_seam_excess
from .structure import compute_range_retention, compute_sobel_structure
from .submission import validate_submission


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _resolve(path: str | None, base: Path) -> Path | None:
    if path is None or not str(path).strip():
        return None
    value = Path(path)
    return value if value.is_absolute() else (base / value).resolve()


def _grid(frozen: dict) -> tuple[tuple[int, int], tuple[float, ...], str]:
    grid = frozen["grid"]
    return (int(grid["height"]), int(grid["width"])), tuple(float(v) for v in grid["transform"]), str(grid["crs"])


def _read_raster(path: Path, expected_shape, expected_transform, expected_crs):
    if not path.is_file():
        raise FileNotFoundError(f"MISSING_ARTIFACT:{path}")
    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise ValueError(f"INVALID_BAND_COUNT:{path}:{dataset.count}")
        if not np.issubdtype(np.dtype(dataset.dtypes[0]), np.number) or np.issubdtype(np.dtype(dataset.dtypes[0]), np.complexfloating):
            raise ValueError(f"INVALID_PIXEL_DTYPE:{path}:{dataset.dtypes[0]}")
        if (dataset.height, dataset.width) != expected_shape:
            raise ValueError(f"INVALID_SHAPE:{path}:{dataset.height}x{dataset.width}")
        if dataset.crs is None or dataset.crs.to_string() != expected_crs:
            raise ValueError(f"INVALID_CRS:{path}:{dataset.crs}")
        actual = tuple(dataset.transform)[:6]
        if not np.allclose(actual, expected_transform, rtol=0.0, atol=1e-9):
            raise ValueError(f"INVALID_TRANSFORM:{path}:{actual}")
        values = dataset.read(1)
        valid = dataset.dataset_mask() > 0
        if np.issubdtype(values.dtype, np.floating):
            valid &= np.isfinite(values)
        return values, valid, dataset.dtypes[0], dataset.nodata


def _mask_sha(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(mask, bool), bitorder="little").tobytes()).hexdigest()


def _resource_record(submission: dict, pixel_count: int, scene_count: int) -> dict:
    runtime = submission.get("runtime") or {}
    budget = submission.get("resource_budget") or {}
    estimated = _estimate_evaluator_working_set(pixel_count, scene_count)
    max_rss = budget.get("max_rss_bytes")
    budget_status = "PASS" if isinstance(max_rss, (int, float)) and math.isfinite(float(max_rss)) and max_rss >= estimated else "BUDGET_MISSING_OR_BELOW_ESTIMATE"
    wall = runtime.get("wall_seconds")
    wall_source = runtime.get("wall_measurement_source") or "NOT_MEASURED"
    rss_status = runtime.get("rss_status") or "NOT_MEASURED"
    rss_source = runtime.get("rss_measurement_source") or "NOT_MEASURED"
    samples = int(runtime.get("rss_sample_count") or 0)
    if rss_status == "MEASURED" and (samples <= 0 or runtime.get("peak_rss_bytes") is None):
        rss_status = "INVALID_PROVENANCE"
    gpu_status = runtime.get("gpu_status", "NOT_MEASURED")
    gpu_source = runtime.get("gpu_measurement_source") or "NOT_MEASURED"
    gpu_samples = int(runtime.get("gpu_sample_count") or 0)
    if gpu_status == "MEASURED" and (gpu_samples <= 0 or runtime.get("peak_gpu_memory_bytes") is None):
        gpu_status = "INVALID_PROVENANCE"
    wall_valid = isinstance(wall, (int, float)) and wall >= 0 and wall_source != "NOT_MEASURED"
    rss_valid = (rss_status == "MEASURED" and samples > 0 and runtime.get("peak_rss_bytes") is not None and rss_source != "NOT_MEASURED") or (rss_status == "NOT_MEASURED" and rss_source == "NOT_MEASURED")
    gpu_valid = (gpu_status == "MEASURED" and gpu_samples > 0 and runtime.get("peak_gpu_memory_bytes") is not None and gpu_source != "NOT_MEASURED") or (gpu_status == "NOT_MEASURED" and gpu_source == "NOT_MEASURED")
    return {
        "runtime_status": runtime.get("status", "NOT_MEASURED"),
        "wall_seconds": wall,
        "wall_measurement_source": wall_source,
        "peak_rss_bytes": runtime.get("peak_rss_bytes"),
        "rss_status": rss_status,
        "rss_sample_count": samples,
        "rss_measurement_source": rss_source,
        "peak_gpu_memory_bytes": runtime.get("peak_gpu_memory_bytes"),
        "gpu_status": gpu_status,
        "gpu_sample_count": gpu_samples,
        "gpu_measurement_source": gpu_source,
        "resource_budget": dict(budget),
        "estimated_evaluator_working_set_bytes": estimated,
        "budget_preflight_status": budget_status,
        "hard_os_memory_limit": bool(runtime.get("hard_os_memory_limit", False)),
        "measurement_provenance_status": "PASS" if wall_valid and rss_valid and gpu_valid and "hard_os_memory_limit" in runtime else "INCOMPLETE",
    }


def _estimate_evaluator_working_set(pixel_count: int, scene_count: int) -> int:
    # Conservative planning estimate for float64 conversions, masks, exact W1
    # sorting and structure/mosaic temporaries. It is not an OS memory limit.
    return int(pixel_count * (scene_count * 24 + scene_count * 2 + 128))


def _available_ram_bytes():
    try:
        import psutil
        return int(psutil.virtual_memory().available), "psutil.virtual_memory().available"
    except Exception:
        return None, "NOT_MEASURED"


def _evaluate_submission(freeze_root, submission: dict, *, submission_dir=None, regimes=("A", "B1", "B2")) -> dict:
    """Score a method's canonical-grid GeoTIFF artifacts against frozen inputs.

    For the official freeze, input masks, all 78 pairs and all 1,533 windows are
    read as-is. Fixture freezes may contain smaller inventories for unit tests.
    """
    freeze_root = Path(freeze_root).resolve()
    base = Path(submission_dir or ".").resolve()
    regimes = tuple(dict.fromkeys(str(r).upper() for r in regimes))
    invalid = sorted(set(regimes) - {"A", "B1", "B2", "B3"})
    if invalid:
        raise ValueError(f"INVALID_REGIME:{invalid}")
    lock_result = verify_lock(freeze_root)
    frozen_path = freeze_root / "frozen_manifest.json"
    if not lock_result.valid:
        return {"status": "FREEZE_INVALID", "errors": list(lock_result.errors)}
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    audit = validate_submission(submission, {
        "input_freeze_sha256": frozen.get("input_freeze_sha256"),
        "geometry_freeze_sha256": frozen.get("geometry_freeze_sha256"),
        "band": "B9",
    })
    schema_errors = []
    for name in ("schema_version", "method_id", "run_id", "method_kind", "pixel_value_unit", "status"):
        if name not in submission:
            schema_errors.append(f"MISSING_{name.upper()}")
    for name in ("method_id", "run_id", "status"):
        if name in submission and (not isinstance(submission[name], str) or not submission[name].strip()):
            schema_errors.append(f"INVALID_{name.upper()}")
    if submission.get("schema_version") != 1:
        schema_errors.append("INVALID_SCHEMA_VERSION")
    if submission.get("pixel_value_unit") != "UNKNOWN":
        schema_errors.append("INVALID_PIXEL_VALUE_UNIT")
    if submission.get("method_kind") not in {"FULL_MOSAIC", "SCENE_OUTPUTS", "PAIR_ONLY", "SYNTHETIC_FIXTURE"}:
        schema_errors.append("INVALID_METHOD_KIND")
    if submission.get("seam_type") is not None and submission.get("seam_type") not in {"HARD", "SOFT", "NONE"}:
        schema_errors.append("INVALID_SEAM_TYPE")
    for name in ("output_mosaic_path", "output_mosaic_mask_path", "source_label_path"):
        if name in submission and submission[name] is not None and not isinstance(submission[name], str):
            schema_errors.append(f"INVALID_{name.upper()}")
    if "corrected_scene_paths" in submission and (
        not isinstance(submission["corrected_scene_paths"], dict)
        or any(not isinstance(value, str) for value in submission["corrected_scene_paths"].values())
    ):
        schema_errors.append("INVALID_CORRECTED_SCENE_PATHS")
    if not audit["eligible"] or schema_errors:
        return {"status": "SUBMISSION_INVALID", "submission_audit": {**audit, "errors": audit["errors"] + schema_errors, "eligible": False}}
    if regimes == ("B3",):
        return {
            "protocol_id": frozen.get("protocol_id"),
            "status": "EVALUATED",
            "submission_audit": {**audit, "status": "PASS", "errors": [], "eligible": True,
                                  "artifact_status": "NOT_REQUIRED_FOR_B3"},
            "B3": {"status": "NOT_RUN_SYNTHETIC_METHOD_OUTPUT_REQUIRED",
                   "reason": "B3 is a separate known-truth simulation; real overlap metrics are not a substitute."},
            "ranking_eligible": False,
        }

    pair_rows = _read_csv(freeze_root / "pair_overlap_inventory.csv")
    scenes = []
    for row in pair_rows:
        for name in ("scene_i", "scene_j"):
            if row[name] not in scenes:
                scenes.append(row[name])
    if len(scenes) != int(frozen.get("scene_count", len(scenes))):
        return {"status": "FREEZE_INVENTORY_MISMATCH", "scene_count": len(scenes)}
    shape, transform, crs = _grid(frozen)
    estimated_bytes = _estimate_evaluator_working_set(shape[0] * shape[1], len(scenes))
    budget = submission.get("resource_budget") or {}
    max_rss = budget.get("max_rss_bytes")
    available_ram, available_source = _available_ram_bytes()
    preflight_reasons = []
    if not isinstance(max_rss, (int, float)) or not math.isfinite(float(max_rss)) or max_rss < estimated_bytes:
        preflight_reasons.append("BUDGET_MISSING_OR_BELOW_ESTIMATE")
    if available_ram is not None and available_ram < estimated_bytes:
        preflight_reasons.append("AVAILABLE_RAM_BELOW_ESTIMATE")
    if preflight_reasons:
        return {"status": "RESOURCE_PREFLIGHT_FAILED", "resource_preflight": {
            "status": preflight_reasons[0], "reasons": preflight_reasons,
            "estimated_evaluator_working_set_bytes": estimated_bytes,
            "declared_max_rss_bytes": max_rss, "available_ram_bytes": available_ram,
            "available_ram_measurement_source": available_source,
            "hard_os_memory_limit": False,
        }}
    scene_index = {name: idx for idx, name in enumerate(scenes)}

    input_values, input_masks, output_values, output_masks, output_dtypes = [], [], [], [], []
    corrected_paths = submission.get("corrected_scene_paths") or {}
    if set(corrected_paths) != set(scenes):
        missing = sorted(set(scenes) - set(corrected_paths))
        extra = sorted(set(corrected_paths) - set(scenes))
        return {"status": "SCENE_OUTPUT_IDENTITY_MISMATCH", "missing_scene_ids": missing, "extra_scene_ids": extra}
    for scene_id in scenes:
        idx = scene_index[scene_id]
        # Use the locked inventory's canonical relative location; never trust an
        # arbitrary path embedded in a submission as the reference input.
        input_path = freeze_root / "aligned_inputs" / f"scene_{idx:03d}_B9_input.tif"
        mask_path = freeze_root / "aligned_inputs" / f"scene_{idx:03d}_input_mask.tif"
        raw, raw_valid, _, _ = _read_raster(input_path, shape, transform, crs)
        frozen_mask_values, frozen_mask_valid, _, _ = _read_raster(mask_path, shape, transform, crs)
        frozen_mask = (frozen_mask_values > 0) & frozen_mask_valid
        # The input's own mask must agree with the separately frozen mask.
        if np.any(frozen_mask & ~raw_valid):
            return {"status": "INVALID_FROZEN_INPUT_MASK", "scene_id": scene_id}
        raw_valid &= frozen_mask
        input_values.append(raw)
        input_masks.append(frozen_mask)
        out_path = _resolve(corrected_paths.get(scene_id), base)
        if out_path is None or not out_path.is_file():
            return {"status": "MISSING_SCENE_OUTPUT", "scene_id": scene_id}
        corrected, corrected_valid, dtype, _ = _read_raster(out_path, shape, transform, crs)
        output_values.append(corrected)
        output_masks.append(corrected_valid)
        output_dtypes.append(dtype)

    input_union = np.logical_or.reduce(input_masks)
    for row in pair_rows:
        support = input_masks[scene_index[row["scene_i"]]] & input_masks[scene_index[row["scene_j"]]]
        observed = int(support.sum())
        expected = int(row.get("aligned_valid_overlap_pixels") or 0)
        expected_status = "OVERLAP" if observed else "NO_VALID_OVERLAP"
        if observed != expected or row.get("status") != expected_status:
            return {"status": "FROZEN_PAIR_SUPPORT_MISMATCH", "pair_id": row["pair_id"],
                    "inventory_pixels": expected, "mask_intersection_pixels": observed,
                    "inventory_status": row.get("status"), "mask_status": expected_status}
    resource = _resource_record(submission, int(input_union.size), len(scenes))
    result = {
        "protocol_id": frozen.get("protocol_id"),
        "status": "EVALUATED",
        "submission_audit": {**audit, "status": "PASS", "errors": [], "eligible": True,
                              "artifact_status": "PASS", "scene_count": len(scenes),
                              "scene_ids": scenes, "corrected_scene_dtypes": dict(zip(scenes, output_dtypes)),
                              "method_status": submission.get("status")},
        "runtime": resource,
        "A": None,
        "B1": None,
        "B2": None,
        "B3": None,
    }
    if "B1" in regimes or "B2" in regimes:
        result["scene_structure"] = _structure_by_scene(scenes, input_values, output_values, input_masks, output_masks)

    if "A" in regimes:
        mosaic_path = _resolve(submission.get("output_mosaic_path"), base)
        mosaic_mask_path = _resolve(submission.get("output_mosaic_mask_path"), base)
        if mosaic_path is None or mosaic_mask_path is None:
            result["A"] = {"status": "MISSING_MOSAIC_OR_MASK"}
        else:
            mosaic, mosaic_valid_from_file, _, _ = _read_raster(mosaic_path, shape, transform, crs)
            mask_values, mask_valid, _, _ = _read_raster(mosaic_mask_path, shape, transform, crs)
            mosaic_valid = mosaic_valid_from_file & mask_valid & (mask_values > 0) & np.isfinite(mosaic)
            coverage = compute_mosaic_coverage(input_union, mosaic_valid)
            unique_source_support = np.sum(np.stack(input_masks), axis=0) == 1
            scene_outputs_complete = all(np.all(output_masks[k][unique_source_support & input_masks[k]]) for k in range(len(scenes)))
            if scene_outputs_complete:
                structure = compute_single_source_structure(output_values, input_masks, mosaic, mosaic_valid)
            else:
                structure = {"status": "PARTIAL_OUTPUT", "unique_source_pixels": int(unique_source_support.sum()),
                             "valid_pixels": 0, "structure": None}
            expected_ids = set(range(len(scenes)))
            labels_path = _resolve(submission.get("source_label_path"), base)
            if labels_path is None:
                seam = {"status": "NOT_EVALUABLE_NO_PROVENANCE", "edge_count": 0, "mean_excess": None}
            else:
                labels, labels_valid, _, _ = _read_raster(labels_path, shape, transform, crs)
                safe_labels = np.where(np.isin(labels, list(expected_ids)), labels, 0).astype(np.int64)
                source_valid = np.zeros(shape, dtype=bool)
                for source_idx, source_mask in enumerate(input_masks):
                    source_valid |= (safe_labels == source_idx) & source_mask & output_masks[source_idx]
                if np.any(mosaic_valid & ~labels_valid) or np.any(mosaic_valid & ~np.isin(labels, list(expected_ids))) or np.any(mosaic_valid & ~source_valid):
                    seam = {"status": "INVALID_SOURCE_LABELS", "edge_count": 0, "mean_excess": None}
                else:
                    # Source ids in the frozen inventory are assigned by stable
                    # first appearance; the mapping is returned for audit.
                    seam = compute_seam_excess(mosaic, labels, mosaic_valid)
                    seam["status"] = "EVALUABLE" if seam.get("status") == "EVALUABLE" else seam.get("status")
                    seam["source_id_mapping"] = {str(i): scenes[i] for i in range(len(scenes))}
            result["A"] = {
                "status": "EVALUABLE" if coverage.coverage == 1.0 and coverage.extras == 0 else "INCOMPLETE_MOSAIC_SUPPORT",
                "mosaic_coverage": {"status": coverage.status, "coverage": coverage.coverage,
                                    "input_union_pixels": int(input_union.sum()),
                                    "output_valid_pixels": int(mosaic_valid.sum()),
                                    "holes": coverage.holes, "extras": coverage.extras},
                "single_source_structure": structure,
                "structure": {"structure_by_scene": result.get("scene_structure") or _structure_by_scene(scenes, input_values, output_values, input_masks, output_masks)},
                "seam": seam,
            }

    if "B1" in regimes:
        scored = []
        for row in pair_rows:
            i, j = scene_index[row["scene_i"]], scene_index[row["scene_j"]]
            support = input_masks[i] & input_masks[j]
            pid = row["pair_id"]
            if not support.any():
                scored.append({"pair_id": pid, "scene_i": row["scene_i"], "scene_j": row["scene_j"],
                               "status": "NO_VALID_OVERLAP", "n": 0, "metrics": {"direct": None, "distribution": None}})
                continue
            both_valid = output_masks[i] & output_masks[j]
            output_pixels = int(np.count_nonzero(support & both_valid))
            full = output_pixels == int(support.sum())
            if not full:
                scored.append({"pair_id": pid, "scene_i": row["scene_i"], "scene_j": row["scene_j"],
                               "status": "OUTPUT_MISSING" if output_pixels == 0 else "PARTIAL_OUTPUT", "n": int(support.sum()),
                               "output_pixels": output_pixels,
                               "coverage": float(output_pixels / support.sum()),
                               "metrics": {"direct": None, "distribution": None}})
                continue
            direct = compute_pair_direct_metrics(output_values[i], output_values[j], support)
            dist = compute_pair_distribution_metrics(output_values[i], output_values[j], support)
            scored.append({"pair_id": pid, "scene_i": row["scene_i"], "scene_j": row["scene_j"],
                           "status": "EVALUABLE", "n": direct.n,
                           "metrics": {"direct": direct.__dict__, "distribution": dist.__dict__}})
        metric_names = {"mae": "direct.mae", "rmse": "direct.rmse", "bias": "direct.bias", "p95ae": "direct.p95ae",
                        "mamd": "distribution.mamd", "msdd": "distribution.msdd", "rdd": "distribution.rdd"}
        summaries = {}
        for name, field in metric_names.items():
            key = field.split(".")[1]
            agg_rows = [{"pair_id": r["pair_id"], "status": r["status"], "value": r["metrics"].get(field.split(".")[0], {}).get(key) if r["metrics"].get(field.split(".")[0]) else None, "n": r["n"]} for r in scored]
            positive_pair_ids = [r["pair_id"] for r in pair_rows if r.get("status") == "OVERLAP"]
            summaries[name] = aggregate_pair_rows(agg_rows, positive_pair_ids)
        result["B1"] = {"status": "EVALUABLE" if all(r["status"] in {"EVALUABLE", "NO_VALID_OVERLAP"} for r in scored) and all(r["status"] != "PARTIAL_OUTPUT" for r in scored) else "PARTIAL_OUTPUT",
                         "expected_pair_rows": len(pair_rows),
                         "positive_overlap_pairs": sum(1 for r in pair_rows if r.get("status") == "OVERLAP"),
                         "no_valid_overlap_pairs": sum(1 for r in pair_rows if r.get("status") == "NO_VALID_OVERLAP"),
                         "pair_rows": scored, "metric_summaries": summaries,
                         "metric_summaries_by_time_stratum": _b1_stratified_summaries(scored, pair_rows, metric_names),
                         "metric_summaries_by_quality": _b1_quality_summaries(scored, pair_rows, metric_names)}

    if "B2" in regimes:
        window_inventory = _read_csv(freeze_root / "fixed_windows.csv")
        windows = []
        structures = []
        for row in window_inventory:
            pair_for_window = next(p for p in pair_rows if p["pair_id"] == row["pair_id"])
            i, j = scene_index[pair_for_window["scene_i"]], scene_index[pair_for_window["scene_j"]]
            r0, c0, h, w = (int(row[k]) for k in ("row_off", "col_off", "height", "width"))
            sl = np.s_[r0:r0+h, c0:c0+w]
            support = input_masks[i][sl] & input_masks[j][sl]
            expected_count = int(row["valid_count"])
            if int(support.sum()) != expected_count or _mask_sha(support) != row["support_sha256"]:
                return {"status": "FROZEN_WINDOW_SUPPORT_MISMATCH", "window_id": row["window_id"]}
            both_valid = output_masks[i][sl] & output_masks[j][sl]
            window_output_pixels = int(np.count_nonzero(both_valid & support))
            full = window_output_pixels == expected_count
            if not full:
                windows.append({"window_id": row["window_id"], "pair_id": row["pair_id"],
                                "status": "OUTPUT_MISSING" if window_output_pixels == 0 else "PARTIAL_OUTPUT",
                                "n": expected_count, "output_pixels": window_output_pixels,
                                "coverage": window_output_pixels / expected_count, "metrics": None})
            else:
                direct = compute_pair_direct_metrics(output_values[i][sl], output_values[j][sl], support)
                dist = compute_pair_distribution_metrics(output_values[i][sl], output_values[j][sl], support)
                windows.append({"window_id": row["window_id"], "pair_id": row["pair_id"], "status": "EVALUABLE", "n": direct.n,
                                "metrics": {"direct": direct.__dict__, "distribution": dist.__dict__}})
            for idx, scene_id in ((i, scenes[i]), (j, scenes[j])):
                scene_support = input_masks[idx][sl]
                if np.any(scene_support & ~output_masks[idx][sl]):
                    structures.append({"window_id": row["window_id"], "pair_id": row["pair_id"], "scene_id": scene_id,
                                       "status": "PARTIAL_OUTPUT", "sobel": None, "range_retention": None})
                else:
                    st = compute_sobel_structure(input_values[idx][sl], output_values[idx][sl], scene_support)
                    rr = compute_range_retention(input_values[idx][sl], output_values[idx][sl], scene_support)
                    structures.append({"window_id": row["window_id"], "pair_id": row["pair_id"], "scene_id": scene_id,
                                       "status": st.status, "sobel": st.__dict__, "range_retention": rr.__dict__})
        window_summaries = {}
        for key, prefix in (("mae", "direct"), ("rmse", "direct"), ("bias", "direct"), ("p95ae", "direct"),
                            ("mamd", "distribution"), ("msdd", "distribution"), ("rdd", "distribution")):
            rows = [{"pair_id": w["pair_id"], "status": w["status"], "value": w["metrics"][prefix][key] if w.get("metrics") else None, "n": w["n"]} for w in windows]
            window_summaries[key] = _window_distribution_summary(rows, [r["pair_id"] for r in pair_rows])
        result["B2"] = {"status": "EVALUABLE" if all(w["status"] == "EVALUABLE" for w in windows) else "PARTIAL_OUTPUT",
                         "expected_window_rows": len(window_inventory), "window_count": len(windows),
                         "window_rows": windows, "metric_distributions": window_summaries,
                         "structure_by_scene_window": structures,
                         "structure_population": _distribution_summary(structures, "sobel.direction_loss"),
                         "range_retention_population": _distribution_summary(structures, "range_retention.value")}

    if "B3" in regimes:
        result["B3"] = {"status": "NOT_RUN_SYNTHETIC_METHOD_OUTPUT_REQUIRED", "reason": "B3 is a separate known-truth simulation; real overlap metrics are not a substitute."}
    seam_eligible = True
    if "A" in regimes and result.get("A"):
        seam_type = submission.get("seam_type", "NONE")
        seam_status = result["A"].get("seam", {}).get("status")
        seam_eligible = seam_type == "NONE" or seam_status == "EVALUABLE"
    result["ranking_eligible"] = bool(
        result.get("submission_audit", {}).get("artifact_status") == "PASS"
        and submission.get("status") == "COMPLETED"
        and resource["budget_preflight_status"] == "PASS"
        and resource["measurement_provenance_status"] == "PASS"
        and seam_eligible
        and all(result.get(r) and result[r].get("status") == "EVALUABLE" for r in ("A", "B1", "B2") if r in regimes)
        and ("A" not in regimes or (result["A"].get("mosaic_coverage", {}).get("coverage") == 1.0
             and result["A"].get("mosaic_coverage", {}).get("extras") == 0
             and result["A"].get("single_source_structure", {}).get("status") == "EVALUABLE"))
        and ("B1" not in regimes or result["B1"].get("positive_overlap_pairs") == result["B1"].get("metric_summaries", {}).get("mae", {}).get("evaluable_pairs"))
        and ("B2" not in regimes or (result["B2"].get("window_count", 0) > 0
             and result["B2"].get("window_count") == result["B2"].get("expected_window_rows")))
    )
    return result


def _b1_stratified_summaries(scored, inventory, metric_names):
    metadata = {row["pair_id"]: row for row in inventory}
    output = {}
    for metric_name, path in metric_names.items():
        section, key = path.split(".")
        for dimension, field in (("time_stratum", "time_stratum"),):
            groups = {}
            for row in scored:
                source = metadata[row["pair_id"]]
                if source.get("status") != "OVERLAP":
                    continue
                group = source.get(field, "UNKNOWN")
                metric = row.get("metrics", {}).get(section)
                groups.setdefault(group, []).append({"pair_id": row["pair_id"], "status": row["status"],
                                                       "value": metric.get(key) if metric else None, "n": row["n"]})
            output.setdefault(dimension, {})[metric_name] = {
                group: aggregate_pair_rows(rows, [p["pair_id"] for p in inventory
                                                    if p.get("status") == "OVERLAP" and p.get(field, "UNKNOWN") == group])
                for group, rows in sorted(groups.items())
            }
    return output


def _b1_quality_summaries(scored, inventory, metric_names):
    metadata = {row["pair_id"]: row for row in inventory}
    groups = {"ALL_PASS": [], "RISK_CONTAINING": [], "UNKNOWN": []}
    output = {}
    for metric_name, path in metric_names.items():
        section, key = path.split(".")
        for group in groups:
            selected = []
            for row in scored:
                source = metadata[row["pair_id"]]
                if source.get("status") != "OVERLAP":
                    continue
                qi, qj = source.get("quality_i", "UNKNOWN"), source.get("quality_j", "UNKNOWN")
                actual = "RISK_CONTAINING" if "RISK" in (qi, qj) else "ALL_PASS" if qi == qj == "PASS" else "UNKNOWN"
                if actual == group:
                    metric = row.get("metrics", {}).get(section)
                    selected.append({"pair_id": row["pair_id"], "status": row["status"],
                                     "value": metric.get(key) if metric else None, "n": row["n"]})
            expected = []
            for pair in inventory:
                if pair.get("status") != "OVERLAP":
                    continue
                qi, qj = pair.get("quality_i", "UNKNOWN"), pair.get("quality_j", "UNKNOWN")
                pair_group = "RISK_CONTAINING" if "RISK" in (qi, qj) else "ALL_PASS" if qi == qj == "PASS" else "UNKNOWN"
                if pair_group == group:
                    expected.append(pair["pair_id"])
            output.setdefault(metric_name, {})[group] = aggregate_pair_rows(selected, expected)
    return output


def _window_distribution_summary(rows, pair_ids):
    aggregated = aggregate_windows(rows, pair_ids)
    per_pair = aggregated["per_pair"]
    macro = {}
    for statistic in ("median", "mean", "p90", "max"):
        values = [float(row[statistic]) for row in per_pair.values() if row.get(statistic) is not None]
        macro[statistic] = float(np.mean(values)) if values else None
    return {**aggregated, "pair_macro": macro}


def evaluate_submission(freeze_root, submission: dict, *, submission_dir=None, regimes=("A", "B1", "B2")) -> dict:
    """Fail-closed public wrapper returning machine-readable artifact errors."""
    try:
        return _evaluate_submission(freeze_root, submission, submission_dir=submission_dir, regimes=regimes)
    except FileNotFoundError as exc:
        return {"status": "MISSING_ARTIFACT", "error": str(exc)}
    except (OSError, ValueError, KeyError) as exc:
        return {"status": "INVALID_ARTIFACT", "error": f"{type(exc).__name__}:{exc}"}


def _structure_by_scene(scene_ids, inputs, outputs, masks, output_masks):
    rows = []
    for scene_id, x, y, mask, outmask in zip(scene_ids, inputs, outputs, masks, output_masks):
        if np.any(np.asarray(mask, bool) & ~np.asarray(outmask, bool)):
            rows.append({"scene_id": scene_id, "status": "PARTIAL_OUTPUT", "support_pixels": int(mask.sum()),
                         "sobel": None, "range_retention": None})
            continue
        st = compute_sobel_structure(x, y, mask)
        rr = compute_range_retention(x, y, mask)
        rows.append({"scene_id": scene_id, "status": st.status, "support_pixels": int(mask.sum()),
                     "sobel": st.__dict__, "range_retention": rr.__dict__})
    return rows


def _distribution_summary(rows, dotted_key):
    parts = dotted_key.split(".")
    values = []
    for row in rows:
        item = row
        for part in parts:
            item = item.get(part) if isinstance(item, dict) else None
        if item is not None and math.isfinite(float(item)):
            values.append(float(item))
    if not values:
        return {"status": "NOT_EVALUABLE", "n": 0, "median": None, "p95": None, "mean": None}
    return {"status": "EVALUABLE", "n": len(values), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95, method="linear")), "mean": float(np.mean(values))}
