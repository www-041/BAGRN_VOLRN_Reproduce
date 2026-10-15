from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio

from src.benchmark_v1_unified.freeze import verify_lock
from src.benchmark_v1_unified.identity import sha256_file
from src.benchmark_v1_unified.raw_mosaic import build_fixed_source_mosaic
from src.benchmark_v1_unified.reporting import write_json_atomic
from src.benchmark_v1_unified.resource_guard import ProcessResourceMonitor
from src.benchmark_v1_unified.streaming_submission_scoring import (
    DISK_RESERVE_BYTES,
    evaluate_submission_streaming,
    _read_csv,
    _resource_preflight,
    _validate_grid,
)


DEFAULT_FREEZE = "data/output/benchmark_v1_unified/freeze_20261013_v3"
DEFAULT_OUTPUT = "data/output/benchmark_v1_unified/raw_baseline_v1_20261014"


def _lock_index(freeze_root):
    lock = json.loads((freeze_root / "freeze_lock.json").read_text(encoding="utf-8"))
    return {str(Path(item["path"]).resolve()).casefold(): item["sha256"] for item in lock["files"]}


def _preflight(freeze_root: Path, output_root: Path, ram_budget_bytes: int,
               ram_reserve_bytes: int, disk_reserve_bytes: int, block_shape):
    locked = verify_lock(freeze_root)
    if not locked.valid:
        return {"status": "FREEZE_INVALID", "errors": list(locked.errors)}
    frozen = json.loads((freeze_root / "frozen_manifest.json").read_text(encoding="utf-8"))
    pairs = _read_csv(freeze_root / "pair_overlap_inventory.csv")
    windows = _read_csv(freeze_root / "fixed_windows.csv")
    scenes = []
    for row in pairs:
        for field in ("scene_i", "scene_j"):
            if row[field] not in scenes:
                scenes.append(row[field])
    if len(scenes) != 13 or int(frozen.get("scene_count", -1)) != 13:
        return {"status": "SCENE_INVENTORY_MISMATCH", "scene_count": len(scenes)}
    if len(pairs) != 78 or sum(row["status"] == "OVERLAP" for row in pairs) != 48 or sum(row["status"] == "NO_VALID_OVERLAP" for row in pairs) != 30:
        return {"status": "PAIR_INVENTORY_MISMATCH", "pair_rows": len(pairs),
                "overlap": sum(row["status"] == "OVERLAP" for row in pairs),
                "no_valid_overlap": sum(row["status"] == "NO_VALID_OVERLAP" for row in pairs)}
    if len(windows) != 1533 or any(not row.get("support_sha256") for row in windows):
        return {"status": "WINDOW_INVENTORY_MISMATCH", "window_rows": len(windows),
                "missing_support_hashes": sum(not row.get("support_sha256") for row in windows)}
    if output_root.exists():
        return {"status": "OUTPUT_ROOT_ALREADY_EXISTS", "path": str(output_root)}
    shape = (int(frozen["grid"]["height"]), int(frozen["grid"]["width"]))
    transform = tuple(float(x) for x in frozen["grid"]["transform"])
    crs = frozen["grid"]["crs"]
    lock_hashes = _lock_index(freeze_root)
    inputs, masks = [], []
    for idx in range(len(scenes)):
        inputs.append(freeze_root / "aligned_inputs" / f"scene_{idx:03d}_B9_input.tif")
        masks.append(freeze_root / "aligned_inputs" / f"scene_{idx:03d}_input_mask.tif")
    header_rows = []
    try:
        for kind, paths in (("RAW_INPUT", inputs), ("FROZEN_MASK", masks)):
            for idx, path in enumerate(paths):
                if not path.is_file() or str(path.resolve()).casefold() not in lock_hashes:
                    return {"status": "INPUT_NOT_LOCKED_OR_MISSING", "path": str(path)}
                with rasterio.open(path) as ds:
                    _validate_grid(ds, path, shape, transform, crs)
                    header_rows.append({"kind": kind, "scene_id": scenes[idx], "path": str(path.resolve()),
                                        "dtype": ds.dtypes[0], "nodata": ds.nodata,
                                        "width": ds.width, "height": ds.height,
                                        "crs": ds.crs.to_string(), "transform": tuple(ds.transform)[:6],
                                        "sha256": lock_hashes[str(path.resolve()).casefold()]})
    except (OSError, ValueError) as exc:
        return {"status": "INPUT_HEADER_INVALID", "error": f"{type(exc).__name__}:{exc}"}

    submission = {"resource_budget": {"max_rss_bytes": ram_budget_bytes}}
    # Scratch is created only after all resource checks pass.
    scratch_probe = output_root / "scratch"
    resource = _resource_preflight(submission, shape, pairs, scratch_probe,
                                   ram_budget_bytes, ram_reserve_bytes, disk_reserve_bytes)
    if resource["status"] != "PASS":
        return {"status": resource["status"], "resource": resource}
    output_estimate = int(shape[0] * shape[1] * 6)  # float32 mosaic + uint8 mask + uint8 labels
    extra_disk_needed = output_estimate + disk_reserve_bytes
    if resource["available_scratch_disk_bytes"] < resource["estimated_peak_scratch_bytes"] + extra_disk_needed:
        resource["status"] = "RESOURCE_PREFLIGHT_FAILED"
        resource["reasons"].append("OUTPUT_DISK_BELOW_ESTIMATE_PLUS_RESERVE")
        return {"status": resource["status"], "resource": resource,
                "estimated_output_bytes": output_estimate}
    return {
        "status": "PASS", "protocol_id": frozen.get("protocol_id"),
        "freeze_root": str(freeze_root.resolve()), "freeze_lock_status": "PASS",
        "freeze_lock_file_count": len(json.loads((freeze_root / "freeze_lock.json").read_text(encoding="utf-8"))["files"]),
        "scene_count": len(scenes), "scene_ids": scenes,
        "grid": {"shape": shape, "transform": transform, "crs": crs},
        "pair_rows": len(pairs), "overlap_pairs": 48, "no_valid_overlap_pairs": 30,
        "window_rows": len(windows), "support_sha256_rows": len(windows),
        "input_headers": header_rows, "resource": resource,
        "estimated_output_bytes": output_estimate,
        "estimated_total_free_disk_required_bytes": resource["estimated_peak_scratch_bytes"] + extra_disk_needed,
        "block_shape": list(block_shape), "fixed_priority": list(range(len(scenes))),
    }


def _flatten_pair_rows(rows):
    output = []
    for row in rows:
        direct = row.get("metrics", {}).get("direct") or {}
        distribution = row.get("metrics", {}).get("distribution") or {}
        output.append({"pair_id": row["pair_id"], "scene_i": row["scene_i"], "scene_j": row["scene_j"],
                       "status": row["status"], "n": row["n"], "output_pixels": row.get("output_pixels"),
                       "coverage": row.get("coverage"), **{f"direct_{k}": v for k, v in direct.items()},
                       **{f"distribution_{k}": v for k, v in distribution.items()}})
    return output


def _flatten_window_rows(rows):
    output = []
    for row in rows:
        direct = row.get("metrics", {}).get("direct") or {}
        distribution = row.get("metrics", {}).get("distribution") or {}
        output.append({"window_id": row["window_id"], "pair_id": row["pair_id"],
                       "status": row["status"], "n": row["n"],
                       **{f"direct_{k}": v for k, v in direct.items()},
                       **{f"distribution_{k}": v for k, v in distribution.items()}})
    return output


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_log(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_runtime_report(*, stage_reports, active_stage, active_monitor, scratch_dir, failure=None):
    measurements = dict(stage_reports or {})
    if active_monitor is not None and active_stage:
        try:
            measurements[active_stage] = active_monitor.report()
        except Exception as exc:
            measurements[active_stage] = {"status": "MONITOR_ERROR",
                                          "error": f"{type(exc).__name__}:{exc}"}
    scratch = Path(scratch_dir)
    files = []
    if scratch.is_dir():
        for path in scratch.rglob("*"):
            if path.is_file():
                try:
                    files.append({"path": str(path), "bytes": path.stat().st_size})
                except OSError:
                    continue
    total_bytes = sum(item["bytes"] for item in files)
    return {
        "stage_measurements": measurements,
        "method_mosaic_runtime": measurements.get("mosaic"),
        "evaluator_runtime": measurements.get("evaluator"),
        "hard_os_memory_limit": False,
        "scratch": {"path": str(scratch.resolve()), "file_count": len(files),
                    "bytes": total_bytes, "files": files,
                    "cleanup_status": "RETAINED_ON_FAILURE" if files and failure else
                                     "REMAINS" if files else "PASS_EMPTY"},
        "failure": failure,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run read-only Unified V1 RAW baseline A/B1/B2 with bounded memory.")
    parser.add_argument("--freeze", default=DEFAULT_FREEZE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--block-height", type=int, default=512)
    parser.add_argument("--block-width", type=int, default=512)
    parser.add_argument("--ram-budget-mib", type=int, default=512)
    parser.add_argument("--ram-reserve-mib", type=int, default=256)
    parser.add_argument("--disk-reserve-gib", type=int, default=2)
    args = parser.parse_args()

    freeze_root = Path(args.freeze).resolve()
    output_root = Path(args.output_root).resolve()
    block_shape = (args.block_height, args.block_width)
    ram_budget = args.ram_budget_mib * 1024**2
    ram_reserve = args.ram_reserve_mib * 1024**2
    disk_reserve = args.disk_reserve_gib * 1024**3
    preflight = _preflight(freeze_root, output_root, ram_budget, ram_reserve, disk_reserve, block_shape)
    if preflight["status"] != "PASS":
        print(json.dumps(preflight, ensure_ascii=False))
        return 2

    output_root.mkdir(parents=True)
    scratch_dir = output_root / "scratch"
    log_dir = output_root / "logs"
    log_dir.mkdir()
    logging.basicConfig(filename=log_dir / "run.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("preflight=%s", json.dumps(preflight, ensure_ascii=False))
    write_json_atomic(output_root / "resource_preflight.json", preflight)
    _json_log(log_dir / "events.jsonl", {"event": "PREFLIGHT_PASS", "time_utc": datetime.now(timezone.utc).isoformat()})
    frozen = json.loads((freeze_root / "frozen_manifest.json").read_text(encoding="utf-8"))
    pair_rows = _read_csv(freeze_root / "pair_overlap_inventory.csv")
    scenes = preflight["scene_ids"]
    input_paths = [freeze_root / "aligned_inputs" / f"scene_{idx:03d}_B9_input.tif" for idx in range(len(scenes))]
    mask_paths = [freeze_root / "aligned_inputs" / f"scene_{idx:03d}_input_mask.tif" for idx in range(len(scenes))]
    mosaic_path = output_root / "mosaic_B9_raw.tif"
    mosaic_mask_path = output_root / "mosaic_valid_mask.tif"
    labels_path = output_root / "source_labels.tif"
    status_path = output_root / "solver_status.json"
    runtime_path = output_root / "runtime.json"
    stage_reports = {}
    active_stage = None
    active_monitor = None
    failure_record = None
    evaluation = None
    mosaic_record = None
    try:
        with ProcessResourceMonitor(ram_budget, [scratch_dir], sample_interval=0.2) as monitor:
            active_stage, active_monitor = "mosaic", monitor
            mosaic_record = build_fixed_source_mosaic(
                input_paths, mask_paths, mosaic_path, mosaic_mask_path, labels_path,
                scene_ids=scenes, expected_shape=tuple(preflight["grid"]["shape"]),
                expected_transform=preflight["grid"]["transform"], expected_crs=preflight["grid"]["crs"],
                priority=preflight["fixed_priority"], block_shape=block_shape, check=monitor.check,
            )
        mosaic_runtime = monitor.report()
        stage_reports["mosaic"] = mosaic_runtime
        active_stage = active_monitor = None
        if mosaic_runtime["status"] != "MEASURED":
            raise RuntimeError(f"MOSAIC_RESOURCE_GUARD:{mosaic_runtime['status']}")
        write_json_atomic(output_root / "mosaic_backend.json", {
            **mosaic_record,
            "algorithm_claim": "shared deterministic fixed-priority hard-source backend; not a paper-specific BAGRN/VOLRN seam algorithm",
            "source_priority_policy": "ascending frozen scene index; first valid source wins",
            "input_support_policy": "frozen mask AND GeoTIFF dataset mask AND finite value",
            "mosaic_runtime_measurement": mosaic_runtime,
            "hard_os_memory_limit": False,
        })
        _json_log(log_dir / "events.jsonl", {"event": "MOSAIC_COMPLETED", "record": mosaic_record,
                    "runtime": mosaic_runtime, "time_utc": datetime.now(timezone.utc).isoformat()})

        input_hashes = {scene: sha256_file(path) for scene, path in zip(scenes, input_paths)}
        output_hashes = {name: sha256_file(path) for name, path in (
            ("mosaic", mosaic_path), ("mask", mosaic_mask_path), ("source_labels", labels_path))}
        method_submission = {
            "schema_version": 1, "method_id": "RAW_FIXED_PRIORITY",
            "run_id": "raw_baseline_v1_20261014", "method_kind": "FULL_MOSAIC",
            "input_freeze_sha256": frozen["input_freeze_sha256"],
            "geometry_freeze_sha256": frozen["geometry_freeze_sha256"],
            "band": "B9", "pixel_value_unit": "UNKNOWN", "status": "COMPLETED",
            "corrected_scene_paths": {scene: str(path.resolve()) for scene, path in zip(scenes, input_paths)},
            "output_mosaic_path": mosaic_path.name,
            "output_mosaic_mask_path": mosaic_mask_path.name,
            "source_label_path": labels_path.name,
            "seam_type": "HARD",
            "resource_budget": {"max_rss_bytes": ram_budget, "gpu_memory_bytes": None},
            "runtime": {
                "status": "MEASURED", "wall_seconds": mosaic_runtime["wall_seconds"],
                "wall_measurement_source": mosaic_runtime["wall_measurement_source"],
                "peak_rss_bytes": mosaic_runtime["peak_rss_bytes"],
                "rss_status": "MEASURED", "rss_sample_count": mosaic_runtime["rss_sample_count"],
                "rss_measurement_source": mosaic_runtime["rss_measurement_source"],
                "gpu_status": "NOT_MEASURED", "gpu_memory_measurement_source": "NOT_APPLICABLE_CPU_ONLY",
                "hard_os_memory_limit": False,
                "algorithm_runtime_status": "NOT_APPLICABLE_RAW_IDENTITY",
                "mosaic_backend_seconds": mosaic_record["elapsed_seconds"],
                "output_write_seconds": mosaic_record["output_write_seconds"],
            },
            "raw_input_sha256": input_hashes,
            "output_artifacts_sha256": output_hashes,
            "fixed_source_priority": preflight["fixed_priority"],
            "frozen_scene_id_to_label": mosaic_record["scene_id_to_label"],
        }
        write_json_atomic(output_root / "method_submission.json", method_submission)

        with ProcessResourceMonitor(ram_budget, [scratch_dir], sample_interval=0.2) as monitor:
            active_stage, active_monitor = "evaluator", monitor
            evaluation = evaluate_submission_streaming(
                freeze_root, method_submission, submission_dir=output_root, regimes=("A", "B1", "B2", "B3"),
                scratch_dir=scratch_dir, block_shape=block_shape, ram_budget_bytes=ram_budget,
                ram_reserve_bytes=ram_reserve, disk_reserve_bytes=disk_reserve, check=monitor.check,
            )
        evaluator_runtime = monitor.report()
        stage_reports["evaluator"] = evaluator_runtime
        active_stage = active_monitor = None
        evaluation["evaluator_runtime"] = evaluator_runtime
        evaluation["runtime_separation"] = {
            "method_stage": "RAW mosaic backend only; no radiometric algorithm executed",
            "evaluator_stage": "A/B1/B2 scoring, separate from method runtime",
        }
        write_json_atomic(output_root / "evaluation_A_B1_B2.json", evaluation)
        if evaluation.get("status") != "EVALUATED" or evaluator_runtime["status"] != "MEASURED":
            raise RuntimeError(f"EVALUATION_OR_RESOURCE_GUARD:{evaluation.get('status')}:{evaluator_runtime['status']}")
        _write_csv(output_root / "b1_pairs.csv", _flatten_pair_rows(evaluation["B1"]["pair_rows"]))
        _write_csv(output_root / "b2_windows.csv", _flatten_window_rows(evaluation["B2"]["window_rows"]))
        _write_csv(output_root / "scene_structure.csv", [
            {"scene_id": row["scene_id"], "status": row["status"], "support_pixels": row["support_pixels"],
             "sobel": json.dumps(row["sobel"], ensure_ascii=False),
             "range_retention": json.dumps(row["range_retention"], ensure_ascii=False)}
            for row in evaluation["scene_structure"]])
        _write_csv(output_root / "b2_scene_structure.csv", [
            {"window_id": row["window_id"], "pair_id": row["pair_id"], "scene_id": row["scene_id"],
             "status": row["status"], "sobel": json.dumps(row["sobel"], ensure_ascii=False),
             "range_retention": json.dumps(row["range_retention"], ensure_ascii=False)}
            for row in evaluation["B2"]["structure_by_scene_window"]])
        status = {
            "status": "COMPLETED" if evaluation["ranking_eligible"] else "COMPLETED_DIAGNOSTIC_NOT_RANKING_ELIGIBLE",
            "ranking_eligible": evaluation["ranking_eligible"],
            "A_status": evaluation["A"]["status"], "B1_status": evaluation["B1"]["status"],
            "B2_status": evaluation["B2"]["status"], "B3_status": evaluation["B3"]["status"],
            "pair_rows": evaluation["B1"]["expected_pair_rows"],
            "overlap_pairs": evaluation["B1"]["positive_overlap_pairs"],
            "no_valid_overlap_pairs": evaluation["B1"]["no_valid_overlap_pairs"],
            "window_rows": evaluation["B2"]["window_count"],
            "coverage": evaluation["A"]["mosaic_coverage"],
            "seam_status": evaluation["A"]["seam"]["status"],
            "method": "RAW input identity; no BAGRN/VOLRN or external algorithm run",
            "report": str((output_root / "evaluation_A_B1_B2.json").resolve()),
        }
        write_json_atomic(status_path, status)
        _json_log(log_dir / "events.jsonl", {"event": "SCORING_COMPLETED", "status": status,
                    "time_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps(status, ensure_ascii=False))
        return 0 if evaluation["ranking_eligible"] else 3
    except BaseException as exc:
        failure = {"status": "STOPPED", "error": f"{type(exc).__name__}:{exc}",
                   "freeze_root": str(freeze_root), "output_root": str(output_root),
                   "time_utc": datetime.now(timezone.utc).isoformat(),
                   "no_bagrn_volrn_or_external_methods_run": True}
        failure_record = failure
        if output_root.exists():
            write_json_atomic(status_path, failure)
            _json_log(log_dir / "events.jsonl", {"event": "STOPPED", **failure})
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 2
    finally:
        runtime = build_runtime_report(stage_reports=stage_reports, active_stage=active_stage,
                                       active_monitor=active_monitor, scratch_dir=scratch_dir,
                                       failure=failure_record)
        runtime.update({
            "mosaic_backend_elapsed_seconds": (mosaic_record or {}).get("elapsed_seconds"),
            "mosaic_output_write_seconds": (mosaic_record or {}).get("output_write_seconds"),
            "evaluator_scratch": evaluation.get("scratch") if isinstance(evaluation, dict) else None,
            "rss_threshold_mib": args.ram_budget_mib,
            "gpu_status": "NOT_MEASURED",
        })
        if output_root.exists():
            write_json_atomic(runtime_path, runtime)


if __name__ == "__main__":
    raise SystemExit(main())
