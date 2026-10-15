"""Windowed A/B1/B2 evaluator for frozen Unified Benchmark V1 submissions."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .aggregation import aggregate_pair_rows, aggregate_windows
from .freeze import verify_lock
from .streaming_metrics import (
    compute_pair_metrics_from_arrays,
    compute_range_retention_from_readers,
    compute_seam_excess_from_readers,
    compute_sobel_structure_from_readers,
)
from .submission import validate_submission


RAM_BUDGET_BYTES = 512 * 1024**2
RAM_RESERVE_BYTES = 256 * 1024**2
DISK_RESERVE_BYTES = 2 * 1024**3


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _resolve(path, base):
    if path is None or not str(path).strip():
        return None
    value = Path(path)
    return value if value.is_absolute() else (Path(base) / value).resolve()


def _mask_sha(mask):
    return hashlib.sha256(np.packbits(np.asarray(mask, dtype=bool), bitorder="little").tobytes()).hexdigest()


def _cleanup_memmap(path, array):
    if array is not None:
        try:
            array.flush()
        except Exception:
            pass
        try:
            array._mmap.close()
        except Exception:
            pass
    Path(path).unlink(missing_ok=True)


def _windows(shape, block_shape):
    height, width = map(int, shape)
    block_h, block_w = map(int, block_shape)
    if block_h < 1 or block_w < 1:
        raise ValueError("INVALID_BLOCK_SHAPE")
    for row in range(0, height, block_h):
        for col in range(0, width, block_w):
            yield Window(col, row, min(block_w, width-col), min(block_h, height-row))


def _validate_grid(ds, path, shape, transform, crs):
    if not Path(path).is_file():
        raise FileNotFoundError(f"MISSING_ARTIFACT:{path}")
    if ds.count != 1:
        raise ValueError(f"INVALID_BAND_COUNT:{path}:{ds.count}")
    if (ds.height, ds.width) != tuple(shape):
        raise ValueError(f"INVALID_SHAPE:{path}:{ds.height}x{ds.width}")
    if ds.crs is None or ds.crs.to_string() != crs:
        raise ValueError(f"INVALID_CRS:{path}:{ds.crs}")
    if not np.allclose(tuple(ds.transform)[:6], transform, rtol=0.0, atol=1e-9):
        raise ValueError(f"INVALID_TRANSFORM:{path}:{tuple(ds.transform)[:6]}")
    dtype = np.dtype(ds.dtypes[0])
    if not np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.complexfloating):
        raise ValueError(f"INVALID_PIXEL_DTYPE:{path}:{ds.dtypes[0]}")


def _read_mask(mask_ds, window):
    values = mask_ds.read(1, window=window)
    return (values > 0) & (mask_ds.dataset_mask(window=window) > 0)


def _read_valid_values(ds, window):
    values = ds.read(1, window=window)
    valid = ds.dataset_mask(window=window) > 0
    if np.issubdtype(values.dtype, np.floating):
        valid &= np.isfinite(values)
    return values, valid


def _summaries(scored, inventory):
    metric_names = {
        "mae": ("direct", "mae"), "rmse": ("direct", "rmse"),
        "bias": ("direct", "bias"), "p95ae": ("direct", "p95ae"),
        "mamd": ("distribution", "mamd"), "msdd": ("distribution", "msdd"),
        "rdd": ("distribution", "rdd"),
    }
    output = {}
    for metric_name, (section, key) in metric_names.items():
        rows = []
        for row in scored:
            metric_values = (row.get("metrics") or {}).get(section) or {}
            rows.append({"pair_id": row["pair_id"], "status": row["status"], "n": row["n"],
                         "value": metric_values.get(key)})
        expected = [r["pair_id"] for r in inventory if r["status"] == "OVERLAP"]
        output[metric_name] = aggregate_pair_rows(rows, expected)
    return output


def _time_summaries(scored, inventory):
    metadata = {r["pair_id"]: r for r in inventory}
    keys = {"mae": ("direct", "mae"), "rmse": ("direct", "rmse"),
            "bias": ("direct", "bias"), "p95ae": ("direct", "p95ae"),
            "mamd": ("distribution", "mamd"), "msdd": ("distribution", "msdd"),
            "rdd": ("distribution", "rdd")}
    result = {}
    for metric_name, (section, key) in keys.items():
        groups = {}
        for row in scored:
            source = metadata[row["pair_id"]]
            if source["status"] != "OVERLAP":
                continue
            stratum = source.get("time_stratum") or "UNKNOWN"
            metric_values = (row.get("metrics") or {}).get(section) or {}
            groups.setdefault(stratum, []).append({
                "pair_id": row["pair_id"], "status": row["status"], "n": row["n"],
                "value": metric_values.get(key),
            })
        result[metric_name] = {
            stratum: aggregate_pair_rows(rows, [p["pair_id"] for p in inventory
                if p["status"] == "OVERLAP" and (p.get("time_stratum") or "UNKNOWN") == stratum])
            for stratum, rows in sorted(groups.items())
        }
    return result


def _quality_summaries(scored, inventory):
    metadata = {r["pair_id"]: r for r in inventory}
    metrics = {"mae": ("direct", "mae"), "rmse": ("direct", "rmse"),
               "bias": ("direct", "bias"), "p95ae": ("direct", "p95ae"),
               "mamd": ("distribution", "mamd"), "msdd": ("distribution", "msdd"),
               "rdd": ("distribution", "rdd")}
    result = {}
    for metric_name, (section, key) in metrics.items():
        groups = {"ALL_PASS": [], "RISK_CONTAINING": [], "UNKNOWN": []}
        expected = {name: [] for name in groups}
        for pair in inventory:
            if pair["status"] != "OVERLAP":
                continue
            qi, qj = pair.get("quality_i", "UNKNOWN"), pair.get("quality_j", "UNKNOWN")
            group = "RISK_CONTAINING" if "RISK" in (qi, qj) else "ALL_PASS" if qi == qj == "PASS" else "UNKNOWN"
            expected[group].append(pair["pair_id"])
        for row in scored:
            source = metadata[row["pair_id"]]
            if source["status"] != "OVERLAP":
                continue
            qi, qj = source.get("quality_i", "UNKNOWN"), source.get("quality_j", "UNKNOWN")
            group = "RISK_CONTAINING" if "RISK" in (qi, qj) else "ALL_PASS" if qi == qj == "PASS" else "UNKNOWN"
            metric_values = (row.get("metrics") or {}).get(section) or {}
            groups[group].append({"pair_id": row["pair_id"], "status": row["status"], "n": row["n"],
                                  "value": metric_values.get(key)})
        result[metric_name] = {group: aggregate_pair_rows(groups[group], expected[group]) for group in groups}
    return result


def _window_summary(rows, pair_ids):
    aggregated = aggregate_windows(rows, pair_ids)
    per_pair = aggregated["per_pair"]
    macro = {}
    for statistic in ("median", "mean", "p90", "max"):
        values = [float(row[statistic]) for row in per_pair.values() if row.get(statistic) is not None]
        macro[statistic] = float(np.mean(values)) if values else None
    return {**aggregated, "pair_macro": macro}


def _distribution_summary(rows, dotted_key):
    _, section, key = dotted_key.split(".")
    values = [float(row[section][key]) for row in rows
              if row.get(section) and row[section].get(key) is not None
              and math.isfinite(float(row[section][key]))]
    if not values:
        return {"status": "NOT_EVALUABLE", "n": 0, "median": None, "p95": None, "mean": None}
    return {"status": "EVALUABLE", "n": len(values), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95, method="linear")), "mean": float(np.mean(values))}


def _resource_preflight(submission, shape, pair_rows, scratch_dir, ram_budget_bytes, ram_reserve_bytes,
                        disk_reserve_bytes):
    reasons = []
    scratch_path = Path(scratch_dir).resolve()
    volume_probe = scratch_path
    while not volume_probe.exists() and volume_probe != volume_probe.parent:
        volume_probe = volume_probe.parent
    try:
        import psutil
        available_ram = int(psutil.virtual_memory().available)
        ram_source = "psutil.virtual_memory().available"
        disk = shutil.disk_usage(volume_probe)
        available_disk = int(disk.free)
        disk_source = "shutil.disk_usage(scratch_path_or_nearest_existing_parent).free"
    except Exception as exc:
        return {"status": "RESOURCE_MONITOR_UNAVAILABLE", "reasons": [f"RESOURCE_MONITOR_UNAVAILABLE:{exc}"]}
    n_pixels = int(shape[0] * shape[1])
    overlap_counts = [int(r.get("aligned_valid_overlap_pixels") or 0) for r in pair_rows]
    pair_spool_bytes = 2 * sum(overlap_counts) * np.dtype(np.float64).itemsize
    max_pair_count = max(overlap_counts, default=0)
    exact_order_scratch = max(n_pixels * 2 * 8, max_pair_count * 8)
    scratch_estimate = pair_spool_bytes + max(max_pair_count * 8, exact_order_scratch)
    if ram_budget_bytes < 128 * 1024**2:
        reasons.append("RAM_BUDGET_BELOW_STREAMING_ESTIMATE")
    if available_ram < ram_budget_bytes + ram_reserve_bytes:
        reasons.append("AVAILABLE_RAM_BELOW_BUDGET_PLUS_RESERVE")
    if available_disk < scratch_estimate + disk_reserve_bytes:
        reasons.append("SCRATCH_DISK_BELOW_ESTIMATE_PLUS_RESERVE")
    method_budget = (submission.get("resource_budget") or {}).get("max_rss_bytes")
    return {
        "status": "PASS" if not reasons else "RESOURCE_PREFLIGHT_FAILED",
        "reasons": reasons,
        "available_ram_bytes": available_ram,
        "available_ram_measurement_source": ram_source,
        "evaluator_ram_budget_bytes": int(ram_budget_bytes),
        "ram_safety_reserve_bytes": int(ram_reserve_bytes),
        "declared_method_max_rss_bytes": method_budget,
        "hard_os_memory_limit": False,
        "available_scratch_disk_bytes": available_disk,
        "scratch_measurement_source": disk_source,
        "scratch_path": str(scratch_path),
        "scratch_volume_probe_path": str(volume_probe),
        "estimated_pair_sample_scratch_bytes": pair_spool_bytes,
        "estimated_exact_order_scratch_bytes": exact_order_scratch,
        "estimated_peak_scratch_bytes": scratch_estimate,
        "disk_safety_reserve_bytes": int(disk_reserve_bytes),
    }


def _schema_check(submission, frozen):
    audit = validate_submission(submission, {
        "input_freeze_sha256": frozen.get("input_freeze_sha256"),
        "geometry_freeze_sha256": frozen.get("geometry_freeze_sha256"),
        "band": "B9",
    })
    errors = []
    for name in ("schema_version", "method_id", "run_id", "method_kind", "pixel_value_unit", "status"):
        if name not in submission:
            errors.append(f"MISSING_{name.upper()}")
    for name in ("method_id", "run_id", "status"):
        if name in submission and (not isinstance(submission[name], str) or not submission[name].strip()):
            errors.append(f"INVALID_{name.upper()}")
    if submission.get("schema_version") != 1:
        errors.append("INVALID_SCHEMA_VERSION")
    if submission.get("pixel_value_unit") != "UNKNOWN":
        errors.append("INVALID_PIXEL_VALUE_UNIT")
    if submission.get("method_kind") not in {"FULL_MOSAIC", "SCENE_OUTPUTS", "PAIR_ONLY", "SYNTHETIC_FIXTURE"}:
        errors.append("INVALID_METHOD_KIND")
    if submission.get("seam_type") is not None and submission.get("seam_type") not in {"HARD", "SOFT", "NONE"}:
        errors.append("INVALID_SEAM_TYPE")
    if not audit["eligible"] or errors:
        return {**audit, "errors": audit["errors"] + errors, "eligible": False}
    return {**audit, "errors": [], "eligible": True}


def evaluate_submission_streaming(freeze_root, submission, *, submission_dir=None,
                                  regimes=("A", "B1", "B2"), scratch_dir,
                                  block_shape=(512, 512), ram_budget_bytes=RAM_BUDGET_BYTES,
                                  ram_reserve_bytes=RAM_RESERVE_BYTES,
                                  disk_reserve_bytes=DISK_RESERVE_BYTES, check=None,
                                  freeze_verifier=None):
    """Score a submission without materializing the full canonical raster grid."""
    freeze_root = Path(freeze_root).resolve()
    base = Path(submission_dir or ".").resolve()
    scratch_dir = Path(scratch_dir).resolve()
    regimes = tuple(dict.fromkeys(str(r).upper() for r in regimes))
    if set(regimes) - {"A", "B1", "B2", "B3"}:
        raise ValueError(f"INVALID_REGIME:{sorted(set(regimes) - {'A', 'B1', 'B2', 'B3'})}")
    locked = (freeze_verifier or verify_lock)(freeze_root)
    if not locked.valid:
        return {"status": "FREEZE_INVALID", "errors": list(locked.errors)}
    frozen = json.loads((freeze_root / "frozen_manifest.json").read_text(encoding="utf-8"))
    audit = _schema_check(submission, frozen)
    if not audit["eligible"]:
        return {"status": "SUBMISSION_INVALID", "submission_audit": audit}
    if regimes == ("B3",):
        return {"protocol_id": frozen.get("protocol_id"), "status": "EVALUATED",
                "submission_audit": {**audit, "artifact_status": "NOT_REQUIRED_FOR_B3"},
                "B3": {"status": "NOT_RUN_SYNTHETIC_METHOD_OUTPUT_REQUIRED",
                       "reason": "B3 is a separate known-truth simulation; real overlap metrics are not a substitute."},
                "ranking_eligible": False}

    pair_rows = _read_csv(freeze_root / "pair_overlap_inventory.csv")
    scenes = []
    for row in pair_rows:
        for field in ("scene_i", "scene_j"):
            if row[field] not in scenes:
                scenes.append(row[field])
    if len(scenes) != int(frozen.get("scene_count", len(scenes))):
        return {"status": "FREEZE_INVENTORY_MISMATCH", "scene_count": len(scenes)}
    grid = frozen["grid"]
    shape = (int(grid["height"]), int(grid["width"]))
    transform = tuple(float(v) for v in grid["transform"])
    crs = str(grid["crs"])
    preflight = _resource_preflight(submission, shape, pair_rows, scratch_dir,
                                    ram_budget_bytes, ram_reserve_bytes, disk_reserve_bytes)
    if preflight["status"] != "PASS":
        return {"status": "RESOURCE_PREFLIGHT_FAILED", "resource_preflight": preflight}

    corrected_paths = submission.get("corrected_scene_paths") or {}
    if set(corrected_paths) != set(scenes):
        return {"status": "SCENE_OUTPUT_IDENTITY_MISMATCH",
                "missing_scene_ids": sorted(set(scenes) - set(corrected_paths)),
                "extra_scene_ids": sorted(set(corrected_paths) - set(scenes))}
    input_paths = [freeze_root / "aligned_inputs" / f"scene_{idx:03d}_B9_input.tif" for idx in range(len(scenes))]
    input_mask_paths = [freeze_root / "aligned_inputs" / f"scene_{idx:03d}_input_mask.tif" for idx in range(len(scenes))]
    output_paths = [_resolve(corrected_paths[scene], base) for scene in scenes]
    for idx, path in enumerate(output_paths):
        if path is None or not path.is_file():
            return {"status": "MISSING_SCENE_OUTPUT", "scene_id": scenes[idx], "path": str(path)}

    scene_index = {scene: i for i, scene in enumerate(scenes)}
    shape_size = shape[0] * shape[1]
    scene_structure = []
    b1_rows = []
    b2_rows = []
    b2_structures = []
    A_result = None
    input_union_pixels = 0
    scratch_peak_observed = 0
    corrected_dtypes = {}
    mosaic_path = _resolve(submission.get("output_mosaic_path"), base)
    mosaic_mask_path = _resolve(submission.get("output_mosaic_mask_path"), base)
    labels_path = _resolve(submission.get("source_label_path"), base)

    try:
        with ExitStack() as stack:
            input_ds = [stack.enter_context(rasterio.open(p)) for p in input_paths]
            mask_ds = [stack.enter_context(rasterio.open(p)) for p in input_mask_paths]
            output_ds = []
            open_by_path = {str(p.resolve()): ds for p, ds in zip(input_paths, input_ds)}
            for p in output_paths:
                key = str(p.resolve())
                if key in open_by_path:
                    output_ds.append(open_by_path[key])
                else:
                    ds = stack.enter_context(rasterio.open(p))
                    open_by_path[key] = ds
                    output_ds.append(ds)
            for ds, path in zip(input_ds, input_paths):
                _validate_grid(ds, path, shape, transform, crs)
            for ds, path in zip(mask_ds, input_mask_paths):
                _validate_grid(ds, path, shape, transform, crs)
                if ds.dtypes[0] not in {"uint8", "int8", "uint16", "int16", "uint32", "int32"}:
                    raise ValueError(f"INVALID_FROZEN_MASK_DTYPE:{path}:{ds.dtypes[0]}")
            for scene, ds, path in zip(scenes, output_ds, output_paths):
                _validate_grid(ds, path, shape, transform, crs)
                corrected_dtypes[scene] = ds.dtypes[0]

            mosaic_ds = maskout_ds = label_ds = None
            if "A" in regimes:
                if mosaic_path is None or mosaic_mask_path is None:
                    return {"status": "MISSING_MOSAIC_OR_MASK"}
                for path in (mosaic_path, mosaic_mask_path):
                    if not path.is_file():
                        return {"status": "MISSING_ARTIFACT", "error": f"MISSING_ARTIFACT:{path}"}
                mosaic_ds = stack.enter_context(rasterio.open(mosaic_path))
                maskout_ds = stack.enter_context(rasterio.open(mosaic_mask_path))
                _validate_grid(mosaic_ds, mosaic_path, shape, transform, crs)
                _validate_grid(maskout_ds, mosaic_mask_path, shape, transform, crs)
                if labels_path is not None:
                    if not labels_path.is_file():
                        return {"status": "MISSING_ARTIFACT", "error": f"MISSING_ARTIFACT:{labels_path}"}
                    label_ds = stack.enter_context(rasterio.open(labels_path))
                    _validate_grid(label_ds, labels_path, shape, transform, crs)

            def read_scene_block(index, window):
                raw, raw_valid = _read_valid_values(input_ds[index], window)
                frozen_mask = _read_mask(mask_ds[index], window)
                if np.any(frozen_mask & ~raw_valid):
                    raise ValueError(f"INVALID_FROZEN_INPUT_MASK:{scenes[index]}")
                output, output_valid = _read_valid_values(output_ds[index], window)
                return raw, frozen_mask, output, output_valid

            def iter_all_blocks():
                for window in _windows(shape, block_shape):
                    if check:
                        check()
                    yield window, [read_scene_block(i, window) for i in range(len(scenes))]

            # Streamed per-scene support precheck and structural diagnostics.
            scene_support_counts = np.zeros(len(scenes), dtype=np.int64)
            scene_output_counts = np.zeros(len(scenes), dtype=np.int64)
            union_count = 0
            if "B1" in regimes:
                expected_counts = [int(r.get("aligned_valid_overlap_pixels") or 0) for r in pair_rows]
                pair_offsets = np.zeros(len(pair_rows), dtype=np.int64)
                total_pairs = sum(expected_counts)
                spool_path = scratch_dir / "b1_pair_samples.f64"
                scratch_dir.mkdir(parents=True, exist_ok=True)
                pair_map = np.memmap(spool_path, mode="w+", dtype=np.float64, shape=(2, total_pairs)) if total_pairs else None
                if pair_map is not None:
                    stack.callback(_cleanup_memmap, spool_path, pair_map)
                pair_support_counts = np.zeros(len(pair_rows), dtype=np.int64)
                pair_output_counts = np.zeros(len(pair_rows), dtype=np.int64)
                for window, scene_blocks in iter_all_blocks():
                    masks = [block[1] for block in scene_blocks]
                    out_masks = [block[3] for block in scene_blocks]
                    union = np.logical_or.reduce(masks)
                    union_count += int(union.sum())
                    for idx, mask in enumerate(masks):
                        scene_support_counts[idx] += int(mask.sum())
                        scene_output_counts[idx] += int(np.count_nonzero(mask & out_masks[idx]))
                    for k, pair in enumerate(pair_rows):
                        i, j = scene_index[pair["scene_i"]], scene_index[pair["scene_j"]]
                        support = masks[i] & masks[j]
                        count = int(support.sum())
                        pair_support_counts[k] += count
                        both = support & out_masks[i] & out_masks[j]
                        n_out = int(both.sum())
                        pair_output_counts[k] += n_out
                        if n_out and pair_map is not None:
                            offset = int(pair_offsets[k])
                            end = offset + n_out
                            start = sum(expected_counts[:k])
                            pair_map[0, start + offset:start + end] = np.asarray(scene_blocks[i][2][both], dtype=np.float64)
                            pair_map[1, start + offset:start + end] = np.asarray(scene_blocks[j][2][both], dtype=np.float64)
                            pair_offsets[k] = end
                for k, pair in enumerate(pair_rows):
                    expected = expected_counts[k]
                    observed = int(pair_support_counts[k])
                    expected_status = "OVERLAP" if observed else "NO_VALID_OVERLAP"
                    if observed != expected or pair.get("status") != expected_status:
                        if pair_map is not None:
                            pair_map._mmap.close()
                            pair_map = None
                            spool_path.unlink(missing_ok=True)
                        return {"status": "FROZEN_PAIR_SUPPORT_MISMATCH", "pair_id": pair["pair_id"],
                                "inventory_pixels": expected, "mask_intersection_pixels": observed,
                                "inventory_status": pair.get("status"), "mask_status": expected_status}
                if pair_map is not None:
                    pair_map.flush()
                    scratch_peak_observed = pair_map.nbytes
                for k, pair in enumerate(pair_rows):
                    i, j = scene_index[pair["scene_i"]], scene_index[pair["scene_j"]]
                    expected = expected_counts[k]
                    output_pixels = int(pair_output_counts[k])
                    if expected == 0:
                        b1_rows.append({"pair_id": pair["pair_id"], "scene_i": pair["scene_i"], "scene_j": pair["scene_j"],
                                        "status": "NO_VALID_OVERLAP", "n": 0,
                                        "metrics": {"direct": None, "distribution": None}})
                        continue
                    if output_pixels != expected:
                        b1_rows.append({"pair_id": pair["pair_id"], "scene_i": pair["scene_i"], "scene_j": pair["scene_j"],
                                        "status": "OUTPUT_MISSING" if output_pixels == 0 else "PARTIAL_OUTPUT",
                                        "n": expected, "output_pixels": output_pixels,
                                        "coverage": output_pixels / expected,
                                        "metrics": {"direct": None, "distribution": None}})
                        continue
                    start = sum(expected_counts[:k])
                    stop = start + expected
                    pair_metrics = compute_pair_metrics_from_arrays(
                        pair_map[0, start:stop], pair_map[1, start:stop], scratch_dir, sort_in_place=True)
                    scratch_peak_observed = max(scratch_peak_observed,
                                                pair_map.nbytes + pair_metrics["scratch_peak_bytes"])
                    b1_rows.append({"pair_id": pair["pair_id"], "scene_i": pair["scene_i"], "scene_j": pair["scene_j"],
                                    "status": "EVALUABLE", "n": expected,
                                    "metrics": {"direct": pair_metrics["direct"],
                                                "distribution": pair_metrics["distribution"]}})
                if pair_map is not None:
                    pair_map.flush()
                    pair_map._mmap.close()
                    pair_map = None
                    spool_path.unlink(missing_ok=True)
            else:
                # One support pass still verifies RAW input validity for A-only runs.
                for _, scene_blocks in iter_all_blocks():
                    masks = [block[1] for block in scene_blocks]
                    union_count += int(np.logical_or.reduce(masks).sum())
                    for idx, block in enumerate(scene_blocks):
                        scene_support_counts[idx] += int(block[1].sum())
                        scene_output_counts[idx] += int(np.count_nonzero(block[1] & block[3]))

            input_union_pixels = union_count
            scene_structure = []
            if "B1" in regimes or "B2" in regimes:
                for idx, scene in enumerate(scenes):
                    if scene_output_counts[idx] != scene_support_counts[idx]:
                        scene_structure.append({"scene_id": scene, "status": "PARTIAL_OUTPUT",
                                                "support_pixels": int(scene_support_counts[idx]),
                                                "sobel": None, "range_retention": None})
                        continue
                    xreader = lambda win, ds=input_ds[idx]: ds.read(1, window=win)
                    yreader = lambda win, ds=output_ds[idx]: ds.read(1, window=win)
                    mreader = lambda win, ds=mask_ds[idx]: _read_mask(ds, win)
                    sobel = compute_sobel_structure_from_readers(shape, xreader, yreader, mreader,
                                  block_shape=block_shape, scratch_dir=scratch_dir)
                    ranges = compute_range_retention_from_readers(shape, xreader, yreader, mreader,
                                  block_shape=block_shape, scratch_dir=scratch_dir)
                    scratch_peak_observed = max(scratch_peak_observed,
                        int(sobel.get("scratch_peak_bytes") or 0), int(ranges.get("scratch_peak_bytes") or 0))
                    sobel.pop("scratch_peak_bytes", None)
                    ranges.pop("scratch_peak_bytes", None)
                    scene_structure.append({"scene_id": scene, "status": sobel["status"],
                                            "support_pixels": int(scene_support_counts[idx]),
                                            "sobel": sobel, "range_retention": ranges})

            if "B1" in regimes:
                summaries = _summaries(b1_rows, pair_rows)
                b1_result = {
                    "status": "EVALUABLE" if all(r["status"] in {"EVALUABLE", "NO_VALID_OVERLAP"} for r in b1_rows) else "PARTIAL_OUTPUT",
                    "expected_pair_rows": len(pair_rows),
                    "positive_overlap_pairs": sum(r["status"] == "OVERLAP" for r in pair_rows),
                    "no_valid_overlap_pairs": sum(r["status"] == "NO_VALID_OVERLAP" for r in pair_rows),
                    "pair_rows": b1_rows,
                    "metric_summaries": summaries,
                    "metric_summaries_by_time_stratum": _time_summaries(b1_rows, pair_rows),
                    "metric_summaries_by_quality": _quality_summaries(b1_rows, pair_rows),
                }
            else:
                b1_result = None

            if "B2" in regimes:
                windows = _read_csv(freeze_root / "fixed_windows.csv")
                pair_by_id = {p["pair_id"]: p for p in pair_rows}
                for row in windows:
                    if check:
                        check()
                    pair = pair_by_id[row["pair_id"]]
                    i, j = scene_index[pair["scene_i"]], scene_index[pair["scene_j"]]
                    r0, c0, h, w = (int(row[k]) for k in ("row_off", "col_off", "height", "width"))
                    window = Window(c0, r0, w, h)
                    mask_i, mask_j = _read_mask(mask_ds[i], window), _read_mask(mask_ds[j], window)
                    support = mask_i & mask_j
                    expected_count = int(row["valid_count"])
                    if int(support.sum()) != expected_count or _mask_sha(support) != row["support_sha256"]:
                        return {"status": "FROZEN_WINDOW_SUPPORT_MISMATCH", "window_id": row["window_id"]}
                    xi, valid_i = _read_valid_values(output_ds[i], window)
                    xj, valid_j = _read_valid_values(output_ds[j], window)
                    output_support = support & valid_i & valid_j
                    output_pixels = int(output_support.sum())
                    if output_pixels != expected_count:
                        b2_rows.append({"window_id": row["window_id"], "pair_id": row["pair_id"],
                                        "status": "OUTPUT_MISSING" if output_pixels == 0 else "PARTIAL_OUTPUT",
                                        "n": expected_count, "output_pixels": output_pixels,
                                        "coverage": output_pixels / expected_count, "metrics": None})
                    else:
                        metric = compute_pair_metrics_from_arrays(
                            np.asarray(xi[support], dtype=np.float64), np.asarray(xj[support], dtype=np.float64),
                            scratch_dir, sort_in_place=False)
                        scratch_peak_observed = max(scratch_peak_observed, metric["scratch_peak_bytes"])
                        b2_rows.append({"window_id": row["window_id"], "pair_id": row["pair_id"],
                                        "status": "EVALUABLE", "n": expected_count,
                                        "metrics": {"direct": metric["direct"], "distribution": metric["distribution"]}})
                    for idx, scene in ((i, scenes[i]), (j, scenes[j])):
                        scene_support = _read_mask(mask_ds[idx], window)
                        input_values, input_valid = _read_valid_values(input_ds[idx], window)
                        output_values, output_valid = _read_valid_values(output_ds[idx], window)
                        if np.any(scene_support & (~input_valid | ~output_valid)):
                            b2_structures.append({"window_id": row["window_id"], "pair_id": row["pair_id"],
                                                  "scene_id": scene, "status": "PARTIAL_OUTPUT",
                                                  "sobel": None, "range_retention": None})
                        else:
                            from .structure import compute_range_retention, compute_sobel_structure
                            st = compute_sobel_structure(input_values, output_values, scene_support)
                            rr = compute_range_retention(input_values, output_values, scene_support)
                            b2_structures.append({"window_id": row["window_id"], "pair_id": row["pair_id"],
                                                  "scene_id": scene, "status": st.status,
                                                  "sobel": st.__dict__, "range_retention": rr.__dict__})
                window_summaries = {}
                for key, section in (("mae", "direct"), ("rmse", "direct"), ("bias", "direct"), ("p95ae", "direct"),
                                     ("mamd", "distribution"), ("msdd", "distribution"), ("rdd", "distribution")):
                    rows = [{"pair_id": w["pair_id"], "status": w["status"], "n": w["n"],
                             "value": (w.get("metrics") or {}).get(section, {}).get(key)} for w in b2_rows]
                    window_summaries[key] = _window_summary(rows, [r["pair_id"] for r in pair_rows])
                b2_result = {"status": "EVALUABLE" if len(b2_rows) == len(windows) and all(w["status"] == "EVALUABLE" for w in b2_rows) else "PARTIAL_OUTPUT",
                             "expected_window_rows": len(windows), "window_count": len(b2_rows),
                             "window_rows": b2_rows, "metric_distributions": window_summaries,
                             "structure_by_scene_window": b2_structures,
                             "structure_population": _distribution_summary(b2_structures, "unused.sobel.direction_loss"),
                             "range_retention_population": _distribution_summary(b2_structures, "unused.range_retention.value")}
                # Keep the frozen consumer-visible nesting for population summaries.
                for key, section, metric in (("structure_population", "sobel", "direction_loss"),
                                              ("range_retention_population", "range_retention", "value")):
                    vals = [float(r[section][metric]) for r in b2_structures if r.get(section) and r[section].get(metric) is not None]
                    b2_result[key] = ({"status": "NOT_EVALUABLE", "n": 0, "median": None, "p95": None, "mean": None}
                                      if not vals else {"status": "EVALUABLE", "n": len(vals), "median": float(np.median(vals)),
                                                        "p95": float(np.quantile(vals, .95, method="linear")), "mean": float(np.mean(vals))})
            else:
                b2_result = None

            if "A" in regimes:
                coverage_holes = coverage_extras = output_valid_pixels = unique_pixels = 0
                invalid_labels = 0
                for window in _windows(shape, block_shape):
                    if check:
                        check()
                    masks = [_read_mask(ds, window) for ds in mask_ds]
                    union = np.logical_or.reduce(masks)
                    values, output_valid = _read_valid_values(mosaic_ds, window)
                    explicit_values, explicit_mask_valid = _read_valid_values(maskout_ds, window)
                    mosaic_valid = output_valid & explicit_mask_valid & (explicit_values > 0) & np.isfinite(values)
                    output_valid_pixels += int(mosaic_valid.sum())
                    coverage_holes += int((union & ~mosaic_valid).sum())
                    coverage_extras += int((~union & mosaic_valid).sum())
                    unique_pixels += int((np.sum(np.stack(masks), axis=0) == 1).sum())
                    if label_ds is not None:
                        labels, label_valid = _read_valid_values(label_ds, window)
                        valid_labels = label_valid & (labels >= 0) & (labels < len(scenes))
                        invalid_labels += int(np.count_nonzero(mosaic_valid & ~valid_labels))
                        for idx, scene_mask in enumerate(masks):
                            output_mask = _read_valid_values(output_ds[idx], window)[1]
                            invalid_labels += int(np.count_nonzero(mosaic_valid & (labels == idx) & ~scene_mask))
                            invalid_labels += int(np.count_nonzero(mosaic_valid & (labels == idx) & ~output_mask))
                coverage = output_valid_pixels / input_union_pixels if input_union_pixels else None
                if labels_path is None:
                    seam = {"status": "NOT_EVALUABLE_NO_PROVENANCE", "edge_count": 0, "mean_excess": None}
                elif invalid_labels:
                    seam = {"status": "INVALID_SOURCE_LABELS", "invalid_label_pixels": invalid_labels,
                            "edge_count": 0, "mean_excess": None}
                else:
                    mosaic_reader = lambda win: mosaic_ds.read(1, window=win)
                    label_reader = lambda win: label_ds.read(1, window=win)
                    def valid_mosaic_reader(win):
                        mvalues, mvalid = _read_valid_values(mosaic_ds, win)
                        exvalues, exvalid = _read_valid_values(maskout_ds, win)
                        return mvalid & exvalid & (exvalues > 0) & np.isfinite(mvalues)
                    seam = compute_seam_excess_from_readers(shape, mosaic_reader, label_reader,
                                   valid_mosaic_reader, scratch_dir=scratch_dir, check=check)
                    seam["source_id_mapping"] = {str(i): scenes[i] for i in range(len(scenes))}
                    scratch_peak_observed = max(scratch_peak_observed, int(seam.get("scratch_peak_bytes") or 0))
                    seam.pop("scratch_peak_bytes", None)
                if np.array_equal(scene_output_counts, scene_support_counts):
                    def unique_block(window):
                        valid_masks = [_read_mask(ds, window) for ds in mask_ds]
                        stack_masks = np.stack(valid_masks)
                        unique = stack_masks.sum(axis=0) == 1
                        ref = np.zeros(unique.shape, dtype=np.float64)
                        for idx, ds in enumerate(input_ds):
                            take = unique & valid_masks[idx]
                            if take.any():
                                ref[take] = ds.read(1, window=window)[take]
                        mvalues, mvalid = _read_valid_values(mosaic_ds, window)
                        ev, evvalid = _read_valid_values(maskout_ds, window)
                        mosaic_valid = mvalid & evvalid & (ev > 0) & np.isfinite(mvalues)
                        return ref, unique & mosaic_valid
                    ref_reader = lambda win: unique_block(win)[0]
                    unique_support_reader = lambda win: unique_block(win)[1]
                    mosaic_reader = lambda win: mosaic_ds.read(1, window=win)
                    sobel = compute_sobel_structure_from_readers(shape, ref_reader, mosaic_reader,
                                unique_support_reader, block_shape=block_shape, scratch_dir=scratch_dir, check=check)
                    ranges = compute_range_retention_from_readers(shape, ref_reader, mosaic_reader,
                                unique_support_reader, block_shape=block_shape, scratch_dir=scratch_dir, check=check)
                    scratch_peak_observed = max(scratch_peak_observed, int(sobel.get("scratch_peak_bytes") or 0),
                                                int(ranges.get("scratch_peak_bytes") or 0))
                    sobel.pop("scratch_peak_bytes", None); ranges.pop("scratch_peak_bytes", None)
                    single_source = {"status": sobel["status"], "unique_source_pixels": unique_pixels,
                                     "valid_pixels": unique_pixels, "structure": sobel}
                    single_source["range_retention"] = ranges
                else:
                    single_source = {"status": "PARTIAL_OUTPUT", "unique_source_pixels": unique_pixels,
                                     "valid_pixels": 0, "structure": None}
                structure_rows = scene_structure
                A_result = {
                    "status": "EVALUABLE" if coverage == 1.0 and coverage_holes == 0 and coverage_extras == 0 else "INCOMPLETE_MOSAIC_SUPPORT",
                    "mosaic_coverage": {"status": "EVALUABLE" if input_union_pixels else "NO_INPUT_UNION",
                                        "coverage": coverage, "input_union_pixels": input_union_pixels,
                                        "output_valid_pixels": output_valid_pixels,
                                        "holes": coverage_holes, "extras": coverage_extras},
                    "single_source_structure": single_source,
                    "structure": {"structure_by_scene": structure_rows},
                    "seam": seam,
                }
    except (OSError, ValueError, KeyError) as exc:
        return {"status": "INVALID_ARTIFACT", "error": f"{type(exc).__name__}:{exc}"}

    resource = {
        "runtime_status": (submission.get("runtime") or {}).get("status", "NOT_MEASURED"),
        "wall_seconds": (submission.get("runtime") or {}).get("wall_seconds"),
        "wall_measurement_source": (submission.get("runtime") or {}).get("wall_measurement_source", "NOT_MEASURED"),
        "peak_rss_bytes": (submission.get("runtime") or {}).get("peak_rss_bytes"),
        "rss_status": (submission.get("runtime") or {}).get("rss_status", "NOT_MEASURED"),
        "rss_sample_count": int((submission.get("runtime") or {}).get("rss_sample_count") or 0),
        "rss_measurement_source": (submission.get("runtime") or {}).get("rss_measurement_source", "NOT_MEASURED"),
        "resource_budget": dict(submission.get("resource_budget") or {}),
        "hard_os_memory_limit": False,
        "measurement_provenance_status": "PASS" if ((submission.get("runtime") or {}).get("wall_measurement_source") not in (None, "NOT_MEASURED")
             and (submission.get("runtime") or {}).get("rss_measurement_source") not in (None, "NOT_MEASURED")
             and int((submission.get("runtime") or {}).get("rss_sample_count") or 0) > 0) else "INCOMPLETE",
    }
    result = {
        "protocol_id": frozen.get("protocol_id"), "status": "EVALUATED",
        "submission_audit": {**audit, "status": "PASS", "artifact_status": "PASS",
                              "scene_count": len(scenes), "scene_ids": scenes,
                              "corrected_scene_dtypes": corrected_dtypes,
                              "method_status": submission.get("status")},
        "runtime": resource, "resource_preflight": preflight,
        "scratch": {"estimated_peak_bytes": preflight["estimated_peak_scratch_bytes"],
                    "observed_peak_bytes": scratch_peak_observed,
                    "cleanup_status": "PASS" if not list(scratch_dir.glob("*.f64")) and not (scratch_dir / "b1_pair_samples.f64").exists() else "REMAINS"},
        "A": A_result if "A" in regimes else None,
        "B1": b1_result if "B1" in regimes else None,
        "B2": b2_result if "B2" in regimes else None,
        "B3": ({"status": "NOT_RUN_SYNTHETIC_METHOD_OUTPUT_REQUIRED",
                "reason": "B3 is a separate known-truth simulation; real overlap metrics are not a substitute."}
               if "B3" in regimes else None),
        "scene_structure": scene_structure if ("B1" in regimes or "B2" in regimes) else None,
    }
    seam_eligible = True
    if "A" in regimes and A_result:
        seam_eligible = submission.get("seam_type", "NONE") == "NONE" or A_result.get("seam", {}).get("status") == "EVALUABLE"
    result["ranking_eligible"] = bool(
        submission.get("status") == "COMPLETED"
        and preflight["status"] == "PASS"
        and resource["measurement_provenance_status"] == "PASS"
        and seam_eligible
        and all(result.get(r) and result[r].get("status") == "EVALUABLE" for r in ("A", "B1", "B2") if r in regimes)
        and ("A" not in regimes or (A_result["mosaic_coverage"]["coverage"] == 1.0
             and A_result["mosaic_coverage"]["extras"] == 0
             and A_result["single_source_structure"].get("status") == "EVALUABLE"))
        and ("B1" not in regimes or b1_result["positive_overlap_pairs"] == b1_result["metric_summaries"]["mae"]["evaluable_pairs"])
        and ("B2" not in regimes or b2_result["window_count"] == b2_result["expected_window_rows"])
    )
    return result
