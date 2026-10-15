"""Strict external A/B1/B2 scoring without changing Phase-1 artifacts."""

from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np


def score_pair_arrays(values_i, values_j, valid_i, valid_j, output_valid_i, output_valid_j):
    from src.benchmark_v1_unified.radiometric import (
        compute_pair_direct_metrics, compute_pair_distribution_metrics, metric_dict,
    )

    vi, vj = np.asarray(valid_i, bool), np.asarray(valid_j, bool)
    oi, oj = np.asarray(output_valid_i, bool), np.asarray(output_valid_j, bool)
    xi, xj = np.asarray(values_i), np.asarray(values_j)
    if not (vi.shape == vj.shape == oi.shape == oj.shape == xi.shape == xj.shape):
        raise ValueError("pair metric arrays must have identical shapes")
    support = vi & vj
    n = int(support.sum())
    complete = support & oi & oj & np.isfinite(xi) & np.isfinite(xj)
    n_eval = int(complete.sum())
    missing = n - n_eval
    if n == 0:
        return {"status": "NO_VALID_OVERLAP", "input_support_pixels": 0, "evaluated_pixels": 0,
                "missing_pixels": 0, "metrics": None}
    if missing:
        return {"status": "PARTIAL_OUTPUT", "input_support_pixels": n, "evaluated_pixels": n_eval,
                "missing_pixels": missing, "metrics": None}
    all_valid = np.ones(n, dtype=bool)
    return {"status": "EVALUABLE", "input_support_pixels": n, "evaluated_pixels": n,
            "missing_pixels": 0,
            "metrics": {"direct": metric_dict(compute_pair_direct_metrics(xi[support], xj[support], all_valid)),
                        "distribution": metric_dict(compute_pair_distribution_metrics(xi[support], xj[support], all_valid))}}


def audit_mosaic_arrays(expected_union, values, output_valid):
    expected, data, valid = np.asarray(expected_union, bool), np.asarray(values), np.asarray(output_valid, bool)
    if expected.shape != data.shape or expected.shape != valid.shape:
        raise ValueError("mosaic arrays must have identical shapes")
    actual = valid & np.isfinite(data)
    n_expected = int(expected.sum())
    inside = int((expected & actual).sum())
    missing = int((expected & ~actual).sum())
    extra = int((~expected & actual).sum())
    return {"status": "PASS" if not missing and not extra else "COVERAGE_INVALID",
            "expected_pixels": n_expected, "output_valid_inside": inside,
            "missing_pixels": missing, "extra_pixels": extra,
            "coverage": inside / n_expected if n_expected else 0.0}


def _grid_status(dataset, grid):
    expected_crs = grid.get("crs")
    expected_transform = tuple(grid.get("transform", []))
    expected_shape = (int(grid.get("height", -1)), int(grid.get("width", -1)))
    actual_crs = dataset.crs.to_string() if dataset.crs else None
    if (actual_crs != expected_crs or tuple(dataset.transform)[:6] != expected_transform[:6]
            or (dataset.height, dataset.width) != expected_shape or dataset.count != 1):
        return "INVALID_GRID"
    return "PASS"


def _open_validity(path, explicit_mask_path, stack, grid):
    import rasterio

    if not Path(path).is_file():
        raise FileNotFoundError(f"MISSING_OUTPUT:{path}")
    dataset = stack.enter_context(rasterio.open(path))
    if _grid_status(dataset, grid) != "PASS":
        raise ValueError("INVALID_GRID")
    mask_dataset = None
    if explicit_mask_path:
        mask_path = Path(explicit_mask_path)
        if not mask_path.is_file():
            raise FileNotFoundError(f"MISSING_OUTPUT_MASK:{mask_path}")
        mask_dataset = stack.enter_context(rasterio.open(mask_path))
        if _grid_status(mask_dataset, grid) != "PASS":
            raise ValueError("INVALID_GRID_MASK")
    elif dataset.nodata is None and all(
        getattr(flag, "name", str(flag)).lower() == "all_valid" for flag in dataset.mask_flag_enums[0]
    ):
        raise ValueError("AMBIGUOUS_OUTPUT_MASK")
    return dataset, mask_dataset


def _valid_values(dataset, mask_dataset, window):
    values = dataset.read(1, window=window)
    if mask_dataset is None:
        valid = dataset.dataset_mask(window=window) > 0
    else:
        valid = mask_dataset.read(1, window=window) > 0
    valid &= np.isfinite(values)
    if dataset.nodata is not None and np.isfinite(dataset.nodata):
        valid &= values != dataset.nodata
    return values, valid


def _scene_indexes(manifest):
    result = {}
    for item in manifest.get("aligned_input_artifacts", []):
        source = Path(item.get("source_path", "")).stem
        scene_id = source[:-3] if source.endswith("_B9") else source
        result[scene_id] = int(item["scene_index"])
    return result


def _audit_phase1_freeze(root):
    from src.benchmark_v1_spectralmatch_phase2.freeze_audit import audit_frozen_inputs

    return audit_frozen_inputs(root, check_rasters=True)


def _input_paths(manifest):
    images, masks = {}, {}
    for item in manifest.get("aligned_input_artifacts", []):
        index = int(item["scene_index"])
        images[index] = item["aligned_path"]
        masks[index] = item["mask_path"]
    return images, masks


def _score_b1(freeze_root, manifest, submission):
    import rasterio
    from rasterio.windows import Window
    from src.benchmark_v1_unified.aggregation import aggregate_pair_rows

    grid = manifest["grid"]
    scene_map = _scene_indexes(manifest)
    input_images, input_masks = _input_paths(manifest)
    output_paths = submission.get("corrected_scene_paths") or {}
    if not output_paths:
        return {"status": "MISSING_OUTPUT", "missing_scene_outputs": list(range(13)), "pairs": []}
    resolved = {scene_map.get(scene): path for scene, path in output_paths.items() if scene_map.get(scene) is not None}
    rows = list(csv.DictReader((freeze_root / "pair_overlap_inventory.csv").open(encoding="utf-8-sig", newline="")))
    positive = [row for row in rows if row["status"] == "OVERLAP" and int(row["aligned_valid_overlap_pixels"] or 0) > 0]
    expected_pairs = manifest.get("pair_counts", {}).get("positive_overlap_pairs")
    if expected_pairs is not None and len(positive) != int(expected_pairs):
        return {"status": "FROZEN_PAIR_INVENTORY_INVALID", "observed_positive_pairs": len(positive),
                "expected_positive_pairs": int(expected_pairs), "pair_results": []}
    pair_results = []
    for row in positive:
        pair_id = row["pair_id"]
        try:
            i, j = scene_map[row["scene_i"]], scene_map[row["scene_j"]]
            oi_path, oj_path = resolved[i], resolved[j]
        except KeyError:
            pair_results.append({"pair_id": pair_id, "status": "MISSING_OUTPUT", "input_support_pixels": int(row["aligned_valid_overlap_pixels"]), "evaluated_pixels": 0, "missing_pixels": int(row["aligned_valid_overlap_pixels"]), "metrics": None})
            continue
        with ExitStack() as stack:
            ai = stack.enter_context(rasterio.open(input_masks[i])); aj = stack.enter_context(rasterio.open(input_masks[j]))
            out_i, mask_i = _open_validity(oi_path, None, stack, grid)
            out_j, mask_j = _open_validity(oj_path, None, stack, grid)
            if _grid_status(ai, grid) != "PASS" or _grid_status(aj, grid) != "PASS":
                pair_results.append({"pair_id": pair_id, "status": "INVALID_INPUT_GRID"}); continue
            metric_chunks = {"i": [], "j": []}
            expected = observed = 0
            for row_off in range(0, int(grid["height"]), 512):
                for col_off in range(0, int(grid["width"]), 512):
                    window = Window(col_off, row_off, min(512, int(grid["width"])-col_off), min(512, int(grid["height"])-row_off))
                    vi, vj = ai.read(1, window=window) > 0, aj.read(1, window=window) > 0
                    support = vi & vj
                    if not support.any():
                        continue
                    expected += int(support.sum())
                    xi, vout_i = _valid_values(out_i, mask_i, window)
                    xj, vout_j = _valid_values(out_j, mask_j, window)
                    complete = support & vout_i & vout_j
                    observed += int(complete.sum())
                    if complete.any():
                        metric_chunks["i"].append(xi[complete].astype(np.float64, copy=False))
                        metric_chunks["j"].append(xj[complete].astype(np.float64, copy=False))
            frozen_n = int(row["aligned_valid_overlap_pixels"])
            if expected != frozen_n:
                pair_results.append({"pair_id": pair_id, "status": "FROZEN_SUPPORT_MISMATCH", "input_support_pixels": expected, "frozen_expected_pixels": frozen_n, "evaluated_pixels": observed, "missing_pixels": expected-observed, "metrics": None})
            elif observed != expected:
                pair_results.append({"pair_id": pair_id, "status": "PARTIAL_OUTPUT", "input_support_pixels": expected, "evaluated_pixels": observed, "missing_pixels": expected-observed, "metrics": None})
            else:
                from src.benchmark_v1_unified.radiometric import compute_pair_direct_metrics, compute_pair_distribution_metrics, metric_dict
                x = np.concatenate(metric_chunks["i"]); y = np.concatenate(metric_chunks["j"]); support = np.ones(x.shape, bool)
                pair_results.append({"pair_id": pair_id, "status": "EVALUABLE", "input_support_pixels": expected,
                                     "evaluated_pixels": observed, "missing_pixels": 0,
                                     "metrics": {"direct": metric_dict(compute_pair_direct_metrics(x, y, support)),
                                                 "distribution": metric_dict(compute_pair_distribution_metrics(x, y, support))}})
    metrics = ("mae", "rmse", "mamd", "msdd", "rdd")
    macro = {}
    for metric in metrics:
        metric_rows = []
        for row in pair_results:
            direct = row.get("metrics", {}).get("direct", {})
            dist = row.get("metrics", {}).get("distribution", {})
            value = direct.get(metric) if metric in ("mae", "rmse") else dist.get(metric)
            metric_rows.append({"pair_id": row["pair_id"], "status": row["status"], "value": value, "n": row.get("input_support_pixels", 0)})
        macro[metric] = aggregate_pair_rows(metric_rows, [row["pair_id"] for row in positive])
    status = "FULL_SUPPORT" if len(pair_results) == len(positive) and all(r.get("status") == "EVALUABLE" for r in pair_results) else "PARTIAL_OUTPUT"
    return {"status": status, "expected_positive_pairs": len(positive), "evaluated_pairs": sum(r.get("status") == "EVALUABLE" for r in pair_results),
            "pair_results": pair_results, "pair_macro": macro, "scope": "FORMAL_FULL"}


def _score_b2(freeze_root, manifest, submission):
    import rasterio
    from rasterio.windows import Window

    # The frozen window inventory is read verbatim. No windows are regenerated.
    windows_path = Path(freeze_root) / "fixed_windows.csv"
    records = list(csv.DictReader(windows_path.open(encoding="utf-8-sig", newline="")))
    expected_windows = manifest.get("window_count")
    if expected_windows is not None and len(records) != int(expected_windows):
        return {"status": "FROZEN_WINDOW_INVENTORY_INVALID", "observed_windows": len(records),
                "expected_windows": int(expected_windows), "scope": "FORMAL_FULL"}
    scene_map = _scene_indexes(manifest); images, masks = _input_paths(manifest)
    corrected = submission.get("corrected_scene_paths") or {}
    outputs = {scene_map.get(scene): path for scene, path in corrected.items() if scene_map.get(scene) is not None}
    by_pair = {}
    for row in records:
        by_pair.setdefault(row["pair_id"], []).append(row)
    pair_rows = list(csv.DictReader((Path(freeze_root) / "pair_overlap_inventory.csv").open(encoding="utf-8-sig", newline="")))
    pair_lookup = {r["pair_id"]: r for r in pair_rows}
    details = []
    for pair_id, candidates in by_pair.items():
        if len(candidates) > 32:
            return {"status": "FROZEN_WINDOW_PROTOCOL_INVALID", "reason": f"more than 32 frozen windows for {pair_id}", "scope": "FORMAL_FULL"}
        rows = candidates[:32]
        for entry in rows:
            pair = pair_lookup[pair_id]
            i, j = scene_map[pair["scene_i"]], scene_map[pair["scene_j"]]
            try:
                path_i, path_j = outputs[i], outputs[j]
            except KeyError:
                details.append({"window_id": entry["window_id"], "pair_id": pair_id, "status": "MISSING_OUTPUT", "expected_pixels": int(entry["valid_count"]), "evaluated_pixels": 0, "missing_pixels": int(entry["valid_count"])})
                continue
            win = Window(int(entry["col_off"]), int(entry["row_off"]), int(entry["width"]), int(entry["height"]))
            with ExitStack() as stack:
                mi = stack.enter_context(rasterio.open(masks[i])); mj = stack.enter_context(rasterio.open(masks[j]))
                if _grid_status(mi, manifest["grid"]) != "PASS" or _grid_status(mj, manifest["grid"]) != "PASS":
                    details.append({"window_id": entry["window_id"], "pair_id": pair_id, "status": "INVALID_INPUT_MASK_GRID", "expected_pixels": int(entry["valid_count"]), "evaluated_pixels": 0, "missing_pixels": int(entry["valid_count"])})
                    continue
                out_i, out_mask_i = _open_validity(path_i, None, stack, manifest["grid"])
                out_j, out_mask_j = _open_validity(path_j, None, stack, manifest["grid"])
                xi, vi = _valid_values(out_i, out_mask_i, win)
                xj, vj = _valid_values(out_j, out_mask_j, win)
                si, sj = mi.read(1, window=win) > 0, mj.read(1, window=win) > 0
                support = si & sj
                import hashlib
                actual_hash = hashlib.sha256(np.packbits(support, bitorder="little").tobytes()).hexdigest()
                if int(support.sum()) != int(entry["valid_count"]) or actual_hash != entry["support_sha256"]:
                    details.append({"window_id": entry["window_id"], "pair_id": pair_id, "status": "FROZEN_SUPPORT_MISMATCH", "expected_pixels": int(entry["valid_count"]), "evaluated_pixels": 0, "missing_pixels": int(entry["valid_count"])})
                    continue
                result = score_pair_arrays(xi, xj, si, sj, vi, vj)
                result.update({"window_id": entry["window_id"], "pair_id": pair_id, "support_sha256": actual_hash})
                details.append(result)
    from src.benchmark_v1_unified.aggregation import aggregate_windows
    aggregate = aggregate_windows([{"pair_id": r["pair_id"], "status": r["status"], "value": r.get("metrics", {}).get("direct", {}).get("mae")} for r in details], sorted(by_pair))
    return {"status": "FULL_SUPPORT" if details and all(r.get("status") == "EVALUABLE" for r in details) else "PARTIAL_OUTPUT",
            "expected_windows": len(records), "evaluated_windows": sum(r.get("status") == "EVALUABLE" for r in details),
            "windows": details, "frozen_window_aggregation": aggregate, "scope": "FORMAL_FULL"}


def _score_a(freeze_root, manifest, submission):
    import rasterio
    from rasterio.windows import Window
    from src.benchmark_v1_unified.structure import compute_sobel_structure

    grid = manifest["grid"]
    mosaic_path = submission.get("output_mosaic_path")
    if not mosaic_path:
        return {"status": "MISSING_OUTPUT", "formal_ranking_eligible": False, "reason": "output_mosaic_path is required"}
    image_paths, mask_paths = _input_paths(manifest)
    expected_indexes = set(range(13))
    if set(image_paths) != expected_indexes or set(mask_paths) != expected_indexes:
        return {"status": "FROZEN_INPUT_INCOMPLETE", "formal_ranking_eligible": False}
    scenes = submission.get("corrected_scene_paths") or {}
    scene_map = _scene_indexes(manifest)
    resolved = {scene_map.get(scene): path for scene, path in scenes.items() if scene_map.get(scene) is not None}
    with ExitStack() as stack:
        mosaic, mosaic_mask = _open_validity(mosaic_path, submission.get("output_mosaic_mask_path"), stack, grid)
        source_masks = [stack.enter_context(rasterio.open(mask_paths[i])) for i in range(13)]
        if any(_grid_status(dataset, grid) != "PASS" for dataset in source_masks):
            return {"status": "INVALID_INPUT_MASK_GRID", "formal_ranking_eligible": False}
        source_images = {}
        for index, path in image_paths.items():
            dataset = stack.enter_context(rasterio.open(path))
            if _grid_status(dataset, grid) != "PASS":
                return {"status": "INVALID_INPUT_GRID", "formal_ranking_eligible": False, "scene_index": index}
            source_images[index] = dataset
        expected_pixels = inside = missing = extras = single_source_pixels = 0
        structure_tiles = []
        for row_off in range(0, int(grid["height"]), 512):
            for col_off in range(0, int(grid["width"]), 512):
                win = Window(col_off, row_off, min(512, int(grid["width"])-col_off), min(512, int(grid["height"])-row_off))
                masks = np.stack([dataset.read(1, window=win) > 0 for dataset in source_masks])
                source_count = masks.sum(axis=0)
                union = source_count > 0
                output, output_valid = _valid_values(mosaic, mosaic_mask, win)
                expected_pixels += int(union.sum())
                inside += int((union & output_valid).sum())
                missing += int((union & ~output_valid).sum())
                extras += int((~union & output_valid).sum())
                single_source_pixels += int((source_count == 1).sum())
                if not scenes:
                    continue
                for index in range(13):
                    unique = masks[index] & (source_count == 1)
                    if int(unique.sum()) < 9 or index not in resolved:
                        continue
                    source_values = source_images[index].read(1, window=win)
                    structure = compute_sobel_structure(source_values, output, unique & output_valid)
                    structure_tiles.append({"scene_index": index, "row_off": row_off, "col_off": col_off,
                                            "stats": structure.__dict__})
    coverage = inside / expected_pixels if expected_pixels else 0.0
    status = "PASS" if missing == 0 and extras == 0 else "COVERAGE_INVALID"
    provenance = submission.get("source_label_path")
    provenance_status = "NOT_AVAILABLE_NO_SOURCE_LABEL" if not provenance else ("AVAILABLE" if Path(provenance).is_file() else "MISSING_SOURCE_LABEL")
    return {
        "status": status,
        "scope": "FORMAL_FULL",
        "expected_union_pixels": expected_pixels,
        "mosaic_valid_inside_union_pixels": inside,
        "missing_pixels": missing,
        "extra_pixels": extras,
        "coverage": coverage,
        "single_source_region_pixels": single_source_pixels,
        "single_source_structure": {"status": "EVALUABLE" if structure_tiles else
                                    "NOT_EVALUABLE_NO_SINGLE_SOURCE_PIXELS" if single_source_pixels == 0 else
                                    "MISSING_CORRECTED_SCENE_OUTPUTS",
                                    "aggregation": "per 512x512 tile, frozen Sobel function; tile rows retained without pooled reinterpretation",
                                    "tiles": structure_tiles},
        "seam_provenance": {"status": provenance_status,
                             "metric": "NOT_AVAILABLE_NO_FROZEN_SOURCE_LABEL_METRIC"},
        "formal_ranking_eligible": status == "PASS" and bool(structure_tiles),
    }


def score_submission(freeze_root, submission, regime, *, expected_scope="FORMAL_FULL", scope=None):
    scope = scope or submission.get("scope", "FORMAL_FULL")
    if regime == "B3":
        return {"status": "NOT_RUN_B3", "formal_ranking_eligible": False}
    if regime not in {"A", "B1", "B2"}:
        return {"status": "UNSUPPORTED_REGIME", "formal_ranking_eligible": False}
    if scope == "DIAGNOSTIC_CROP_ONLY" and expected_scope == "FORMAL_FULL":
        return {"status": "SCOPE_REJECTED", "formal_ranking_eligible": False,
                "reason": "crop outputs are diagnostic only and cannot enter formal A/B ranking"}
    if scope != expected_scope:
        return {"status": "SCOPE_MISMATCH", "formal_ranking_eligible": False}
    root = Path(freeze_root)
    try:
        manifest = json.loads((root / "frozen_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "INVALID_FREEZE", "reason": str(exc), "formal_ranking_eligible": False}
    if manifest.get("status") != "FROZEN" or manifest.get("protocol_id") != "DZ01V_UNIFIED_A_B_V1_20261013":
        return {"status": "INVALID_FREEZE", "reason": "freeze status or protocol identity mismatch", "formal_ranking_eligible": False}
    freeze_audit = _audit_phase1_freeze(root)
    if freeze_audit.get("status") != "PASS":
        return {"status": "INVALID_FREEZE", "reason": "locked freeze or code content verification failed",
                "freeze_errors": freeze_audit.get("errors", []), "formal_ranking_eligible": False}
    if submission.get("input_freeze_sha256") != manifest.get("input_freeze_sha256"):
        return {"status": "INPUT_HASH_MISMATCH", "formal_ranking_eligible": False}
    if submission.get("geometry_freeze_sha256") != manifest.get("geometry_freeze_sha256"):
        return {"status": "GEOMETRY_HASH_MISMATCH", "formal_ranking_eligible": False}
    if (submission.get("schema_version") != 1 or not submission.get("method_id") or not submission.get("run_id")
            or submission.get("method_kind") not in {"FULL_MOSAIC", "SCENE_OUTPUTS", "PAIR_ONLY", "SYNTHETIC_FIXTURE"}
            or submission.get("band") != "B9" or submission.get("pixel_value_unit") != "UNKNOWN"
            or submission.get("status") is None):
        return {"status": "INVALID_SUBMISSION", "formal_ranking_eligible": False}
    try:
        if regime == "A":
            result = _score_a(root, manifest, submission)
        elif regime == "B1":
            result = _score_b1(root, manifest, submission)
        else:
            result = _score_b2(root, manifest, submission)
    except ModuleNotFoundError as exc:
        return {"status": "SCORER_DEPENDENCY_UNAVAILABLE", "reason": str(exc), "formal_ranking_eligible": False}
    except FileNotFoundError as exc:
        return {"status": "MISSING_OUTPUT", "reason": str(exc), "formal_ranking_eligible": False}
    except ValueError as exc:
        return {"status": str(exc), "formal_ranking_eligible": False}
    regime_eligible = result.get("formal_ranking_eligible", result["status"] == "FULL_SUPPORT")
    result["formal_ranking_eligible"] = bool(regime_eligible) and scope == "FORMAL_FULL"
    return result
