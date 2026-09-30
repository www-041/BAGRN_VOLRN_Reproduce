"""Task13A frozen protocol verification and one pairwise feasibility batch.

This entry point reads frozen BAGRN rasters only; no upstream method is run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.seam_local.pipeline import PairResult, process_pair
BASE = Path("data/output/b9_five_scene_validation")
SOURCE_CONFIG = BASE / "04_frozen_five_scene_config_1024.json"
GEOMETRY = BASE / "global_runs_1024/efficient_loftr/translation_l2/global_transforms.json"
ACCEPTED_GRAPH = BASE / "matcher_runs_1024_globalready/efficient_loftr/accepted_graph.json"
EDGE_SUMMARY = BASE / "global_runs_1024/efficient_loftr/translation_l2/05_translation_edge_summary.csv"
GRID = BASE / "mosaic_runs_1024/protocol/canonical_output_grid.json"
CACHE = BASE / "final_replay_v2_hardened/efficient_loftr_translation_l2_bagrn"
OUTPUT = BASE / "seam_local_task13a/protocol.json"
OUTPUT_ROOT = OUTPUT.parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(f"HARD_STOP_PROVENANCE_MISMATCH: {detail}")


def _scene_checks(scene: dict, scene_id: str, grid: dict) -> dict:
    path = CACHE / "normalized_scenes" / scene["path"]
    _require(scene["scene_id"] == scene_id, f"scene ID {scene['scene_index']}")
    _require(_sha256(path) == scene["sha256"], f"scene raster hash {path}")
    valid_count = 0
    finite_count = 0
    with rasterio.open(ROOT / path) as src:
        _require(src.count == 1, f"single band {path}")
        _require(src.crs is not None and src.crs.to_string() == grid["crs"], f"CRS {path}")
        _require(src.width == grid["width"] and src.height == grid["height"], f"shape {path}")
        _require(list(src.transform)[:6] == grid["transform"], f"transform {path}")
        for row in range(0, src.height, 256):
            window = Window(0, row, src.width, min(256, src.height - row))
            pixels = src.read(1, window=window)
            valid = src.read_masks(1, window=window) > 0
            valid_count += int(valid.sum())
            finite_count += int(np.isfinite(pixels[valid]).sum())
    _require(valid_count == scene["valid_pixels"], f"valid pixel count {path}")
    _require(finite_count == valid_count and valid_count > 0, f"nonfinite valid pixels {path}")
    return {
        "scene_index": scene["scene_index"],
        "scene_id": scene_id,
        "path": path.as_posix(),
        "sha256": scene["sha256"],
        "valid_pixels": valid_count,
        "finite_valid_pixels": finite_count,
    }


def build_protocol() -> dict:
    """Verify every frozen dependency before producing a single protocol record."""
    config = _load_json(SOURCE_CONFIG)
    grid = _load_json(GRID)
    manifest_path = CACHE / "normalized_scenes_manifest.json"
    manifest = _load_json(manifest_path)
    geometry_source = _load_json(CACHE / "geometry_source.json")
    radiometric_method = _load_json(CACHE / "radiometric_method.json")
    graph = _load_json(ACCEPTED_GRAPH)
    scene_ids = [scene["scene_id"] for scene in config["scenes"]]
    _require(config["selection"]["frozen"] is True, "source selection not frozen")
    _require(config["selection"]["manifest_indices"] == [2, 3, 5, 8, 10], "manifest indices")
    _require(len(scene_ids) == 5 and scene_ids == manifest["scene_ids"], "scene IDs")
    _require(manifest["method"] == radiometric_method["method"] == "BAGRN", "radiometric method")
    _require(manifest["geometry_run"] == geometry_source["geometry_run"] == "efficient_loftr_translation_l2", "geometry run")
    _require(geometry_source["matcher"] == "efficient_loftr" and geometry_source["global_method"] == "translation_l2", "geometry method")
    _require(geometry_source["geometry_mutable"] is False, "geometry mutability")
    hashes = {
        "source_config_sha256": _sha256(SOURCE_CONFIG),
        "global_transforms_sha256": _sha256(GEOMETRY),
        "canonical_grid_sha256": _sha256(GRID),
    }
    for field, expected in (
        ("source_config_sha256", manifest["source_config_sha256"]),
        ("global_transforms_sha256", manifest["global_transforms_sha256"]),
        ("canonical_grid_sha256", manifest["output_grid_sha256"]),
    ):
        _require(hashes[field] == expected, field)
    _require(hashes["source_config_sha256"] == geometry_source["source_config_sha256"], "geometry source config")
    _require(hashes["global_transforms_sha256"] == geometry_source["global_transforms_sha256"], "geometry transforms")
    _require(hashes["canonical_grid_sha256"] == geometry_source["output_grid_sha256"], "geometry grid")
    _require(grid["crs"] == "EPSG:32650" and grid["width"] == 5116 and grid["height"] == 6134, "canonical grid identity")
    _require(grid["transform"] == manifest["grid"]["transform"], "manifest grid transform")
    _require(grid["resolution"] == 14.0, "grid resolution")
    scenes = [_scene_checks(scene, scene_ids[index], grid) for index, scene in enumerate(manifest["scenes"])]
    _require([scene["scene_index"] for scene in scenes] == list(range(5)), "scene order")
    _require(graph["status"] == "CONNECTED" and graph["matcher"] == "efficient_loftr", "accepted graph")
    pairs = sorted((int(edge["idx_i"]), int(edge["idx_j"])) for edge in graph["accepted_edges"] if edge["status"] == "OK")
    _require(len(pairs) == 10 and pairs == [(i, j) for i in range(5) for j in range(i + 1, 5)], "10 accepted pairs")
    with (ROOT / EDGE_SUMMARY).open(newline="", encoding="utf-8") as stream:
        edge_pairs = sorted((int(row["edge_i"]), int(row["edge_j"])) for row in csv.DictReader(stream))
    _require(pairs == edge_pairs, "accepted graph/global edge disagreement")
    _require(_sha256(CACHE / "radiometric_summary.json") == manifest["radiometric_summary_sha256"], "radiometric summary hash")
    _require(_sha256(CACHE / "bagrn_parameters.npz") == manifest["bagrn_parameters_sha256"], "BAGRN parameter hash")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "schema_version": 1,
        "task": "Task13A pairwise seam-local feasibility",
        "branch": branch,
        "head": head,
        "input_policy": "frozen read-only; no matcher, global geometry, BAGRN, or VOLRN invocation",
        "radiometric_input": "BAGRN normalized_scenes only; VOLRN outputs excluded",
        "paths": {
            "source_config": SOURCE_CONFIG.as_posix(),
            "global_transforms": GEOMETRY.as_posix(),
            "accepted_graph": ACCEPTED_GRAPH.as_posix(),
            "global_edge_summary": EDGE_SUMMARY.as_posix(),
            "canonical_grid": GRID.as_posix(),
            "bagrn_cache": CACHE.as_posix(),
            "bagrn_manifest": manifest_path.as_posix(),
        },
        "hashes": {
            **hashes,
            "accepted_graph_sha256": _sha256(ACCEPTED_GRAPH),
            "global_edge_summary_sha256": _sha256(EDGE_SUMMARY),
            "bagrn_manifest_sha256": _sha256(manifest_path),
            "bagrn_parameters_sha256": manifest["bagrn_parameters_sha256"],
            "radiometric_summary_sha256": manifest["radiometric_summary_sha256"],
        },
        "grid": manifest["grid"],
        "band": "B9",
        "manifest_indices": [2, 3, 5, 8, 10],
        "scenes": scenes,
        "accepted_pairs": [
            {"pair_id": f"{i:02d}_{j:02d}", "scene_i": i, "scene_j": j,
             "scene_id_i": scene_ids[i], "scene_id_j": scene_ids[j]}
            for i, j in pairs
        ],
        "variants": ["BAGRN_WEIGHTED", "BAGRN_SEAM_ONLY", "BAGRN_LOCAL_REFINED_SEAM"],
        "parameters": {
            "cost": {"gradient": "existing Sobel magnitude", "alpha": 0.5, "beta": 0.5, "p95_normalization": True, "eps": 1e-6},
            "seam": {"model": "monotonic dynamic programming", "orientation": "vertical if overlap_height >= overlap_width else horizontal", "neighbor_steps": [-1, 0, 1], "coarse_factor": 4, "refine_half_width_px": 64, "full_resolution_if_dimension_below_px": 256},
            "local_moment": {"correction_half_width_px": 128, "segment_length_px": 256, "minimum_valid_pair_pixels_per_segment": 4096, "percentile_clip": [1, 99], "target": "symmetric mean and standard deviation", "degenerate_fallback": "nearest valid segment", "gain_unstable_below": 0.5, "gain_unstable_above": 2.0, "correction_taper": "0.5*(1+cos(pi*abs(d)/128))"},
            "blend": {"half_width_px": 64, "transition": "cosine", "source_side": "exclusive-valid geometry"},
        },
        "evaluation": {
            "fixed_initial_seam_corridor": ["MAE", "RMSE", "RDD_exact_W1", "mean_difference", "std_difference"],
            "actual_seam_half_width_px": 64,
            "actual_seam": ["MAE", "RMSE", "RDD_exact_W1", "mean_absolute_cost", "p95_cost"],
            "structure": ["existing_Task10D_CGL", "gradient_magnitude_NCC", "gradient_orientation_cosine"],
            "mosaic_diagnostics": ["valid_pixels", "finite_pixels", "NaN_Inf", "blend_pixels", "seam_length", "seam_orientation"],
        },
        "gate": {
            "v1_min_success_pairs": 8,
            "v2_min_success_pairs": 8,
            "v2_vs_v1_median_seam_MAE_RMSE_RDD": "decrease all three",
            "local_vs_bagrn_fixed_corridor_median_MAE_RDD": "decrease both",
            "CGL": "finite with no systematic clear deterioration",
            "max_unstable_gain_pairs": 2,
            "formal_outputs": "no Inf",
            "hard_stop_if_unsupported_topology_pairs_gt": 2,
            "hard_stop_if_unstable_gain_pairs_gt": 2,
            "decisions": ["PROMISING_FOR_MULTISCENE", "MIXED_NEEDS_REVIEW", "NOT_WORTH_CONTINUING"],
        },
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_ready) + "\n", encoding="utf-8")


def _flatten(prefix: str, value: Any, result: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten(f"{prefix}_{key}" if prefix else str(key), nested, result)
    elif isinstance(value, (tuple, list, np.ndarray)):
        result[prefix] = json.dumps(value, default=_json_ready)
    elif isinstance(value, np.generic):
        result[prefix] = value.item()
    else:
        result[prefix] = value


def _pair_record(pair: dict, result: PairResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "pair_id": pair["pair_id"], "scene_i": pair["scene_i"], "scene_j": pair["scene_j"],
        "status": result.status, "v0_status": result.v0_status,
        "v1_status": result.v1_status, "v2_status": result.v2_status,
    }
    if result.valid_union is not None:
        row["union_valid_pixels"] = int(np.count_nonzero(result.valid_union))
    for name in ("v0", "v1", "v2"):
        mosaic = getattr(result, f"{name}_mosaic")
        if mosaic is not None and result.valid_union is not None:
            valid = result.valid_union
            row[f"{name}_finite_valid_pixels"] = int(np.count_nonzero(np.isfinite(mosaic) & valid))
            row[f"{name}_nan_valid_pixels"] = int(np.count_nonzero(np.isnan(mosaic) & valid))
            row[f"{name}_inf_valid_pixels"] = int(np.count_nonzero(np.isinf(mosaic) & valid))
    for name, seam in (("initial", result.initial_seam), ("refined", result.refined_seam)):
        if seam is not None:
            row[f"{name}_seam_orientation"] = seam.orientation
            row[f"{name}_seam_length_pixels"] = len(seam.row_col_path)
            row[f"{name}_seam_search_mode"] = seam.search_mode
            row[f"{name}_seam_total_cost"] = seam.total_cost
            row[f"{name}_seam_mean_cost"] = seam.mean_cost
            row[f"{name}_seam_p95_cost"] = seam.p95_cost
    _flatten("metric", result.metrics, row)
    _flatten("diagnostic", result.diagnostics, row)
    return row


def _write_metrics(rows: list[dict[str, Any]]) -> None:
    path = ROOT / OUTPUT_ROOT / "pair_metrics.csv"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_seam(path: Path, seam: Any, origin: tuple[int, int], transform: Any) -> None:
    from rasterio.transform import xy

    coordinates = []
    for row, col in seam.row_col_path:
        x, y = xy(transform, int(origin[0] + row), int(origin[1] + col), offset="center")
        coordinates.append([float(x), float(y)])
    _write_json(path, {
        "type": "FeatureCollection", "name": path.stem,
        "crs": {"type": "name", "properties": {"name": "EPSG:32650"}},
        "features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": coordinates},
                      "properties": {"orientation": seam.orientation, "search_mode": seam.search_mode,
                                     "total_cost": seam.total_cost, "mean_cost": seam.mean_cost,
                                     "p95_cost": seam.p95_cost}}],
    })


def _write_mosaic(path: Path, mosaic: np.ndarray, result: PairResult, grid: dict) -> None:
    from affine import Affine

    assert result.crop_origin is not None and result.valid_union is not None
    valid = result.valid_union
    if not np.isfinite(mosaic[valid]).all():
        raise RuntimeError(f"NUMERICAL_INVALID: nonfinite valid mosaic pixels in {path.name}")
    offset = Affine.translation(result.crop_origin[1], result.crop_origin[0])
    transform = Affine(*grid["transform"]) * offset
    with rasterio.open(path, "w", driver="GTiff", width=mosaic.shape[1], height=mosaic.shape[0],
                       count=1, dtype="float32", crs=grid["crs"], transform=transform,
                       nodata=np.nan, compress="deflate", predictor=3, tiled=True) as dst:
        for first in range(0, mosaic.shape[0], 256):
            last = min(mosaic.shape[0], first + 256)
            tile = np.asarray(mosaic[first:last], dtype=np.float32).copy()
            tile[~valid[first:last]] = np.nan
            dst.write(tile, 1, window=Window(0, first, mosaic.shape[1], last - first))


def _write_pair_artifacts(pair: dict, result: PairResult, grid: dict) -> list[str]:
    pair_dir = ROOT / OUTPUT_ROOT / "pairs" / pair["pair_id"]
    if result.initial_seam is None and result.v1_mosaic is None:
        return []
    pair_dir.mkdir(parents=True, exist_ok=False)
    output_files: list[str] = []
    from affine import Affine

    transform = Affine(*grid["transform"])
    for name, seam in (("seam_initial.geojson", result.initial_seam),
                       ("seam_refined.geojson", result.refined_seam)):
        if seam is not None and result.crop_origin is not None:
            _write_seam(pair_dir / name, seam, result.crop_origin, transform)
            output_files.append(name)
    if result.local_segments:
        name = "local_coefficients.csv"
        with (pair_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(asdict(result.local_segments[0])))
            writer.writeheader()
            writer.writerows(asdict(segment) for segment in result.local_segments)
        output_files.append(name)
    for name, mosaic in (("v1_seam_only.tif", result.v1_mosaic),
                         ("v2_local_refined.tif", result.v2_mosaic)):
        if mosaic is not None:
            _write_mosaic(pair_dir / name, mosaic, result, grid)
            output_files.append(name)
    return output_files


def run_batch() -> dict:
    """Run each accepted frozen pair once, stopping only at fixed hard-stop counts."""
    frozen = _load_json(OUTPUT)
    current = build_protocol()
    current["head"] = frozen["head"]
    _require(current == frozen, "frozen protocol differs from current inputs")
    manifest_path = ROOT / OUTPUT_ROOT / "run_manifest.json"
    metrics_path = ROOT / OUTPUT_ROOT / "pair_metrics.csv"
    _require(not manifest_path.exists() and not metrics_path.exists(), "formal batch already started")
    manifest: dict[str, Any] = {
        "task": "Task13A frozen ten-pair feasibility batch",
        "status": "RUNNING", "started_utc": _utc_now(), "finished_utc": None,
        "protocol_sha256": _sha256(OUTPUT),
        "source_raster_sha256": {scene["scene_index"]: scene["sha256"] for scene in frozen["scenes"]},
        "no_upstream_rerun_policy": "Read frozen BAGRN scenes only; no matcher, global, BAGRN, or VOLRN invocation",
        "planned_pair_ids": [pair["pair_id"] for pair in frozen["accepted_pairs"]],
        "completed_pairs": [], "hard_stop": None,
    }
    _write_json(manifest_path, manifest)
    rows: list[dict[str, Any]] = []
    unsupported = unstable = 0
    for pair in frozen["accepted_pairs"]:
        pair_id = pair["pair_id"]
        try:
            arrays = []
            masks = []
            for index in (pair["scene_i"], pair["scene_j"]):
                scene = frozen["scenes"][index]
                with rasterio.open(ROOT / scene["path"]) as src:
                    arrays.append(src.read(1))
                    masks.append(src.read_masks(1) > 0)
            result = process_pair(arrays[0], arrays[1], masks[0], masks[1])
            del arrays, masks
            row = _pair_record(pair, result)
            files = _write_pair_artifacts(pair, result, frozen["grid"])
            row["artifact_files"] = json.dumps(files)
            rows.append(row)
            _write_metrics(rows)
            manifest["completed_pairs"].append({"pair_id": pair_id, "status": result.status,
                                                 "v1_status": result.v1_status,
                                                 "v2_status": result.v2_status,
                                                 "artifact_files": files})
            unsupported += int(result.status == "UNSUPPORTED_TOPOLOGY")
            unstable += int(result.status == "UNSTABLE_LOCAL_GAIN")
            if unsupported > 2:
                manifest["hard_stop"] = "HARD_STOP_PAIRWISE_TOPOLOGY_TOO_BRITTLE"
            if unstable > 2:
                manifest["hard_stop"] = "HARD_STOP_LOCAL_MODEL_UNSTABLE"
            _write_json(manifest_path, manifest)
            print(f"{pair_id}: {result.status} (V1={result.v1_status}, V2={result.v2_status})", flush=True)
            del result
            if manifest["hard_stop"]:
                break
        except Exception as exc:
            manifest["status"] = "ENGINEERING_ERROR"
            manifest["error_pair_id"] = pair_id
            manifest["error_type"] = type(exc).__name__
            manifest["error_message"] = str(exc)
            manifest["finished_utc"] = _utc_now()
            _write_json(manifest_path, manifest)
            raise
    manifest["status"] = manifest["hard_stop"] or "COMPLETED"
    manifest["finished_utc"] = _utc_now()
    manifest["pair_count_completed"] = len(rows)
    manifest["unsupported_topology_pairs"] = unsupported
    manifest["unstable_local_gain_pairs"] = unstable
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--verify-protocol", action="store_true")
    parser.add_argument("--run-batch", action="store_true")
    args = parser.parse_args()
    if sum((args.freeze_protocol, args.verify_protocol, args.run_batch)) != 1:
        parser.error("choose exactly one of --freeze-protocol, --verify-protocol, or --run-batch")
    if args.run_batch:
        manifest = run_batch()
        print(f"{manifest['status']}: {manifest['pair_count_completed']} pairs; run manifest at {(OUTPUT_ROOT / 'run_manifest.json').as_posix()}")
        return
    protocol = build_protocol()
    if args.freeze_protocol:
        destination = ROOT / OUTPUT
        destination.parent.mkdir(parents=True, exist_ok=True)
        _require(not destination.exists(), "protocol already frozen")
        destination.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"PASS: froze {OUTPUT.as_posix()} with {len(protocol['scenes'])} scenes and {len(protocol['accepted_pairs'])} pairs")
    else:
        frozen = _load_json(OUTPUT)
        # The recorded starting HEAD stays frozen even if Task13A code is committed.
        protocol["head"] = frozen["head"]
        _require(frozen == protocol, "frozen protocol differs from current inputs")
        print("PASS: frozen Task13A protocol and all five scene rasters verified")


if __name__ == "__main__":
    main()
