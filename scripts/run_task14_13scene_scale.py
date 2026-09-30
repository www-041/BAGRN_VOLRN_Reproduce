"""Task14: bounded, resumable 13-scene scalability validation.

The runner deliberately stops at the first frozen-pipeline hard stop.  It
does not silently replace EfficientLoFTR, relax graph gates, or infer large-
scale performance from the five-scene experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multiscene_sift.b9_dataset import discover_b9_scenes
from src.multiscene_sift.b9_overlap import build_b9_overlap_graph, connected_components
from src.multiscene_sift.models import OverlapEdge, PairwiseRegistration, Scene
from src.overlap import detect_multi_overlap

DEFAULT_INPUT_ROOT = Path(r"D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\input\B9_reference_13scenes_full")
DEFAULT_OUTPUT_ROOT = ROOT / "data/output/b9_13scene_task14"
TASK14_TILE_SIZE = 1024
TASK14_HALO = 128
HARD_STOP_RESUME_PROVENANCE_MISMATCH = "HARD_STOP_RESUME_PROVENANCE_MISMATCH"
HARD_STOP_STREAMING_REFERENCE_MISSING = "HARD_STOP_STREAMING_REFERENCE_MISSING"
HARD_STOP_MATCHER_UNAVAILABLE = "HARD_STOP_MATCHER_UNAVAILABLE"
HARD_STOP_TASK14_STAGE_NOT_IMPLEMENTED = "HARD_STOP_TASK14_STAGE_NOT_IMPLEMENTED"
HARD_STOP_PREFLIGHT_INVALID = "HARD_STOP_PREFLIGHT_INVALID"
HARD_STOP_REGISTRATION_GRAPH_DISCONNECTED = "HARD_STOP_REGISTRATION_GRAPH_DISCONNECTED"
HARD_STOP_TASK14_RESOURCE_LIMIT = "HARD_STOP_TASK14_RESOURCE_LIMIT"

STAGES = (
    "-1_streaming_equivalence",
    "00_preflight",
    "01_spatial_index_graph",
    "02_matching_ransac",
    "03_global_adjustment",
    "04_canonical_warp",
    "05_valid_distance_cache",
    "06_bagrn",
    "07_pairwise_seam_local",
    "08_multiscene_labeling",
    "09_v0_weighted_feather",
    "10_v1_v2_tile_mosaic",
    "11_metrics_figures",
    "12_scalability_metrics",
)


def _task14_downstream_stages() -> tuple[str, ...]:
    """Stages that are downstream of the completed Stage 06 BAGRN gate."""
    return STAGES[8:]


def unavailable_value() -> str:
    """Represent an unmeasurable resource without fabricating zero."""
    return "UNAVAILABLE"


def canonical_sha256(path: str | Path) -> str:
    """Hash one file in bounded chunks for provenance and resume safety."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): canonical_sha256(path) for path in paths if path.is_file()}


def stage_is_reusable(stage_dir: str | Path, input_hashes: dict[str, str], output_hashes: dict[str, str]) -> bool:
    """Validate a successful stage before ``--resume`` skips it."""
    stage_dir = Path(stage_dir)
    marker = stage_dir / "_SUCCESS.json"
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("status") not in {"SUCCESS", "SKIPPED_HARD_STOP"}:
        return False
    if payload.get("input_hashes") != input_hashes:
        raise RuntimeError(HARD_STOP_RESUME_PROVENANCE_MISMATCH)
    expected = payload.get("output_hashes", {})
    if expected != output_hashes:
        raise RuntimeError(HARD_STOP_RESUME_PROVENANCE_MISMATCH)
    for path, digest in expected.items():
        if not Path(path).is_file() or canonical_sha256(path) != digest:
            raise RuntimeError(HARD_STOP_RESUME_PROVENANCE_MISMATCH)
    return True


def _snapshot_resource() -> dict[str, Any]:
    try:
        import psutil  # type: ignore
        rss = float(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0))
    except Exception:
        rss = unavailable_value()
    gpu_alloc = unavailable_value()
    gpu_reserved = unavailable_value()
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            gpu_alloc = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
            gpu_reserved = float(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0))
    except Exception:
        pass
    return {
        "peak_cpu_rss_mb": rss,
        "torch_cuda_max_memory_allocated_mb": gpu_alloc,
        "torch_cuda_max_memory_reserved_mb": gpu_reserved,
    }


def _record_manifest(records: list[dict]) -> list[dict]:
    rows = []
    for index, record in enumerate(records):
        path = Path(record["b9_path"])
        mtl_path = Path(record["mtl_path"])
        with rasterio.open(path) as src:
            valid = int(np.count_nonzero(src.read_masks(1) > 0))
            transform = src.transform
            nodata = src.nodata
            dtype = src.dtypes[0]
        rows.append({
            "scene_index": index,
            "scene_id": record["scene_id"],
            "band": "B9",
            "acquisition_time": record.get("acquisition_time"),
            "crs": record.get("crs"),
            "pixel_size_x_m": float(record.get("resolution_x_m", abs(transform.a))),
            "pixel_size_y_m": float(record.get("resolution_y_m", abs(transform.e))),
            "width": int(record["width"]),
            "height": int(record["height"]),
            "valid_pixel_count": valid,
            "source_file_size_bytes": path.stat().st_size,
            "source_file_sha256": canonical_sha256(path),
            "b9_path": str(path),
            "mtl_path": str(mtl_path),
            "mtl_file_size_bytes": mtl_path.stat().st_size,
            "mtl_file_sha256": canonical_sha256(mtl_path),
            "left": float(record["left"]), "bottom": float(record["bottom"]),
            "right": float(record["right"]), "top": float(record["top"]),
            "dtype": dtype, "nodata": nodata,
            "transform": [float(v) for v in transform[:]],
        })
    return rows


def _bbox_candidates(records: list[dict], *, return_backend: bool = False):
    """Return bbox-overlap candidates using a real spatial index.

    Shapely's STRtree is the frozen Task14 index.  The explicit sweep fallback
    is retained only for environments without Shapely and is recorded in the
    graph summary rather than being silently presented as an indexed run.
    """
    candidates: set[tuple[int, int]] = set()
    backend = "STRtree"
    try:
        from shapely.geometry import box
        from shapely.strtree import STRtree

        geoms = [box(float(r["left"]), float(r["bottom"]), float(r["right"]), float(r["top"])) for r in records]
        tree = STRtree(geoms)
        by_id = {id(g): i for i, g in enumerate(geoms)}
        by_wkb = {g.wkb: i for i, g in enumerate(geoms)}
        for i, geom in enumerate(geoms):
            for hit in tree.query(geom):
                if isinstance(hit, (int, np.integer)):
                    j = int(hit)
                else:
                    j = by_id.get(id(hit), by_wkb.get(hit.wkb, -1))
                if j <= i or j < 0:
                    continue
                # Keep the exact strict-intersection semantics used by the
                # exhaustive audit (touching bounds are not overlap).
                if geom.intersection(geoms[j]).area > 0.0:
                    candidates.add((i, j))
    except Exception:
        backend = "bbox_sweep_fallback"
        for i, a in enumerate(records):
            for j in range(i + 1, len(records)):
                if min(float(a["right"]), float(records[j]["right"])) > max(float(a["left"]), float(records[j]["left"])) and min(float(a["top"]), float(records[j]["top"])) > max(float(a["bottom"]), float(records[j]["bottom"])):
                    candidates.add((i, j))
    return (candidates, backend) if return_backend else candidates


def graph_statistics(records: list[dict], pairs: list[dict], *, bbox_candidate_pairs: set[tuple[int, int]], spatial_index_backend: str = "STRtree") -> dict:
    """Summarize exact overlap graph and spatial-index accounting."""
    exact = {(int(p["idx_i"]), int(p["idx_j"])) for p in pairs if p.get("has_overlap")}
    components = connected_components(len(records), pairs)
    degrees = [0] * len(records)
    for i, j in exact:
        degrees[i] += 1; degrees[j] += 1
    areas = sorted(float(p["intersection_area"]) for p in pairs if p.get("has_overlap"))
    n_possible = len(records) * (len(records) - 1) // 2
    false_negative = exact - bbox_candidate_pairs
    false_positive = bbox_candidate_pairs - exact
    return {
        "scene_count": len(records),
        "possible_pair_count": n_possible,
        "spatial_index_bbox_candidate_count": len(bbox_candidate_pairs),
        "spatial_index_backend": spatial_index_backend,
        "exact_overlap_edge_count": len(exact),
        "non_overlap_pair_count": n_possible - len(exact),
        "spatial_index_false_negative_count": len(false_negative),
        "spatial_index_false_positive_bbox_count": len(false_positive),
        "pair_pruning_fraction": (1.0 - len(bbox_candidate_pairs) / n_possible) if n_possible else 0.0,
        "graph_density": (2.0 * len(exact) / (len(records) * (len(records) - 1))) if len(records) > 1 else 0.0,
        "connected_component_count": len(components),
        "connected_components": components,
        "degree_min": min(degrees) if degrees else 0,
        "degree_median": float(median(degrees)) if degrees else 0.0,
        "degree_mean": float(np.mean(degrees)) if degrees else 0.0,
        "degree_p95": float(np.percentile(degrees, 95)) if degrees else 0.0,
        "degree_max": max(degrees) if degrees else 0,
        "cycle_rank": len(exact) - len(records) + len(components),
        "overlap_area_min": min(areas) if areas else 0.0,
        "overlap_area_median": float(median(areas)) if areas else 0.0,
        "overlap_area_p95": float(np.percentile(areas, 95)) if areas else 0.0,
        "overlap_area_max": max(areas) if areas else 0.0,
    }


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _stage_marker(stage_dir: Path, *, input_hashes: dict[str, str], output_paths: list[Path], params: dict, started: float, status: str, error: str | None = None) -> None:
    output_hashes = _hashes(output_paths)
    payload = {
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "stage_parameters": params,
        "wall_time_sec": time.perf_counter() - started,
        "resource": _snapshot_resource(),
        "status": status,
    }
    if error:
        payload["error"] = error
    _write_json(stage_dir / "_SUCCESS.json", payload)


def _scene_models(records: list[dict]) -> list[Scene]:
    scenes = []
    for i, record in enumerate(records):
        from affine import Affine
        from rasterio.crs import CRS
        tf = Affine(float(record["resolution_x_m"]), 0.0, float(record["left"]), 0.0, -float(record["resolution_y_m"]), float(record["top"]))
        scenes.append(Scene(index=i, name=record["scene_id"], directory=record["scene_dir"], band_paths={"B9": record["b9_path"]}, crs=CRS.from_string(record["crs"]) if record.get("crs") else None, transforms={"B9": tf}, shapes={"B9": (int(record["height"]), int(record["width"]))}, nodata={"B9": None}, bounds={"B9": (float(record["left"]), float(record["bottom"]), float(record["right"]), float(record["top"]))}))
    return scenes


def _edge_models(pairs: list[dict]) -> list[OverlapEdge]:
    return [
        OverlapEdge(
            idx_i=int(p["idx_i"]), idx_j=int(p["idx_j"]),
            intersection_area=float(p["intersection_area"]),
            overlap_ratio_i=float(p.get("overlap_area_i_ratio", p.get("symmetric_overlap_ratio", 0.0))),
            overlap_ratio_j=float(p.get("overlap_area_j_ratio", p.get("symmetric_overlap_ratio", 0.0))),
        )
        for p in pairs if p.get("has_overlap")
    ]


def _write_stage_runtime(root: Path, rows: list[dict]) -> None:
    fields = ["stage", "status", "wall_time_sec", "peak_cpu_rss_mb", "torch_cuda_max_memory_allocated_mb", "torch_cuda_max_memory_reserved_mb", "error"]
    numeric = [r.get("wall_time_sec") for r in rows if isinstance(r.get("wall_time_sec"), (int, float))]
    rss = [r.get("peak_cpu_rss_mb") for r in rows if isinstance(r.get("peak_cpu_rss_mb"), (int, float))]
    total_status = "SUCCESS" if all(r.get("status") == "SUCCESS" for r in rows) else "PARTIAL_HARD_STOP"
    total_error = ";".join(dict.fromkeys(str(r.get("error")) for r in rows if r.get("error")))
    total = {
        "stage": "TOTAL", "status": total_status,
        "wall_time_sec": float(sum(numeric)),
        "peak_cpu_rss_mb": max(rss) if rss else unavailable_value(),
        "torch_cuda_max_memory_allocated_mb": unavailable_value(),
        "torch_cuda_max_memory_reserved_mb": unavailable_value(),
        "error": total_error,
    }
    output_rows = [*rows, total]
    _write_csv(root / "scalability/stage_runtime.csv", output_rows, fields)
    _write_csv(root / "scalability/resource_usage.csv", output_rows, fields)


def _validate_resume_markers(output_root: Path) -> dict | None:
    """Validate every completed marker and return the first hard-stop payload."""
    for stage_name in STAGES:
        marker = output_root / f"stages/{stage_name}/_SUCCESS.json"
        if not marker.is_file():
            continue
        payload = json.loads(marker.read_text(encoding="utf-8"))
        input_hashes = {path: canonical_sha256(path) for path in payload.get("input_hashes", {}) if Path(path).is_file()}
        if set(input_hashes) != set(payload.get("input_hashes", {})):
            raise RuntimeError(HARD_STOP_RESUME_PROVENANCE_MISMATCH)
        stage_is_reusable(output_root / f"stages/{stage_name}", input_hashes, payload.get("output_hashes", {}))
        if payload.get("status") == "HARD_STOP":
            return payload
    return None


def load_task14_prefix(output_root: str | Path) -> dict:
    """Load Stage -1/00/01 artifacts without rediscovering or recomputing them."""
    output_root = Path(output_root)
    protocol = json.loads((output_root / "protocol.json").read_text(encoding="utf-8"))
    manifest_rows = list(csv.DictReader((output_root / "scene_manifest.csv").open(encoding="utf-8", newline="")))
    records = []
    for row in manifest_rows:
        b9_path = Path(row["b9_path"])
        nodata_text = row.get("nodata", "")
        try:
            nodata = float(nodata_text) if nodata_text not in {"", "None", "nan"} else None
        except ValueError:
            nodata = None
        records.append({
            "scene_index": int(row["scene_index"]),
            "scene_id": row["scene_id"],
            "b9_path": row["b9_path"],
            "mtl_path": row.get("mtl_path", ""),
            "scene_dir": str(b9_path.parent),
            "crs": row.get("crs") or None,
            "resolution_x_m": float(row.get("pixel_size_x_m", 14.0)),
            "resolution_y_m": float(row.get("pixel_size_y_m", 14.0)),
            "width": int(row["width"]), "height": int(row["height"]),
            "left": float(row["left"]), "bottom": float(row["bottom"]),
            "right": float(row["right"]), "top": float(row["top"]),
            "nodata": nodata,
        })
    pair_rows = []
    overlap_path = output_root / "overlap_edges.csv"
    if overlap_path.is_file():
        for row in csv.DictReader(overlap_path.open(encoding="utf-8", newline="")):
            item = dict(row)
            item["idx_i"] = int(item["idx_i"]); item["idx_j"] = int(item["idx_j"])
            item["intersection_area"] = float(item.get("intersection_area", 0.0))
            item["has_overlap"] = str(item.get("has_overlap", "False")).lower() == "true"
            pair_rows.append(item)
    graph_path = output_root / "graph_summary.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.is_file() else None
    return {"protocol": protocol, "records": records, "pairs": pair_rows, "graph": graph}


def validate_task14_prefix(output_root: str | Path) -> bool:
    """Validate the frozen Stage -1/00/01 markers without executing them."""
    output_root = Path(output_root)
    required = STAGES[:3]
    for stage_name in required:
        marker_path = output_root / f"stages/{stage_name}/_SUCCESS.json"
        if not marker_path.is_file():
            raise RuntimeError(HARD_STOP_RESUME_PROVENANCE_MISMATCH)
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        if payload.get("status") != "SUCCESS":
            raise RuntimeError(HARD_STOP_RESUME_PROVENANCE_MISMATCH)
        input_hashes = {path: canonical_sha256(path) for path in payload.get("input_hashes", {}) if Path(path).is_file()}
        if set(input_hashes) != set(payload.get("input_hashes", {})):
            raise RuntimeError(HARD_STOP_RESUME_PROVENANCE_MISMATCH)
        stage_is_reusable(output_root / f"stages/{stage_name}", input_hashes, payload.get("output_hashes", {}))
    return True


def _task14_registration_rows(results: list[PairwiseRegistration], records: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        transform = np.asarray(result.pair_pixel_matrix, dtype=np.float64)
        transform_hash = _json_sha256(transform.tolist())
        common = result.pair_common_transform
        common_values = list(common)[:6] if common is not None else None
        rows.append({
            "pair_id": f"{result.idx_i:02d}_{result.idx_j:02d}",
            "scene_i": records[result.idx_i]["scene_id"], "scene_j": records[result.idx_j]["scene_id"],
            "status": result.status, "raw_match_count": result.raw_matches,
            "valid_match_count": result.raw_matches, "ransac_inlier_count": result.inliers,
            "inlier_ratio": result.inlier_ratio, "coverage": result.coverage,
            "pairwise_rmse_px": None if not np.isfinite(result.residual_rmse) else result.residual_rmse,
            "pairwise_p95_px": None if not np.isfinite(result.residual_p95) else result.residual_p95,
            "pairwise_max_px": None, "matcher_runtime_sec": result.matcher_runtime_sec,
            "ransac_runtime_sec": result.geometry_runtime_sec,
            "total_edge_runtime_sec": result.runtime_sec,
            "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
            "transform": transform.tolist(), "transform_hash": transform_hash,
            "pair_common_transform": common_values,
            "accepted_for_global": result.status == "OK",
        })
    return rows


def _persist_task14_geometry(results: list[PairwiseRegistration], records: list[dict], output_root: Path) -> list[Path]:
    from src.multiscene_sift.geometry_artifacts import save_pair_geometry_bundle
    geometry_dir = output_root / "registration/geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    config_hash = canonical_sha256(output_root / "protocol.json")
    scenes = _scene_models(records)
    outputs = []
    for result in results:
        if result.pair_common_transform is None or result.inlier_ref_xy is None or result.inlier_tgt_xy is None:
            continue
        sidecar = geometry_dir / f"pair_{result.idx_i:02d}_{result.idx_j:02d}.json"
        if sidecar.exists():
            outputs.extend([sidecar, sidecar.with_suffix(".npz")])
            continue
        path = save_pair_geometry_bundle(
            geometry_dir, matcher="efficient_loftr", idx_i=result.idx_i, idx_j=result.idx_j,
            scene_i=scenes[result.idx_i].name, scene_j=scenes[result.idx_j].name,
            accepted=result.status == "OK", status=result.status,
            inlier_ref_xy=result.inlier_ref_xy, inlier_tgt_xy=result.inlier_tgt_xy,
            pair_pixel_matrix=result.pair_pixel_matrix, pair_common_transform=result.pair_common_transform,
            crs=str(scenes[result.idx_i].crs) if scenes[result.idx_i].crs else None,
            pixel_size_x_m=14.0, pixel_size_y_m=14.0,
            coordinate_frame="pair_common_grid", transform_direction="target_to_reference",
            config_path=str(output_root / "protocol.json"), config_sha256=config_hash, match_max_side=1024,
        )
        outputs.extend([path, path.with_suffix(".npz")])
    return outputs


def resume_task14_stage02(
    output_root: str | Path,
    *,
    repo_dir: str | Path,
    weights_path: str | Path,
    device: str = "cuda",
) -> dict:
    """Resume from the frozen Task14 Stage 02 checkpoint only."""
    output_root = Path(output_root)
    validate_task14_prefix(output_root)
    prefix = load_task14_prefix(output_root)
    records = prefix["records"]
    pairs = prefix["pairs"]
    edge_pairs = [pair for pair in pairs if pair.get("has_overlap")]
    if len(records) != 13 or len(edge_pairs) != 60:
        raise RuntimeError(HARD_STOP_RESUME_PROVENANCE_MISMATCH)
    os.environ["EFFICIENT_LOFTR_REPO"] = str(Path(repo_dir).resolve())
    os.environ["EFFICIENT_LOFTR_WEIGHTS"] = str(Path(weights_path).resolve())
    from src.registration_benchmark.matchers.efficient_loftr import is_efficient_loftr_available
    if not is_efficient_loftr_available(repo_dir, weights_path):
        raise RuntimeError(HARD_STOP_MATCHER_UNAVAILABLE)
    from src.multiscene_sift.pairwise import run_all_pairs
    started = time.perf_counter()
    results = run_all_pairs(
        _scene_models(records), _edge_models(edge_pairs), output_root / "registration",
        band="B9", match_max_side=1024, ransac_threshold=2.0, random_seed=0,
        matcher="efficient_loftr", device=device,
    )
    rows = _task14_registration_rows(results, records)
    _write_csv(output_root / "registration_pair_metrics.csv", rows)
    geometry_outputs = _persist_task14_geometry(results, records, output_root)
    registration = _registration_metrics(rows)
    accepted = [result for result in results if result.status == "OK"]
    try:
        adj, _ = __import__("src.multiscene_sift.global_registration", fromlist=["build_accepted_graph"]).build_accepted_graph(results, matcher_name="EfficientLoFTR", n_scenes=len(records))
        components = 1
        accepted_graph_status = "CONNECTED"
    except RuntimeError:
        adj = {}
        components = len(connected_components(len(records), [{"idx_i": r.idx_i, "idx_j": r.idx_j, "has_overlap": r.status == "OK"} for r in results]))
        accepted_graph_status = "DISCONNECTED"
    registration.update({"registration_graph_components": components, "registration_graph_cycle_rank": len(accepted) - len(records) + components, "accepted_graph_status": accepted_graph_status})
    _write_json(output_root / "global_geometry_summary.json", registration)
    sdir = output_root / "stages/02_matching_ransac"; sdir.mkdir(parents=True, exist_ok=True)
    outputs = [output_root / "registration_pair_metrics.csv", output_root / "global_geometry_summary.json", *geometry_outputs]
    _stage_marker(sdir, input_hashes=_hashes([output_root / "overlap_edges.csv", output_root / "runtime_recovery/02_five_scene_matcher_replay.json"]), output_paths=outputs, params={"matcher": "EfficientLoFTR", "match_max_side": 1024, "device": device, "resume_from": "Stage -1/00/01"}, started=started, status="SUCCESS")
    return {"status": "SUCCESS", "accepted_edges": len(accepted), "registration": registration, "runtime_sec": time.perf_counter() - started}


def load_task14_registration_results(output_root: str | Path, records: list[dict]) -> list[PairwiseRegistration]:
    """Rehydrate Stage 02 results from persisted CSV/geometry bundles."""
    output_root = Path(output_root)
    rows = list(csv.DictReader((output_root / "registration_pair_metrics.csv").open(encoding="utf-8", newline="")))
    geometry_dir = output_root / "registration/geometry"
    results = []
    from affine import Affine
    for row in rows:
        pair_id = row["pair_id"]
        idx_i, idx_j = map(int, pair_id.split("_"))
        matrix = json.loads(row["transform"])
        common_values = json.loads(row["pair_common_transform"]) if row.get("pair_common_transform") not in {"", "None"} else None
        common = Affine(*common_values) if common_values else None
        ref_xy = tgt_xy = None
        sidecar = geometry_dir / f"pair_{pair_id}.json"
        if sidecar.is_file():
            side = json.loads(sidecar.read_text(encoding="utf-8"))
            common = Affine(*np.asarray(side["pair_common_transform"]["matrix"], dtype=np.float64).reshape(3, 3)[:2, :].reshape(-1))
            arrays = np.load(sidecar.with_suffix(".npz"))
            ref_xy = np.asarray(arrays["inlier_ref_xy"], dtype=np.float64)
            tgt_xy = np.asarray(arrays["inlier_tgt_xy"], dtype=np.float64)
        def _float_or_nan(name: str) -> float:
            value = row.get(name, "")
            return float(value) if value not in {"", "None", "nan"} else float("nan")
        peak = row.get("peak_gpu_memory_mb", "")
        results.append(PairwiseRegistration(
            idx_i=idx_i, idx_j=idx_j, status=row["status"], raw_matches=int(row["raw_match_count"]),
            inliers=int(row["ransac_inlier_count"]), inlier_ratio=float(row["inlier_ratio"]),
            coverage=float(row["coverage"]), residual_median=float("nan"), residual_rmse=_float_or_nan("pairwise_rmse_px"),
            residual_p95=_float_or_nan("pairwise_p95_px"), pair_pixel_matrix=matrix,
            pair_common_transform=common, runtime_sec=float(row.get("total_edge_runtime_sec", 0.0)),
            inlier_ref_xy=ref_xy, inlier_tgt_xy=tgt_xy, matcher="efficient_loftr",
            matcher_runtime_sec=float(row.get("matcher_runtime_sec", 0.0)), geometry_runtime_sec=float(row.get("ransac_runtime_sec", 0.0)),
            peak_gpu_memory_mb=float(peak) if peak not in {"", "None", "nan"} else None,
        ))
    return results


def _task14_source_config(output_root: Path, records: list[dict]) -> Path:
    payload_records = []
    for record in records:
        item = dict(record)
        item["shape"] = [int(record["height"]), int(record["width"])]
        item["transform"] = [float(record["resolution_x_m"]), 0.0, float(record["left"]), 0.0, -float(record["resolution_y_m"]), float(record["top"])]
        payload_records.append(item)
    path = output_root / "stages/04_canonical_warp/task14_source_config.json"
    _write_json(path, {"schema_version": 1, "band": "B9", "scenes": payload_records})
    return path


def _task14_canonical_grid(output_root: Path, records: list[dict], transforms: list[np.ndarray]) -> dict:
    from affine import Affine
    from src.multiscene_sift.mosaic_protocol import transformed_scene_bounds
    inventory = []
    for record in records:
        item = dict(record); item["shape"] = [int(record["height"]), int(record["width"])]
        item["transform"] = [float(record["resolution_x_m"]), 0.0, float(record["left"]), 0.0, -float(record["resolution_y_m"]), float(record["top"])]
        inventory.append(item)
    bounds = [transformed_scene_bounds(item, matrix) for item, matrix in zip(inventory, transforms)]
    resolution = 14.0
    left = np.floor(min(b[0] for b in bounds) / resolution) * resolution
    bottom = np.floor(min(b[1] for b in bounds) / resolution) * resolution
    right = np.ceil(max(b[2] for b in bounds) / resolution) * resolution
    top = np.ceil(max(b[3] for b in bounds) / resolution) * resolution
    width = int(round((right - left) / resolution)); height = int(round((top - bottom) / resolution))
    transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)
    return {
        "schema_version": 1, "crs": "EPSG:32650", "resolution": resolution, "pixel_size": resolution,
        "pixel_size_m": resolution, "origin": [float(left), float(top)], "width": width, "height": height,
        "transform": list(transform)[:6], "bounds": [float(left), float(bottom), float(right), float(top)],
        "source_union_bounds": [min(r["left"] for r in records), min(r["bottom"] for r in records), max(r["right"] for r in records), max(r["top"] for r in records)],
        "transformed_union_bounds": [float(left), float(bottom), float(right), float(top)],
        "grid_identity": {"crs": "EPSG:32650", "resolution": resolution, "pixel_size": resolution, "origin": [float(left), float(top)], "width": width, "height": height, "bounds": [float(left), float(bottom), float(right), float(top)]},
    }


def _task14_window_from_bounds(
    bounds: tuple[float, float, float, float], grid: dict
) -> tuple[int, int, int, int]:
    """Return a clipped row/column window on the frozen north-up grid."""
    from affine import Affine

    left, bottom, right, top = (float(value) for value in bounds)
    transform = Affine(*grid["transform"])
    resolution_x = abs(float(transform.a))
    resolution_y = abs(float(transform.e))
    col_start = int(np.floor((left - transform.c) / resolution_x))
    col_stop = int(np.ceil((right - transform.c) / resolution_x))
    row_start = int(np.floor((transform.f - top) / resolution_y))
    row_stop = int(np.ceil((transform.f - bottom) / resolution_y))
    row_start = max(0, min(int(grid["height"]), row_start))
    row_stop = max(0, min(int(grid["height"]), row_stop))
    col_start = max(0, min(int(grid["width"]), col_start))
    col_stop = max(0, min(int(grid["width"]), col_stop))
    return row_start, row_stop, col_start, col_stop


def _task14_local_registered_inputs(
    source_config: Path, global_run_dir: Path, grid: dict
) -> tuple[list[np.ndarray], list, list, list[str], list[np.ndarray], list[tuple[int, int, int, int]]]:
    """Register one scene at a time, retaining only its local canonical footprint."""
    from affine import Affine
    from scripts.run_b9_weighted_mosaic import _load_global_transforms, _scene_path
    from src.multiscene_sift.band_geometry import apply_world_correction_to_transform

    payload = json.loads(source_config.read_text(encoding="utf-8"))
    scenes = payload["scenes"]
    global_transforms = _load_global_transforms(global_run_dir, len(scenes))
    destination_transform = Affine(*grid["transform"])
    arrays, transforms, bounds, scene_ids, valid_masks, crop_slices = [], [], [], [], [], []
    for index, record in enumerate(scenes):
        path = _scene_path(source_config, record)
        with rasterio.open(path) as source:
            source_array = source.read(1)
            source_crs = source.crs.to_string() if source.crs else grid["crs"]
            nodata = source.nodata if source.nodata is not None else record.get("nodata")
            corrected = apply_world_correction_to_transform(
                source.transform, global_transforms[index]
            )
            source_bounds = array_bounds(source.height, source.width, corrected)
        row_start, row_stop, col_start, col_stop = _task14_window_from_bounds(source_bounds, grid)
        if row_stop <= row_start or col_stop <= col_start:
            raise ValueError(f"scene {index} has no intersection with the canonical grid")
        local_transform = destination_transform * Affine.translation(col_start, row_start)
        projected = np.full(
            (row_stop - row_start, col_stop - col_start), np.nan, dtype=np.float64
        )
        reproject(
            source=source_array.astype(np.float64, copy=False),
            destination=projected,
            src_transform=corrected,
            src_crs=source_crs,
            dst_transform=local_transform,
            dst_crs=grid["crs"],
            src_nodata=nodata,
            dst_nodata=np.nan,
            init_dest_nodata=True,
            resampling=Resampling.bilinear,
        )
        valid = np.isfinite(projected)
        rows, cols = np.where(valid)
        if rows.size == 0 or cols.size == 0:
            raise ValueError(f"scene {index} has no valid canonical pixels")
        local_row_start, local_row_stop = int(rows.min()), int(rows.max()) + 1
        local_col_start, local_col_stop = int(cols.min()), int(cols.max()) + 1
        arrays.append(
            projected[local_row_start:local_row_stop, local_col_start:local_col_stop]
            .astype(np.float32, copy=True)[np.newaxis, :, :]
        )
        local_transform = local_transform * Affine.translation(local_col_start, local_row_start)
        transforms.append(local_transform)
        bounds.append(
            tuple(
                float(value)
                for value in array_bounds(local_row_stop - local_row_start, local_col_stop - local_col_start, local_transform)
            )
        )
        scene_ids.append(str(record.get("scene_id", path.stem)))
        valid_masks.append(valid[local_row_start:local_row_stop, local_col_start:local_col_stop].copy())
        crop_slices.append(
            (
                row_start + local_row_start,
                row_start + local_row_stop,
                col_start + local_col_start,
                col_start + local_col_stop,
            )
        )
        del source_array, projected, valid
    return arrays, transforms, bounds, scene_ids, valid_masks, crop_slices


def _task14_write_full_raster_window(
    path: Path, data: np.ndarray, crop: tuple[int, int, int, int], grid: dict, *, dtype: str, nodata
) -> None:
    """Write a local scene array into a full-grid GeoTIFF without a full canvas buffer."""
    from src.multiscene_sift.mosaic_protocol import grid_transform

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": int(grid["height"]), "width": int(grid["width"]),
        "count": 1, "dtype": dtype, "crs": grid["crs"], "transform": grid_transform(grid),
        "nodata": nodata, "compress": "deflate", "tiled": True,
        "blockxsize": 256, "blockysize": 256, "sparse_ok": True,
    }
    r0, r1, c0, c1 = crop
    with rasterio.open(path, "w", **profile) as dst:
        fill_value = np.nan if dtype == "float32" else 0
        for row in range(0, int(grid["height"]), 256):
            height = min(256, int(grid["height"]) - row)
            dst.write(
                np.full((height, int(grid["width"])), fill_value, dtype=dtype),
                1,
                window=Window(0, row, int(grid["width"]), height),
            )
        dst.write(np.asarray(data, dtype=dtype), 1, window=Window(c0, r0, c1 - c0, r1 - r0))


def _task14_stream_bagrn(
    output_root: Path, source_config: Path, global_run_dir: Path, grid: dict
) -> dict:
    """Run the existing BAGRN equations on local arrays and stream its artifacts."""
    from src.bagrn import (
        _apply_moment_matching,
        _compute_overlap_moment_params,
        _overlap_means_stds,
        _solve_compensation,
    )
    from src.multiscene_sift.mosaic_protocol import grid_transform

    arrays, transforms, bounds, scene_ids, valid_masks, crop_slices = _task14_local_registered_inputs(
        source_config, global_run_dir, grid
    )
    nodata_values = [None] * len(arrays)
    overlaps = detect_multi_overlap(bounds, transforms, min_pixels=100)
    if not overlaps:
        raise RuntimeError("fixed geometry has no usable radiometric overlaps")
    pair_means, pair_stds, pair_pixels, clear_valid_counts = _overlap_means_stds(
        arrays, nodata_values, overlaps, [0]
    )
    valid_pairs = np.all(clear_valid_counts > 0, axis=2)
    for band_index in range(arrays[0].shape[0]):
        valid_indices = np.flatnonzero(valid_pairs[:, band_index])
        adjacency = {index: set() for index in range(len(arrays))}
        for pair_index in valid_indices:
            overlap = overlaps[int(pair_index)]
            left, right = int(overlap["idx_i"]), int(overlap["idx_j"])
            adjacency[left].add(right); adjacency[right].add(left)
        visited, stack = set(), [0]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current); stack.extend(adjacency[current] - visited)
        if len(visited) != len(arrays):
            raise RuntimeError("BAGRN_RADIO_NETWORK_DISCONNECTED")
    theta_mu = _solve_compensation(len(arrays), overlaps, pair_means, pair_pixels, 0, 1, valid_pairs)
    theta_sigma = _solve_compensation(len(arrays), overlaps, pair_stds, pair_pixels, 0, 1, valid_pairs)
    bagrn_dir = output_root / "stages/06_bagrn/bagrn"
    bagrn_dir.mkdir(parents=True, exist_ok=True)
    normalized_paths = []
    support_paths = []
    for index, (array, crop) in enumerate(zip(arrays, crop_slices)):
        omega, upsilon = _compute_overlap_moment_params(
            index, 0, overlaps, pair_means, pair_stds, pair_pixels,
            theta_mu, theta_sigma, 1, valid_pairs
        )
        normalized = _apply_moment_matching(array, None, omega, upsilon, [0])
        normalized_path = bagrn_dir / "normalized_scenes" / f"scene_{index:03d}.tif"
        support_path = bagrn_dir / "normalized_scenes/input_valid_masks" / f"scene_{index:03d}.tif"
        _task14_write_full_raster_window(normalized_path, normalized[0].astype(np.float32), crop, grid, dtype="float32", nodata=np.nan)
        _task14_write_full_raster_window(support_path, valid_masks[index].astype(np.uint8), crop, grid, dtype="uint8", nodata=0)
        normalized_paths.append(normalized_path)
        support_paths.append(support_path)
        del normalized
    np.savez_compressed(bagrn_dir / "bagrn_parameters.npz", theta_mu=theta_mu, theta_sigma=theta_sigma)
    distance_paths = [
        output_root / "stages/05_valid_distance_cache" / f"distance_scene_{index:03d}.tif"
        for index in range(len(arrays))
    ]
    mosaic_path = bagrn_dir / "mosaic.tif"
    _stream_blend(normalized_paths, distance_paths, mosaic_path)
    manifest = {
        "schema_version": 1, "method": "BAGRN", "geometry_run": "efficient_loftr_translation_l2",
        "radiometric_control_idx": 0, "scene_ids": scene_ids,
        "grid": grid.get("grid_identity", grid),
        "scenes": [
            {"scene_index": index, "scene_id": scene_ids[index], "path": str(path), "support_path": str(support_paths[index]),
             "valid_pixels": int(valid_masks[index].sum())}
            for index, path in enumerate(normalized_paths)
        ],
    }
    _write_json(bagrn_dir / "normalized_scenes_manifest.json", manifest)
    summary = {
        "schema_version": 2, "dataset": "B9", "geometry_run": "efficient_loftr_translation_l2",
        "radiometric_method": "BAGRN", "science_status": "PASS_BAGRN",
        "radiometric_control_idx": 0, "scene_ids": scene_ids,
        "overlap_count": len(overlaps), "bagrn": {"theta_mu": theta_mu.tolist(), "theta_sigma": theta_sigma.tolist()},
        "outputs": {"mosaic.tif": "mosaic.tif", "bagrn_parameters.npz": "bagrn_parameters.npz", "normalized_scenes_manifest.json": "normalized_scenes_manifest.json"},
    }
    _write_json(bagrn_dir / "radiometric_summary.json", summary)
    return {"summary": summary, "output_dir": str(bagrn_dir)}


def continue_task14_after_stage02(output_root: str | Path, *, repo_dir: str | Path, weights_path: str | Path, device: str = "cuda") -> dict:
    """Execute frozen global/BAGRN stages after a validated Stage 02 resume.

    The repository currently has no 13-scene Task13A/13B adapter.  Stages
    07 onward therefore terminate with an explicit hard stop after preserving
    valid Stage 03-06 artifacts; no five-scene seams are extrapolated.
    """
    output_root = Path(output_root)
    prefix = load_task14_prefix(output_root)
    records = prefix["records"]
    results = load_task14_registration_results(output_root, records)
    accepted = [r for r in results if r.status == "OK"]
    if len(accepted) == 0:
        raise RuntimeError(HARD_STOP_REGISTRATION_GRAPH_DISCONNECTED)
    from src.multiscene_sift.global_execution import run_global_connections
    scenes = _scene_models(records)
    stage03 = output_root / "stages/03_global_adjustment"
    stage03.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    global_result = run_global_connections({"results": results, "scenes": scenes}, stage03 / "global", matcher="efficient_loftr", pixel_size_m=14.0)
    _write_json(stage03 / "global_adjustment_summary.json", global_result)
    _stage_marker(stage03, input_hashes=_hashes([output_root / "registration_pair_metrics.csv"]), output_paths=[stage03 / "global_adjustment_summary.json", stage03 / "global/translation_l2/global_transforms.json"], params={"global": "Translation-L2", "matcher": "EfficientLoFTR"}, started=started, status="SUCCESS")
    stage04 = output_root / "stages/04_canonical_warp"; stage04.mkdir(parents=True, exist_ok=True)
    matrices_payload = json.loads((stage03 / "global/translation_l2/global_transforms.json").read_text(encoding="utf-8"))
    transforms = [np.asarray(item["matrix"], dtype=np.float64) for item in sorted(matrices_payload["transforms"], key=lambda item: int(item["scene"]))]
    grid = _task14_canonical_grid(output_root, records, transforms)
    _write_json(stage04 / "canonical_output_grid.json", grid)
    source_config = _task14_source_config(output_root, records)
    _stage_marker(stage04, input_hashes=_hashes([stage03 / "global/translation_l2/global_transforms.json"]), output_paths=[stage04 / "canonical_output_grid.json", source_config], params={"resolution_m": 14.0, "crs": "EPSG:32650"}, started=time.perf_counter(), status="SUCCESS")
    stage05 = output_root / "stages/05_valid_distance_cache"; stage05.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    from src.multiscene_sift.radiometric_runner import _registered_inputs
    from src.multiscene_sift.mosaic_protocol import grid_transform
    registered, _, _, scene_ids, _, valid_masks = _registered_inputs(source_config, stage03 / "global/translation_l2", grid)
    cache_paths = []
    for index, mask in enumerate(valid_masks):
        dist = distance_transform_edt(mask).astype(np.float32); dist[mask] += 1e-6
        path = stage05 / f"distance_scene_{index:03d}.tif"
        profile = {"driver": "GTiff", "height": mask.shape[0], "width": mask.shape[1], "count": 1, "dtype": "float32", "crs": grid["crs"], "transform": grid_transform(grid), "nodata": np.nan, "compress": "deflate"}
        with rasterio.open(path, "w", **profile) as dst: dst.write(dist, 1)
        cache_paths.append(path)
    _write_json(stage05 / "distance_cache_summary.json", {"scene_count": len(valid_masks), "union_valid_pixels": int(np.any(np.stack(valid_masks), axis=0).sum()), "paths": [str(p) for p in cache_paths]})
    _stage_marker(stage05, input_hashes=_hashes([stage04 / "canonical_output_grid.json", stage03 / "global/translation_l2/global_transforms.json"]), output_paths=[stage05 / "distance_cache_summary.json", *cache_paths], params={"distance_transform": "full-grid EDT + 1e-6"}, started=started, status="SUCCESS")
    del registered, valid_masks
    stage06 = output_root / "stages/06_bagrn"; stage06.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    bagrn_out = stage06 / "bagrn"
    try:
        bagrn = _task14_stream_bagrn(output_root, source_config, stage03 / "global/translation_l2", grid)
    except MemoryError as exc:
        hard_stop = HARD_STOP_TASK14_RESOURCE_LIMIT
        _stage_marker(stage06, input_hashes=_hashes([stage04 / "task14_source_config.json", stage03 / "global/translation_l2/global_transforms.json", stage04 / "canonical_output_grid.json"]), output_paths=[], params={"method": "BAGRN", "reason": "full-canvas mosaic allocation exceeded available memory"}, started=started, status="HARD_STOP", error=f"{hard_stop}: {exc}")
        for later in _task14_downstream_stages():
            ldir = output_root / f"stages/{later}"; ldir.mkdir(parents=True, exist_ok=True)
            _stage_marker(ldir, input_hashes={}, output_paths=[], params={"blocked_by": [hard_stop]}, started=time.perf_counter(), status="SKIPPED_HARD_STOP", error=hard_stop)
        return {"decision": "NOT_READY_FOR_LARGE_SCALE", "hard_stops": [hard_stop], "accepted_edges": len(accepted), "global": global_result, "bagrn": None}
    _stage_marker(stage06, input_hashes=_hashes([stage04 / "task14_source_config.json", stage03 / "global/translation_l2/global_transforms.json", stage04 / "canonical_output_grid.json"]), output_paths=[bagrn_out / "radiometric_summary.json", bagrn_out / "mosaic.tif", bagrn_out / "normalized_scenes_manifest.json"], params={"method": "BAGRN", "geometry": "Translation-L2", "memory_mode": "local_scene_windows"}, started=started, status="SUCCESS")
    hard_stop = "HARD_STOP_TASK14_MULTISCENE_ADAPTER_UNAVAILABLE"
    for later in _task14_downstream_stages():
        ldir = output_root / f"stages/{later}"; ldir.mkdir(parents=True, exist_ok=True)
        _stage_marker(ldir, input_hashes={}, output_paths=[], params={"blocked_by": [hard_stop], "reason": "No frozen 13-scene Task13A/13B adapter is present"}, started=time.perf_counter(), status="SKIPPED_HARD_STOP", error=hard_stop)
    return {"decision": "NOT_READY_FOR_LARGE_SCALE", "hard_stops": [hard_stop], "accepted_edges": len(accepted), "global": global_result, "bagrn": bagrn}


def _stream_blend(scene_paths: list[Path], weight_paths: list[Path], output: Path) -> None:
    """Blend windows with no full-grid value stack in memory."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(scene_paths[0]) as first:
        profile = first.profile.copy()
        h, w = first.height, first.width
        profile.update(dtype="float32", count=1, nodata=np.nan, tiled=True, blockxsize=256, blockysize=256, compress="deflate", predictor=3)
    sources = [rasterio.open(p) for p in scene_paths]
    weights = [rasterio.open(p) for p in weight_paths]
    try:
        with rasterio.open(output, "w", **profile) as dst:
            for r0 in range(0, h, TASK14_TILE_SIZE):
                for c0 in range(0, w, TASK14_TILE_SIZE):
                    hh = min(TASK14_TILE_SIZE, h - r0); ww = min(TASK14_TILE_SIZE, w - c0)
                    win = Window(c0, r0, ww, hh)
                    num = np.zeros((hh, ww), dtype=np.float64); den = np.zeros((hh, ww), dtype=np.float64)
                    for src, wt in zip(sources, weights):
                        a = src.read(1, window=win).astype(np.float64)
                        q = wt.read(1, window=win).astype(np.float64)
                        ok = np.isfinite(a) & np.isfinite(q) & (q > 0)
                        num[ok] += a[ok] * q[ok]; den[ok] += q[ok]
                    out = np.full((hh, ww), np.nan, dtype=np.float32)
                    ok = den > 0
                    out[ok] = (num[ok] / den[ok]).astype(np.float32)
                    dst.write(out, 1, window=win)
    finally:
        for src in sources + weights:
            src.close()


def _write_distance_weights(scene_paths: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, scene_path in enumerate(scene_paths):
        with rasterio.open(scene_path) as src:
            mask = src.read_masks(1) > 0
            profile = src.profile.copy()
            profile.update(dtype="float32", count=1, nodata=np.nan, tiled=True, blockxsize=256, blockysize=256, compress="deflate", predictor=3)
        dist = distance_transform_edt(mask).astype(np.float32)
        dist[mask] += 1e-6
        path = output_dir / f"weight_scene_{index:03d}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(dist, 1)
        outputs.append(path)
    return outputs


def _raster_difference(reference: Path, candidate: Path) -> dict:
    with rasterio.open(reference) as ref, rasterio.open(candidate) as got:
        if (ref.width, ref.height, ref.crs, tuple(ref.transform)[:6]) != (got.width, got.height, got.crs, tuple(got.transform)[:6]):
            raise RuntimeError("HARD_STOP_STREAMING_SEMANTICS_CHANGED")
        a = ref.read(1); b = got.read(1)
    va = np.isfinite(a); vb = np.isfinite(b)
    if not np.array_equal(va, vb):
        raise RuntimeError("HARD_STOP_STREAMING_SEMANTICS_CHANGED")
    d = np.abs(a[va].astype(np.float64) - b[va].astype(np.float64))
    max_abs = float(d.max(initial=0.0))
    max_reference = float(np.max(np.abs(a[va].astype(np.float64)))) if np.any(va) else 0.0
    if not np.allclose(a[va], b[va], rtol=1e-6, atol=1e-3):
        raise RuntimeError("HARD_STOP_STREAMING_SEMANTICS_CHANGED")
    return {"max_abs_diff": max_abs, "mean_abs_diff": float(d.mean()) if d.size else 0.0, "finite_support": int(va.sum()), "allclose_rtol": 1e-6, "allclose_atol": 1e-3}


def _streaming_equivalence(root: Path) -> tuple[str, str, list[Path], list[Path]]:
    """Run the five-scene 1024/128 tile-stream equivalence gate."""
    ref_root = ROOT / "data/output/b9_five_scene_validation"
    bag_root = ref_root / "final_replay_v2_hardened/efficient_loftr_translation_l2_bagrn"
    cont_root = ref_root / "multiscene_task13b1_tie_resolution/continuation"
    scene_paths = [bag_root / "normalized_scenes" / f"scene_{i:03d}.tif" for i in range(5)]
    refs = {
        "v0": bag_root / "weighted_feather/mosaic.tif",
        "v1": cont_root / "v1_multiscene_label_blend.tif",
        "v2": cont_root / "v2_multiscene_local_blend.tif",
    }
    payload = {"tile_size": TASK14_TILE_SIZE, "halo": TASK14_HALO, "references": {k: str(v) for k, v in refs.items()}}
    source_inputs = [*scene_paths, *refs.values()]
    if not all(p.is_file() for p in source_inputs):
        payload.update({"status": HARD_STOP_STREAMING_REFERENCE_MISSING, "reason": "frozen five-scene source/reference raster is incomplete"})
        _write_json(root / "stages/-1_streaming_equivalence/streaming_equivalence.json", payload)
        return "HARD_STOP", HARD_STOP_STREAMING_REFERENCE_MISSING, [root / "stages/-1_streaming_equivalence/streaming_equivalence.json"], [p for p in source_inputs if p.is_file()]
    stage_dir = root / "stages/-1_streaming_equivalence"
    weight_dirs = stage_dir / "weights_v0"
    v0_weights = _write_distance_weights(scene_paths, weight_dirs)
    v0_out = stage_dir / "v0_tile_stream.tif"
    _stream_blend(scene_paths, v0_weights, v0_out)
    weight_paths = [cont_root / "weights_final" / f"weight_scene_{i:03d}.tif" for i in range(5)]
    corrected = [cont_root / "_temporary_corrected" / f"corrected_{i:03d}.tif" for i in range(5)]
    stream_inputs = [*source_inputs, *weight_paths, *corrected]
    if not all(p.is_file() for p in [*weight_paths, *corrected]):
        payload.update({"status": HARD_STOP_STREAMING_REFERENCE_MISSING, "reason": "frozen V1/V2 weight/correction raster is incomplete"})
        _write_json(root / "stages/-1_streaming_equivalence/streaming_equivalence.json", payload)
        return "HARD_STOP", HARD_STOP_STREAMING_REFERENCE_MISSING, [root / "stages/-1_streaming_equivalence/streaming_equivalence.json", *v0_weights, v0_out], [p for p in stream_inputs if p.is_file()]
    v1_out = stage_dir / "v1_tile_stream.tif"; v2_out = stage_dir / "v2_tile_stream.tif"
    _stream_blend(scene_paths, weight_paths, v1_out); _stream_blend(corrected, weight_paths, v2_out)
    diffs = {"v0": _raster_difference(refs["v0"], v0_out), "v1": _raster_difference(refs["v1"], v1_out), "v2": _raster_difference(refs["v2"], v2_out)}
    payload.update({"status": "PASS", "diffs": diffs})
    _write_json(root / "stages/-1_streaming_equivalence/streaming_equivalence.json", payload)
    return "SUCCESS", "", [root / "stages/-1_streaming_equivalence/streaming_equivalence.json", *v0_weights, v0_out, v1_out, v2_out], stream_inputs


def _registration_metrics(rows: list[dict]) -> dict:
    accepted = [r for r in rows if r.get("accepted_for_global")]
    return {
        "overlap_edges_total": len(rows),
        "accepted_edges": len(accepted),
        "rejected_edges": len(rows) - len(accepted),
        "accepted_edge_fraction": len(accepted) / len(rows) if rows else 0.0,
        "pairwise_rmse_median": float(np.nanmedian([r["pairwise_rmse_px"] for r in accepted])) if accepted else None,
        "pairwise_rmse_p95_across_edges": float(np.nanpercentile([r["pairwise_rmse_px"] for r in accepted], 95)) if accepted else None,
        "pairwise_p95_median": float(np.nanmedian([r["pairwise_p95_px"] for r in accepted])) if accepted else None,
        "pairwise_p95_worst": float(np.nanmax([r["pairwise_p95_px"] for r in accepted])) if accepted else None,
        "coverage_median": float(np.nanmedian([r["coverage"] for r in accepted])) if accepted else None,
        "coverage_worst": float(np.nanmin([r["coverage"] for r in accepted])) if accepted else None,
        "registration_graph_components": None,
        "registration_graph_cycle_rank": None,
    }


def _final_report(root: Path, *, decision: str, hard_stops: list[str], graph: dict | None, registration: dict | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    graph = graph or {}
    registration = registration or {}
    report = [
        "# Task14 13-Scene Scalable Mosaic Validation",
        "",
        f"Final decision: **{decision}**.",
        "",
        "The attached Task14 plan is the binding implementation specification; the user request was to execute it strictly.",
        "No matcher fallback, parameter tuning, geometry relaxation, BAGRN/VOLRN substitution, or 1000-scene extrapolation was performed.",
        "",
        "## Workload / graph",
        f"- Scenes: {graph.get('scene_count', 'UNAVAILABLE')}; possible pairs: {graph.get('possible_pair_count', 'UNAVAILABLE')}; exact overlap edges: {graph.get('exact_overlap_edge_count', 'UNAVAILABLE')}.",
        f"- BBox candidates: {graph.get('spatial_index_bbox_candidate_count', 'UNAVAILABLE')}; false negatives: {graph.get('spatial_index_false_negative_count', 'UNAVAILABLE')}; graph components: {graph.get('connected_component_count', 'UNAVAILABLE')}; cycle rank: {graph.get('cycle_rank', 'UNAVAILABLE')}.",
        f"- Registration accepted edges: {registration.get('accepted_edges', 'UNAVAILABLE')} / {registration.get('overlap_edges_total', 'UNAVAILABLE')}.",
        "",
        "## Hard stops",
    ]
    report += [f"- `{stop}`" for stop in hard_stops] if hard_stops else ["- none"]
    stream_path = root / "stages/-1_streaming_equivalence/streaming_equivalence.json"
    stream = json.loads(stream_path.read_text(encoding="utf-8")) if stream_path.is_file() else {}
    stream_lines = [
        "",
        "## Five-scene streaming gate",
        f"- Status: `{stream.get('status', 'UNAVAILABLE')}`; tile {stream.get('tile_size', TASK14_TILE_SIZE)} px / halo {stream.get('halo', TASK14_HALO)} px.",
    ]
    if stream.get("diffs"):
        for name, diff in stream["diffs"].items():
            stream_lines.append(f"- {name}: max abs diff {diff.get('max_abs_diff', 'UNAVAILABLE')}; mean abs diff {diff.get('mean_abs_diff', 'UNAVAILABLE')}; finite support {diff.get('finite_support', 'UNAVAILABLE')}; allclose rtol={diff.get('allclose_rtol', 'UNAVAILABLE')}, atol={diff.get('allclose_atol', 'UNAVAILABLE')}.")
    else:
        stream_lines.append(f"- Reason: {stream.get('reason', 'UNAVAILABLE')}.")
    report += stream_lines + [
        "",
        "## Interpretation",
        "The 13-scene scientific quality and scale gates were not graded as PASS because the frozen execution prerequisites were not complete. This run stops at the first blocking stage and preserves all completed artifacts for resume.",
        "The next permitted action is to provide the official EfficientLoFTR checkout/checkpoint, then resume with hash validation; no algorithmic parameter change is authorized.",
    ]
    (root / "TASK14_13SCENE_SCALE_VALIDATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run_task14(*, stage: str = "all", resume: bool = False, input_root: Path = DEFAULT_INPUT_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT, device: str = "auto", expected_scene_count: int = 13, scene_selection: tuple[int, ...] | None = None, include_streaming_equivalence: bool = True) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    if resume:
        hard_payload = _validate_resume_markers(output_root)
        if hard_payload is not None:
            graph_path = output_root / "graph_summary.json"
            registration_path = output_root / "global_geometry_summary.json"
            graph = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.is_file() else None
            registration = json.loads(registration_path.read_text(encoding="utf-8")) if registration_path.is_file() else None
            result = {"decision": "NOT_READY_FOR_LARGE_SCALE", "hard_stops": [hard_payload.get("error", HARD_STOP_MATCHER_UNAVAILABLE)], "graph": graph, "registration": registration, "resumed": True}
            return result
    runtime_rows: list[dict] = []
    hard_stops: list[str] = []
    graph = None
    registration = None

    # Stage -1 is intentionally run before the 13-scene stages.
    if include_streaming_equivalence and stage in {"all", "-1", "-1_streaming_equivalence"}:
        sdir = output_root / "stages/-1_streaming_equivalence"; sdir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter(); status, error, stream_outputs, stream_inputs = _streaming_equivalence(output_root)
        _stage_marker(sdir, input_hashes=_hashes(stream_inputs), output_paths=stream_outputs, params={"tile_size": TASK14_TILE_SIZE, "halo": TASK14_HALO}, started=started, status="SUCCESS" if status == "SUCCESS" else status, error=error or None)
        runtime_rows.append({"stage": "-1_streaming_equivalence", "status": status, "wall_time_sec": time.perf_counter() - started, **_snapshot_resource(), "error": error})
        if status != "SUCCESS": hard_stops.append(error)
        if status != "SUCCESS":
            _write_stage_runtime(output_root, runtime_rows)
            _final_report(output_root, decision="NOT_READY_FOR_LARGE_SCALE", hard_stops=hard_stops, graph=None, registration=None)
            return {"decision": "NOT_READY_FOR_LARGE_SCALE", "hard_stops": hard_stops}
        if stage != "all":
            _write_stage_runtime(output_root, runtime_rows); _final_report(output_root, decision="NOT_READY_FOR_LARGE_SCALE", hard_stops=hard_stops, graph=None, registration=None); return {"decision": "NOT_READY_FOR_LARGE_SCALE", "hard_stops": hard_stops}

    records = discover_b9_scenes(input_root)
    if scene_selection is not None:
        try:
            records = [records[index] for index in scene_selection]
        except IndexError as exc:
            raise RuntimeError(f"HARD_STOP_INPUT_INVALID: scene_selection index {exc}") from exc
    preflight_errors: list[str] = []
    if len(records) != expected_scene_count:
        preflight_errors.append(f"scene_count={len(records)} (expected {expected_scene_count})")
    for record in records:
        if record.get("crs") != "EPSG:32650":
            preflight_errors.append(f"{record.get('scene_id')}: CRS {record.get('crs')} != EPSG:32650")
        if not np.isclose(float(record.get("resolution_x_m", 0.0)), 14.0) or not np.isclose(float(record.get("resolution_y_m", 0.0)), 14.0):
            preflight_errors.append(f"{record.get('scene_id')}: pixel size is not 14 m")
        if not Path(record["b9_path"]).is_file() or not Path(record["mtl_path"]).is_file():
            preflight_errors.append(f"{record.get('scene_id')}: B9/MTL source missing")
    manifest_rows = _record_manifest(records)
    manifest_paths = [Path(r["b9_path"]) for r in manifest_rows] + [Path(r["mtl_path"]) for r in manifest_rows]

    # Stage 00: immutable source/provenance inventory.
    sdir = output_root / "stages/00_preflight"; sdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    protocol = {"task": "Task14", "scene_count": len(records), "band": "B9", "crs": records[0].get("crs") if records else None, "pixel_size_m": 14.0, "matcher": "EfficientLoFTR", "match_max_side": 1024, "global": "Translation-L2", "tile_size": TASK14_TILE_SIZE, "halo": TASK14_HALO, "input_root": str(input_root), "git_head": _git_head(), "git_status": _git_status()}
    _write_json(output_root / "protocol.json", protocol)
    _write_csv(output_root / "scene_manifest.csv", manifest_rows)
    _write_json(sdir / "preflight_summary.json", {"protocol": protocol, "scene_count": len(records), "source_hashes": _hashes(manifest_paths)})
    outputs = [output_root / "protocol.json", output_root / "scene_manifest.csv", sdir / "preflight_summary.json"]
    preflight_status = "HARD_STOP" if preflight_errors else "SUCCESS"
    preflight_error = HARD_STOP_PREFLIGHT_INVALID if preflight_errors else None
    _write_json(sdir / "preflight_summary.json", {"protocol": protocol, "scene_count": len(records), "source_hashes": _hashes(manifest_paths), "errors": preflight_errors})
    _stage_marker(sdir, input_hashes=_hashes(manifest_paths), output_paths=outputs, params=protocol, started=started, status=preflight_status, error=preflight_error)
    runtime_rows.append({"stage": "00_preflight", "status": preflight_status, "wall_time_sec": time.perf_counter() - started, **_snapshot_resource(), "error": preflight_error or ""})
    if preflight_errors:
        hard_stops.append(HARD_STOP_PREFLIGHT_INVALID)
        _write_stage_runtime(output_root, runtime_rows)
        _final_report(output_root, decision="NOT_READY_FOR_LARGE_SCALE", hard_stops=hard_stops, graph=None, registration=None)
        return {"decision": "NOT_READY_FOR_LARGE_SCALE", "hard_stops": hard_stops}
    if stage == "00_preflight":
        _write_stage_runtime(output_root, runtime_rows); _final_report(output_root, decision="NOT_READY_FOR_LARGE_SCALE", hard_stops=hard_stops, graph=None, registration=None); return {"decision": "NOT_READY_FOR_LARGE_SCALE", "hard_stops": hard_stops}

    # Stage 01: bbox index + exhaustive exact pair audit.
    sdir = output_root / "stages/01_spatial_index_graph"; sdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); pairs = build_b9_overlap_graph(records); bbox, spatial_backend = _bbox_candidates(records, return_backend=True); graph = graph_statistics(records, pairs, bbox_candidate_pairs=bbox, spatial_index_backend=spatial_backend)
    overlap_fields = ["idx_i", "idx_j", "scene_i", "scene_j", "intersection_area", "overlap_area_i_ratio", "overlap_area_j_ratio", "symmetric_overlap_ratio", "has_overlap"]
    _write_csv(output_root / "overlap_edges.csv", pairs, overlap_fields); _write_json(output_root / "graph_summary.json", graph); _write_json(sdir / "graph_summary.json", graph)
    _stage_marker(sdir, input_hashes=_hashes([output_root / "scene_manifest.csv"]), output_paths=[output_root / "overlap_edges.csv", output_root / "graph_summary.json", sdir / "graph_summary.json"], params={"exhaustive_pairs": len(pairs), "bbox_candidates": len(bbox)}, started=started, status="SUCCESS")
    runtime_rows.append({"stage": "01_spatial_index_graph", "status": "SUCCESS", "wall_time_sec": time.perf_counter() - started, **_snapshot_resource(), "error": ""})
    if graph["spatial_index_false_negative_count"] != 0:
        hard_stops.append("HARD_STOP_SPATIAL_INDEX_FALSE_NEGATIVE")
    if stage == "01_spatial_index_graph":
        _write_stage_runtime(output_root, runtime_rows); _final_report(output_root, decision="NOT_READY_FOR_LARGE_SCALE", hard_stops=hard_stops, graph=graph, registration=None); return {"decision": "NOT_READY_FOR_LARGE_SCALE", "hard_stops": hard_stops, "graph": graph}
    if hard_stops:
        _write_stage_runtime(output_root, runtime_rows); _final_report(output_root, decision="NOT_READY_FOR_LARGE_SCALE", hard_stops=hard_stops, graph=graph, registration=None); return {"decision": "NOT_READY_FOR_LARGE_SCALE", "hard_stops": hard_stops, "graph": graph}

    # Stage 02: fixed EfficientLoFTR only.  No SIFT/LoFTR substitution.
    sdir = output_root / "stages/02_matching_ransac"; sdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); rows = []
    try:
        from src.registration_benchmark.matchers.efficient_loftr import is_efficient_loftr_available
        available = bool(is_efficient_loftr_available())
    except Exception:
        # Import-time dependency failures (for example missing cv2) mean the
        # fixed matcher is unavailable.  They are never treated as permission
        # to substitute another matcher.
        available = False
    edge_pairs = [p for p in pairs if p.get("has_overlap")]
    if not available:
        hard_stops.append(HARD_STOP_MATCHER_UNAVAILABLE)
        rows = [{"pair_id": f"{int(p['idx_i']):02d}_{int(p['idx_j']):02d}", "scene_i": p["scene_i"], "scene_j": p["scene_j"], "raw_match_count": 0, "valid_match_count": 0, "ransac_inlier_count": 0, "inlier_ratio": 0.0, "pairwise_rmse_px": None, "pairwise_p95_px": None, "pairwise_max_px": None, "coverage": 0.0, "matcher_runtime_sec": 0.0, "ransac_runtime_sec": 0.0, "status": HARD_STOP_MATCHER_UNAVAILABLE, "accepted_for_global": False} for p in edge_pairs]
    else:
        from src.multiscene_sift.pairwise import run_all_pairs
        results = run_all_pairs(_scene_models(records), _edge_models(edge_pairs), output_root / "registration", band="B9", match_max_side=1024, ransac_threshold=2.0, random_seed=0, matcher="efficient_loftr", device=device)
        rows = _task14_registration_rows(results, records)
        _persist_task14_geometry(results, records, output_root)
    _write_csv(output_root / "registration_pair_metrics.csv", rows)
    registration = _registration_metrics(rows); _write_json(output_root / "global_geometry_summary.json", registration); _write_json(sdir / "matcher_availability.json", {"efficient_loftr_available": available, "matcher": "EfficientLoFTR", "match_max_side": 1024})
    _stage_marker(sdir, input_hashes=_hashes([output_root / "overlap_edges.csv"]), output_paths=[output_root / "registration_pair_metrics.csv", output_root / "global_geometry_summary.json", sdir / "matcher_availability.json"], params={"matcher": "EfficientLoFTR", "match_max_side": 1024, "device": device}, started=started, status="HARD_STOP" if not available else "SUCCESS", error=HARD_STOP_MATCHER_UNAVAILABLE if not available else None)
    runtime_rows.append({"stage": "02_matching_ransac", "status": "HARD_STOP" if not available else "SUCCESS", "wall_time_sec": time.perf_counter() - started, **_snapshot_resource(), "error": HARD_STOP_MATCHER_UNAVAILABLE if not available else ""})

    # The repository has no wired Task14 adapters for stages 03-12 yet.  If
    # matching ever becomes available, stop explicitly and preserve a valid
    # package instead of raising an unstructured exception.
    if available:
        hard_stops.append(HARD_STOP_TASK14_STAGE_NOT_IMPLEMENTED)

    # Preserve explicit stage checkpoint rows after a hard stop; never pretend skipped work succeeded.
    if hard_stops:
        for later in STAGES[4:]:
            ldir = output_root / f"stages/{later}"; ldir.mkdir(parents=True, exist_ok=True)
            _stage_marker(ldir, input_hashes={}, output_paths=[], params={"blocked_by": hard_stops}, started=time.perf_counter(), status="SKIPPED_HARD_STOP", error=";".join(hard_stops))
            runtime_rows.append({"stage": later, "status": "SKIPPED_HARD_STOP", "wall_time_sec": 0.0, **_snapshot_resource(), "error": ";".join(hard_stops)})
        _write_stage_runtime(output_root, runtime_rows)
        not_run = {"status": "NOT_RUN_HARD_STOP", "hard_stops": hard_stops}
        for name in ("radiometric_global_metrics.json", "labeling_summary.json", "final_metrics.json"):
            _write_json(output_root / f"metrics/{name}", not_run)
        for name in ("radiometric_edge_metrics.csv", "boundary_metrics.csv", "label_method_counts.csv", "tile_runtime.csv"):
            _write_csv(output_root / f"metrics/{name}", [], ["status", "reason"])
        _write_csv(output_root / "scalability/tile_runtime.csv", [], ["status", "reason"])
        _write_json(output_root / "scalability/scalability_summary.json", {
            "decision": "NOT_READY_FOR_LARGE_SCALE", "N_scenes": len(records), "E_overlap_edges": len(edge_pairs),
            "total_wall_time_sec": float(sum(float(r.get("wall_time_sec", 0.0)) for r in runtime_rows)),
            "accepted_registration_E": registration.get("accepted_edges", 0), "graph_density": graph["graph_density"],
            "canonical_megapixels": unavailable_value(), "union_valid_megapixels": unavailable_value(),
            "pixel_scene_exposure": unavailable_value(), "matching_sec_per_edge": unavailable_value(),
            "seam_sec_per_edge": unavailable_value(), "labeling_sec_per_megapixel": unavailable_value(),
            "mosaic_sec_per_megapixel": unavailable_value(), "peak_cpu_rss_mb": max((r.get("peak_cpu_rss_mb") for r in runtime_rows if isinstance(r.get("peak_cpu_rss_mb"), (int, float))), default=unavailable_value()),
            "peak_gpu_allocated_mb": unavailable_value(), "peak_gpu_reserved_mb": unavailable_value(), "disk_gb": unavailable_value(), "hard_stops": hard_stops,
        })
        total_bytes = sum(p.stat().st_size for p in output_root.rglob("*") if p.is_file())
        _write_json(output_root / "scalability/disk_usage.json", {
            "registered_scene_cache_bytes": unavailable_value(), "valid_distance_cache_bytes": unavailable_value(),
            "bagrn_scene_cache_bytes": unavailable_value(), "pair_artifact_bytes": unavailable_value(),
            "label_artifact_bytes": unavailable_value(), "mosaic_bytes": unavailable_value(),
            "task14_total_output_bytes": total_bytes,
        })
        _final_report(output_root, decision="NOT_READY_FOR_LARGE_SCALE", hard_stops=hard_stops, graph=graph, registration=registration)
        return {"decision": "NOT_READY_FOR_LARGE_SCALE", "hard_stops": hard_stops, "graph": graph, "registration": registration}

    # All permitted Task14 execution paths terminate above until the frozen
    # stage 03-12 adapters are explicitly wired.
    return {"decision": "NOT_READY_FOR_LARGE_SCALE", "hard_stops": hard_stops, "graph": graph, "registration": registration}


def _git_head() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return unavailable_value()


def _git_status() -> list[str]:
    import subprocess
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
    except Exception:
        return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=["all", *STAGES, "-1"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    try:
        result = run_task14(stage=args.stage, resume=args.resume, input_root=args.input_root, output_root=args.output_root, device=args.device)
    except RuntimeError as exc:
        print(json.dumps({"status": "HARD_STOP", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
