"""Five-image B14 pipeline: register to a reference, then BAGRN/VOLRN/mosaic.

The registration and normalization formulas remain in the existing ``src``
modules.  This script only orchestrates the five input scenes and writes
per-scene, per-method, and mosaic products.
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bagrn import bagrn_normalize
from src.coregistration import (
    collect_block_matches,
    multi_image_network_adjustment,
    phase_correlation,
    warp_with_displacement_field,
)
from src.io_utils import read_geotiff, write_geotiff
from src.metrics import compute_all
from src.mosaic import create_mosaic
from src.overlap import detect_multi_overlap
from src.volrn import volrn_normalize


DEFAULT_INPUT_DIR = Path("data/input/flat")
DEFAULT_OUTPUT_DIR = Path("data/output/five_image_float")
DEFAULT_BAND = "B14"
DEFAULT_REFERENCE_SCENE = (
    "DZ01V_L2_E113.4_N36.6_20260810030932_01_T1"
)
EXPECTED_SCENE_COUNT = 5


@dataclass
class SceneData:
    """One single-band scene in its current pixel/geographic coordinate system."""

    name: str
    path: str
    array: np.ndarray
    transform: Any
    crs: str
    nodata: Optional[float]


def _json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _as_2d(array: np.ndarray) -> np.ndarray:
    """Use the first band from a single-band B14 GeoTIFF."""
    array = np.asarray(array)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[0] >= 1:
        return array[0]
    raise ValueError(f"Expected a 2D or 3D GeoTIFF array, got {array.shape}")


def _band_file(scene_dir: Path, band: str) -> Optional[Path]:
    """Find the scene's band file, accepting TIFF suffix case differences."""
    expected_stem = f"{scene_dir.name}_{band}".lower()
    matches = sorted(
        path for path in scene_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
        and path.stem.lower() == expected_stem
    )
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple {band} TIFF files found in {scene_dir}")
    return matches[0]


def discover_scene_paths(
    input_dir: Path,
    band: str = DEFAULT_BAND,
    reference_scene: str = DEFAULT_REFERENCE_SCENE,
    expected_count: int = EXPECTED_SCENE_COUNT,
) -> List[Path]:
    """Return exactly ``expected_count`` scene files with the reference first."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    scene_files = {}
    for scene_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        band_path = _band_file(scene_dir, band)
        if band_path is not None:
            scene_files[scene_dir.name] = band_path

    if len(scene_files) != expected_count:
        raise ValueError(
            f"expected {expected_count} scenes with {band}, found {len(scene_files)}"
        )
    if reference_scene not in scene_files:
        raise ValueError(
            f"reference scene {reference_scene!r} was not found in {input_dir}"
        )

    others = sorted(name for name in scene_files if name != reference_scene)
    return [scene_files[reference_scene]] + [scene_files[name] for name in others]


def load_scene(path: Path, band: str = DEFAULT_BAND) -> SceneData:
    """Read one B14 scene and normalize its array to two dimensions."""
    array, transform, crs, nodata = read_geotiff(str(path))
    return SceneData(
        name=Path(path).parent.name,
        path=str(path),
        array=_as_2d(array),
        transform=transform,
        crs=crs,
        nodata=nodata,
    )


def _scene_valid_mask(scene: SceneData) -> np.ndarray:
    """Build the validity mask used by the existing phase fallback."""
    valid = np.isfinite(scene.array)
    if scene.nodata is not None:
        valid &= scene.array != scene.nodata
    return valid


def _measure_registration_edge(
    scene_i: SceneData,
    scene_j: SceneData,
    overlap: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Measure one geometric overlap edge, or return an explicit rejection."""
    i = int(overlap["idx_i"])
    j = int(overlap["idx_j"])
    matches, screening = collect_block_matches(
        scene_i.array, scene_i.transform,
        scene_j.array, scene_j.transform,
        scene_i.nodata, scene_j.nodata,
        block_size=512,
        max_global_shift=40,
        confidence_threshold=0.5,
    )
    screening = dict(screening or {})
    if matches:
        confidences = np.asarray(
            [float(match.get("confidence", 0.0)) for match in matches],
            dtype=float,
        )
        dx_values = np.asarray([float(match["shift_dx"]) for match in matches])
        dy_values = np.asarray([float(match["shift_dy"]) for match in matches])
        weights = np.maximum(confidences, 1e-12)
        shift_dx = float(np.average(dx_values, weights=weights))
        shift_dy = float(np.average(dy_values, weights=weights))
        residuals = np.hypot(dx_values - shift_dx, dy_values - shift_dy)
        return {
            "idx_i": i,
            "idx_j": j,
            "shift_dx": shift_dx,
            "shift_dy": shift_dy,
            "confidence": float(np.mean(confidences)),
            "n_blocks": len(matches),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "p95": float(np.percentile(residuals, 95)),
            "matches": _json_safe(matches),
            "screening": _json_safe(screening),
            "method": "block_match",
        }, None

    try:
        shift_y, shift_x, phase_confidence = phase_correlation(
            scene_i.array,
            scene_j.array,
            valid_ref=_scene_valid_mask(scene_i),
            valid_tgt=_scene_valid_mask(scene_j),
        )
    except Exception as exc:
        shift_y, shift_x, phase_confidence = 0.0, 0.0, 0.0
        phase_error = str(exc)
    else:
        phase_error = None

    if phase_confidence > 0.3 and abs(shift_y) < 40 and abs(shift_x) < 40:
        return {
            "idx_i": i,
            "idx_j": j,
            "shift_dx": float(shift_x),
            "shift_dy": float(shift_y),
            "confidence": float(phase_confidence),
            "n_blocks": 1,
            "rmse": 0.0,
            "p95": 0.0,
            "matches": [],
            "screening": _json_safe(screening),
            "method": "phase_correlation",
        }, None

    reasons = ["block_match无匹配且phase_correlation失败"]
    if phase_error:
        reasons.append(f"phase_error={phase_error}")
    for key in ("low_valid", "low_texture", "low_conf", "large_shift"):
        if screening.get(key, 0):
            reasons.append(f"{key}={screening[key]}")
    return None, {
        "idx_i": i,
        "idx_j": j,
        "reason": "; ".join(reasons),
        "screening": _json_safe(screening),
    }


def match_all_overlap_edges(
    scenes: Sequence[SceneData],
    overlaps: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Match every geometric overlap and keep rejected edges out of the graph."""
    pair_measurements: List[Dict[str, Any]] = []
    rejected_edges: List[Dict[str, Any]] = []
    for overlap in overlaps:
        i = int(overlap["idx_i"])
        j = int(overlap["idx_j"])
        measurement, rejection = _measure_registration_edge(
            scenes[i], scenes[j], overlap
        )
        if measurement is not None:
            pair_measurements.append(measurement)
        elif rejection is not None:
            rejection["scene_i"] = scenes[i].name
            rejection["scene_j"] = scenes[j].name
            rejected_edges.append(rejection)
    return pair_measurements, rejected_edges


def build_registration_graph(
    pair_measurements: Sequence[Dict[str, Any]],
    n_images: int,
    reference_idx: int = 0,
) -> Dict[str, Any]:
    """Build a deterministic reference-rooted graph from reliable edges."""
    if not 0 <= reference_idx < n_images:
        raise ValueError(f"reference_idx {reference_idx} is outside 0..{n_images - 1}")

    adjacency = {index: set() for index in range(n_images)}
    for pair in pair_measurements:
        i = int(pair["idx_i"])
        j = int(pair["idx_j"])
        if i == j or not (0 <= i < n_images and 0 <= j < n_images):
            raise ValueError(f"invalid registration edge: {i}-{j}")
        adjacency[i].add(j)
        adjacency[j].add(i)

    from collections import deque

    parent: Dict[int, Optional[int]] = {reference_idx: None}
    queue = deque([reference_idx])
    spanning_tree_edges: List[Tuple[int, int]] = []
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency[node]):
            if neighbor in parent:
                continue
            parent[neighbor] = node
            spanning_tree_edges.append((node, neighbor))
            queue.append(neighbor)

    reachable = sorted(parent)
    unreachable = [index for index in range(n_images) if index not in parent]
    return {
        "adjacency": {
            index: sorted(neighbors) for index, neighbors in adjacency.items()
        },
        "parent_map": parent,
        "parent": parent,
        "reachable": reachable,
        "unreachable": unreachable,
        "spanning_tree_edges": spanning_tree_edges,
    }


def solve_registration_network(
    pair_measurements: Sequence[Dict[str, Any]],
    n_images: int,
    reference_idx: int,
) -> Dict[str, Any]:
    """Solve all reliable pair equations with the configured reference fixed."""
    result = multi_image_network_adjustment(
        list(pair_measurements), n_images, reference_idx=reference_idx
    )
    result["global_shifts"] = np.asarray(result["global_shifts"], dtype=float)
    result["global_shifts"][reference_idx] = [0.0, 0.0]
    return result


def build_reference_paths(
    parent: Dict[int, Optional[int]],
    reference_idx: int,
    n_images: int,
) -> Dict[int, Optional[List[int]]]:
    """Expand BFS parents into reference-rooted paths with cycle protection."""
    paths: Dict[int, Optional[List[int]]] = {}
    for index in range(n_images):
        if index not in parent:
            paths[index] = None
            continue
        path: List[int] = []
        current: Optional[int] = index
        seen = set()
        while current is not None:
            if current in seen:
                raise RuntimeError(
                    f"registration graph parent cycle at scene {current}"
                )
            seen.add(current)
            path.append(current)
            if current == reference_idx:
                break
            current = parent.get(current)
        if not path or path[-1] != reference_idx:
            paths[index] = None
        else:
            paths[index] = list(reversed(path))
    return paths


def apply_network_shifts_from_original(
    scenes: Sequence[SceneData],
    global_shifts: np.ndarray,
    reference_idx: int,
) -> Tuple[List[SceneData], Dict[int, str]]:
    """Warp each original scene once using its network-adjusted displacement."""
    shifts = np.asarray(global_shifts, dtype=float)
    if shifts.shape != (len(scenes), 2):
        raise ValueError(
            f"global_shifts must have shape {(len(scenes), 2)}, got {shifts.shape}"
        )

    registered: List[SceneData] = []
    statuses: Dict[int, str] = {}
    for index, scene in enumerate(scenes):
        global_dx = float(shifts[index, 0])
        global_dy = float(shifts[index, 1])
        if index == reference_idx:
            array = scene.array.astype(np.float64, copy=True)
            statuses[index] = "reference_anchor"
        elif abs(global_dx) < 1e-6 and abs(global_dy) < 1e-6:
            array = scene.array.astype(np.float64, copy=True)
            statuses[index] = "network_solution_zero"
        else:
            local_dx = np.zeros_like(scene.array, dtype=np.float64)
            local_dy = np.zeros_like(scene.array, dtype=np.float64)
            array = warp_with_displacement_field(
                scene.array,
                global_dx,
                global_dy,
                local_dx,
                local_dy,
                scene.nodata,
            )
            statuses[index] = "network_adjusted"
        registered.append(
            SceneData(
                name=scene.name,
                path=scene.path,
                array=array,
                transform=scene.transform,
                crs=scene.crs,
                nodata=scene.nodata,
            )
        )
    return registered, statuses


def summarize_registration_edges(
    geometric_edges: Sequence[Tuple[int, int]],
    pair_measurements: Sequence[Dict[str, Any]],
    rejected_edges: Sequence[Dict[str, Any]],
    radiometric_overlap_count: Optional[int] = None,
) -> Dict[str, int]:
    """Return separate counts for geometry, registration, rejection, and radiometry."""
    return {
        "geometric_overlap_pairs": len(geometric_edges),
        "reliable_registration_edges": len(pair_measurements),
        "rejected_registration_edges": len(rejected_edges),
        "radiometric_overlap_pairs": (
            len(geometric_edges)
            if radiometric_overlap_count is None
            else int(radiometric_overlap_count)
        ),
    }


def run_registration_stage(
    scenes: Sequence[SceneData],
    overlaps: Sequence[Dict[str, Any]],
    reference_idx: int,
) -> Dict[str, Any]:
    """Run formal content registration before normalization or mosaicking."""
    if not overlaps and len(scenes) > 1:
        raise RuntimeError("No geometric overlap pairs were found for registration")

    pair_measurements, rejected_edges = match_all_overlap_edges(scenes, overlaps)
    graph = build_registration_graph(pair_measurements, len(scenes), reference_idx)
    if graph["unreachable"]:
        unreachable_names = [scenes[index].name for index in graph["unreachable"]]
        raise RuntimeError(
            "Reliable registration graph is disconnected from reference: "
            + ", ".join(unreachable_names)
        )

    network_result = solve_registration_network(
        pair_measurements, len(scenes), reference_idx
    )
    global_shifts = network_result["global_shifts"]
    if not np.allclose(global_shifts[reference_idx], [0.0, 0.0], atol=1e-8):
        raise RuntimeError("Network adjustment violated reference anchor")
    reference_paths = build_reference_paths(
        graph["parent_map"], reference_idx, len(scenes)
    )
    registered, registration_statuses = apply_network_shifts_from_original(
        scenes, global_shifts, reference_idx
    )

    geometric_edges = [
        (int(overlap["idx_i"]), int(overlap["idx_j"])) for overlap in overlaps
    ]
    pair_by_edge = {
        (int(pair["idx_i"]), int(pair["idx_j"])): pair
        for pair in pair_measurements
    }
    rejected_by_edge = {
        (int(item["idx_i"]), int(item["idx_j"])): item
        for item in rejected_edges
    }
    print(
        f"Reference anchor: [{reference_idx}] {scenes[reference_idx].name}"
    )
    print(f"Geometric overlap pairs: {len(geometric_edges)}")
    print("--- Pairwise registration edges ---")
    for i, j in geometric_edges:
        pair = pair_by_edge.get((i, j))
        rejection = rejected_by_edge.get((i, j))
        if pair is not None:
            print(
                f"[{i}]-[{j}] ACCEPTED method={pair.get('method', 'unknown')} "
                f"dx={pair['shift_dx']:.4f} dy={pair['shift_dy']:.4f} "
                f"confidence={pair['confidence']:.3f} blocks={pair['n_blocks']}"
            )
        else:
            reason = rejection.get("reason", "unavailable") if rejection else "unavailable"
            print(f"[{i}]-[{j}] REJECTED {reason}")
    print(f"Reliable registration edges: {len(pair_measurements)}")
    print(f"Rejected registration edges: {len(rejected_edges)}")
    print(f"Reference-connected scenes: {len(graph['reachable'])}/{len(scenes)}")
    print("Registration paths:")
    for index in range(len(scenes)):
        path = reference_paths[index]
        path_text = "->".join(map(str, path)) if path else "unreachable"
        print(f"  [{index}] {scenes[index].name} path={path_text}")
    print("Network-adjusted global residual shifts:")
    for index, scene in enumerate(scenes):
        dx, dy = global_shifts[index]
        print(
            f"  [{index}] dx={dx:.4f} dy={dy:.4f} "
            f"{registration_statuses[index]}"
        )

    return {
        "registered": registered,
        "pair_measurements": pair_measurements,
        "rejected_edges": rejected_edges,
        "graph": graph,
        "network_result": network_result,
        "reference_paths": reference_paths,
        "registration_statuses": registration_statuses,
        "geometric_edges": geometric_edges,
    }


def write_registration_network_artifacts(
    output_dir: Path,
    scenes: Sequence[SceneData],
    reference_idx: int,
    overlaps: Sequence[Dict[str, Any]],
    pair_measurements: Sequence[Dict[str, Any]],
    rejected_edges: Sequence[Dict[str, Any]],
    graph: Dict[str, Any],
    network_result: Dict[str, Any],
    reference_paths: Dict[int, Optional[List[int]]],
    registration_statuses: Dict[int, str],
) -> Dict[str, Path]:
    """Write the auditable registration-network JSON and CSV artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometric_edges = [
        (int(overlap["idx_i"]), int(overlap["idx_j"])) for overlap in overlaps
    ]
    pair_by_edge = {
        (int(pair["idx_i"]), int(pair["idx_j"])): pair
        for pair in pair_measurements
    }
    rejected_by_edge = {
        (int(item["idx_i"]), int(item["idx_j"])): item
        for item in rejected_edges
    }
    counts = summarize_registration_edges(
        geometric_edges,
        pair_measurements,
        rejected_edges,
        radiometric_overlap_count=len(overlaps),
    )
    shifts = np.asarray(network_result["global_shifts"], dtype=float)

    global_shift_records = []
    global_shift_payload = []
    for index, scene in enumerate(scenes):
        path = reference_paths.get(index)
        global_dx = float(shifts[index, 0])
        global_dy = float(shifts[index, 1])
        global_shift_records.append({
            "scene_index": index,
            "scene_id": scene.name,
            "is_reference": index == reference_idx,
            "reference_path": path,
            "global_dx_pixels": global_dx,
            "global_dy_pixels": global_dy,
            "global_magnitude_pixels": float(np.hypot(global_dx, global_dy)),
            "registration_status": registration_statuses.get(index, "unknown"),
        })
        global_shift_payload.append({
            "scene_index": index,
            "scene_id": scene.name,
            "reference_path": path,
            "dx": global_dx,
            "dy": global_dy,
            "magnitude_pixels": float(np.hypot(global_dx, global_dy)),
            "status": registration_statuses.get(index, "unknown"),
        })

    network_payload = {
        "reference_scene_index": int(reference_idx),
        "reference_scene_id": scenes[reference_idx].name,
        "reference_role": "anchor_only",
        "geometric_edges": [list(edge) for edge in geometric_edges],
        "geometric_overlap_count": counts["geometric_overlap_pairs"],
        "reliable_edges": [
            [int(pair["idx_i"]), int(pair["idx_j"])]
            for pair in pair_measurements
        ],
        "rejected_edges": [
            [int(item["idx_i"]), int(item["idx_j"])]
            for item in rejected_edges
        ],
        "reliable_registration_edge_count": counts["reliable_registration_edges"],
        "rejected_registration_edge_count": counts["rejected_registration_edges"],
        "radiometric_overlap_pairs": counts["radiometric_overlap_pairs"],
        "connected": not bool(graph["unreachable"]),
        "unreachable": list(graph["unreachable"]),
        "unreachable_scene_indices": list(graph["unreachable"]),
        "spanning_tree_edges": [list(edge) for edge in graph["spanning_tree_edges"]],
        "reference_paths": {
            str(index): path for index, path in reference_paths.items()
        },
        "global_shifts": global_shift_payload,
        "network_adjustment": {
            "n_edges": int(network_result.get("n_edges", len(pair_measurements))),
            "is_tree": bool(network_result.get("is_tree", False)),
            "loop_errors": _json_safe(network_result.get("loop_errors", [])),
            "pair_results": _json_safe(network_result.get("pair_results", [])),
        },
        "rejected_edge_records": _json_safe(list(rejected_edges)),
    }
    json_path = output_dir / "registration_network.json"
    json_path.write_text(
        json.dumps(_json_safe(network_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pair_fields = [
        "idx_i", "scene_i", "idx_j", "scene_j", "geometric_overlap",
        "registration_available", "method", "shift_dx", "shift_dy",
        "confidence", "n_blocks", "rmse", "p95", "reject_reason",
    ]
    pairs_csv = output_dir / "registration_pair_edges.csv"
    with pairs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        for i, j in geometric_edges:
            pair = pair_by_edge.get((i, j))
            rejection = rejected_by_edge.get((i, j))
            writer.writerow({
                "idx_i": i,
                "scene_i": scenes[i].name,
                "idx_j": j,
                "scene_j": scenes[j].name,
                "geometric_overlap": True,
                "registration_available": pair is not None,
                "method": pair.get("method") if pair else "rejected",
                "shift_dx": pair.get("shift_dx") if pair else None,
                "shift_dy": pair.get("shift_dy") if pair else None,
                "confidence": pair.get("confidence") if pair else None,
                "n_blocks": pair.get("n_blocks") if pair else None,
                "rmse": pair.get("rmse") if pair else None,
                "p95": pair.get("p95") if pair else None,
                "reject_reason": rejection.get("reason") if rejection else None,
            })

    scene_fields = [
        "scene_index", "scene_id", "is_reference", "reference_path",
        "global_dx_pixels", "global_dy_pixels", "global_magnitude_pixels",
        "registration_status",
    ]
    scenes_csv = output_dir / "registration_scene_shifts.csv"
    with scenes_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scene_fields)
        writer.writeheader()
        for record in global_shift_records:
            path = record["reference_path"]
            writer.writerow({
                **record,
                "reference_path": " -> ".join(map(str, path)) if path else "",
            })

    return {"json": json_path, "pairs_csv": pairs_csv, "scenes_csv": scenes_csv}


def _write_registration_matches(path: Path, matches: Iterable[Dict[str, Any]]) -> None:
    """Write accepted block matches for one target scene."""
    fields = [
        "match_index", "ref_x", "ref_y", "tgt_x", "tgt_y",
        "shift_dx_pixels", "shift_dy_pixels", "confidence",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, match in enumerate(matches):
            writer.writerow({
                "match_index": index,
                "ref_x": match.get("ref_x"),
                "ref_y": match.get("ref_y"),
                "tgt_x": match.get("tgt_x"),
                "tgt_y": match.get("tgt_y"),
                "shift_dx_pixels": match.get("shift_dx"),
                "shift_dy_pixels": match.get("shift_dy"),
                "confidence": match.get("confidence"),
            })


def _compute_bounds(transform: Any, shape: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """Return rasterio-style (left, bottom, right, top) bounds."""
    height, width = shape
    return (
        transform.c,
        transform.f + transform.e * height,
        transform.c + transform.a * width,
        transform.f,
    )


def run_pipeline(
    scene_paths: Sequence[Path],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    band: str = DEFAULT_BAND,
    reference_scene: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the reference-anchored network registration and mosaic workflow."""
    if len(scene_paths) != EXPECTED_SCENE_COUNT:
        raise ValueError(f"expected {EXPECTED_SCENE_COUNT} scene paths, got {len(scene_paths)}")

    output_band_dir = Path(output_dir) / band
    output_band_dir.mkdir(parents=True, exist_ok=True)
    loaded = [load_scene(Path(path), band=band) for path in scene_paths]
    if reference_scene is None:
        reference_idx = 0
    else:
        matching_indices = [
            index for index, scene in enumerate(loaded)
            if scene.name == reference_scene
        ]
        if not matching_indices:
            raise ValueError(
                f"reference scene {reference_scene!r} was not found in loaded scenes"
            )
        reference_idx = matching_indices[0]
    reference = loaded[reference_idx]
    registration_dir = output_band_dir / "registration"
    registered_dir = output_band_dir / "registered"
    bagrn_dir = output_band_dir / "bagrn"
    volrn_dir = output_band_dir / "volrn"
    registration_dir.mkdir(parents=True, exist_ok=True)
    registered_dir.mkdir(parents=True, exist_ok=True)
    bagrn_dir.mkdir(parents=True, exist_ok=True)
    volrn_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reference scene: {reference.name}")
    print(f"Scenes: {len(loaded)}")
    original_bounds = [
        _compute_bounds(scene.transform, scene.array.shape) for scene in loaded
    ]
    overlaps = detect_multi_overlap(original_bounds, [scene.transform for scene in loaded], min_pixels=100)
    if not overlaps:
        raise RuntimeError("No valid overlap pairs were found for the five scenes")

    registration = run_registration_stage(loaded, overlaps, reference_idx)
    registered = registration["registered"]
    pair_measurements = registration["pair_measurements"]
    rejected_edges = registration["rejected_edges"]
    network_result = registration["network_result"]
    graph = registration["graph"]
    reference_paths = registration["reference_paths"]
    registration_statuses = registration["registration_statuses"]
    artifact_paths = write_registration_network_artifacts(
        output_band_dir,
        loaded,
        reference_idx,
        overlaps,
        pair_measurements,
        rejected_edges,
        graph,
        network_result,
        reference_paths,
        registration_statuses,
    )

    pair_by_edge = {
        (int(pair["idx_i"]), int(pair["idx_j"])): pair
        for pair in pair_measurements
    }
    rejected_by_edge = {
        (int(item["idx_i"]), int(item["idx_j"])): item
        for item in rejected_edges
    }
    for i, j in registration["geometric_edges"]:
        pair = pair_by_edge.get((i, j))
        record = pair if pair is not None else rejected_by_edge[(i, j)]
        edge_stem = f"{i:02d}_{loaded[i].name}__{j:02d}_{loaded[j].name}_{band}"
        _write_registration_matches(
            registration_dir / f"{edge_stem}_matches.csv",
            record.get("matches", []),
        )
        (registration_dir / f"{edge_stem}_metrics.json").write_text(
            json.dumps(_json_safe(record), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    arrays = [scene.array[np.newaxis, :, :] for scene in registered]
    nodatas = [scene.nodata for scene in registered]
    transforms = [scene.transform for scene in registered]
    bounds = [_compute_bounds(scene.transform, scene.array.shape) for scene in registered]

    for index, scene in enumerate(registered):
        write_geotiff(
            registered_dir / f"{index:02d}_{scene.name}_{band}_registered.tif",
            scene.array,
            scene.transform,
            scene.crs,
            nodata=scene.nodata,
        )

    original_metrics = compute_all(arrays, arrays, nodatas, overlaps, [reference_idx])
    start = time.time()
    bagrn_result, bagrn_coeffs, bagrn_info = bagrn_normalize(
        arrays, nodatas, overlaps, control_idx=reference_idx
    )
    bagrn_seconds = time.time() - start

    start = time.time()
    volrn_result, volrn_coeffs = volrn_normalize(
        bagrn_result,
        transforms,
        bounds,
        nodatas,
        block_size_pixels=200,
        lambda_param=0.1,
        rho=1.0,
        max_iter=200,
        tol=1e-4,
        verbose=False,
    )
    volrn_seconds = time.time() - start
    bagrn_metrics = compute_all(
        arrays, bagrn_result, nodatas, overlaps, [reference_idx]
    )
    volrn_metrics = compute_all(
        arrays, volrn_result, nodatas, overlaps, [reference_idx]
    )

    for method, normalized in (("bagrn", bagrn_result), ("volrn", volrn_result)):
        method_dir = bagrn_dir if method == "bagrn" else volrn_dir
        for index, scene_array in enumerate(normalized):
            write_geotiff(
                method_dir / f"{index:02d}_{registered[index].name}_{band}_{method}.tif",
                scene_array,
                registered[index].transform,
                registered[index].crs,
                nodata=registered[index].nodata,
            )

    mosaic_paths = {
        "bagrn": output_band_dir / f"mosaic_{band}_bagrn.tif",
        "volrn": output_band_dir / f"mosaic_{band}_volrn.tif",
    }
    create_mosaic(bagrn_result, transforms, reference.crs, nodatas, str(mosaic_paths["bagrn"]))
    create_mosaic(volrn_result, transforms, reference.crs, nodatas, str(mosaic_paths["volrn"]))

    result = {
        "band": band,
        "scene_count": len(registered),
        "reference_scene": reference.name,
        "reference_scene_index": reference_idx,
        "scene_order": [scene.name for scene in registered],
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "registration": _json_safe(pair_measurements),
        "registration_rejected": _json_safe(rejected_edges),
        "registration_network": _json_safe({
            "geometric_edges": registration["geometric_edges"],
            "reliable_edges": [
                [int(pair["idx_i"]), int(pair["idx_j"])]
                for pair in pair_measurements
            ],
            "rejected_edges": [
                [int(item["idx_i"]), int(item["idx_j"])]
                for item in rejected_edges
            ],
            "reachable": graph["reachable"],
            "unreachable": graph["unreachable"],
            "spanning_tree_edges": graph["spanning_tree_edges"],
            "reference_paths": reference_paths,
            "global_shifts": network_result["global_shifts"],
            "registration_statuses": registration_statuses,
        }),
        "registration_artifacts": {
            key: str(path) for key, path in artifact_paths.items()
        },
        "metrics": {
            "original": original_metrics,
            "bagrn": bagrn_metrics,
            "volrn": volrn_metrics,
        },
        "normalization": {
            "bagrn_coefficients": bagrn_coeffs,
            "bagrn_info": bagrn_info,
            "volrn_coefficients": volrn_coeffs,
            "bagrn_seconds": bagrn_seconds,
            "volrn_seconds": volrn_seconds,
        },
        "outputs": {
            "bagrn_mosaic": str(mosaic_paths["bagrn"]),
            "volrn_mosaic": str(mosaic_paths["volrn"]),
        },
    }
    summary_path = output_band_dir / "five_image_summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nBAGRN mosaic: {mosaic_paths['bagrn']}")
    print(f"VOLRN mosaic: {mosaic_paths['volrn']}")
    print(f"Summary: {summary_path}")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Five-scene B14 registration and mosaic pipeline")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--band", default=DEFAULT_BAND)
    parser.add_argument("--reference-scene", default=DEFAULT_REFERENCE_SCENE)
    args = parser.parse_args(argv)
    paths = discover_scene_paths(args.input_dir, args.band, args.reference_scene)
    run_pipeline(paths, args.output_dir, args.band, args.reference_scene)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
