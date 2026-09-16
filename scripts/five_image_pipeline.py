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
    build_local_residual_controls,
    collect_block_matches,
    compute_shifts_from_overlap,
    fit_local_rbf,
    multi_image_network_adjustment,
    phase_correlation,
    spatial_cross_validate,
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


def _residual_summary(matches: Sequence[Dict[str, Any]], dx: float, dy: float) -> Dict[str, Any]:
    """Summarize accepted block residuals relative to the global shift."""
    if not matches:
        return {"count": 0, "rmse_pixels": None, "mean_pixels": None,
                "median_pixels": None, "p95_pixels": None, "max_pixels": None}
    residual_dx = np.asarray([m["shift_dx"] for m in matches], dtype=float) - dx
    residual_dy = np.asarray([m["shift_dy"] for m in matches], dtype=float) - dy
    errors = np.hypot(residual_dx, residual_dy)
    return {
        "count": int(errors.size),
        "rmse_pixels": float(np.sqrt(np.mean(errors ** 2))),
        "mean_pixels": float(np.mean(errors)),
        "median_pixels": float(np.median(errors)),
        "p95_pixels": float(np.percentile(errors, 95)),
        "max_pixels": float(np.max(errors)),
    }


def _registration_metrics(
    matches: Sequence[Dict[str, Any]],
    screening: Dict[str, Any],
    dx: float,
    dy: float,
    confidence: float,
) -> Dict[str, Any]:
    """Build concise per-target registration evidence."""
    accepted = len(matches)
    candidate = int(screening.get("total", accepted))
    if accepted:
        residual_dx = np.asarray([m["shift_dx"] for m in matches], dtype=float) - dx
        residual_dy = np.asarray([m["shift_dy"] for m in matches], dtype=float) - dy
        med_dx = np.median(residual_dx)
        med_dy = np.median(residual_dy)
        mad_dx = np.median(np.abs(residual_dx - med_dx))
        mad_dy = np.median(np.abs(residual_dy - med_dy))
        inliers = (
            (np.abs(residual_dx - med_dx) < max(3 * mad_dx, 0.3))
            & (np.abs(residual_dy - med_dy) < max(3 * mad_dy, 0.3))
        )
    else:
        inliers = np.zeros(0, dtype=bool)
    return {
        "offset": {
            "dx_pixels": float(dx),
            "dy_pixels": float(dy),
            "magnitude_pixels": float(np.hypot(dx, dy)),
            "direction_image_degrees": float(np.degrees(np.arctan2(dy, dx)))
            if dx or dy else 0.0,
        },
        "phase_confidence": float(confidence),
        "matching": {
            "candidate_blocks": candidate,
            "accepted_matches": accepted,
            "inlier_matches": int(inliers.sum()),
            "accepted_match_ratio": float(accepted / candidate) if candidate else 0.0,
            "inlier_ratio": float(inliers.sum() / accepted) if accepted else 0.0,
            "screening": _json_safe(screening),
        },
        "residual_relative_to_global_model": _residual_summary(matches, dx, dy),
    }


def _select_translation_or_rbf(summary: Dict[str, Any]) -> str:
    """Use the same 10% P95 improvement rule as the two-image script."""
    translation = summary.get("translation", {}).get("p95")
    rbf = summary.get("rbf", {}).get("p95")
    if translation is None or rbf is None:
        return "translation"
    if np.isfinite(translation) and np.isfinite(rbf) and translation > 0:
        if (translation - rbf) / translation >= 0.10:
            return "rbf"
    return "translation"


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


def _build_local_fields(
    target_shape: Tuple[int, int],
    controls: Dict[str, Any],
    rbf_dx: Any,
    rbf_dy: Any,
    coord_range: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the selected local RBF only inside its control-point hull."""
    from scipy.spatial import Delaunay

    height, width = target_shape
    hull = Delaunay(controls["points_xy"])
    yy, xx = np.mgrid[0:height, 0:width]
    points = np.column_stack([xx.ravel(), yy.ravel()])
    dx_field = np.zeros(points.shape[0], dtype=np.float64)
    dy_field = np.zeros(points.shape[0], dtype=np.float64)
    xmin, ymin, xmax, ymax = coord_range
    for start in range(0, len(points), 50000):
        stop = start + 50000
        chunk = points[start:stop]
        inside = hull.find_simplex(chunk) >= 0
        tx = (chunk[:, 0] - xmin) / max(xmax - xmin, 1e-10)
        ty = (chunk[:, 1] - ymin) / max(ymax - ymin, 1e-10)
        normalized = np.column_stack([tx, ty])
        dx_chunk = np.clip(rbf_dx(normalized), -2.5, 2.5)
        dy_chunk = np.clip(rbf_dy(normalized), -2.5, 2.5)
        dx_field[start:stop] = np.where(inside, dx_chunk, 0.0)
        dy_field[start:stop] = np.where(inside, dy_chunk, 0.0)
    return dx_field.reshape(target_shape), dy_field.reshape(target_shape)


def register_scene_to_reference(
    reference: SceneData,
    target: SceneData,
) -> Tuple[SceneData, Dict[str, Any]]:
    """Apply the existing two-image global/RBF registration flow to one target."""
    if reference.crs != target.crs:
        raise ValueError(f"CRS mismatch: {reference.crs} vs {target.crs}")

    matches, screening = collect_block_matches(
        reference.array, reference.transform,
        target.array, target.transform,
        reference.nodata, target.nodata,
    )
    lag_y, lag_x, confidence, translation_stats = compute_shifts_from_overlap(
        reference.array, reference.transform,
        target.array, target.transform,
        reference.nodata, target.nodata,
    )
    controls = build_local_residual_controls(
        matches, lag_x, lag_y, confidence_threshold=0.75
    )

    use_local = False
    model = "translation"
    rbf_dx = rbf_dy = None
    coord_range = None
    cv_summary: Dict[str, Any] = {}
    best_smoothing = None
    if controls["n_valid"] >= 30:
        cv_summary, _ = spatial_cross_validate(
            controls["points_xy"], controls["residual_dx"], controls["residual_dy"],
            lag_x, lag_y, matches, rbf_smoothing=0.1,
        )
        if _select_translation_or_rbf(cv_summary) == "rbf":
            smoothing_rows = []
            for smoothing in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
                try:
                    summary_s, _ = spatial_cross_validate(
                        controls["points_xy"], controls["residual_dx"], controls["residual_dy"],
                        lag_x, lag_y, matches, rbf_smoothing=smoothing,
                    )
                    smoothing_rows.append({
                        "smoothing": smoothing,
                        "p95": summary_s.get("rbf", {}).get("p95"),
                        "rmse": summary_s.get("rbf", {}).get("rmse"),
                    })
                except Exception as exc:
                    smoothing_rows.append({"smoothing": smoothing, "p95": None, "rmse": None,
                                           "error": str(exc)})
            valid_rows = [row for row in smoothing_rows
                          if row["p95"] is not None and np.isfinite(row["p95"])]
            if valid_rows:
                best_smoothing = min(valid_rows, key=lambda row: row["p95"])["smoothing"]
                rbf_dx, rbf_dy, xmin, xmax = fit_local_rbf(
                    controls["points_xy"], controls["residual_dx"],
                    controls["residual_dy"], smoothing=best_smoothing,
                    neighbors=min(20, controls["n_valid"]),
                )
                coord_range = (xmin[0], xmin[1], xmax[0], xmax[1])
                use_local = True
                model = "rbf"
        else:
            smoothing_rows = []
    else:
        smoothing_rows = []

    local_dx = np.zeros_like(target.array, dtype=np.float64)
    local_dy = np.zeros_like(target.array, dtype=np.float64)
    if use_local and controls["n_valid"] >= 3:
        local_dx, local_dy = _build_local_fields(
            target.array.shape, controls, rbf_dx, rbf_dy, coord_range
        )
    warped = warp_with_displacement_field(
        target.array,
        lag_x,
        lag_y,
        local_dx,
        local_dy,
        target.nodata,
    )
    diagnostics = _registration_metrics(matches, screening, lag_x, lag_y, confidence)
    diagnostics.update({
        "target_scene": target.name,
        "model_used": model,
        "matches": _json_safe(matches),
        "local_control_points": int(controls["n_valid"]),
        "translation_stats": _json_safe(translation_stats),
        "local_cross_validation": _json_safe(cv_summary),
        "rbf_smoothing": best_smoothing,
        "smoothing_candidates": _json_safe(smoothing_rows),
    })
    return SceneData(
        name=target.name,
        path=target.path,
        array=warped,
        transform=target.transform,
        crs=target.crs,
        nodata=target.nodata,
    ), diagnostics


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
) -> Dict[str, Any]:
    """Run the five-scene registration, normalization, and mosaic workflow."""
    if len(scene_paths) != EXPECTED_SCENE_COUNT:
        raise ValueError(f"expected {EXPECTED_SCENE_COUNT} scene paths, got {len(scene_paths)}")

    output_band_dir = Path(output_dir) / band
    output_band_dir.mkdir(parents=True, exist_ok=True)
    loaded = [load_scene(Path(path), band=band) for path in scene_paths]
    reference = loaded[0]
    registered = [reference]
    registration_records = []
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
    for target in loaded[1:]:
        print(f"\n--- Co-registration: {reference.name} -> {target.name} ---")
        registered_target, diagnostics = register_scene_to_reference(reference, target)
        registered.append(registered_target)
        registration_records.append(diagnostics)
        print(
            f"  Translation: dx={diagnostics['offset']['dx_pixels']:.4f}, "
            f"dy={diagnostics['offset']['dy_pixels']:.4f}, "
            f"matches={diagnostics['matching']['accepted_matches']}/"
            f"{diagnostics['matching']['candidate_blocks']}, model={diagnostics['model_used']}"
        )
        _write_registration_matches(
            registration_dir / f"{len(registered) - 1:02d}_{target.name}_{band}_matches.csv",
            diagnostics.get("matches", []),
        )
        (registration_dir / f"{len(registered) - 1:02d}_{target.name}_{band}_metrics.json").write_text(
            json.dumps(_json_safe(diagnostics), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    arrays = [scene.array[np.newaxis, :, :] for scene in registered]
    nodatas = [scene.nodata for scene in registered]
    transforms = [scene.transform for scene in registered]
    bounds = [_compute_bounds(scene.transform, scene.array.shape) for scene in registered]
    overlaps = detect_multi_overlap(bounds, transforms, min_pixels=100)
    if not overlaps:
        raise RuntimeError("No valid overlap pairs were found for the five registered scenes")
    print(f"\nOverlap pairs: {len(overlaps)}")

    for index, scene in enumerate(registered):
        write_geotiff(
            registered_dir / f"{index:02d}_{scene.name}_{band}_registered.tif",
            scene.array,
            scene.transform,
            scene.crs,
            nodata=scene.nodata,
        )

    original_metrics = compute_all(arrays, arrays, nodatas, overlaps, [0])
    start = time.time()
    bagrn_result, bagrn_coeffs, bagrn_info = bagrn_normalize(
        arrays, nodatas, overlaps, control_idx=0
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
    bagrn_metrics = compute_all(arrays, bagrn_result, nodatas, overlaps, [0])
    volrn_metrics = compute_all(arrays, volrn_result, nodatas, overlaps, [0])

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
        "scene_order": [scene.name for scene in registered],
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "registration": registration_records,
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
    run_pipeline(paths, args.output_dir, args.band)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
