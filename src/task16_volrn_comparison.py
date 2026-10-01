"""Task16 fixed-geometry 13-scene BAGRN/VOLRN comparison.

This module deliberately starts from persisted Task15 Stage06 BAGRN scenes.
It does not contain a geometry or BAGRN execution path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine, array_bounds
from scipy.ndimage import distance_transform_edt
from scipy.stats import wasserstein_distance

from src.multiscene_sift.mosaic_protocol import grid_transform
from src.multiscene_sift.radiometric_metrics import (
    aggregate_weighted_pair_metric,
    compute_pair_mamd,
    compute_pair_msdd,
    compute_pair_rdd,
    compute_seam_zone_metrics,
    compute_cgl,
)
from src.multiscene_sift.radiometric_runner import _task10d_metrics
from src.registration_benchmark.mosaic_diagnostics import build_seam_zone_mask
from src.multiscene_sift.task10d_metric_protocol import load_task10d_metric_protocol
from src.multiscene_sift.structural_metrics import stream_structure_metrics
from src.volrn import volrn_normalize


TASK16_PARAMS = {
    "block_size_pixels": 400,
    "lambda": 0.5,
    "rho": 1.0,
    "max_iter": 200,
    "tol": 1e-4,
}
# Historical value retained only for regression reporting; it is not a gate.
HISTORICAL_EXPECTED_SUPPORT = 62_033_096
EXPECTED_SUPPORT = HISTORICAL_EXPECTED_SUPPORT
EXPECTED_SCENE_COUNT = 13


class ProvenanceMismatch(RuntimeError):
    """Raised when the frozen Task15 input identity is not reproducible."""


def validate_task16_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the version-controlled Task16 YAML without allowing drift."""
    volrn = config.get("volrn")
    if not isinstance(volrn, Mapping):
        raise ValueError("Task16 config requires a volrn mapping")
    expected = {"block_size_pixels": 400, "lambda": 0.5, "rho": 1.0, "max_iter": 200, "tol": 1e-4, "adaptive_rho": False, "preconditioner": "jacobi"}
    for key, value in expected.items():
        if volrn.get(key, value) != value:
            raise ValueError(f"Task16 frozen parameter mismatch: {key}={volrn.get(key)!r}, expected {value!r}")
    strict = config.get("strict_ablation", {})
    e2e = config.get("end_to_end", {})
    if strict.get("layout", "task15_v1") != "task15_v1" or e2e.get("reuse_same_refine_algorithm", True) is not True:
        raise ValueError("Task16 strict ablation/end-to-end semantics are frozen")
    return dict(expected)


def classify_volrn_status(diag: Mapping[str, Any]) -> str:
    """Classify the formal Task16 status without treating x-stability as convergence."""

    finite = bool(diag.get("finite_state", False))
    cg_failed = bool(diag.get("cg_failed", diag.get("cg_failed_any", False)))
    iterations = int(diag.get("iterations", -1))
    if not finite or cg_failed:
        return "NUMERICAL_INVALID_ITER200"
    if iterations != TASK16_PARAMS["max_iter"]:
        return "ITERATION_COUNT_MISMATCH"
    if bool(diag.get("strict_admm_converged", diag.get("converged", False))):
        return "PASS_STRICT_CONVERGED"
    return "COMPLETED_FINITE_NONCONVERGED"


def compare_metric_values(metric: str, baseline: float | None, candidate: float | None) -> dict[str, Any]:
    """Return the fixed comparison-table row for one scalar metric."""

    if baseline is None or candidate is None:
        absolute = relative = None
    else:
        absolute = float(candidate - baseline)
        relative = float(100.0 * absolute / baseline) if baseline != 0 else None
    return {
        "metric": metric,
        "baseline": baseline,
        "candidate": candidate,
        "absolute_change": absolute,
        "relative_change_percent": relative,
    }


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_grid(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    """Verify the canonical grid identity exactly enough for a replay."""

    keys = ("crs", "width", "height")
    if any(expected.get(key) != actual.get(key) for key in keys):
        raise ValueError("HARD_STOP_PROVENANCE_MISMATCH: canonical grid identity differs")
    expected_transform = [float(x) for x in expected.get("transform", [])]
    actual_transform = [float(x) for x in actual.get("transform", [])]
    if expected_transform != actual_transform:
        raise ValueError("HARD_STOP_PROVENANCE_MISMATCH: canonical transform differs")
    expected_res = float(expected.get("resolution", expected.get("pixel_size")))
    actual_res = float(actual.get("resolution", actual.get("pixel_size")))
    if expected_res != actual_res:
        raise ValueError("HARD_STOP_PROVENANCE_MISMATCH: canonical resolution differs")
    return True


def _write_single_band(path: Path, data: np.ndarray, transform, crs: str, nodata, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": int(data.shape[0]), "width": int(data.shape[1]),
        "count": 1, "dtype": dtype, "crs": crs, "transform": transform,
        "nodata": nodata, "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.asarray(data).astype(dtype, copy=False), 1)


def _write_preview(path: Path, raster_path: Path, valid: np.ndarray, lower: float | None = None, upper: float | None = None) -> dict[str, Any]:
    with rasterio.open(raster_path) as src:
        data = src.read(1, out_shape=(1, 1200, 800), masked=False).astype(np.float32)
    mask = np.isfinite(data)
    values = data[mask]
    if lower is None or upper is None:
        lower, upper = (np.percentile(values, [2, 98]).tolist() if values.size else [0.0, 1.0])
    scale = max(float(upper - lower), 1e-12)
    image = np.clip((data - lower) / scale * 255.0, 0, 255).astype(np.uint8)
    image[~mask] = 0
    Image.fromarray(image, mode="L").save(path)
    return {"lower": float(lower), "upper": float(upper), "width": 800, "height": 1200}


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_scene(path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, Any]:
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        transform = src.transform
        if src.crs is None:
            raise ProvenanceMismatch(f"HARD_STOP_PROVENANCE_MISMATCH: missing CRS in {path}")
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(bool)
    if array.shape != mask.shape:
        raise ProvenanceMismatch(f"HARD_STOP_PROVENANCE_MISMATCH: scene/mask shape mismatch {path.name}")
    array[~mask] = np.nan
    return array, mask, transform


def _crop_scene(array: np.ndarray, mask: np.ndarray, transform: Any) -> tuple[np.ndarray, np.ndarray, Any, tuple[int, int, int, int]]:
    rows, cols = np.where(mask)
    if not rows.size:
        raise ValueError("frozen scene has no valid support")
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    local_transform = transform * Affine.translation(c0, r0)
    local_array = array[r0:r1, c0:c1][np.newaxis, :, :].copy()
    local_mask = mask[r0:r1, c0:c1].copy()
    local_array[0][~local_mask] = np.nan
    return local_array, local_mask, local_transform, (r0, r1, c0, c1)


def _support_products(output_dir: Path, masks: list[np.ndarray], grid: Mapping[str, Any]) -> dict[str, Any]:
    union = np.zeros(masks[0].shape, dtype=bool)
    count = np.zeros(masks[0].shape, dtype=np.uint8)
    for mask in masks:
        union |= mask
        count += mask.astype(np.uint8)
    weights = np.zeros(union.shape, dtype=np.float32)
    for mask in masks:
        weight = distance_transform_edt(mask).astype(np.float32)
        weight[mask] += 1e-6
        weights += weight
    _write_single_band(output_dir / "valid_mask.tif", union.astype(np.uint8), grid_transform(grid), str(grid["crs"]), 0, "uint8")
    _write_single_band(output_dir / "contributor_count.tif", count, grid_transform(grid), str(grid["crs"]), 0, "uint8")
    _write_single_band(output_dir / "weight_sum.tif", weights, grid_transform(grid), str(grid["crs"]), 0.0, "float32")
    return {"union_support": int(union.sum()), "contributor_count_max": int(count.max()), "finite_weight_pixels": int(np.isfinite(weights).sum())}


def _stream_weighted_mosaic(
    scene_paths: list[Path], weight_paths: list[Path], output_path: Path,
    grid: Mapping[str, Any], *, tile_size: int = 512,
) -> None:
    """Apply the frozen distance-weighted feather formula without a full projection cache."""

    if len(scene_paths) != len(weight_paths):
        raise ValueError("scene and weight path counts must match")
    with rasterio.open(scene_paths[0]) as first:
        profile = first.profile.copy()
        height, width = first.height, first.width
        profile.update(driver="GTiff", count=1, dtype="float32", nodata=np.nan, compress="lzw")
    expected_transform = grid_transform(grid)
    for scene_path, weight_path in zip(scene_paths, weight_paths):
        with rasterio.open(scene_path) as scene, rasterio.open(weight_path) as weight:
            if (scene.height, scene.width) != (height, width) or (weight.height, weight.width) != (height, width):
                raise ValueError("streaming weighted mosaic grid mismatch")
            if scene.transform != expected_transform or weight.transform != expected_transform:
                raise ValueError("streaming weighted mosaic transform mismatch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenes = [rasterio.open(path) for path in scene_paths]
    weights = [rasterio.open(path) for path in weight_paths]
    try:
        with rasterio.open(output_path, "w", **profile) as dst:
            for r0 in range(0, height, tile_size):
                r1 = min(height, r0 + tile_size)
                for c0 in range(0, width, tile_size):
                    c1 = min(width, c0 + tile_size)
                    window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
                    numerator = np.zeros((r1 - r0, c1 - c0), dtype=np.float64)
                    denominator = np.zeros_like(numerator)
                    for scene, weight in zip(scenes, weights):
                        values = scene.read(1, window=window).astype(np.float64)
                        feather = weight.read(1, window=window).astype(np.float64)
                        valid = np.isfinite(values) & np.isfinite(feather) & (feather > 0)
                        numerator[valid] += values[valid] * feather[valid]
                        denominator[valid] += feather[valid]
                    result = np.full(numerator.shape, np.nan, dtype=np.float32)
                    valid = denominator > 1e-12
                    result[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
                    dst.write(result, 1, window=window)
    finally:
        for handle in scenes + weights:
            handle.close()


def _load_frozen_inputs(task15_root: Path) -> dict[str, Any]:
    canonical_path = task15_root / "stages/04_canonical_warp/canonical_output_grid.json"
    source_path = task15_root / "stages/04_canonical_warp/task14_source_config.json"
    manifest_path = task15_root / "stages/06_bagrn/bagrn/normalized_scenes_manifest.json"
    scene_dir = task15_root / "stages/06_bagrn/bagrn/normalized_scenes"
    mask_dir = scene_dir / "input_valid_masks"
    v2_path = task15_root / "stages/10_mosaics/v2_local_corrected_multiscene.tif"
    weight_root = task15_root / "stages/10_mosaics/weights"
    weight_dir = weight_root / "v1" if (weight_root / "v1").is_dir() else weight_root
    v2_weight_dir = weight_root / "v2" if (weight_root / "v2").is_dir() else weight_dir
    ours_scene_dir = task15_root / "stages/09_correction"
    ours_scene_paths = [ours_scene_dir / f"corrected_scene_{i:03d}.tif" for i in range(EXPECTED_SCENE_COUNT)]
    required = [canonical_path, source_path, manifest_path, v2_path, task15_root / "stages/10_mosaics/mosaic_summary.json"]
    required += [scene_dir / f"scene_{i:03d}.tif" for i in range(EXPECTED_SCENE_COUNT)]
    required += [mask_dir / f"scene_{i:03d}.tif" for i in range(EXPECTED_SCENE_COUNT)]
    required += [weight_dir / f"weight_scene_{i:03d}.tif" for i in range(EXPECTED_SCENE_COUNT)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ProvenanceMismatch("HARD_STOP_PROVENANCE_MISMATCH: missing frozen artifact(s): " + ", ".join(missing))
    grid = _json(canonical_path)
    manifest = _json(manifest_path)
    source = _json(source_path)
    if len(manifest.get("scene_ids", [])) != EXPECTED_SCENE_COUNT or len(manifest.get("scenes", [])) != EXPECTED_SCENE_COUNT:
        raise ProvenanceMismatch("HARD_STOP_PROVENANCE_MISMATCH: Task15 is not a 13-scene manifest")
    if len(source.get("scenes", [])) != EXPECTED_SCENE_COUNT:
        raise ProvenanceMismatch("HARD_STOP_PROVENANCE_MISMATCH: source config scene count differs")
    manifest_grid = {"crs": manifest["grid"]["crs"], "width": manifest["grid"]["width"], "height": manifest["grid"]["height"], "resolution": manifest["grid"].get("resolution", manifest["grid"].get("pixel_size")), "transform": grid["transform"]}
    validate_frozen_grid(grid, manifest_grid)
    summary = _json(task15_root / "stages/10_mosaics/mosaic_summary.json")
    if "weighted" not in str(summary.get("v0", "")).lower() or not (task15_root / "stages/10_mosaics/v0_bagrn_weighted.tif").is_file():
        raise ProvenanceMismatch("HARD_STOP_V0_SEMANTIC_MISMATCH")
    masks: list[np.ndarray] = []
    scene_paths: list[Path] = []
    mask_paths: list[Path] = []
    for index in range(EXPECTED_SCENE_COUNT):
        scene_path = scene_dir / f"scene_{index:03d}.tif"
        mask_path = mask_dir / f"scene_{index:03d}.tif"
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1).astype(bool)
            if (mask_src.width, mask_src.height) != (int(grid["width"]), int(grid["height"])):
                raise ProvenanceMismatch("HARD_STOP_PROVENANCE_MISMATCH: valid-mask grid differs")
        masks.append(mask)
        scene_paths.append(scene_path)
        mask_paths.append(mask_path)
        declared = int(manifest["scenes"][index].get("valid_pixels", -1))
        if declared >= 0 and declared != int(mask.sum()):
            raise ProvenanceMismatch(f"HARD_STOP_PROVENANCE_MISMATCH: scene {index} support declaration differs")
    union = np.zeros((int(grid["height"]), int(grid["width"])), dtype=bool)
    for mask in masks:
        union |= mask
    union_support = int(union.sum())
    return {
        "grid": grid,
        "source_config": source_path,
        "manifest": manifest_path,
        "scene_paths": scene_paths,
        "ours_scene_paths": ours_scene_paths,
        "mask_paths": mask_paths,
        "weight_paths": [weight_dir / f"weight_scene_{i:03d}.tif" for i in range(EXPECTED_SCENE_COUNT)],
        "v1_weight_paths": [weight_dir / f"weight_scene_{i:03d}.tif" for i in range(EXPECTED_SCENE_COUNT)],
        "v2_weight_paths": [v2_weight_dir / f"weight_scene_{i:03d}.tif" for i in range(EXPECTED_SCENE_COUNT)],
        "masks": masks,
        "scene_ids": list(manifest["scene_ids"]),
        "v2_path": v2_path,
        "provenance_files": required,
        "union_support": union_support,
    }


def _save_full_scene(path: Path, array: np.ndarray, grid: Mapping[str, Any]) -> None:
    _write_single_band(path, array, grid_transform(grid), str(grid["crs"]), np.nan, "float32")


def _save_history(output_dir: Path, diagnostics: Mapping[str, Any]) -> None:
    bands = diagnostics.get("band_solver_diagnostics", [])
    payload = {"schema_version": 2, "n_bands": len(bands), "bands": bands}
    (output_dir / "solver_history.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for band, diag in enumerate(bands):
        for item in diag.get("history", []):
            rows.append({"band": band, **item})
    fields = ["band", "iteration", "objective", "primal_residual", "primal_tolerance", "primal_ratio", "dual_residual", "dual_tolerance", "dual_ratio", "relative_change", "x_update_norm", "z_update_norm", "dual_update_norm", "cg_iterations", "cg_residual", "cg_info", "cg_final_residual_norm", "rho", "rho_changed", "finite_state"]
    _write_csv(output_dir / "solver_history.csv", rows, fields)


def _coefficient_diagnostics(coefficients: np.ndarray, scene_audit: Mapping[str, Any], solver_diag: Mapping[str, Any]) -> dict[str, Any]:
    coeff = np.asarray(coefficients, dtype=np.float64)
    flat = coeff.reshape(-1, 2) if coeff.size else np.empty((0, 2))
    a = flat[:, 0] if flat.size else np.array([], dtype=float)
    b = flat[:, 1] if flat.size else np.array([], dtype=float)
    def stats(values: np.ndarray) -> dict[str, Any]:
        return {"min": float(values.min()) if values.size else None, "max": float(values.max()) if values.size else None, "mean": float(values.mean()) if values.size else None, "median": float(np.median(values)) if values.size else None}
    return {
        "n_blocks": int(solver_diag.get("n_blocks", 0)),
        "n_pairs": int(solver_diag.get("n_pairs", 0)),
        "n_variables": int(a.size * 2),
        "a": {**stats(a), "mean_abs_a_minus_1": float(np.mean(np.abs(a - 1))) if a.size else None, "max_abs_a_minus_1": float(np.max(np.abs(a - 1))) if a.size else None},
        "b": {**stats(b), "mean_abs_b": float(np.mean(np.abs(b))) if b.size else None, "max_abs_b": float(np.max(np.abs(b))) if b.size else None},
        "nonidentity_block_count": int(np.count_nonzero((np.abs(a - 1) > 1e-12) | (np.abs(b) > 1e-12))),
        "scene_application": scene_audit.get("scenes", []),
    }


def _finite_scene_outputs(paths: list[Path], masks: list[np.ndarray]) -> dict[str, Any]:
    finite = 0
    invalid = 0
    per_scene = []
    for path, mask in zip(paths, masks):
        with rasterio.open(path) as src:
            data = src.read(1)
        good = mask & np.isfinite(data)
        bad = mask & ~np.isfinite(data)
        finite += int(good.sum())
        invalid += int(bad.sum())
        per_scene.append({"finite_valid_pixels": int(good.sum()), "numerical_invalid_pixels": int(bad.sum())})
    return {"finite_valid_pixels": finite, "numerical_invalid_pixels": invalid, "per_scene": per_scene}


def _structural_ncc(path_a: Path, path_b: Path, mask: np.ndarray, tile: int = 512) -> float:
    metrics = stream_structure_metrics(path_a, path_b, mask, tile_size=tile, halo=1)
    value = metrics.get("gradient_magnitude_ncc")
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else float("nan")


def _stream_cgl(path_a: Path, path_b: Path, mask: np.ndarray, tile: int = 512) -> dict[str, Any]:
    """Compute Task10D CGL in haloed windows to avoid full-canvas temporaries."""

    metrics = stream_structure_metrics(path_a, path_b, mask, tile_size=tile, halo=1)
    count = int(metrics.get("eligible_pixels", 0))
    cgl = metrics.get("cgl_rad")
    return {"status": "PASS" if cgl is not None else "INSUFFICIENT_SUPPORT", "valid_pixels": count,
            "cgl_rad": cgl, "cgl_deg": metrics.get("cgl_deg")}


def _stream_task10d_metrics(
    paths: list[Path], masks: list[np.ndarray], weight_paths: list[Path], scene_ids: list[str],
) -> dict[str, Any]:
    """Task10D pair/local/seam metrics with only one pair/tile in memory."""

    import itertools
    global_rows = {"mamd": [], "msdd": [], "rdd": []}
    local_rows = {"mamd": [], "rdd": []}
    seam_rows = {"seam_mae": [], "seam_rmse": [], "seam_rdd": []}
    height, width = masks[0].shape
    tile_size = 256
    tiles = [(r0, min(height, r0 + tile_size), c0, min(width, c0 + tile_size)) for r0 in range(0, height, tile_size) for c0 in range(0, width, tile_size)]
    for i, j in itertools.combinations(range(len(paths)), 2):
        shared = masks[i] & masks[j]
        rows, cols = np.where(shared)
        if not rows.size:
            continue
        r0, r1, c0, c1 = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        with rasterio.open(paths[i]) as src_i, rasterio.open(paths[j]) as src_j, rasterio.open(weight_paths[i]) as wi, rasterio.open(weight_paths[j]) as wj:
            a = src_i.read(1, window=window).astype(np.float64)
            b = src_j.read(1, window=window).astype(np.float64)
            pair_mask = shared[r0:r1, c0:c1]
            finite = pair_mask & np.isfinite(a) & np.isfinite(b)
            count = int(finite.sum())
            pair_name = f"{scene_ids[i]}::{scene_ids[j]}"
            if count:
                for name, function in (("mamd", compute_pair_mamd), ("msdd", compute_pair_msdd), ("rdd", compute_pair_rdd)):
                    global_rows[name].append({"pair": pair_name, "value": function(a, b, finite), "valid_pixels": count})
            for tr0, tr1, tc0, tc1 in tiles:
                local_mask = shared[tr0:tr1, tc0:tc1]
                if int(local_mask.sum()) < 4096:
                    continue
                local_window = rasterio.windows.Window(tc0, tr0, tc1 - tc0, tr1 - tr0)
                la = src_i.read(1, window=local_window).astype(np.float64)
                lb = src_j.read(1, window=local_window).astype(np.float64)
                local_valid = local_mask & np.isfinite(la) & np.isfinite(lb)
                valid_pixels = int(local_valid.sum())
                if valid_pixels < 4096:
                    continue
                local_rows["mamd"].append({"pair": pair_name, "value": compute_pair_mamd(la, lb, local_valid), "valid_pixels": valid_pixels})
                local_rows["rdd"].append({"pair": pair_name, "value": compute_pair_rdd(la, lb, local_valid), "valid_pixels": valid_pixels})
                wa = wi.read(1, window=local_window).astype(np.float64)
                wb = wj.read(1, window=local_window).astype(np.float64)
                seam_mask = build_seam_zone_mask(local_mask, local_mask, wa, wb)
                seam = compute_seam_zone_metrics(la, lb, seam_mask)
                if seam["valid_pixels"]:
                    for name in seam_rows:
                        seam_rows[name].append({"pair": pair_name, "value": seam[name], "valid_pixels": seam["valid_pixels"]})

    def local_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"median": None, "mean": None, "p95": None, "worst_tile": None, "valid_tile_count": 0, "per_tile": []}
        values = np.asarray([row["value"] for row in rows], dtype=np.float64)
        worst = int(np.argmax(values))
        return {"median": float(np.median(values)), "mean": float(values.mean()), "p95": float(np.percentile(values, 95)), "worst_tile": dict(rows[worst]), "valid_tile_count": len(rows), "per_tile": rows}

    return {
        "mamd": aggregate_weighted_pair_metric(global_rows["mamd"]),
        "msdd": aggregate_weighted_pair_metric(global_rows["msdd"]),
        "rdd": aggregate_weighted_pair_metric(global_rows["rdd"]),
        "local_mamd": local_summary(local_rows["mamd"]),
        "local_rdd": local_summary(local_rows["rdd"]),
        "seam_mae": aggregate_weighted_pair_metric(seam_rows["seam_mae"]),
        "seam_rmse": aggregate_weighted_pair_metric(seam_rows["seam_rmse"]),
        "seam_rdd": aggregate_weighted_pair_metric(seam_rows["seam_rdd"]),
    }


def _metric_scalar(metrics: Mapping[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = metrics
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _task10d_rows(
    a_metrics: Mapping[str, Any], b_metrics: Mapping[str, Any],
    b_structural: Mapping[str, Any], support: int,
    a_structural: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    mapping = [
        ("MAMD", ("mamd", "weighted_mean"), ("mamd", "weighted_mean")),
        ("MSDD", ("msdd", "weighted_mean"), ("msdd", "weighted_mean")),
        ("RDD", ("rdd", "weighted_mean"), ("rdd", "weighted_mean")),
        ("Local MAMD median", ("local_mamd", "median"), ("local_mamd", "median")),
        ("Local RDD median", ("local_rdd", "median"), ("local_rdd", "median")),
        ("Seam MAE", ("seam_mae", "weighted_mean"), ("seam_mae", "weighted_mean")),
        ("Seam RMSE", ("seam_rmse", "weighted_mean"), ("seam_rmse", "weighted_mean")),
        ("Seam RDD", ("seam_rdd", "weighted_mean"), ("seam_rdd", "weighted_mean")),
        ("CGL degrees", ("cgl_rad", "value_deg"), ("cgl_rad", "value_deg")),
    ]
    rows = [compare_metric_values(name, _metric_scalar(a_metrics, a_path), _metric_scalar(b_metrics, b_path)) for name, a_path, b_path in mapping]
    # Identity/no-local baseline is exactly one.  Do not duplicate the
    # measured candidate value in both columns; that hides structure loss.
    rows.append(compare_metric_values("Structure CGL degrees", _metric_scalar(a_structural or {"cgl_deg": 0.0}, ("cgl_deg",)), _metric_scalar(b_structural, ("cgl_deg",))))
    rows.append(compare_metric_values("Gradient NCC mean", _metric_scalar(a_structural or {"mean": 1.0}, ("mean",)), _metric_scalar(b_structural, ("mean",))))
    rows.append(compare_metric_values("Gradient orientation cosine mean", _metric_scalar(a_structural or {"orientation_cosine": 1.0}, ("orientation_cosine",)), _metric_scalar(b_structural, ("orientation_cosine",))))
    rows.append(compare_metric_values("Union support", float(support), float(support)))
    return rows


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _plot_solver_history(history_dir: Path, diagnostics: Mapping[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (history_dir / "plot_status.txt").write_text("matplotlib unavailable; plots not measured\n", encoding="utf-8")
        return
    bands = diagnostics.get("band_solver_diagnostics", [])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for band, diag in enumerate(bands):
        history = diag.get("history", [])
        if not history:
            continue
        x = [row["iteration"] for row in history]
        axes[0, 0].plot(x, [row["objective"] for row in history], alpha=0.7, label=f"band {band}")
        axes[0, 1].plot(x, [row["primal_residual"] for row in history], alpha=0.7)
        axes[0, 1].plot(x, [row.get("primal_tolerance", np.nan) for row in history], "--", alpha=0.5)
        axes[1, 0].plot(x, [row["dual_residual"] for row in history], alpha=0.7)
        axes[1, 0].plot(x, [row.get("dual_tolerance", np.nan) for row in history], "--", alpha=0.5)
        axes[1, 1].plot(x, [row["relative_change"] for row in history], alpha=0.7)
    for ax, title in zip(axes.flat, ("Objective", "Primal residual / tolerance", "Dual residual / tolerance", "Relative x change")):
        ax.set_title(title); ax.set_xlabel("iteration"); ax.grid(alpha=0.25)
        ax.set_yscale("log")
    axes[0, 0].legend(loc="best", fontsize=7)
    fig.savefig(history_dir / "solver_convergence.png", dpi=160)
    plt.close(fig)


def _three_method_figure(path: Path, raster_paths: list[Path], labels: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    images = []
    for raster_path in raster_paths:
        with rasterio.open(raster_path) as src:
            images.append(src.read(1, out_shape=(1, 900, 600), masked=False).astype(np.float32))
    finite = np.isfinite(images[0])
    lower, upper = (np.percentile(images[0][finite], [2, 98]).tolist() if finite.any() else [0, 1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, image, label in zip(axes, images, labels):
        show = np.clip((image - lower) / max(upper - lower, 1e-12), 0, 1)
        ax.imshow(show, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label); ax.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _stream_structure_summary(
    baseline_paths: list[Path], candidate_paths: list[Path], masks: list[np.ndarray],
) -> dict[str, Any]:
    """Aggregate canonical structure metrics scene-by-scene without stacks."""
    rows = []
    for index, (baseline, candidate, mask) in enumerate(zip(baseline_paths, candidate_paths, masks)):
        rows.append({"scene": index, **stream_structure_metrics(baseline, candidate, mask, tile_size=512, halo=1)})
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and np.isfinite(row[key])]
    ncc = values("gradient_magnitude_ncc")
    cgl = values("cgl_deg")
    orientation = values("gradient_orientation_cosine")
    return {
        "per_scene": rows,
        "mean": float(np.mean(ncc)) if ncc else None,
        "min": float(np.min(ncc)) if ncc else None,
        "cgl_deg": float(np.mean(cgl)) if cgl else None,
        "orientation_cosine": float(np.mean(orientation)) if orientation else None,
        "finite": bool(rows) and len(ncc) == len(rows) and len(cgl) == len(rows) and len(orientation) == len(rows),
    }


def run_strict_local_ablation(
    bagrn_scene_paths: list[Path],
    ours_scene_paths: list[Path],
    volrn_scene_paths: list[Path],
    v1_weight_paths: list[Path],
    masks: list[np.ndarray],
    scene_ids: list[str],
    grid: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run A0/A1/A2 with one byte-identical frozen V1 weight layout.

    A0 is BAGRN + V1 weights, A1 is the Task15/Ours corrected scenes + the
    same V1 weights, and A2 is VOLRN corrected scenes + the same V1 weights.
    The helper is deliberately path/hash based so a caller cannot silently
    substitute a route-specific weight directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    routes = {"A0_BAGRN_V1": bagrn_scene_paths, "A1_OURS_V1": ours_scene_paths, "A2_VOLRN_V1": volrn_scene_paths}
    if any(len(paths) != len(v1_weight_paths) for paths in routes.values()):
        raise ValueError("strict ablation route/weight counts differ")
    weight_hashes = [_sha256(path) for path in v1_weight_paths]
    rows = []
    route_metrics = {}
    baseline_structure = _stream_structure_summary(bagrn_scene_paths, bagrn_scene_paths, masks)
    for route, scene_paths in routes.items():
        mosaic = output_dir / f"{route.lower()}.tif"
        _stream_weighted_mosaic(scene_paths, v1_weight_paths, mosaic, grid)
        metrics = _stream_task10d_metrics(scene_paths, masks, v1_weight_paths, scene_ids)
        structure = _stream_structure_summary(bagrn_scene_paths, scene_paths, masks)
        route_metrics[route] = {"mosaic": str(mosaic), "mosaic_sha256": _sha256(mosaic), "metrics": metrics,
                                "structure_preservation": structure,
                                "scene_sha256": [_sha256(path) for path in scene_paths], "weight_sha256": weight_hashes}
    baseline_metrics = route_metrics["A0_BAGRN_V1"]["metrics"]
    support = int(np.count_nonzero(np.logical_or.reduce(masks)))
    for route, payload in route_metrics.items():
        rows.extend({"route": route, **row} for row in _task10d_rows(
            baseline_metrics, payload["metrics"], payload["structure_preservation"], support,
            a_structural=baseline_structure,
        ))
    manifest = {"routes": list(routes), "weight_paths": [str(path) for path in v1_weight_paths],
                "weight_sha256": weight_hashes, "same_weight_hashes": True,
                "same_weight_layout_across_routes": True, "semantics": "A0 BAGRN + V1; A1 Ours corrected + SAME V1; A2 VOLRN corrected + SAME V1"}
    (output_dir / "weight_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "strict_ablation_metrics.json").write_text(json.dumps(route_metrics, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    _write_csv(output_dir / "strict_ablation.csv", rows)
    return {"routes": route_metrics, "weight_manifest": manifest}


def run_volrn_end_to_end(
    corrected_scene_paths: list[Path],
    valid_masks: list[np.ndarray],
    initial_pair_records: list[Mapping[str, Any]],
    grid: Mapping[str, Any],
    output_dir: str | Path,
    *,
    refine_fn: Any | None = None,
    runtime_config: Any | None = None,
) -> dict[str, Any]:
    """Execute VOLRN -> shared refinement -> source-side -> labels -> blend.

    ``refine_fn`` is injectable for the focused contract test and defaults to
    the same ``refine_seam_after_correction`` used by the Ours pipeline.
    """
    from src.seam_local.adapter import _cosine_weights, aggregate_labels_with_ties
    from src.seam_local.config import SeamLocalRuntimeConfig
    from src.seam_local.footprint import footprint_polygon_from_valid_mask
    from src.seam_local.multiscene_label import build_pairwise_preference_field
    from src.seam_local.pipeline import refine_seam_after_correction
    from src.seam_local.source_side import resolve_source_sides

    cfg = runtime_config or SeamLocalRuntimeConfig()
    refine = refine_fn or refine_seam_after_correction
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    masks = [np.asarray(mask, dtype=bool) for mask in valid_masks]
    if not masks or any(mask.ndim != 2 for mask in masks):
        raise ValueError("VOLRN end-to-end valid masks must be nonempty 2D arrays")
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks) or len(corrected_scene_paths) != len(masks):
        raise ValueError("VOLRN end-to-end scene/mask shapes differ")
    transform = grid_transform(grid)
    footprints = [footprint_polygon_from_valid_mask(mask, transform) for mask in masks]

    def seam_for_window(seam, r0, r1, c0, c1):
        from src.seam_local.seam import SeamResult
        path = np.asarray(seam.row_col_path)
        if seam.orientation == "vertical":
            rows = np.arange(r0, r1)
            centers = np.interp(rows, path[:, 0], path[:, 1]) - c0
            local = np.column_stack((np.arange(r1 - r0), np.rint(centers).astype(np.int32)))
        else:
            cols = np.arange(c0, c1)
            centers = np.interp(cols, path[:, 1], path[:, 0]) - r0
            local = np.column_stack((np.rint(centers).astype(np.int32), np.arange(c1 - c0)))
        return SeamResult(seam.orientation, local, seam.total_cost, seam.mean_cost, seam.p95_cost, seam.status, seam.search_mode)

    def seam_for_tile(record, r0, r1, c0, c1):
        return seam_for_window(record["seam"], r0, r1, c0, c1)

    refined_records = []
    refined_count = 0
    refinement_rows = []
    for record in initial_pair_records:
        seam = record.get("initial_seam")
        if seam is None:
            continue
        i, j = int(record["scene_i"]), int(record["scene_j"])
        shared = masks[i] & masks[j]
        rows, cols = np.where(shared)
        if not rows.size:
            refinement_rows.append({"pair_id": record.get("pair_id", f"{i:02d}_{j:02d}"), "status": "NO_FINAL_SHARED_SUPPORT"})
            continue
        wr0, wr1 = int(rows.min()), int(rows.max()) + 1
        wc0, wc1 = int(cols.min()), int(cols.max()) + 1
        window = rasterio.windows.Window(wc0, wr0, wc1 - wc0, wr1 - wr0)
        with rasterio.open(corrected_scene_paths[i]) as left, rasterio.open(corrected_scene_paths[j]) as right:
            a = left.read(1, window=window).astype(np.float64)
            b = right.read(1, window=window).astype(np.float64)
        va, vb = masks[i][wr0:wr1, wc0:wc1], masks[j][wr0:wr1, wc0:wc1]
        a[~va] = np.nan; b[~vb] = np.nan
        local_initial = seam_for_window(seam, wr0, wr1, wc0, wc1)
        refined = refine(a, b, va, vb, local_initial,
                         refine_half_width=cfg.refine_half_width, cost_config=cfg)
        refinement_rows.append({"pair_id": record.get("pair_id", f"{i:02d}_{j:02d}"), "status": refined.status})
        if refined.status != "OK":
            continue
        refined_count += 1
        source = resolve_source_sides(refined, va & vb, va, vb, footprints[i], footprints[j])
        side = source.side_1_source or record.get("initial_side_1_source") or "A"
        from src.seam_local.seam import SeamResult
        local_path = np.asarray(refined.row_col_path).copy()
        if refined.orientation == "vertical":
            global_path = local_path + np.asarray([wr0, wc0])
        else:
            global_path = local_path + np.asarray([wr0, wc0])
        refined_global = SeamResult(refined.orientation, global_path, refined.total_cost, refined.mean_cost, refined.p95_cost, refined.status, refined.search_mode)
        refined_records.append({"scene_i": i, "scene_j": j, "seam": refined_global, "side": side,
                                "window": (wr0, wr1, wc0, wc1), "source_status": source.status})

    raw_dir = output_dir / "raw_edt_cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = []
    p95_values = []
    for index, mask in enumerate(masks):
        distance = distance_transform_edt(mask).astype(np.float32)
        positive = distance[mask]
        p95_values.append(float(np.percentile(positive, 95)) if positive.size else 1.0)
        path = raw_dir / f"distance_scene_{index:03d}.tif"
        _write_single_band(path, distance, transform, str(grid["crs"]), 0.0, "float32")
        raw_paths.append(path)
        del distance, positive

    profile = None
    with rasterio.open(corrected_scene_paths[0]) as src:
        profile = src.profile.copy()
    label_profile = profile.copy(); label_profile.update(dtype="int16", count=1, nodata=-1, compress="deflate")
    method_profile = profile.copy(); method_profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    margin_profile = profile.copy(); margin_profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
    label_path = output_dir / "source_label_map.tif"
    method_path = output_dir / "label_method_map.tif"
    margin_path = output_dir / "score_margin.tif"
    coverage_path = output_dir / "coverage_count.tif"
    stats = {"unresolved_pixels": 0, "invalid_label_pixels": 0, "cycle_pixels": 0,
             "multiscene_pixels": 0, "pairwise_score_tie_pixels": 0,
             "resolved_by_unclipped_interiority": 0, "resolved_by_raw_edt": 0,
             "union_valid_pixels": 0, "label_method_counts": {str(i): 0 for i in range(5)},
             "score_margin_values": []}
    with rasterio.open(label_path, "w", **label_profile) as label_dst, rasterio.open(method_path, "w", **method_profile) as method_dst, rasterio.open(margin_path, "w", **margin_profile) as margin_dst, rasterio.open(coverage_path, "w", **method_profile) as coverage_dst:
        for r0 in range(0, shape[0], 512):
            r1 = min(shape[0], r0 + 512)
            for c0 in range(0, shape[1], 512):
                c1 = min(shape[1], c0 + 512)
                tile_shape = (r1 - r0, c1 - c0)
                tile_masks = np.asarray([mask[r0:r1, c0:c1] for mask in masks], dtype=bool)
                raw_tiles = []
                for raw_path in raw_paths:
                    with rasterio.open(raw_path) as raw_src:
                        raw_tiles.append(raw_src.read(1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)))
                raw_tile = np.asarray(raw_tiles, dtype=np.float32)
                fields = {}
                for record in refined_records:
                    wr0, wr1, wc0, wc1 = record["window"]
                    if record["scene_i"] >= len(masks) or not (wr1 > r0 and wr0 < r1 and wc1 > c0 and wc0 < c1):
                        continue
                    local_path = seam_for_tile(record, r0, r1, c0, c1)
                    field = build_pairwise_preference_field(
                        tile_masks[record["scene_i"]], tile_masks[record["scene_j"]], local_path,
                        record["side"], orientation=record["seam"].orientation,
                        scene_a=record["scene_i"], scene_b=record["scene_j"],
                        seam_half_width=cfg.preference_distance_scale,
                    )
                    domain = np.zeros(tile_shape, dtype=bool)
                    if record["seam"].orientation == "vertical":
                        domain[max(0, wr0 - r0):min(r1, wr1) - r0, :] = True
                    else:
                        domain[:, max(0, wc0 - c0):min(c1, wc1) - c0] = True
                    available = field.available & domain
                    fields[(record["scene_i"], record["scene_j"])] = field.__class__(
                        record["scene_i"], record["scene_j"], np.where(available, field.vote, 0.0).astype(np.float32),
                        available, np.where(available, field.confidence, 0.0).astype(np.float32), field.orientation,
                    )
                lab, methods, diag = aggregate_labels_with_ties(tile_masks, fields, raw_tile, p95_edt=np.asarray(p95_values), tie_tolerance=cfg.tie_tolerance)
                win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
                label_dst.write(lab.astype(np.int16), 1, window=win); method_dst.write(methods.astype(np.uint8), 1, window=win)
                margin_dst.write(np.asarray(diag["score_margin"], dtype=np.float32), 1, window=win); coverage_dst.write(np.count_nonzero(tile_masks, axis=0).astype(np.uint8), 1, window=win)
                coverage = np.count_nonzero(tile_masks, axis=0)
                stats["union_valid_pixels"] += int(np.count_nonzero(coverage)); stats["multiscene_pixels"] += int(np.count_nonzero(coverage >= 3)); stats["cycle_pixels"] += int(diag.get("cycle_pixels", 0)); stats["pairwise_score_tie_pixels"] += int(diag.get("top_score_tie_pixels", 0)); stats["resolved_by_unclipped_interiority"] += int(diag.get("resolved_by_unclipped_interiority", 0)); stats["resolved_by_raw_edt"] += int(diag.get("resolved_by_raw_edt", 0)); stats["unresolved_pixels"] += int(diag.get("unresolved_pixels", 0)); stats["invalid_label_pixels"] += int(np.count_nonzero((coverage > 0) & (lab < 0))); stats["label_method_counts"] = {str(i): stats["label_method_counts"].get(str(i), 0) + int(np.count_nonzero(methods == i)) for i in range(5)}
                margin_values = np.asarray(diag.get("score_margin", []), dtype=np.float32); margin_values = margin_values[np.isfinite(margin_values)]
                if margin_values.size: stats["score_margin_values"].extend(np.percentile(margin_values, [0, 50, 95]).tolist())
    for raw_path in raw_paths:
        raw_path.unlink(missing_ok=True)
    stats["pairwise_score_tie_pixels"] = int(stats["pairwise_score_tie_pixels"])
    stats["cycle_fraction"] = stats["cycle_pixels"] / max(stats["multiscene_pixels"], 1)
    stats["score_margin_min"] = float(min(stats["score_margin_values"])) if stats["score_margin_values"] else None
    stats["score_margin_median"] = float(np.median(stats["score_margin_values"])) if stats["score_margin_values"] else None
    stats["score_margin_p95"] = float(np.percentile(stats["score_margin_values"], 95)) if stats["score_margin_values"] else None
    stats.pop("score_margin_values", None)
    stats["status"] = "SUCCESS" if stats["unresolved_pixels"] == 0 else "HARD_STOP_UNRESOLVED_LABELS"

    weight_paths = []
    with rasterio.open(label_path) as label_src:
        labels = label_src.read(1)
        for index, mask in enumerate(masks):
            region = labels == index
            inside = distance_transform_edt(region)
            outside = distance_transform_edt(~region)
            signed = inside - outside
            weight_path = output_dir / f"weight_scene_{index:03d}.tif"; weight_paths.append(weight_path)
            weight_profile = profile.copy(); weight_profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
            with rasterio.open(weight_path, "w", **weight_profile) as dst:
                for r0 in range(0, shape[0], 512):
                    r1 = min(shape[0], r0 + 512); values = signed[r0:r1]
                    active = mask[r0:r1] & (values > -cfg.blend_half_width)
                    weights = np.zeros(values.shape, dtype=np.float32)
                    weights[active] = np.where(values[active] >= cfg.blend_half_width, 1.0, 0.5 * (1.0 + np.cos(np.pi * np.abs(values[active]) / cfg.blend_half_width)))
                    weights[~mask[r0:r1]] = np.nan
                    dst.write(weights, 1, window=rasterio.windows.Window(0, r0, shape[1], r1 - r0))
            del region, inside, outside, signed
    mosaic_path = output_dir / "mosaic.tif"
    _stream_weighted_mosaic(corrected_scene_paths, weight_paths, mosaic_path, grid)
    result = {"status": "SUCCESS" if refined_count else "NO_REFINED_PAIRS", "refined_pair_count": refined_count,
              "refinement_rows": refinement_rows, "labels": stats, "shared_refine_function": getattr(refine, "__name__", str(refine)),
              "weight_paths": [str(path) for path in weight_paths],
              "outputs": {"labels": str(label_path), "mosaic": str(mosaic_path)}}
    (output_dir / "end_to_end_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return result


def _load_task15_initial_seams(task15_root: Path, grid: Mapping[str, Any], scene_count: int) -> list[dict[str, Any]]:
    """Restore Stage07 initial seams into the canonical pixel grid."""
    from src.seam_local.seam import SeamResult
    pair_root = task15_root / "stages/07_pairwise_seam_local/pairs"
    transform = grid_transform(grid)
    records = []
    for row in _json(task15_root / "stages/07_pairwise_seam_local/pairwise_results.json") if (task15_root / "stages/07_pairwise_seam_local/pairwise_results.json").is_file() else []:
        path = pair_root / str(row.get("pair_id")) / "seam_initial.geojson"
        if not path.is_file():
            continue
        payload = _json(path); feature = payload["features"][0]; orientation = feature["properties"]["orientation"]
        pixels = []
        for x, y in feature["geometry"]["coordinates"]:
            col, line = (~transform) * (float(x), float(y))
            pixels.append((int(round(line - 0.5)), int(round(col - 0.5))))
        path_array = np.asarray(pixels, dtype=np.int32)
        if orientation == "vertical":
            centers = np.interp(np.arange(int(grid["height"])), path_array[:, 0], path_array[:, 1])
            path_array = np.column_stack((np.arange(int(grid["height"])), np.rint(centers).astype(np.int32)))
        else:
            centers = np.interp(np.arange(int(grid["width"])), path_array[:, 1], path_array[:, 0])
            path_array = np.column_stack((np.rint(centers).astype(np.int32), np.arange(int(grid["width"]))))
        records.append({**row, "initial_seam": SeamResult(orientation, path_array, None, None, None, "OK", "task15_initial")})
    return records


def run_task16(task15_root: str | Path, output_root: str | Path, *, protocol_path: str | Path | None = None, resume: bool = False, task16_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Execute the Task16 fixed-geometry comparison and write the full artifact tree."""

    task15_root = Path(task15_root)
    output_root = Path(output_root)
    task_params = validate_task16_config(task16_config) if task16_config is not None else dict(TASK16_PARAMS)
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError(f"Task16 output directory is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("00_protocol", "01_baseline_bagrn_weighted", "02_volrn_solver", "03_volrn_corrected_scenes", "04_volrn_weighted_mosaic", "04_strict_local_ablation", "05_volrn_end_to_end", "05_radiometric_metrics", "06_task15_v2_reference", "07_comparison_tables", "08_figures", "09_report"):
        (output_root / name).mkdir(exist_ok=True)

    frozen = _load_frozen_inputs(task15_root)
    grid = frozen["grid"]
    if protocol_path is None:
        protocol_path = Path(__file__).resolve().parents[1] / "configs/task10d_metric_protocol.json"
    protocol_path = Path(protocol_path)
    metric_protocol = load_task10d_metric_protocol(protocol_path)
    provenance = {
        "task": "Task16",
        "status": "PASS",
        "geometry_mutable": False,
        "source_task15_root": str(task15_root),
        "scene_count": EXPECTED_SCENE_COUNT,
        "scene_ids": frozen["scene_ids"],
        "canonical_grid": grid,
        "union_support_historical_reference": HISTORICAL_EXPECTED_SUPPORT,
        "union_support_measured": int(frozen["union_support"]),
        "volrn_parameters": task_params,
        "rho_mode": "fixed",
        "preconditioner": "canonical Task10D VOLRN Jacobi preconditioner",
        "task10d_protocol_sha256": _sha256(protocol_path),
        "input_sha256": {str(path.relative_to(task15_root)): _sha256(path) for path in frozen["provenance_files"] if path.is_file() and path.suffix.lower() in {".json", ".tif"}},
    }
    (output_root / "00_protocol/provenance_manifest.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    (output_root / "00_protocol/task10d_metric_protocol.json").write_text(json.dumps(metric_protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    canonical_transform = grid_transform(grid)
    full_shape = (int(grid["height"]), int(grid["width"]))
    masks = frozen["masks"]

    # A: rebuild from persisted Stage06 scenes with the frozen weighted formula.
    baseline_dir = output_root / "01_baseline_bagrn_weighted"
    baseline_mosaic = baseline_dir / "bagrn_weighted.tif"
    baseline_complete = all((baseline_dir / name).is_file() for name in ("bagrn_weighted.tif", "valid_mask.tif", "contributor_count.tif", "weight_sum.tif"))
    baseline_arrays = None
    if resume and baseline_complete:
        baseline_runtime = 0.0
        baseline_arrays = None
    else:
        t0 = time.perf_counter()
        _stream_weighted_mosaic(frozen["scene_paths"], frozen["weight_paths"], baseline_mosaic, grid)
        baseline_runtime = time.perf_counter() - t0
    baseline_support = _support_products(baseline_dir, masks, grid)
    if baseline_support["union_support"] != frozen["union_support"]:
        raise ProvenanceMismatch("HARD_STOP_BASELINE_SUPPORT_MISMATCH")
    union_mask = np.zeros(full_shape, dtype=bool)
    for mask in masks:
        union_mask |= mask
    baseline_preview = _write_preview(baseline_dir / "preview.png", baseline_dir / "bagrn_weighted.tif", union_mask)
    if baseline_arrays is not None:
        del baseline_arrays

    # B: VOLRN is the only numerical stage executed by Task16.
    local_arrays: list[np.ndarray] = []
    local_masks: list[np.ndarray] = []
    local_transforms: list[Any] = []
    local_bounds: list[Any] = []
    crops: list[tuple[int, int, int, int]] = []
    for scene_path, mask in zip(frozen["scene_paths"], masks):
        array, _, transform = _read_scene(scene_path, frozen["mask_paths"][len(local_arrays)])
        local_array, local_mask, local_transform, crop = _crop_scene(array, mask, transform)
        local_arrays.append(local_array)
        local_masks.append(local_mask)
        local_transforms.append(local_transform)
        local_bounds.append(tuple(float(x) for x in array_bounds(local_array.shape[1], local_array.shape[2], local_transform)))
        crops.append(crop)
    solver_dir = output_root / "02_volrn_solver"
    corrected_dir = output_root / "03_volrn_corrected_scenes"
    corrected_dir.mkdir(exist_ok=True)
    corrected_scene_paths = [corrected_dir / f"scene_{index:03d}.tif" for index in range(EXPECTED_SCENE_COUNT)]
    reuse_volrn = bool(resume and (solver_dir / "solver_diagnostics.json").is_file() and (solver_dir / "volrn_coefficients.npz").is_file() and all(path.is_file() for path in corrected_scene_paths) and (corrected_dir / "application_audit.json").is_file())
    if reuse_volrn:
        stored = _json(solver_dir / "solver_diagnostics.json")
        solver_diag = stored["diagnostics"]
        formal_status = str(stored["formal_status"])
        band_statuses = list(stored.get("band_statuses", []))
        with np.load(solver_dir / "volrn_coefficients.npz") as stored_coefficients:
            coefficients = stored_coefficients["block_coefficients"]
        corrected_local = []
        solver_runtime = 0.0
    else:
        solver_start = time.perf_counter()
        corrected_local, coefficients, solver_diag = volrn_normalize(
            local_arrays, local_transforms, local_bounds, [None] * EXPECTED_SCENE_COUNT,
            block_size_pixels=task_params["block_size_pixels"], lambda_param=task_params["lambda"], rho=task_params["rho"], max_iter=task_params["max_iter"], tol=task_params["tol"],
            verbose=False, return_diagnostics=True, valid_masks=local_masks, use_preconditioner=True, adaptive_rho=False,
        )
        solver_runtime = time.perf_counter() - solver_start
        band_statuses = [classify_volrn_status(diag) for diag in solver_diag.get("band_solver_diagnostics", [])]
        formal_status = "NUMERICAL_INVALID_ITER200" if any(status == "NUMERICAL_INVALID_ITER200" for status in band_statuses) else ("PASS_STRICT_CONVERGED" if all(status == "PASS_STRICT_CONVERGED" for status in band_statuses) else ("COMPLETED_FINITE_NONCONVERGED" if all(status == "COMPLETED_FINITE_NONCONVERGED" for status in band_statuses) else "ITERATION_COUNT_MISMATCH"))
        _save_history(solver_dir, solver_diag)
        (solver_dir / "solver_diagnostics.json").write_text(json.dumps({"formal_status": formal_status, "band_statuses": band_statuses, "diagnostics": solver_diag}, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        np.savez_compressed(solver_dir / "volrn_coefficients.npz", block_coefficients=coefficients)

    corrected_paths: list[Path] = []
    if reuse_volrn:
        corrected_paths = corrected_scene_paths
    else:
        for index, (local, crop) in enumerate(zip(corrected_local, crops)):
            full = np.full(full_shape, np.nan, dtype=np.float32)
            r0, r1, c0, c1 = crop
            full[r0:r1, c0:c1] = local[0].astype(np.float32)
            out_path = corrected_dir / f"scene_{index:03d}.tif"
            _save_full_scene(out_path, full, grid)
            corrected_paths.append(out_path)
        for index, mask in enumerate(masks):
            _write_single_band(corrected_dir / "input_valid_masks" / f"scene_{index:03d}.tif", mask.astype(np.uint8), canonical_transform, str(grid["crs"]), 0, "uint8")
    finite_outputs = _finite_scene_outputs(corrected_paths, masks)
    from src.multiscene_sift.radiometric_runner import build_volrn_application_audit
    if reuse_volrn:
        application_audit = _json(corrected_dir / "application_audit.json")
        coeff_diag = _json(solver_dir / "coefficient_diagnostics.json")
    else:
        application_audit = build_volrn_application_audit(local_arrays, corrected_local, local_masks, coefficients)
        (corrected_dir / "application_audit.json").write_text(json.dumps(application_audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        coeff_diag = _coefficient_diagnostics(coefficients, application_audit, solver_diag)
        (solver_dir / "coefficient_diagnostics.json").write_text(json.dumps(coeff_diag, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    # B weighted mosaic from the same grid/mode; no label/seam stage is introduced.
    mosaic_dir = output_root / "04_volrn_weighted_mosaic"
    mosaic_start = time.perf_counter()
    # The frozen formula is identical to ``create_mosaic(mode='weighted')``;
    # Task15's persisted distance-weight maps let this replay avoid keeping
    # thirteen projected float64 canvases in memory simultaneously.
    import gc
    del local_arrays, local_masks, local_transforms, local_bounds, corrected_local
    gc.collect()

    strict_dir = output_root / "04_strict_local_ablation"
    if all(path.is_file() for path in frozen["ours_scene_paths"]):
        strict_ablation = run_strict_local_ablation(
            frozen["scene_paths"], frozen["ours_scene_paths"], corrected_paths,
            frozen["v1_weight_paths"], masks, frozen["scene_ids"], grid, strict_dir,
        )
    else:
        strict_ablation = {"status": "HARD_STOP_MISSING_TASK15_OURS_CORRECTED_SCENES"}
        (strict_dir / "strict_ablation_metrics.json").write_text(json.dumps(strict_ablation, indent=2), encoding="utf-8")

    initial_records = _load_task15_initial_seams(task15_root, grid, EXPECTED_SCENE_COUNT)
    e2e_dir = output_root / "05_volrn_end_to_end"
    if initial_records:
        # Both routes use the same injected function and seam/refinement
        # contract; only their corrected scene inputs differ.
        ours_e2e = run_volrn_end_to_end(frozen["ours_scene_paths"], masks, initial_records, grid, e2e_dir / "ours") if all(path.is_file() for path in frozen["ours_scene_paths"]) else {"status": "MISSING_OURS_SCENES"}
        volrn_e2e = run_volrn_end_to_end(corrected_paths, masks, initial_records, grid, e2e_dir / "volrn")
        end_to_end = {"ours": ours_e2e, "volrn": volrn_e2e, "shared_refine_function": "refine_seam_after_correction"}
    else:
        end_to_end = {"status": "HARD_STOP_MISSING_TASK15_INITIAL_SEAMS", "shared_refine_function": "refine_seam_after_correction"}
    (e2e_dir / "end_to_end_manifest.json").write_text(json.dumps(end_to_end, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    end_to_end_metrics = {"status": "NOT_MEASURED", "comparison": []}
    if isinstance(end_to_end.get("ours"), Mapping) and isinstance(end_to_end.get("volrn"), Mapping) and end_to_end["ours"].get("status") == "SUCCESS" and end_to_end["volrn"].get("status") == "SUCCESS":
        ours_weight_paths = [Path(path) for path in end_to_end["ours"]["weight_paths"]]
        volrn_weight_paths = [Path(path) for path in end_to_end["volrn"]["weight_paths"]]
        ours_metrics = _stream_task10d_metrics(frozen["ours_scene_paths"], masks, ours_weight_paths, frozen["scene_ids"])
        volrn_metrics = _stream_task10d_metrics(corrected_paths, masks, volrn_weight_paths, frozen["scene_ids"])
        ours_structure = _stream_structure_summary(frozen["scene_paths"], frozen["ours_scene_paths"], masks)
        volrn_structure = _stream_structure_summary(frozen["scene_paths"], corrected_paths, masks)
        from scripts.run_task14a_resume_13 import _transition_metrics
        fixed_ours_rows, fixed_ours = _transition_metrics(frozen["v1_weight_paths"], frozen["scene_paths"], frozen["ours_scene_paths"], full_shape[0], full_shape[1])
        fixed_volrn_rows, fixed_volrn = _transition_metrics(frozen["v1_weight_paths"], frozen["scene_paths"], corrected_paths, full_shape[0], full_shape[1])
        metric_map = [("MAMD", ("mamd", "weighted_mean")), ("MSDD", ("msdd", "weighted_mean")), ("RDD", ("rdd", "weighted_mean")), ("Local MAMD median", ("local_mamd", "median")), ("Local RDD median", ("local_rdd", "median"))]
        comparison = [{"metric": name, "Ours_E2E": _metric_scalar(ours_metrics, path), "VOLRN_E2E": _metric_scalar(volrn_metrics, path)} for name, path in metric_map]
        comparison.extend([
            {"metric": "Structure CGL degrees", "Ours_E2E": ours_structure.get("cgl_deg"), "VOLRN_E2E": volrn_structure.get("cgl_deg")},
            {"metric": "Gradient NCC mean", "Ours_E2E": ours_structure.get("mean"), "VOLRN_E2E": volrn_structure.get("mean")},
            {"metric": "Gradient orientation cosine mean", "Ours_E2E": ours_structure.get("orientation_cosine"), "VOLRN_E2E": volrn_structure.get("orientation_cosine")},
            {"metric": "Fixed V1 transition weighted MAE", "Ours_E2E": fixed_ours.get("candidate_weighted_mae"), "VOLRN_E2E": fixed_volrn.get("candidate_weighted_mae")},
            {"metric": "Fixed V1 transition weighted RDD", "Ours_E2E": fixed_ours.get("candidate_weighted_rdd"), "VOLRN_E2E": fixed_volrn.get("candidate_weighted_rdd")},
        ])
        end_to_end_metrics = {"status": "SUCCESS", "support_definition": "V1 initial seam / V1 transition zone", "ours": ours_metrics, "volrn": volrn_metrics, "ours_structure": ours_structure, "volrn_structure": volrn_structure, "fixed_support": {"ours": {"rows": fixed_ours_rows, "metrics": fixed_ours}, "volrn": {"rows": fixed_volrn_rows, "metrics": fixed_volrn}}, "comparison": comparison}
        (e2e_dir / "end_to_end_metrics.json").write_text(json.dumps(end_to_end_metrics, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        (e2e_dir / "fixed_support_metrics.json").write_text(json.dumps({"support_definition": "V1 initial seam / V1 transition zone", "ours": end_to_end_metrics["fixed_support"]["ours"], "volrn": end_to_end_metrics["fixed_support"]["volrn"]}, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    else:
        (e2e_dir / "end_to_end_metrics.json").write_text(json.dumps(end_to_end_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    _stream_weighted_mosaic(
        corrected_paths, frozen["weight_paths"], mosaic_dir / "bagrn_volrn_iter200_weighted.tif", grid
    )
    mosaic_runtime = time.perf_counter() - mosaic_start
    b_support = _support_products(mosaic_dir, masks, grid)
    b_preview = _write_preview(mosaic_dir / "preview.png", mosaic_dir / "bagrn_volrn_iter200_weighted.tif", union_mask, baseline_preview["lower"], baseline_preview["upper"])
    if b_support["union_support"] != frozen["union_support"]:
        raise ProvenanceMismatch("HARD_STOP_BASELINE_SUPPORT_MISMATCH: B support differs")

    # A/B Task10D metrics use the same 13-scene masks and canonical tiles.
    metrics_dir = output_root / "05_radiometric_metrics"
    a_metrics = _stream_task10d_metrics(frozen["scene_paths"], masks, frozen["weight_paths"], frozen["scene_ids"])
    b_metrics = _stream_task10d_metrics(corrected_paths, masks, frozen["weight_paths"], frozen["scene_ids"])
    a_diag = {"metric_execution": "streaming_pair_tile", "changed_pixel_fraction": []}
    b_diag = {"metric_execution": "streaming_pair_tile", "changed_pixel_fraction": []}
    a_metrics["cgl_rad"] = {"value_rad": 0.0, "value_deg": 0.0, "per_scene": [], "status": "IDENTICAL_BASELINE"}
    b_cgl_rows = []
    for index, (before, after, mask) in enumerate(zip(frozen["scene_paths"], corrected_paths, masks)):
        b_cgl_rows.append({"scene_id": frozen["scene_ids"][index], **_stream_cgl(before, after, mask)})
    cgl_values = [row["cgl_rad"] for row in b_cgl_rows if row.get("cgl_rad") is not None]
    b_cgl = float(np.mean(cgl_values)) if cgl_values else None
    b_metrics["cgl_rad"] = {"value_rad": b_cgl, "value_deg": float(np.rad2deg(b_cgl)) if b_cgl is not None else None, "per_scene": b_cgl_rows, "status": "STREAMING"}
    structural = _stream_structure_summary(frozen["scene_paths"], corrected_paths, masks)
    ncc_values = [row["gradient_magnitude_ncc"] for row in structural["per_scene"] if isinstance(row.get("gradient_magnitude_ncc"), (int, float)) and np.isfinite(row["gradient_magnitude_ncc"])]
    structural["median"] = float(np.median(ncc_values)) if ncc_values else None
    structural["worst"] = float(min(ncc_values)) if ncc_values else None
    (metrics_dir / "task10d_metrics.json").write_text(json.dumps({"BAGRN_weighted": a_metrics, "BAGRN_VOLRN_iter200_weighted": b_metrics, "diagnostics_A": a_diag, "diagnostics_B": b_diag, "structure_preservation": structural, "finite_outputs": finite_outputs}, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    a_structural = _stream_structure_summary(frozen["scene_paths"], frozen["scene_paths"], masks)
    task10d_rows = _task10d_rows(a_metrics, b_metrics, structural, frozen["union_support"], a_structural=a_structural)
    _write_csv(metrics_dir / "radiometric_ablation.csv", task10d_rows, ["metric", "baseline", "candidate", "absolute_change", "relative_change_percent"])

    # Reuse the exact Task15 source-label boundary definition for A/B.
    from scripts.run_task14a_resume_13 import _boundary_metrics
    label_path = task15_root / "stages/08_multiscene_labeling/v1/source_label_map.tif"
    if not label_path.is_file():
        label_path = task15_root / "stages/08_multiscene_labeling/source_label_map.tif"
    with rasterio.open(label_path) as label_src:
        labels = label_src.read(1)
    boundary_rows, boundary_ab = _boundary_metrics(labels, frozen["scene_paths"], corrected_paths, full_shape[0], full_shape[1])
    stage11 = _json(task15_root / "stages/11_metrics/metrics_summary.json")
    task15_boundary = stage11.get("boundary_metrics", {})
    task15_structural = stage11.get("structural", [])
    c_min_ncc = min((float(row["gradient_magnitude_ncc"]) for row in task15_structural if isinstance(row, Mapping) and isinstance(row.get("gradient_magnitude_ncc"), (int, float))), default=None)
    _write_csv(metrics_dir / "boundary_metrics_A_vs_B.csv", boundary_rows)
    (metrics_dir / "boundary_metrics_summary.json").write_text(json.dumps(boundary_ab, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    reference_dir = output_root / "06_task15_v2_reference"
    shutil.copy2(frozen["v2_path"], reference_dir / "v2_local_corrected_multiscene.tif")
    (reference_dir / "reference_metrics.json").write_text(json.dumps({"source": str(frozen["v2_path"]), "source_sha256": _sha256(frozen["v2_path"]), "stage11_boundary_metrics": task15_boundary, "stage11_structural": task15_structural}, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    legacy_end_rows = []
    def c_value(key: str) -> float | None:
        value = task15_boundary.get(key)
        return float(value) if isinstance(value, (int, float)) else None
    end_map = [
        ("boundary weighted MAE", boundary_ab.get("bagrn_weighted_mae"), boundary_ab.get("v2_weighted_mae"), c_value("v2_weighted_mae")),
        ("boundary weighted RMSE", float(np.average([r["bagrn_rmse"] for r in boundary_rows], weights=[r["pixels"] for r in boundary_rows])) if boundary_rows else None, float(np.average([r["v2_rmse"] for r in boundary_rows], weights=[r["pixels"] for r in boundary_rows])) if boundary_rows else None, None),
        ("boundary weighted RDD", boundary_ab.get("bagrn_weighted_rdd"), boundary_ab.get("v2_weighted_rdd"), c_value("v2_weighted_rdd")),
        ("boundary median MAE", boundary_ab.get("median_bagrn_mae"), boundary_ab.get("median_v2_mae"), c_value("median_v2_mae")),
        ("boundary median RDD", boundary_ab.get("median_bagrn_rdd"), boundary_ab.get("median_v2_rdd"), c_value("median_v2_rdd")),
        ("boundary worst MAE", max((r["bagrn_mae"] for r in boundary_rows), default=None), max((r["v2_mae"] for r in boundary_rows), default=None), None),
        ("minimum scene gradient NCC", 1.0, structural["min"], c_min_ncc),
        ("union support", float(frozen["union_support"]), float(frozen["union_support"]), float(frozen["union_support"])),
        ("runtime total B seconds", None, float(solver_runtime + mosaic_runtime), None),
    ]
    for metric, a_value, b_value, c_value_ in end_map:
        legacy_end_rows.append({"metric": metric, "BAGRN_weighted": a_value, "BAGRN_VOLRN_iter200_weighted": b_value, "Task15_V2": c_value_})
    _write_csv(output_root / "07_comparison_tables/legacy_weighted_comparison.csv", legacy_end_rows)
    endpoint_rows = end_to_end_metrics.get("comparison", []) if isinstance(end_to_end_metrics, Mapping) else []
    if endpoint_rows:
        _write_csv(output_root / "07_comparison_tables/end_to_end_comparison.csv", endpoint_rows, ["metric", "Ours_E2E", "VOLRN_E2E"])
    else:
        _write_csv(output_root / "07_comparison_tables/end_to_end_comparison.csv", [{"metric": "status", "Ours_E2E": end_to_end_metrics.get("status"), "VOLRN_E2E": end_to_end_metrics.get("status")}], ["metric", "Ours_E2E", "VOLRN_E2E"])
    _write_csv(output_root / "07_comparison_tables/radiometric_ablation.csv", task10d_rows, ["metric", "baseline", "candidate", "absolute_change", "relative_change_percent"])

    figures_dir = output_root / "08_figures"
    _plot_solver_history(figures_dir, solver_diag)
    _three_method_figure(figures_dir / "three_method_comparison.png", [baseline_dir / "bagrn_weighted.tif", mosaic_dir / "bagrn_volrn_iter200_weighted.tif", reference_dir / "v2_local_corrected_multiscene.tif"], ["A BAGRN + weighted", "B BAGRN + VOLRN iter200 + weighted", "C Task15 V2"])

    total_runtime = baseline_runtime + solver_runtime + mosaic_runtime
    summary = {
        "task": "Task16",
        "status": formal_status if formal_status != "PASS_STRICT_CONVERGED" else "PASS_STRICT_CONVERGED",
        "scene_count": EXPECTED_SCENE_COUNT,
        "support": {"historical_reference": HISTORICAL_EXPECTED_SUPPORT, "measured": frozen["union_support"], "A": baseline_support, "B": b_support},
        "volrn": {"formal_status": formal_status, "band_statuses": band_statuses, "runtime_sec": solver_runtime, "diagnostics": solver_diag, "coefficient_diagnostics": coeff_diag},
        "runtime_sec": {"baseline_mosaic": baseline_runtime, "solver": solver_runtime, "volrn_mosaic": mosaic_runtime, "total_task16": total_runtime, "ram_vram": "NOT_MEASURED"},
        "structure_preservation": structural,
        "task10d": {"A": a_metrics, "B": b_metrics},
        "strict_local_ablation": strict_ablation,
        "volrn_end_to_end": end_to_end,
        "end_to_end_metrics": end_to_end_metrics,
        "boundary_A_vs_B": boundary_ab,
        "task15_v2_reference": {"boundary_metrics": task15_boundary, "minimum_gradient_ncc": c_min_ncc},
        "numerical_validity": finite_outputs,
    }
    (metrics_dir / "task16_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    report = _build_report(summary, provenance, output_root)
    (output_root / "09_report/TASK16_13SCENE_VOLRN_COMPARISON_REPORT.md").write_text(report, encoding="utf-8")
    (output_root / "task16_status.json").write_text(json.dumps({"status": summary["status"], "summary": "05_radiometric_metrics/task16_summary.json", "report": "09_report/TASK16_13SCENE_VOLRN_COMPARISON_REPORT.md"}, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _build_report(summary: Mapping[str, Any], provenance: Mapping[str, Any], output_root: Path) -> str:
    volrn = summary["volrn"]
    structure = summary["structure_preservation"]
    boundary = summary["boundary_A_vs_B"]
    task15 = summary["task15_v2_reference"]["boundary_metrics"]
    questions = [
        f"1. Same Task15 geometry/BAGRN: **yes**; fixed scene count={summary['scene_count']} and grid/support were replayed from Stage06.",
        "2. Exact VOLRN solver: canonical Task10D ADMM VOLRN implementation with fixed-rho Jacobi-preconditioned CG.",
        f"3. Frozen parameters: `{json.dumps(TASK16_PARAMS, ensure_ascii=False)}`.",
        f"4. Exactly 200 iterations: **{all(int(d.get('iterations', -1)) == 200 for d in volrn['diagnostics'].get('band_solver_diagnostics', []))}**; actual history is preserved.",
        f"5. Strict convergence: **{volrn['formal_status'] == 'PASS_STRICT_CONVERGED'}**; formal status is `{volrn['formal_status']}`.",
        "6. Residual/tolerance: every recorded iteration contains primal/dual residuals, tolerances, and ratios in `02_volrn_solver/solver_history.csv`.",
        f"7. CG failure: **{any(bool(d.get('cg_failed', False)) for d in volrn['diagnostics'].get('band_solver_diagnostics', []))}**.",
        f"8. Finite solver/output state: numerical-invalid valid pixels={summary['numerical_validity']['numerical_invalid_pixels']}.",
        f"9. Block/pair/variable counts: {volrn['coefficient_diagnostics']['n_blocks']} / {volrn['coefficient_diagnostics']['n_pairs']} / {volrn['coefficient_diagnostics']['n_variables']}.",
        f"10. Coefficient ranges: a={volrn['coefficient_diagnostics']['a']}; b={volrn['coefficient_diagnostics']['b']}.",
        f"11. Support equality: A={summary['support']['A']['union_support']}, B={summary['support']['B']['union_support']}, measured frozen support={summary['support']['measured']} (historical reference={summary['support']['historical_reference']}).",
        "12. BAGRN→VOLRN MAMD/MSDD/RDD: see `05_radiometric_metrics/radiometric_ablation.csv`.",
        "13. Local MAMD/RDD: see the same ablation table and full JSON diagnostics.",
        "14. Seam MAE/RMSE/RDD: see `05_radiometric_metrics/boundary_metrics_summary.json` and the ablation table.",
        f"15. Structure preservation: min/median/mean NCC={structure['min']} / {structure['median']} / {structure['mean']}; CGL/orientation={structure.get('cgl_deg')} / {structure.get('orientation_cosine')}; threshold review={structure['min'] is not None and structure['min'] >= 0.99}.",
        f"16. Task15 V2 boundary final MAE/RDD: {task15.get('v2_weighted_mae')} / {task15.get('v2_weighted_rdd')}.",
        f"17. VOLRN weighted boundary MAE/RDD: {boundary.get('v2_weighted_mae')} / {boundary.get('v2_weighted_rdd')}.",
        f"18. Runtime: solver={summary['runtime_sec']['solver']:.3f}s; VOLRN mosaic={summary['runtime_sec']['volrn_mosaic']:.3f}s; total={summary['runtime_sec']['total_task16']:.3f}s; RAM/VRAM=NOT_MEASURED.",
        f"19. Final strict vs finite-nonconverged status: `{volrn['formal_status']}`.",
        "20. Paper wording: this report does not claim paper CD/GL; it reports the frozen Task10D metric definitions.",
        "21. Strict ablation: A and B use identical frozen geometry, scene order, masks, canonical grid, and weighted-feather mode.",
        f"22. Ours E2E vs VOLRN E2E: status={summary.get('end_to_end_metrics', {}).get('status')}; see `05_volrn_end_to_end/end_to_end_metrics.json` and `07_comparison_tables/end_to_end_comparison.csv`.",
        f"23. Numerical invalidity: {'none' if summary['numerical_validity']['numerical_invalid_pixels'] == 0 else 'present; formal B result requires review'}.",
    ]
    return "\n".join([
        "# Task16 — 13-Scene BAGRN + VOLRN Weighted-Feather Radiometric Comparison", "",
        f"Decision/status: **{summary['status']}**.", "",
        "The fixed A/B comparison consumes Task15 Stage06 BAGRN scenes as read-only inputs. The separate endpoint section explicitly reruns shared seam refinement, source-side labeling, weights, and mosaic on Ours/VOLRN corrected scenes.", "",
        "## Final report answers", "", *[f"- {item}" for item in questions], "",
        "## Artifact map", "",
        "- `00_protocol/`: provenance and frozen Task10D protocol.",
        "- `01_baseline_bagrn_weighted/`: rebuilt A mosaic and support products.",
        "- `02_volrn_solver/`: coefficients, complete history, convergence plots, and diagnostics.",
        "- `03_volrn_corrected_scenes/`: B corrected scenes and application audit.",
        "- `04_volrn_weighted_mosaic/`: B mosaic and support products.",
        "- `04_strict_local_ablation/`: A0/A1/A2 using the identical V1 weight layout and actual structure metrics.",
        "- `05_volrn_end_to_end/`: Ours/VOLRN shared-refinement endpoint artifacts, fixed-support metrics, and endpoint comparison.",
        "- `05_radiometric_metrics/`: Task10D metrics, boundary diagnostics, and summary.",
        "- `06_task15_v2_reference/`: copied read-only V2 reference and source metrics.",
        "- `07_comparison_tables/` and `08_figures/`: comparison tables and figures.",
    ])
