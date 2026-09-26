"""Fixed-geometry B9 radiometric experiment runner for Task 10."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine, array_bounds
from scipy.ndimage import distance_transform_edt

from scripts.run_b9_weighted_mosaic import (
    _load_global_transforms,
    _load_sources,
    _project_scene,
)
from src.bagrn import bagrn_normalize
from src.metrics import compute_all
from src.mosaic import create_mosaic
from src.multiscene_sift.band_geometry import apply_world_correction_to_transform
from src.multiscene_sift.mosaic_protocol import grid_transform
from src.multiscene_sift.radiometric_protocol import (
    GEOMETRY_SPECS,
    RADIOMETRIC_METHODS,
    build_task10_config,
    write_run_metadata,
)
from src.overlap import detect_multi_overlap
from src.registration_benchmark.mosaic_diagnostics import (
    build_seam_zone_mask,
    compute_overlap_metrics,
)
from src.volrn import volrn_normalize


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_single_band(path: Path, data: np.ndarray, transform, crs, nodata, dtype: str) -> None:
    profile = {
        "driver": "GTiff", "height": data.shape[0], "width": data.shape[1],
        "count": 1, "dtype": dtype, "crs": crs, "transform": transform,
        "nodata": nodata, "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.asarray(data).astype(dtype, copy=False), 1)


def _write_preview(path: Path, mosaic_path: Path, valid_mask: np.ndarray) -> None:
    with rasterio.open(mosaic_path) as src:
        image = src.read(1).astype(np.float32)
    values = image[valid_mask & np.isfinite(image)]
    if values.size:
        lower, upper = np.percentile(values, [2.0, 98.0])
        if upper <= lower:
            lower, upper = float(values.min()), float(values.max())
    else:
        lower, upper = 0.0, 1.0
    stretched = np.clip((image - lower) / max(float(upper - lower), 1e-12) * 255.0, 0, 255)
    stretched[~valid_mask] = 0
    Image.fromarray(stretched.astype(np.uint8), mode="L").save(path)


def _histogram_distance(values_a: np.ndarray, values_b: np.ndarray, bins: int = 256) -> float:
    if values_a.size < 2 or values_b.size < 2:
        return 0.0
    lower = min(float(values_a.min()), float(values_b.min()))
    upper = max(float(values_a.max()), float(values_b.max()))
    if upper <= lower:
        return 0.0
    hist_a, _ = np.histogram(values_a, bins=bins, range=(lower, upper))
    hist_b, _ = np.histogram(values_b, bins=bins, range=(lower, upper))
    return float(0.5 * np.abs(hist_a / hist_a.sum() - hist_b / hist_b.sum()).sum())


def _pair_metrics(
    before: list[np.ndarray],
    after: list[np.ndarray],
    valid_masks: list[np.ndarray],
) -> dict:
    def _crop(mask: np.ndarray, left: np.ndarray, right: np.ndarray):
        rows, cols = np.where(mask)
        if rows.size == 0:
            return left[:0, :0], right[:0, :0], mask[:0, :0]
        r0, r1 = max(0, int(rows.min()) - 1), min(mask.shape[0], int(rows.max()) + 2)
        c0, c1 = max(0, int(cols.min()) - 1), min(mask.shape[1], int(cols.max()) + 2)
        return left[r0:r1, c0:c1], right[r0:r1, c0:c1], mask[r0:r1, c0:c1]

    rows = []
    seam_rows = []
    weights = [distance_transform_edt(mask).astype(np.float64) for mask in valid_masks]
    for i, j in itertools.combinations(range(len(before)), 2):
        shared = valid_masks[i] & valid_masks[j]
        metric_a, metric_b, metric_mask = _crop(shared, after[i][0], after[j][0])
        metric = compute_overlap_metrics(metric_a, metric_b, metric_mask, min_valid_pixels=1)
        raw_shared = valid_masks[i] & valid_masks[j]
        a = before[i][0][raw_shared].astype(np.float64)
        b = before[j][0][raw_shared].astype(np.float64)
        if a.size:
            histogram = {
                "mean_abs_difference": float(abs(a.mean() - b.mean())),
                "std_abs_difference": float(abs(a.std() - b.std())),
                "histogram_tv_distance": _histogram_distance(a, b),
            }
        else:
            histogram = {
                "mean_abs_difference": None,
                "std_abs_difference": None,
                "histogram_tv_distance": None,
            }
        rows.append({
            "scene_i": i,
            "scene_j": j,
            "valid_overlap_pixels": int(shared.sum()),
            "mae": metric["intensity_MAE"],
            "rmse": metric["intensity_RMSE"],
            "bias": metric["mean_bias"],
            "histogram": histogram,
        })
        seam_zone = build_seam_zone_mask(valid_masks[i], valid_masks[j], weights[i], weights[j])
        seam_a, seam_b, seam_mask = _crop(seam_zone, after[i][0], after[j][0])
        seam_metric = compute_overlap_metrics(seam_a, seam_b, seam_mask, min_valid_pixels=1)
        seam_rows.append({
            "scene_i": i,
            "scene_j": j,
            "seam_zone_pixels": int(seam_zone.sum()),
            "seam_gradient": seam_metric["gradient_magnitude_NCC"],
            "seam_mae": seam_metric["intensity_MAE"],
            "seam_rmse": seam_metric["intensity_RMSE"],
            "seam_bias": seam_metric["mean_bias"],
        })

    def _mean(field: str, rows_: list[dict]) -> float | None:
        values = [row[field] for row in rows_ if row[field] is not None]
        return float(np.mean(values)) if values else None

    histogram_rows = [row["histogram"] for row in rows]
    summary = {
        "pairs": len(rows),
        "mae": _mean("mae", rows),
        "rmse": _mean("rmse", rows),
        "bias": _mean("bias", rows),
        "histogram_mean_abs_difference": _mean("mean_abs_difference", histogram_rows),
        "histogram_std_abs_difference": _mean("std_abs_difference", histogram_rows),
        "histogram_tv_distance": _mean("histogram_tv_distance", histogram_rows),
        "seam_gradient": _mean("seam_gradient", seam_rows),
        "seam_mae": _mean("seam_mae", seam_rows),
        "seam_rmse": _mean("seam_rmse", seam_rows),
        "seam_bias": _mean("seam_bias", seam_rows),
    }
    return {"pairs": rows, "seam_zone": seam_rows, "summary": summary}


def _registered_inputs(
    source_config: Path,
    global_run_dir: Path,
    output_grid: Mapping,
) -> tuple[list[np.ndarray], list, list, list[str], list[tuple[float, float, float, float]], list[np.ndarray]]:
    arrays, source_transforms, nodata_values, scene_ids = _load_sources(source_config)
    transforms = _load_global_transforms(global_run_dir, len(arrays))
    destination_transform = grid_transform(output_grid)
    width, height = int(output_grid["width"]), int(output_grid["height"])
    crs = str(output_grid["crs"])
    corrected = [
        apply_world_correction_to_transform(source_transforms[index], transforms[index])
        for index in range(len(arrays))
    ]
    registered = []
    valid_masks = []
    bounds = []
    for array, transform, nodata in zip(arrays, corrected, nodata_values):
        projected, valid = _project_scene(
            array, transform, nodata, crs, destination_transform, width, height
        )
        registered.append(projected[np.newaxis, :, :])
        valid_masks.append(valid)
        left, bottom, right, top = array_bounds(array.shape[0], array.shape[1], transform)
        bounds.append((float(left), float(bottom), float(right), float(top)))
    registered_transforms = [destination_transform] * len(registered)
    registered_nodata = [None] * len(registered)
    return registered, registered_transforms, registered_nodata, scene_ids, bounds, valid_masks


def _crop_registered_footprints(
    registered: list[np.ndarray],
    valid_masks: list[np.ndarray],
    shared_transform,
) -> tuple[list[np.ndarray], list, list, list[tuple[int, int, int, int]]]:
    """Keep each registered scene on its local shared-grid footprint.

    The canonical canvas is needed for the final mosaic and comparable
    diagnostics, but passing its scene-sized NaN margins into VOLRN needlessly
    multiplies the IDW work.  Local arrays remain on exactly the same north-up
    pixel lattice through translated subwindow transforms.
    """
    arrays = []
    transforms = []
    bounds = []
    slices = []
    for array, valid in zip(registered, valid_masks):
        rows, cols = np.where(valid)
        if rows.size == 0 or cols.size == 0:
            raise ValueError("registered scene has no valid pixels")
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        local_transform = shared_transform * Affine.translation(c0, r0)
        local_array = array[:, r0:r1, c0:c1].copy()
        left, bottom, right, top = array_bounds(local_array.shape[1], local_array.shape[2], local_transform)
        arrays.append(local_array)
        transforms.append(local_transform)
        bounds.append((float(left), float(bottom), float(right), float(top)))
        slices.append((r0, r1, c0, c1))
    return arrays, transforms, bounds, slices


def _expand_registered_footprints(
    local_arrays: list[np.ndarray],
    slices: list[tuple[int, int, int, int]],
    full_shape: tuple[int, int],
) -> list[np.ndarray]:
    expanded = []
    for array, (r0, r1, c0, c1) in zip(local_arrays, slices):
        full = np.full((array.shape[0], full_shape[0], full_shape[1]), np.nan, dtype=np.float64)
        full[:, r0:r1, c0:c1] = array
        expanded.append(full)
    return expanded


def run_fixed_geometry_radiometric(
    source_config: str | Path,
    global_run_dir: str | Path,
    output_grid: str | Path,
    output_dir: str | Path,
    *,
    method: str,
    geometry_run: str = "sift_mst",
    radiometric_control_idx: int = 0,
    block_size_pixels: int = 800,
    lambda_param: float = 0.1,
    rho: float = 1.0,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> dict:
    """Run one RAW/BAGRN/BAGRN_VOLRN experiment without changing geometry."""
    method = str(method).upper()
    if method not in RADIOMETRIC_METHODS:
        raise ValueError(f"unknown radiometric method: {method}")
    if geometry_run not in GEOMETRY_SPECS:
        raise ValueError(f"unknown frozen geometry run: {geometry_run}")
    source_config = Path(source_config)
    global_run_dir = Path(global_run_dir)
    output_grid = Path(output_grid)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = json.loads(output_grid.read_text(encoding="utf-8"))
    registered, transforms, nodata, scene_ids, bounds, valid_masks = _registered_inputs(
        source_config, global_run_dir, grid
    )
    processing_arrays, processing_transforms, processing_bounds, crop_slices = _crop_registered_footprints(
        registered, valid_masks, transforms[0]
    )
    processing_nodata = [None] * len(processing_arrays)
    if not 0 <= radiometric_control_idx < len(registered):
        raise ValueError("radiometric_control_idx is out of range")
    overlaps = detect_multi_overlap(processing_bounds, processing_transforms, min_pixels=100)
    if not overlaps:
        raise RuntimeError("fixed geometry has no usable radiometric overlaps")

    bagrn_result = processing_arrays
    theta_mu = np.zeros((1, len(registered)), dtype=np.float64)
    theta_sigma = np.zeros((1, len(registered)), dtype=np.float64)
    bagrn_runtime = 0.0
    volrn_result = None
    volrn_coeffs = np.empty((0,), dtype=np.float64)
    volrn_diag = {"all_converged": True, "n_blocks": 0, "n_pairs": 0}
    if method in ("BAGRN", "BAGRN_VOLRN"):
        import time

        start = time.perf_counter()
        bagrn_result, theta_mu, theta_sigma = bagrn_normalize(
            processing_arrays, processing_nodata, overlaps, control_idx=radiometric_control_idx
        )
        bagrn_runtime = time.perf_counter() - start
    final_arrays = processing_arrays if method == "RAW" else bagrn_result
    final_transforms = processing_transforms
    final_nodata = processing_nodata
    volrn_runtime = 0.0
    if method == "BAGRN_VOLRN":
        import time

        start = time.perf_counter()
        volrn_result, volrn_coeffs, volrn_diag = volrn_normalize(
            bagrn_result, processing_transforms, processing_bounds, processing_nodata,
            block_size_pixels=block_size_pixels, lambda_param=lambda_param,
            rho=rho, max_iter=max_iter, tol=tol, verbose=False,
            return_diagnostics=True,
        )
        volrn_runtime = time.perf_counter() - start
        final_arrays = volrn_result

    mosaic_path = output_dir / "mosaic.tif"
    _, mosaic_diag = create_mosaic(
        arrays=final_arrays, transforms=final_transforms, crs=str(grid["crs"]),
        nodata_values=final_nodata, output_path=str(mosaic_path),
        resolution=float(grid.get("resolution", grid["pixel_size"])), mode="weighted",
        output_transform=grid_transform(grid), output_width=int(grid["width"]),
        output_height=int(grid["height"]), return_diagnostics=True,
    )
    valid_union = np.any(np.stack(valid_masks), axis=0)
    contributor_count = np.sum(np.stack(valid_masks), axis=0).astype(np.uint8)
    weight_sum = np.zeros_like(valid_masks[0], dtype=np.float64)
    for valid in valid_masks:
        weight = distance_transform_edt(valid).astype(np.float64)
        weight[valid] += 1e-6
        weight_sum += weight
    _write_single_band(output_dir / "valid_mask.tif", valid_union.astype(np.uint8), grid_transform(grid), grid["crs"], 0, "uint8")
    _write_single_band(output_dir / "contributor_count.tif", contributor_count, grid_transform(grid), grid["crs"], 0, "uint8")
    _write_single_band(output_dir / "weight_sum.tif", weight_sum.astype(np.float32), grid_transform(grid), grid["crs"], 0.0, "float32")
    _write_preview(output_dir / "preview.png", mosaic_path, valid_union)

    metric_before = registered
    metric_after = (
        registered
        if method == "RAW"
        else _expand_registered_footprints(final_arrays, crop_slices, registered[0].shape[1:])
    )
    overlap_metrics = _pair_metrics(metric_before, metric_after, valid_masks)
    if method == "RAW":
        registered_metrics = overlap_metrics
    else:
        registered_metrics = _pair_metrics(registered, registered, valid_masks)
    if method in ("BAGRN", "BAGRN_VOLRN"):
        np.savez_compressed(output_dir / "bagrn_parameters.npz", theta_mu=theta_mu, theta_sigma=theta_sigma)
    if method == "BAGRN_VOLRN":
        np.savez_compressed(output_dir / "volrn_parameters.npz", block_coefficients=volrn_coeffs)

    root = Path(__file__).resolve().parents[2]
    config = build_task10_config(root)
    config.update({
        "source_config": str(source_config), "output_grid": str(output_grid),
        "global_root": str(global_run_dir.parent),
        "output_root": str(output_dir.parent.parent.parent),
        "volrn_params": {
            "block_size_pixels": block_size_pixels, "lambda": lambda_param,
            "rho": rho, "max_iter": max_iter, "tol": tol,
        },
    })
    geometry_source = {
        "geometry_run": geometry_run,
        **GEOMETRY_SPECS[geometry_run],
        "source_config": str(source_config),
        "source_config_sha256": _sha256(source_config),
        "global_run_dir": str(global_run_dir),
        "global_transforms_sha256": _sha256(global_run_dir / "global_transforms.json"),
        "output_grid": str(output_grid),
        "output_grid_sha256": _sha256(output_grid),
        "scene_ids": scene_ids,
        "geometry_mutable": False,
    }
    radiometric_method = {
        "method": method, "band": "B9", "radiometric_control_idx": radiometric_control_idx,
        "cloud_mask_enabled": False, "mosaic_mode": "weighted",
        "block_size_pixels": block_size_pixels, "lambda": lambda_param,
        "rho": rho, "max_iter": max_iter, "tol": tol,
    }
    write_run_metadata(
        output_dir, config, geometry_source=geometry_source,
        radiometric_method=radiometric_method,
    )
    summary = {
        "schema_version": 1, "dataset": "B9", "geometry_run": geometry_run,
        "radiometric_method": method, "scene_ids": scene_ids,
        "overlap_count": len(overlaps), "overlaps": overlaps,
        "registered_metrics": registered_metrics,
        "overlap_metrics": overlap_metrics,
        "bagrn": {"runtime_sec": bagrn_runtime, "theta_mu": theta_mu.tolist(), "theta_sigma": theta_sigma.tolist()},
        "volrn": {"runtime_sec": volrn_runtime, "diagnostics": volrn_diag},
        "convergence": {"bagrn": {"completed": method in ("BAGRN", "BAGRN_VOLRN")}, "volrn": volrn_diag},
        "mosaic": {"mode": "weighted", "diagnostics": mosaic_diag},
        "outputs": {name: name for name in (
            "mosaic.tif", "preview.png", "valid_mask.tif", "contributor_count.tif",
            "weight_sum.tif", "run_config.json", "geometry_source.json", "radiometric_method.json",
        )},
    }
    (output_dir / "radiometric_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    return summary


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
