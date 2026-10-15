"""Task12 VOLRN solver-alignment replay using the frozen BAGRN scene cache.

The script deliberately never invokes matcher or global registration.  It validates
and, only when absent, rebuilds the frozen BAGRN cache through ``replay_baseline``;
the actual Run A/Run B work starts from the persisted per-scene BAGRN GeoTIFFs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine, array_bounds

# Allow direct ``python scripts/run_task12_volrn_alignment.py`` execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mosaic import create_mosaic
from src.multiscene_sift.final_baseline_replay import (
    BaselineReplaySpec,
    _cache_support,
    _find_valid_cache,
    _load_json,
    _validate_scene_cache,
    _write_preview,
    replay_baseline,
)
from src.multiscene_sift.mosaic_protocol import grid_transform
from src.multiscene_sift.radiometric_runner import (
    _expand_registered_footprints,
    _pair_metrics,
    _task10d_metrics,
)
from src.volrn import volrn_normalize, image_blocking, _interpolate_and_apply


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data/output/b9_five_scene_validation"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _spec(root: Path) -> BaselineReplaySpec:
    return BaselineReplaySpec(
        name="efficient_loftr_translation_l2_bagrn",
        geometry_run="efficient_loftr_translation_l2",
        # The hardened Task11 cache is locked to the 1024 audit config hash.
        source_config=root / "04_frozen_five_scene_config_1024.json",
        global_run_dir=root / "global_runs_1024/efficient_loftr/translation_l2",
        output_grid=root / "mosaic_runs_1024/protocol/canonical_output_grid.json",
        replay_root=root / "final_replay_v2_hardened",
    )


def _ensure_cache(spec: BaselineReplaySpec) -> tuple[Path, Path, dict]:
    base = spec.replay_root / spec.name
    found = _find_valid_cache(spec, base)
    if found is None:
        # This is the only permitted BAGRN fallback.  Geometry remains frozen by spec.
        artifact = replay_baseline(spec)
        run_dir = artifact.run_dir
        manifest_path = artifact.normalized_scenes_manifest
        summary = _load_json(run_dir / "radiometric_summary.json")
        return run_dir, manifest_path, summary
    run_dir, (manifest_path, summary) = found
    return run_dir, manifest_path, summary


def _load_cached_arrays(run_dir: Path, manifest_path: Path, grid: dict):
    manifest = _load_json(manifest_path)
    base_transform = grid_transform(grid)
    arrays, transforms, bounds, masks, scene_ids, slices = [], [], [], [], [], []
    full_shape = (int(grid["height"]), int(grid["width"]))
    for scene in manifest["scenes"]:
        path = run_dir / "normalized_scenes" / str(scene["path"])
        with rasterio.open(path) as ds:
            data = ds.read(1).astype(np.float64)
        valid = np.isfinite(data)
        rows, cols = np.where(valid)
        if not rows.size:
            raise ValueError(f"cached scene has no finite support: {path}")
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        local = data[r0:r1, c0:c1][None, ...]
        local_mask = valid[r0:r1, c0:c1]
        transform = base_transform * Affine.translation(c0, r0)
        left, bottom, right, top = array_bounds(local.shape[1], local.shape[2], transform)
        arrays.append(local)
        transforms.append(transform)
        bounds.append((left, bottom, right, top))
        masks.append(local_mask)
        scene_ids.append(str(scene["scene_id"]))
        slices.append((r0, r1, c0, c1))
    return arrays, transforms, bounds, masks, scene_ids, full_shape, slices


def _write_solver_history(path: Path, diagnostics: dict) -> None:
    (path / "volrn_solver_history.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = []
    for band, band_diag in enumerate(diagnostics.get("band_solver_diagnostics", [])):
        for item in band_diag.get("history", []):
            row = {"band": band}
            row.update(item)
            rows.append(row)
    fields = [
        "band", "iteration", "objective", "primal_residual", "dual_residual",
        "x_update_norm", "z_update_norm", "dual_update_norm", "cg_iterations",
        "cg_residual", "relative_change", "cg_info", "cg_final_residual_norm",
        "preconditioner_type", "rho", "rho_changed",
    ]
    with (path / "volrn_solver_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_status(diagnostics: dict) -> str:
    bands = diagnostics.get("band_solver_diagnostics", [])
    if not bands or not all(bool(item.get("finite_state", False)) for item in bands):
        return "NUMERICAL_INVALID"
    if any(bool(item.get("cg_failed_any", item.get("cg_failed", False))) for item in bands):
        return "NUMERICAL_INVALID"
    if all(bool(item.get("strict_admm_converged", False)) for item in bands):
        return "STRICT_ADMM_CONVERGED"
    if all(bool(item.get("coefficient_stable", False)) for item in bands):
        return "STABLE_BUT_STRICT_NONCONVERGED"
    return "NONCONVERGED_AFTER_ALIGNMENT"


def run(run: str, root: Path = DEFAULT_ROOT) -> dict:
    if run not in {"fixed", "adaptive"}:
        raise ValueError("run must be fixed or adaptive")
    spec = _spec(root)
    cache_dir, manifest_path, bagrn_summary = _ensure_cache(spec)
    grid = _load_json(spec.output_grid)
    arrays, transforms, bounds, masks, scene_ids, full_shape, crop_slices = _load_cached_arrays(
        cache_dir, manifest_path, grid
    )
    output_root = root / "volrn_solver_alignment_task12"
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / ("pcg_fixed" if run == "fixed" else "pcg_adaptive")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"non-empty Task12 output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    result, coeffs, diagnostics = volrn_normalize(
        arrays, transforms, bounds, [None] * len(arrays),
        block_size_pixels=400, lambda_param=0.5, rho=1.0,
        max_iter=200, tol=1e-4, return_diagnostics=True,
        valid_masks=masks, use_preconditioner=True, adaptive_rho=(run == "adaptive"),
    )
    runtime = time.perf_counter() - start
    final_full = _expand_registered_footprints(result, crop_slices, full_shape)
    before_full = _expand_registered_footprints(arrays, crop_slices, full_shape)
    valid_masks_full = [np.isfinite(a[0]) for a in before_full]
    mosaic_path = out / "mosaic.tif"
    _, mosaic_diag = create_mosaic(
        arrays=result, transforms=transforms, crs=str(grid["crs"]),
        nodata_values=[None] * len(result), output_path=str(mosaic_path),
        resolution=float(grid.get("resolution", grid["pixel_size"])), mode="weighted",
        output_transform=grid_transform(grid), output_width=int(grid["width"]),
        output_height=int(grid["height"]), output_dtype="float32", return_diagnostics=True,
    )
    support = np.any(np.stack(valid_masks_full), axis=0)
    _write_preview(out / "preview.png", mosaic_path, support)
    task10d, task10d_diag = _task10d_metrics(before_full, final_full, valid_masks_full, scene_ids)
    overlap = _pair_metrics(before_full, final_full, valid_masks_full)
    _write_solver_history(out, diagnostics)
    np.savez_compressed(out / "volrn_parameters.npz", block_coefficients=coeffs)
    (out / "radiometric_metrics.json").write_text(json.dumps({
        "task10d_primary_metrics": task10d,
        "task10d_diagnostics": task10d_diag,
        "overlap_metrics": overlap,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "schema_version": 1,
        "run": run,
        "solver": "PCG_JACOBI" if run == "fixed" else "PCG_JACOBI_ADAPTIVE_RHO",
        "cache_run_dir": str(cache_dir),
        "cache_manifest_sha256": _sha256(manifest_path),
        "frozen_geometry": spec.geometry_run,
        "scene_ids": scene_ids,
        "parameters": {"block_size_pixels": 400, "lambda_param": 0.5, "rho": 1.0, "max_iter": 200, "tol": 1e-4},
        "runtime_sec": runtime,
        "solver_diagnostics": diagnostics,
        "task10d_primary_metrics": task10d,
        "task10d_diagnostics": task10d_diag,
        "overlap_metrics": overlap,
        "mosaic": mosaic_diag,
        "bagrn_reference_summary_sha256": _sha256(cache_dir / "radiometric_summary.json"),
        "status": _aggregate_status(diagnostics),
    }
    (out / "solver_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def repair_existing(run: str, root: Path = DEFAULT_ROOT) -> dict:
    """Repair post-processing from saved coefficients without another solver run."""
    spec = _spec(root)
    cache_dir, manifest_path, _ = _ensure_cache(spec)
    grid = _load_json(spec.output_grid)
    arrays, transforms, bounds, masks, scene_ids, full_shape, crop_slices = _load_cached_arrays(
        cache_dir, manifest_path, grid
    )
    out = root / "volrn_solver_alignment_task12" / ("pcg_fixed" if run == "fixed" else "pcg_adaptive")
    coeffs = np.load(out / "volrn_parameters.npz")["block_coefficients"]
    blocks, _ = image_blocking(
        arrays, transforms, bounds, [None] * len(arrays), 400, [0],
        valid_masks=masks,
    )
    all_x = np.zeros((coeffs.shape[0], coeffs.shape[1] * 2), dtype=np.float64)
    all_x[:, 0::2] = coeffs[:, :, 0]
    all_x[:, 1::2] = coeffs[:, :, 1]
    result = _interpolate_and_apply(
        arrays, [None] * len(arrays), blocks, all_x, 400,
        transforms, bounds, [0], valid_masks=masks,
    )
    final_full = _expand_registered_footprints(result, crop_slices, full_shape)
    before_full = _expand_registered_footprints(arrays, crop_slices, full_shape)
    valid_masks_full = [np.isfinite(a[0]) for a in before_full]
    diagnostics = _load_json(out / "solver_summary.json")
    metrics = _task10d_metrics(before_full, final_full, valid_masks_full, scene_ids)
    overlap = _pair_metrics(before_full, final_full, valid_masks_full)
    (out / "radiometric_metrics.json").write_text(json.dumps({
        "task10d_primary_metrics": metrics[0],
        "task10d_diagnostics": metrics[1],
        "overlap_metrics": overlap,
        "postprocessing_repaired_from_saved_coefficients": True,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    diagnostics["task10d_primary_metrics"] = metrics[0]
    diagnostics["task10d_diagnostics"] = metrics[1]
    diagnostics["overlap_metrics"] = overlap
    (out / "solver_summary.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    support = np.any(np.stack(valid_masks_full), axis=0)
    _write_preview(out / "preview.png", out / "mosaic.tif", support)
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["fixed", "adaptive"], required=True)
    parser.add_argument("--repair-existing", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = repair_existing(args.run, args.root) if args.repair_existing else run(args.run, args.root)
    print(json.dumps({
        "run": result.get("run", args.run),
        "status": result.get("status", "POSTPROCESS_REPAIRED"),
        "runtime_sec": result.get("runtime_sec"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
