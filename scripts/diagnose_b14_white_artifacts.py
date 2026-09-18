"""B14-only white-artifact diagnostic pipeline (SIFT → BAGRN → VOLRN).

This script re-runs the B14 registration and radiometric normalization
so it can save every intermediate stage per scene and answer:

  1. At which stage do white pixels first appear?
  2. What is the real DN value of those pixels?
  3. Are they NoData / NaN / Inf holes, or extreme-high valid DN?
  4. If VOLRN produced them, do they align with specific blocks
     (extreme a/b, low valid fraction)?

It does NOT modify production BAGRN / VOLRN math. Only B14 is run.

Usage::

    python -m scripts.diagnose_b14_white_artifacts \\
        --input-root <flat dir> --output-dir <out dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import rasterio

from src.multiscene_sift.dataset import discover_five_scenes
from src.multiscene_sift.overlap_graph import (
    build_geographic_overlap_graph,
)
from src.multiscene_sift.pairwise import run_all_pairs
from src.multiscene_sift.global_registration import (
    build_accepted_sift_graph,
    select_reference_scene,
    build_spanning_tree,
    compose_global_transforms,
    global_consistency_diagnostics,
)
from src.multiscene_sift.mosaicking import (
    compute_shared_mosaic_grid,
    raster_bounds_from_transform,
)
from src.multiscene_sift.band_geometry import apply_world_correction_to_transform
from src.multiscene_sift.white_artifact_diagnostics import (
    stage_statistics,
    count_new_invalid,
    extreme_high_threshold,
    make_invalid_mask,
    make_extreme_high_mask,
    save_stage_tiff,
    block_valid_fraction_stats,
    coefficient_outlier_rows,
    affine_coordinate_diagnostic,
    write_json,
    _is_valid,
)

from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize, image_blocking
from src.overlap import detect_multi_overlap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registration geometry (reuses the fixed modules from this branch)
# ---------------------------------------------------------------------------


def run_registration_geometry(
    input_root: str,
    output_dir: Path,
    scene_names: list[str],
    registration_band: str,
    match_max_side: int,
    ransac_threshold: float,
):
    """Run B14 SIFT geometry; return scenes, G, ref info, grid."""
    out = output_dir

    scenes, manifest = discover_five_scenes(
        input_root, scene_names, bands=(registration_band, "B8", "B5")
    )
    write_json(out / "dataset_manifest.json", manifest)

    edges = build_geographic_overlap_graph(scenes, registration_band)

    pairwise_results = run_all_pairs(
        scenes, edges, out,
        band=registration_band,
        match_max_side=match_max_side,
        ransac_threshold=ransac_threshold,
    )
    adj, accepted = build_accepted_sift_graph(pairwise_results)
    ref_info = select_reference_scene(adj, accepted)
    ref_idx = ref_info["reference_index"]
    ref_name = scenes[ref_idx].name if ref_idx < len(scenes) else "unknown"
    tree_edges = build_spanning_tree(adj, accepted, ref_idx)
    G = compose_global_transforms(scenes, accepted, tree_edges, ref_idx)

    consensus = global_consistency_diagnostics(
        accepted, G, tree_edges,
        pixel_size=abs(scenes[0].transforms[registration_band].a),
    )

    mosaic_grid = compute_shared_mosaic_grid(
        scenes, G, band=registration_band,
    )

    return {
        "scenes": scenes,
        "edges": edges,
        "pairwise_results": pairwise_results,
        "accepted": accepted,
        "ref_info": ref_info,
        "ref_idx": ref_idx,
        "ref_name": ref_name,
        "tree_edges": tree_edges,
        "G": G,
        "consistency": consensus,
        "mosaic_grid": mosaic_grid,
    }


def load_band_stage(
    scenes,
    G,
    band: str,
) -> tuple[list[np.ndarray], list, list, list[float | None]]:
    """Read registered (=geometry-corrected) arrays for one band.

    Returns ``(arrays, corrected_transforms, corrected_bounds, nodata_values)``.
    """
    arrays = []
    transforms = []
    bounds = []
    nodata_values = []

    for s, g in zip(scenes, G):
        with rasterio.open(s.band_paths[band]) as src:
            data = src.read(1)
            orig_tf = src.transform
            nd = src.nodata
        tf = apply_world_correction_to_transform(orig_tf, g)
        h, w = s.shapes[band]
        bbox = raster_bounds_from_transform(tf, h, w)
        arrays.append(data[np.newaxis, :, :])
        transforms.append(tf)
        bounds.append(bbox)
        nodata_values.append(nd)

    return arrays, transforms, bounds, nodata_values


# ---------------------------------------------------------------------------
# Stage persistence + statistics + bags of masks
# ---------------------------------------------------------------------------


def dump_stage(
    arrays: list[np.ndarray],
    transformed_info,
    stage_name: str,
    band: str,
    out_dir: Path,
):
    """Save per-scene TIFFs for a stage into ``diagnostics_white_blocks/``."""
    stage_dir = out_dir / "diagnostics_white_blocks"
    stage_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, arr in enumerate(arrays):
        tf = transformed_info[0][i]
        crs = transformed_info[1]
        nd = transformed_info[2][i]
        path = stage_dir / f"scene{i}_{stage_name}_{band}.tif"
        # Need crs: read from first scene's profile
        save_stage_tiff(arr[0], tf, crs, nd, path)
        saved.append(str(path))
    return saved


def collect_stage_stats(
    arrays: list[np.ndarray],
    nodata_values: list[float | None],
    stage_name: str,
    band: str,
) -> tuple[list[dict], dict]:
    """Per-scene stage statistics; returns (rows, new_invalid summary)."""
    rows = []
    new_invalid = {}
    for i, arr in enumerate(arrays):
        data = arr[0]
        stats = stage_statistics(data, nodata_values[i])
        stats["stage"] = stage_name
        stats["band"] = band
        stats["scene"] = i
        rows.append(stats)
    return rows, {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    t0 = time.perf_counter()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "input_root": args.input_root,
        "output_dir": str(out),
        "registration_band": args.registration_band,
        "match_max_side": args.match_max_side,
        "ransac_threshold": args.ransac_threshold,
        "block_size": args.block_size,
        "lambda_param": args.lambda_param,
        "rho": args.rho,
        "max_iter": args.max_iter,
        "tol": args.tol,
    }
    with open(out / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # =====================================================================
    # 1. Registration geometry (B14)
    # =====================================================================
    scene_names = (args.scene_names.split(",") if args.scene_names
                   else None)
    geo = run_registration_geometry(
        args.input_root, out, scene_names,
        args.registration_band, args.match_max_side, args.ransac_threshold,
    )
    scenes = geo["scenes"]
    G = geo["G"]
    ref_idx = geo["ref_idx"]
    band = args.registration_band

    # =====================================================================
    # 2. Registered stage
    # =====================================================================
    logger.info("=== Registered stage ===")
    arrays, transforms, bounds, nodata_values = load_band_stage(
        scenes, G, band,
    )
    for s in scenes:
        logger.info("scene %d: %s", s.index, s.name)
    with __import__("rasterio").open(scenes[0].band_paths[band]) as src:
        crs = src.crs
    registered_crs = crs
    reg_tif = dump_stage(
        arrays, (transforms, crs, nodata_values), "registered", band, out,
    )

    registered_rows, _ = collect_stage_stats(arrays, nodata_values, "registered", band)

    # =====================================================================
    # 3. Radiometric overlaps (recomputed after registration)
    # =====================================================================
    overlaps = detect_multi_overlap(bounds, transforms, min_pixels=100)
    logger.info("Radiometric overlap pairs: %d", len(overlaps))

    # =====================================================================
    # 4. BAGRN
    # =====================================================================
    logger.info("=== BAGRN stage ===")
    t_bagrn = time.perf_counter()
    bagrn_result, theta_mu, theta_sigma = bagrn_normalize(
        arrays, nodata_values, overlaps, control_idx=ref_idx,
    )
    logger.info("BAGRN took %.1f s, theta_mu=%s, theta_sigma=%s",
                time.perf_counter() - t_bagrn,
                list(np.asarray(theta_mu).ravel())[:6],
                list(np.asarray(theta_sigma).ravel())[:6])

    bagrn_tif = dump_stage(
        bagrn_result, (transforms, crs, nodata_values), "BAGRN", band, out,
    )
    bagrn_rows, _ = collect_stage_stats(bagrn_result, nodata_values, "BAGRN", band)

    # BAGRN must not fabricate invalid pixels
    bagrn_new_invalid = []
    for i, arr in enumerate(bagrn_result):
        d = count_new_invalid(arrays[i][0], nodata_values[i],
                              arr[0], nodata_values[i])
        d["scene"] = i
        bagrn_new_invalid.append(d)
    _write_csv(out / "bagrn_new_invalid.csv",
               ["scene", "new_nodata", "new_nan", "new_inf"],
               [{**r} for r in bagrn_new_invalid])

    # =====================================================================
    # 5. VOLRN with diagnostics
    # =====================================================================
    logger.info("=== VOLRN stage ===")
    t_volrn = time.perf_counter()
    volrn_result, block_coeffs, volrn_diag = volrn_normalize(
        bagrn_result,
        transforms,
        bounds,
        nodata_values,
        block_size_pixels=args.block_size,
        lambda_param=args.lambda_param,
        rho=args.rho,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=False,
        return_diagnostics=True,
    )
    logger.info("VOLRN took %.1f s", time.perf_counter() - t_volrn)

    volrn_tif = dump_stage(
        volrn_result, (transforms, crs, nodata_values), "VOLRN", band, out,
    )
    volrn_rows, _ = collect_stage_stats(volrn_result, nodata_values, "VOLRN", band)

    volrn_new_invalid = []
    for i, arr in enumerate(volrn_result):
        d = count_new_invalid(arrays[i][0], nodata_values[i],
                              arr[0], nodata_values[i])
        d["scene"] = i
        volrn_new_invalid.append(d)

    # ---- Save VOLRN diagnostics -----------------------------------------
    diag_out = {
        "n_blocks": volrn_diag["n_blocks"],
        "n_pairs": volrn_diag["n_pairs"],
        "all_converged": volrn_diag["all_converged"],
        "band_converged": volrn_diag["band_converged"],
        "band_iterations": volrn_diag["band_iterations"],
        "block_details": _sanitize_block_details(volrn_diag["block_details"]),
    }
    write_json(out / "volrn_diagnostics_B14.json", diag_out)

    # ---- Rebuild blocks to get valid fraction without changing selection ---
    blocks, pairs = image_blocking(
        bagrn_result, transforms, bounds, nodata_values,
        args.block_size, [0],
    )
    vf_stats = block_valid_fraction_stats(blocks, n_bands=1)

    # ---- block coefficients CSV -----------------------------------------
    write_block_coefficients_csv(
        out / "volrn_block_coefficients_B14.csv",
        blocks, volrn_diag["block_details"],
    )

    # ---- valid fraction statistics ---------------------------------------
    write_json(out / "volrn_block_valid_fraction.json", vf_stats)

    # ---- a / b extremes + outliers --------------------------------------
    outlier_info = coefficient_outlier_rows(volrn_diag["block_details"])
    _write_csv(
        out / "volrn_coefficient_outliers.csv",
        ["block_id", "image_idx", "grid_m", "grid_n", "a", "b",
         "is_a_outlier", "is_b_outlier"],
        outlier_info["flagged"],
    )
    write_json(out / "volrn_coefficient_summary.json",
               {"per_scene": outlier_info["per_scene"],
                "n_flagged": outlier_info["n_flagged"]})

    # =====================================================================
    # 5.5 Combine all stage stats -> b14_stage_statistics.csv / .json
    # =====================================================================
    all_rows = registered_rows + bagrn_rows + volrn_rows
    _write_csv(
        out / "b14_stage_statistics.csv",
        ["scene", "stage", "band", "total_pixels", "valid_pixels",
         "nodata_pixels", "nan_pixels", "inf_pixels",
         "valid_min", "valid_max", "mean", "std",
         "p0_1", "p1", "p50", "p99", "p99_9"],
        all_rows,
    )
    write_json(out / "b14_stage_statistics.json", {
        "stages": ["registered", "BAGRN", "VOLRN"],
        "rows": all_rows,
        "new_invalid_after_bagrn": bagrn_new_invalid,
        "new_invalid_after_volrn": volrn_new_invalid,
    })

    # =====================================================================
    # 5.6 Block overlay PNG (Task 9)
    # =====================================================================
    try:
        _draw_block_overlay(
            out / "volrn_block_overlay_B14.png",
            volrn_result, volrn_diag["block_details"], transforms,
            crs,
            outlier_blocks={r["block_id"] for r in outlier_info["flagged"]},
            low_valid_blocks=_low_valid_block_ids(blocks),
        )
    except Exception as exc:  # diagnostic-only, never crash the run
        logger.warning("Block overlay failed: %s", exc)

    # =====================================================================
    # 6. invalid / extreme-high masks (per VOLRN scene)
    # =====================================================================
    logger.info("=== Masks ===")
    mask_dir = out / "diagnostics_white_blocks"
    for i, arr in enumerate(volrn_result):
        data = arr[0]
        nodata = nodata_values[i]
        inv = make_invalid_mask(data, nodata)
        thr = extreme_high_threshold(arrays[i][0], nodata_values[i], percentile=99.9)
        high = make_extreme_high_mask(data, nodata, thr)

        save_mask_png(mask_dir / f"scene{i}_invalid_mask.png", inv)
        save_mask_png(mask_dir / f"scene{i}_extreme_high_mask.png", high)
        write_json(
            mask_dir / f"scene{i}_extreme_high_threshold.json",
            {"threshold_gt_registered_p99_9": thr,
             "n_invalid_pixels": int(inv.sum()),
             "n_extreme_high_pixels": int(high.sum())},
        )

    # =====================================================================
    # 7. Extreme valid DN distribution in final VOLRN
    # =====================================================================
    final_high_rows = []
    for i, arr in enumerate(volrn_result):
        data = arr[0]
        valid = _is_valid(data, nodata_values[i])
        if valid.any():
            p999 = float(np.percentile(data[valid].astype(np.float64), 99.9))
            n_over_p999 = int((valid & (data > p999)).sum())
        else:
            p999 = None
            n_over_p999 = 0
        final_high_rows.append({
            "scene": i,
            "volrn_p99_9": p999,
            "n_over_volrn_p99_9": n_over_p999,
        })
    write_json(out / "volrn_extreme_high_summary.json",
               {"rows": final_high_rows})

    # =====================================================================
    # 8. Affine / IDW coordinate diagnostic (Task 10)
    # =====================================================================
    heights = [s.shapes[band][0] for s in scenes]
    widths = [s.shapes[band][1] for s in scenes]
    res_x = abs(transforms[0].a)
    res_y = abs(transforms[0].e)
    affine_diag = affine_coordinate_diagnostic(
        transforms, heights, widths, res_x, res_y, args.block_size,
    )
    write_json(out / "affine_idw_coordinate_diagnostic.json", affine_diag)

    # =====================================================================
    # 9. Weighted mosaic + union/final mask comparison (Task 11)
    # =====================================================================
    logger.info("=== Weighted mosaic ===")
    from src.mosaic import create_mosaic

    mosaic_path = out / "mosaic_B14_SIFT_BAGRN_VOLRN_weighted.tif"
    create_mosaic(
        arrays=volrn_result,
        transforms=transforms,
        crs=str(crs),
        nodata_values=nodata_values,
        output_path=str(mosaic_path),
        resolution=geo["mosaic_grid"].resolution,
        mode="weighted",
        output_transform=geo["mosaic_grid"].transform,
        output_width=geo["mosaic_grid"].width,
        output_height=geo["mosaic_grid"].height,
    )

    # union vs final valid masks
    union_stats = _union_vs_final(
        arrays, transforms, nodata_values,
        geo["mosaic_grid"], crs, mosaic_path, out,
    )
    write_json(out / "mosaic_union_valid_stats.json", union_stats)

    elapsed = time.perf_counter() - t0
    logger.info("Done in %.1f s", elapsed)

    print("\n" + "=" * 60)
    print("B14 white-artifact diagnosis complete")
    print("=" * 60)
    print(f"Outputs under: {out}")
    print(f"  b14_stage_statistics: {out/'b14_stage_statistics'}.csv / .json")
    print(f"  volrn_diagnostics_B14.json")
    print(f"  volrn_block_coefficients_B14.csv")
    print(f"  affine_idw_coordinate_diagnostic.json")
    print(f"  mosaic_union_but_final_invalid_mask.png")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="B14-only white-artifact diagnosis (SIFT + BAGRN + VOLRN)"
    )
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--registration-band", default="B14")
    p.add_argument("--match-max-side", type=int, default=1600)
    p.add_argument("--ransac-threshold", type=float, default=2.0)
    p.add_argument("--scene-names", default=None,
                   help="comma-separated scene dir names (default: 5 data scenes)")
    p.add_argument("--block-size", type=int, default=200)
    p.add_argument("--lambda-param", type=float, default=0.1)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-4)
    return p


def _sanitize_block_details(block_details: list[dict]) -> list[dict]:
    """Keep only JSON-safe scalar fields from VOLRN block details."""
    out = []
    for b in block_details:
        out.append({
            "block_id": b["block_id"],
            "image_idx": b["image_idx"],
            "grid_m": b["grid_m"],
            "grid_n": b["grid_n"],
            "center_x": b["center_x"],
            "center_y": b["center_y"],
            "mu": b.get("mu"),
            "sigma": b.get("sigma"),
            "a": b.get("a"),
            "b": b.get("b"),
            "converged": b.get("converged"),
            "iterations": b.get("iterations"),
        })
    return out


def write_block_coefficients_csv(path: Path, blocks, block_details) -> None:
    """Write per-block coefficients CSV with valid-fraction diagnostics."""
    # block_details is expanded over (band, block); single band here.
    rows = []
    for k, blk in enumerate(blocks):
        detail = block_details[k]  # single band -> aligned index
        vm = blk.valid_mask[0] if blk.valid_mask.ndim > 1 else blk.valid_mask
        window = blk.window
        n_total = max(window[1] - window[0], 1) * max(window[3] - window[2], 1)
        n_valid = int(vm.sum())
        rows.append({
            "block_id": detail["block_id"],
            "image_idx": detail["image_idx"],
            "grid_m": detail["grid_m"],
            "grid_n": detail["grid_n"],
            "center_x": round(float(detail["center_x"]), 2),
            "center_y": round(float(detail["center_y"]), 2),
            "a": round(float(detail["a"]), 6),
            "b": round(float(detail["b"]), 6),
            "valid_count": n_valid,
            "total_count": n_total,
            "valid_fraction": round(n_valid / n_total, 6) if n_total > 0 else 0.0,
        })
    _write_csv(
        path,
        ["block_id", "image_idx", "grid_m", "grid_n",
         "center_x", "center_y", "a", "b",
         "valid_count", "total_count", "valid_fraction"],
        rows,
    )


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    """Save a boolean mask as a simple grayscale PNG."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mask, cmap="gray", interpolation="nearest")
    ax.set_title(path.stem)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved mask: %s", path)


def _low_valid_block_ids(blocks, threshold: float = 0.10) -> set[int]:
    """Block ids whose valid_fraction is below *threshold*."""
    low = set()
    for blk in blocks:
        vm = blk.valid_mask[0] if blk.valid_mask.ndim > 1 else blk.valid_mask
        window = blk.window
        n_total = (window[1] - window[0]) * (window[3] - window[2])
        if n_total <= 0:
            continue
        frac = int(vm.sum()) / n_total
        if frac < threshold:
            low.add(blk.block_id)
    return low


def _draw_block_overlay(
    path: Path,
    volrn_result: list[np.ndarray],
    block_details: list[dict],
    transforms,
    crs,
    outlier_blocks: set[int],
    low_valid_blocks: set[int],
) -> None:
    """Render VOLRN result thumbnail with 200px block grid + flagged blocks."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    # Downscale each scene synoptically for display (thumbnail)
    import cv2

    n = len(volrn_result)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    cmap = "gray"
    for i, arr in enumerate(volrn_result):
        ax = axes[0][i]
        data = arr[0].astype(np.float64)
        data = np.nan_to_num(data, nan=np.nanmin(data) if np.isfinite(data).any() else 0)
        if data.max() - data.min() <= 0:
            disp = np.zeros_like(data)
        else:
            disp = (data - data.min()) / (data.max() - data.min())
        # scale display to a legible size
        max_side = 512
        h, w = disp.shape
        scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
        disp = cv2.resize(disp, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
        ax.imshow(disp, cmap=cmap)
        ax.set_title(f"scene {i}")

        # Draw block grid for this scene
        for blk in block_details:
            if blk["image_idx"] != i:
                continue
            # convert block center to display pixel
            gx = blk["center_x"]
            gy = blk["center_y"]
            px, py = ~transforms[i] * (gx, gy)
            # block is ~200px in source pixel space; scale down for display
            px *= scale
            py *= scale
            half = 200 * scale / 2
            fc = "none"
            ec = "orange"
            lw = 0.6
            if blk["block_id"] in outlier_blocks:
                ec = "red"
                lw = 1.5
            if blk["block_id"] in low_valid_blocks:
                ec = "blue"
                lw = 1.0
            rect = Rectangle((px - half, py - half), half * 2, half * 2,
                             fill=fc, edgecolor=ec, linewidth=lw)
            ax.add_patch(rect)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved block overlay: %s", path)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Saved CSV: %s", path)


def _union_vs_final(
    arrays,
    transforms,
    nodata_values,
    mosaic_grid,
    crs,
    mosaic_path: Path,
    out: Path,
) -> dict:
    """Reproject every scene's valid mask to the shared grid, OR them, and
    compare with the final weighted mosaic's valid mask."""
    import rasterio.warp
    from rasterio.enums import Resampling

    gw = mosaic_grid.width
    gh = mosaic_grid.height
    union = np.zeros((gh, gw), dtype=bool)

    for i, arr in enumerate(arrays):
        data = arr[0]
        valid = _is_valid(data, nodata_values[i]).astype(np.uint8)
        proj = np.zeros((gh, gw), dtype=np.uint8)
        rasterio.warp.reproject(
            source=valid,
            destination=proj,
            src_transform=transforms[i],
            src_crs=crs,
            dst_transform=mosaic_grid.transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )
        union |= proj > 0

    with rasterio.open(mosaic_path) as src:
        final_data = src.read(1)
        nd = src.nodata
    final_valid = _is_valid(final_data, nd)

    union_but_not_final = union & ~final_valid
    stats = {
        "union_valid_pixels": int(union.sum()),
        "final_valid_pixels": int(final_valid.sum()),
        "union_but_final_invalid_pixels": int(union_but_not_final.sum()),
    }
    save_mask_png(out / "mosaic_union_but_final_invalid_mask.png",
                  union_but_not_final)
    return stats


if __name__ == "__main__":
    main()