"""Five-scene SIFT multi-image registration and mosaic runner."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from src.multiscene_sift.dataset import discover_five_scenes, save_manifest
from src.multiscene_sift.overlap_graph import (
    build_geographic_overlap_graph,
    save_overlap_graph,
)
from src.multiscene_sift.pairwise import run_all_pairs
from src.multiscene_sift.global_registration import (
    build_accepted_sift_graph,
    select_reference_scene,
    build_spanning_tree,
    compose_global_transforms,
    global_consistency_diagnostics,
    save_global_registration_info,
    save_consistency_diagnostics,
)
from src.multiscene_sift.mosaicking import (
    compute_shared_mosaic_grid,
    make_mosaic,
    make_mosaic_original_transforms,
)
from src.multiscene_sift.rgb import stack_three_band_geotiff, make_rgb_preview
from src.multiscene_sift.diagnostics import (
    draw_footprints,
    draw_overlap_graph,
    draw_accepted_sift_graph,
    draw_spanning_tree,
    draw_global_consistency,
)

from src.multiscene_sift.radiometric import (
    BandRadiometricResult,
    normalize_registered_band,
    release_band_result,
)
from src.multiscene_sift.radiometric_reporting import (
    save_radiometric_metrics,
    save_normalization_info,
)

logger = logging.getLogger(__name__)

# Default scene names (DZ01V flat-terrain)
DEFAULT_SCENE_NAMES = [
    "DZ01V_L2_E113.4_N36.6_20260810030932_01_T1",
    "DZ01V_L2_E113.6_N36.3_20260616031133_01_T1",
    "DZ01V_L2_E113.0_N36.4_20260222031837_01_T1",
    "DZ01V_L2_E114.0_N36.4_20260714030410_01_T1",
    "DZ01V_L2_E113.7_N36.6_20260616031127_01_T1",
]


def run_five_scene_mosaic(
    input_root: str,
    output_dir: str,
    registration_band: str = "B14",
    bands: tuple[str, ...] = ("B14", "B8", "B5"),
    match_max_side: int = 1600,
    ransac_threshold: float = 2.0,
    scene_names: list[str] | None = None,
    save_diagnostics: bool = False,
    block_size: int = 200,
    lambda_param: float = 0.1,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> dict:
    """Run the full five-scene SIFT + BAGRN + VOLRN mosaic pipeline.

    Args:
        input_root: Path to the ``flat/`` input directory.
        output_dir: Output directory for all results.
        registration_band: Band used for SIFT registration.
        bands: All bands to process.
        match_max_side: Max side for match-view images.
        ransac_threshold: RANSAC inlier threshold in pixels.
        scene_names: Override default scene names.
        save_diagnostics: If True, save diagnostic plots and detail mosaics.
        block_size: VOLRN block size.
        lambda_param: VOLRN lambda.
        rho: VOLRN rho.
        max_iter: VOLRN max iterations.
        tol: VOLRN tolerance.

    Returns:
        Summary dict with status and output paths.
    """
    t_total = time.perf_counter()

    if scene_names is None:
        scene_names = DEFAULT_SCENE_NAMES

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = {
        "input_root": input_root,
        "output_dir": str(out),
        "registration_band": registration_band,
        "bands": list(bands),
        "match_max_side": match_max_side,
        "ransac_threshold": ransac_threshold,
        "scene_names": scene_names,
    }
    with open(out / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # =====================================================================
    # 1. Dataset discovery
    # =====================================================================
    logger.info("=== Step 1: Dataset Discovery ===")
    scenes, manifest = discover_five_scenes(input_root, scene_names, bands=bands)
    save_manifest(manifest, out / "dataset_manifest.json")
    logger.info("Discovered %d scenes", len(scenes))

    # =====================================================================
    # 2. Geographic overlap graph
    # =====================================================================
    logger.info("=== Step 2: Geographic Overlap Graph ===")
    edges = build_geographic_overlap_graph(scenes, registration_band)
    save_overlap_graph(edges, out)
    logger.info("Geographic graph: %d edges, connected", len(edges))

    # =====================================================================
    # 3. Pairwise B14 SIFT registration
    # =====================================================================
    logger.info("=== Step 3: Pairwise B14 SIFT Registration ===")
    pairwise_results = run_all_pairs(
        scenes, edges, out,
        band=registration_band,
        match_max_side=match_max_side,
        ransac_threshold=ransac_threshold,
    )
    n_ok = sum(1 for r in pairwise_results if r.status == "OK")
    n_fail = len(pairwise_results) - n_ok
    logger.info("Pairwise: %d OK, %d failed", n_ok, n_fail)

    # =====================================================================
    # 4. Accepted SIFT graph
    # =====================================================================
    logger.info("=== Step 4: Accepted SIFT Graph ===")
    try:
        adj, accepted = build_accepted_sift_graph(pairwise_results)
    except RuntimeError as exc:
        logger.error("STOP: %s", exc)
        # Save accepted graph components anyway
        components_data = {"error": str(exc), "connected": False}
        with open(out / "accepted_graph_components.json", "w") as f:
            json.dump(components_data, f, indent=2)
        elapsed = time.perf_counter() - t_total
        disconnected_summary = {
            "status": "DISCONNECTED",
            "error": str(exc),
            "output_dir": str(out),
            "runtime_sec": elapsed,
            "n_scenes": len(scenes),
            "geographic_edges": len(edges),
            "accepted_sift_edges": 0,
            "failed_sift_edges": len(pairwise_results),
            "reference_index": None,
            "reference_name": None,
            "spanning_tree_edges": 0,
            "outputs": {},
            "worst_global_p95_px": None,
        }
        with open(out / "run_summary.json", "w") as f:
            json.dump(disconnected_summary, f, indent=2)
        return disconnected_summary
    logger.info("Accepted graph: %d edges, connected", len(accepted))

    # =====================================================================
    # 5. Reference selection
    # =====================================================================
    logger.info("=== Step 5: Reference Selection ===")
    ref_info = select_reference_scene(adj, accepted)
    ref_idx = ref_info["reference_index"]
    ref_name = scenes[ref_idx].name if ref_idx < len(scenes) else "unknown"
    ref_info["reference_name"] = ref_name
    logger.info("Reference: [%d] %s (degree=%d)",
                ref_idx, ref_name, ref_info["degree"])

    # =====================================================================
    # 6. Spanning tree
    # =====================================================================
    logger.info("=== Step 6: Maximum Spanning Tree ===")
    tree_edges = build_spanning_tree(adj, accepted, ref_idx)
    logger.info("Spanning tree: %d edges", len(tree_edges))

    # =====================================================================
    # 7. Global world transforms
    # =====================================================================
    logger.info("=== Step 7: Global World Transforms ===")
    G = compose_global_transforms(scenes, accepted, tree_edges, ref_idx)
    save_global_registration_info(ref_info, tree_edges, G, out)
    logger.info("Global transforms for %d scenes", len(G))

    # =====================================================================
    # 8. Global consistency
    # =====================================================================
    logger.info("=== Step 8: Global Consistency Diagnostics ===")
    # Get pixel size from first scene B14
    with __import__("rasterio").open(scenes[0].band_paths[registration_band]) as src:
        pixel_size = abs(src.transform.a)
    consistency = global_consistency_diagnostics(
        accepted, G, tree_edges, pixel_size=pixel_size,
    )
    save_consistency_diagnostics(consistency, out)

    # Flag P95 > 2px
    flagged = [c for c in consistency if c["global_p95_px"] > 2.0]
    if flagged:
        logger.warning("Global consistency: %d edges with P95 > 2 px", len(flagged))
    else:
        logger.info("Global consistency: all edges within 2 px")

    # =====================================================================
    # 9. Shared mosaic grid
    # =====================================================================
    logger.info("=== Step 9: Shared Mosaic Grid ===")
    mosaic_grid = compute_shared_mosaic_grid(scenes, G, band=registration_band)
    logger.info("Grid: %d×%d px @ %.2f m", mosaic_grid.width,
                mosaic_grid.height, mosaic_grid.resolution)

    # =====================================================================
    # 10. Diagnostic mosaics (only if --save-diagnostics)
    # =====================================================================
    if save_diagnostics:
        logger.info("=== Step 10: Diagnostic Mosaics ===")
        make_mosaic_original_transforms(
            scenes, registration_band, mosaic_grid,
            out / "mosaic_B14_before_registration_source_selection.tif",
            mode="source_selection",
        )
        make_mosaic(
            scenes, G, registration_band, mosaic_grid,
            out / "mosaic_B14_after_registration_source_selection.tif",
            mode="source_selection",
        )

    # =====================================================================
    # 11. Radiometric normalization (BAGRN → VOLRN per band)
    # =====================================================================
    logger.info("=== Step 11: Radiometric Normalization ===")
    band_results: dict[str, BandRadiometricResult] = {}
    mosaic_paths: dict[str, str] = {}
    total_bagrn_time = 0.0
    total_volrn_time = 0.0

    for band in bands:
        logger.info("--- Processing band: %s ---", band)

        # A. Radiometric normalization
        result = normalize_registered_band(
            scenes, G, band, ref_idx,
            block_size_pixels=block_size,
            lambda_param=lambda_param,
            rho=rho,
            max_iter=max_iter,
            tol=tol,
        )
        band_results[band] = result
        total_bagrn_time += result.bagrn_runtime_sec
        total_volrn_time += result.volrn_runtime_sec

        # B. Final VOLRN weighted mosaic on shared grid
        from src.mosaic import create_mosaic
        mosaic_path = out / f"mosaic_{band}_SIFT_BAGRN_VOLRN_weighted.tif"
        create_mosaic(
            arrays=result.normalized_arrays,
            transforms=result.corrected_transforms,
            crs=str(mosaic_grid.crs),
            nodata_values=result.nodata_values,
            output_path=str(mosaic_path),
            resolution=mosaic_grid.resolution,
            mode="weighted",
            output_transform=mosaic_grid.transform,
            output_width=mosaic_grid.width,
            output_height=mosaic_grid.height,
        )
        mosaic_paths[band] = str(mosaic_path)

        # C. Release memory
        release_band_result(result)
        logger.info("--- Band %s complete ---", band)

    # ---- Save radiometric reports ------------------------------------------
    save_radiometric_metrics(band_results, out)
    save_normalization_info(band_results, ref_idx, ref_name, out)

    # =====================================================================
    # 12. Three-band GeoTIFF
    # =====================================================================
    logger.info("=== Step 12: B14-B8-B5 Three-Band GeoTIFF ===")
    rgb_path = out / "mosaic_RGB_B14_B8_B5_SIFT_BAGRN_VOLRN.tif"
    stack_three_band_geotiff(
        {"R": mosaic_paths["B14"],
         "G": mosaic_paths["B8"],
         "B": mosaic_paths["B5"]},
        rgb_path,
    )

    # =====================================================================
    # 13. RGB preview PNG
    # =====================================================================
    logger.info("=== Step 13: RGB Preview PNG ===")
    preview_path = out / "mosaic_RGB_B14_B8_B5_SIFT_BAGRN_VOLRN_preview.png"
    stretch_path = out / "rgb_preview_stretch.json"
    make_rgb_preview(rgb_path, preview_path, stretch_path)

    # =====================================================================
    # 14. Diagnostic plots (only if --save-diagnostics)
    # =====================================================================
    if save_diagnostics:
        logger.info("=== Step 14: Diagnostic Plots ===")
        diag_dir = out / "diagnostics"
        diag_dir.mkdir(exist_ok=True)
        try:
            draw_footprints(scenes, diag_dir / "scene_footprints_before.png",
                            "Scene Footprints (Before Registration)")
            draw_footprints(scenes, diag_dir / "scene_footprints_after.png",
                            "Scene Footprints (After Registration)",
                            corrected=True, G=G)
            draw_overlap_graph(edges, scenes,
                              diag_dir / "overlap_graph.png")
            draw_accepted_sift_graph(accepted, pairwise_results, scenes,
                                    diag_dir / "accepted_sift_graph.png")
            draw_spanning_tree(tree_edges, scenes,
                              diag_dir / "spanning_tree.png")
            draw_global_consistency(consistency,
                                   diag_dir / "global_edge_consistency.png")
        except Exception as exc:
            logger.warning("Diagnostic plot generation failed: %s", exc)

    # =====================================================================
    # Summary
    # =====================================================================
    elapsed = time.perf_counter() - t_total
    worst_p95 = max((c["global_p95_px"] for c in consistency), default=0)

    summary = {
        "status": "OK",
        "n_scenes": len(scenes),
        "geographic_edges": len(edges),
        "accepted_sift_edges": n_ok,
        "failed_sift_edges": n_fail,
        "reference_index": ref_idx,
        "reference_name": ref_name,
        "spanning_tree_edges": len(tree_edges),
        "worst_global_p95_px": worst_p95,
        "bagrn_runtime_sec": round(total_bagrn_time, 1),
        "volrn_runtime_sec": round(total_volrn_time, 1),
        "radiometric_overlap_pairs": sum(
            len(r.overlaps) for r in band_results.values()
        ),
        "outputs": {
            "b14_normalized": str(out / "mosaic_B14_SIFT_BAGRN_VOLRN_weighted.tif"),
            "b8_normalized": str(out / "mosaic_B8_SIFT_BAGRN_VOLRN_weighted.tif"),
            "b5_normalized": str(out / "mosaic_B5_SIFT_BAGRN_VOLRN_weighted.tif"),
            "rgb_geotiff": str(rgb_path),
            "rgb_preview": str(preview_path),
        },
        "runtime_sec": elapsed,
    }
    with open(out / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    _print_summary(summary, consistency, tree_edges, band_results)

    return summary


def _print_summary(summary, consistency, tree_edges, band_results=None):
    """Print end-of-run console summary."""
    print("\n" + "=" * 60)
    print("Five-scene SIFT + BAGRN + VOLRN mosaic complete")
    print("=" * 60)
    print(f"\nScenes: {summary['n_scenes']}")
    print(f"Reference: [{summary['reference_index']}] {summary['reference_name']}")
    print(f"Accepted SIFT edges: {summary['accepted_sift_edges']}")
    print(f"Radiometric overlap pairs: {summary.get('radiometric_overlap_pairs', 'N/A')}")
    print(f"\nBAGRN runtime: {summary.get('bagrn_runtime_sec', 'N/A')} s")
    print(f"VOLRN runtime: {summary.get('volrn_runtime_sec', 'N/A')} s")
    print(f"Total runtime: {summary['runtime_sec']:.1f} s")
    print(f"\nFinal outputs:")
    for k, v in summary["outputs"].items():
        print(f"  {k}: {v}")
    print("=" * 60)