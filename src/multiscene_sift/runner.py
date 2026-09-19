"""Five-scene matcher-selectable registration and radiometric mosaic runner."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from src.cloud_mask import generate_cloud_masks, DEFAULT_CLOUD_MASK_PARAMS
from src.multiscene_sift.dataset import discover_five_scenes, save_manifest
from src.multiscene_sift.overlap_graph import (
    build_geographic_overlap_graph,
    save_overlap_graph,
)
from src.multiscene_sift.pairwise import run_all_pairs, SUPPORTED_MATCHERS
from src.multiscene_sift.global_registration import (
    build_accepted_graph,
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
from src.multiscene_sift.summary import (
    collect_registration_summary,
    save_registration_summary,
    save_mosaic_coverage_summary,
)

logger = logging.getLogger(__name__)

DEFAULT_SCENE_NAMES = [
    "DZ01V_L2_E113.4_N36.6_20260810030932_01_T1",
    "DZ01V_L2_E113.6_N36.3_20260616031133_01_T1",
    "DZ01V_L2_E113.0_N36.4_20260222031837_01_T1",
    "DZ01V_L2_E114.0_N36.4_20260714030410_01_T1",
    "DZ01V_L2_E113.7_N36.6_20260616031127_01_T1",
]


def _normalize_matcher(matcher: str) -> str:
    matcher = str(matcher).strip().lower()
    if matcher not in SUPPORTED_MATCHERS:
        raise ValueError(
            f"matcher={matcher!r} is invalid; expected one of {SUPPORTED_MATCHERS}"
        )
    return matcher


def _matcher_tag(matcher: str) -> str:
    return "SIFT" if matcher == "sift" else "LoFTR"


def run_five_scene_mosaic(
    input_root: str,
    output_dir: str,
    registration_band: str = "B14",
    bands: tuple[str, ...] = ("B14", "B8", "B5"),
    match_max_side: int = 1600,
    ransac_threshold: float = 2.0,
    random_seed: int = 0,
    radiometric_control_idx: int = 0,
    scene_names: list[str] | None = None,
    save_diagnostics: bool = False,
    block_size: int = 200,
    lambda_param: float = 0.1,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-4,
    matcher: str = "sift",
) -> dict:
    """Run the shared five-scene pipeline with SIFT or LoFTR tie points.

    The selected matcher affects only tie-point generation. Both methods share
    the same common-grid preprocessing, affine RANSAC, graph/MST/global
    transforms, Scheme-A north-up warp, cloud-aware BAGRN/VOLRN, metrics, and
    weighted mosaic.
    """
    matcher = _normalize_matcher(matcher)
    matcher_tag = _matcher_tag(matcher)
    t_total = time.perf_counter()

    if scene_names is None:
        scene_names = DEFAULT_SCENE_NAMES
    bands = tuple(dict.fromkeys(bands))
    discovery_bands = (
        bands if registration_band in bands else (registration_band,) + bands
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "input_root": input_root,
        "output_dir": str(out),
        "matcher": matcher,
        "registration_band": registration_band,
        "bands": list(bands),
        "match_max_side": match_max_side,
        "ransac_threshold": ransac_threshold,
        "random_seed": random_seed,
        "radiometric_control_index": radiometric_control_idx,
        "cloud_mask_enabled": True,
        "cloud_mask_bands": ["B2", "B5", "B7", "B14"],
        "cloud_mask_params": DEFAULT_CLOUD_MASK_PARAMS,
        "scene_names": scene_names,
    }
    with open(out / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 1. Dataset discovery -------------------------------------------------
    logger.info("=== Step 1: Dataset Discovery ===")
    scenes, manifest = discover_five_scenes(
        input_root, scene_names, bands=discovery_bands
    )
    save_manifest(manifest, out / "dataset_manifest.json")
    logger.info("Discovered %d scenes", len(scenes))

    cloud_masks = generate_cloud_masks(scenes, output_dir=out / "cloud_masks")
    cloud_mask_summary = [
        {
            "scene_index": idx,
            "scene_name": scenes[idx].name,
            "detected_cloud_fraction": rec.cloud_fraction,
            "metadata_cloud_cover": rec.metadata_cloud_cover,
        }
        for idx, rec in enumerate(cloud_masks)
    ]
    with open(out / "cloud_mask_summary.json", "w") as f:
        json.dump(cloud_mask_summary, f, indent=2)

    if radiometric_control_idx < 0 or radiometric_control_idx >= len(scenes):
        raise ValueError(
            f"radiometric_control_idx={radiometric_control_idx} out of range "
            f"[0, {len(scenes)})"
        )
    radiometric_control_name = scenes[radiometric_control_idx].name
    config["radiometric_control_name"] = radiometric_control_name
    with open(out / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info(
        "Radiometric control: [%d] %s",
        radiometric_control_idx,
        radiometric_control_name,
    )

    # 2. Geographic overlap graph ----------------------------------------
    logger.info("=== Step 2: Geographic Overlap Graph ===")
    edges = build_geographic_overlap_graph(scenes, registration_band)
    save_overlap_graph(edges, out)
    logger.info("Geographic graph: %d edges", len(edges))

    # 3. Pairwise matcher + shared affine RANSAC -------------------------
    logger.info(
        "=== Step 3: Pairwise %s Registration on %s ===",
        matcher_tag,
        registration_band,
    )
    pairwise_results = run_all_pairs(
        scenes,
        edges,
        out,
        band=registration_band,
        match_max_side=match_max_side,
        ransac_threshold=ransac_threshold,
        random_seed=random_seed,
        matcher=matcher,
    )
    n_ok = sum(1 for r in pairwise_results if r.status == "OK")
    n_fail = len(pairwise_results) - n_ok
    logger.info("Pairwise: %d OK, %d failed", n_ok, n_fail)

    # 4. Accepted graph ---------------------------------------------------
    logger.info("=== Step 4: Accepted %s Graph ===", matcher_tag)
    try:
        adj, accepted = build_accepted_graph(
            pairwise_results, matcher_name=matcher_tag
        )
    except RuntimeError as exc:
        logger.error("STOP: %s", exc)
        components_data = {
            "error": str(exc),
            "connected": False,
            "matcher": matcher,
        }
        with open(out / "accepted_graph_components.json", "w") as f:
            json.dump(components_data, f, indent=2)

        elapsed = time.perf_counter() - t_total
        reg_summary = collect_registration_summary(
            matcher=matcher,
            random_seed=random_seed,
            geographic_edges=len(edges),
            pairwise_results=pairwise_results,
            consistency=[],
            bagrn_runtime_sec=0.0,
            volrn_runtime_sec=0.0,
            total_runtime_sec=elapsed,
        )
        save_registration_summary(reg_summary, out)

        disconnected_summary = {
            "status": "DISCONNECTED",
            "error": str(exc),
            "matcher": matcher,
            "output_dir": str(out),
            "runtime_sec": elapsed,
            "n_scenes": len(scenes),
            "geographic_edges": len(edges),
            "accepted_edges": n_ok,
            "failed_edges": n_fail,
            "random_seed": random_seed,
            "geometry_reference_index": None,
            "geometry_reference_name": None,
            "radiometric_control_index": radiometric_control_idx,
            "radiometric_control_name": radiometric_control_name,
            "cloud_mask_enabled": True,
            "cloud_masks": cloud_mask_summary,
            "reference_index": None,
            "reference_name": None,
            "spanning_tree_edges": 0,
            "outputs": {},
            "worst_global_p95_px": None,
        }
        if matcher == "sift":
            disconnected_summary["accepted_sift_edges"] = n_ok
            disconnected_summary["failed_sift_edges"] = n_fail
        with open(out / "run_summary.json", "w") as f:
            json.dump(disconnected_summary, f, indent=2)
        return disconnected_summary

    logger.info("Accepted graph: %d edges, connected", len(accepted))

    # 5. Geometry reference ----------------------------------------------
    logger.info("=== Step 5: Reference Selection ===")
    ref_info = select_reference_scene(adj, accepted)
    geometry_ref_idx = ref_info["reference_index"]
    geometry_ref_name = (
        scenes[geometry_ref_idx].name
        if geometry_ref_idx < len(scenes)
        else "unknown"
    )
    ref_info["reference_name"] = geometry_ref_name
    logger.info(
        "Geometry reference: [%d] %s (degree=%d)",
        geometry_ref_idx,
        geometry_ref_name,
        ref_info["degree"],
    )

    # 6. Maximum spanning tree -------------------------------------------
    logger.info("=== Step 6: Maximum Spanning Tree ===")
    tree_edges = build_spanning_tree(adj, accepted, geometry_ref_idx)
    logger.info("Spanning tree: %d edges", len(tree_edges))

    # 7. Global transforms ------------------------------------------------
    logger.info("=== Step 7: Global World Transforms ===")
    G = compose_global_transforms(
        scenes, accepted, tree_edges, geometry_ref_idx
    )
    save_global_registration_info(ref_info, tree_edges, G, out)

    # 8. Point-level global consistency ----------------------------------
    logger.info("=== Step 8: Global Consistency Diagnostics ===")
    with __import__("rasterio").open(
        scenes[0].band_paths[registration_band]
    ) as src:
        pixel_size = abs(src.transform.a)
    consistency = global_consistency_diagnostics(
        accepted, G, tree_edges, pixel_size=pixel_size
    )
    save_consistency_diagnostics(consistency, out)
    flagged = [c for c in consistency if c["global_p95_px"] > 2.0]
    if flagged:
        logger.warning(
            "Global consistency: %d edges with P95 > 2 px", len(flagged)
        )
    else:
        logger.info("Global consistency: all evaluated edges within 2 px")

    # 9. Shared north-up mosaic grid -------------------------------------
    logger.info("=== Step 9: Shared Mosaic Grid ===")
    mosaic_grid = compute_shared_mosaic_grid(
        scenes, G, band=registration_band
    )
    logger.info(
        "Grid: %dx%d px @ %.2f m",
        mosaic_grid.width,
        mosaic_grid.height,
        mosaic_grid.resolution,
    )

    # 10. Optional geometry diagnostics ----------------------------------
    if save_diagnostics:
        logger.info("=== Step 10: Diagnostic Mosaics ===")
        make_mosaic_original_transforms(
            scenes,
            registration_band,
            mosaic_grid,
            out / f"mosaic_{registration_band}_before_registration_source_selection.tif",
            mode="source_selection",
        )
        make_mosaic(
            scenes,
            G,
            registration_band,
            mosaic_grid,
            out / f"mosaic_{registration_band}_after_{matcher_tag}_registration_source_selection.tif",
            mode="source_selection",
        )

    # 11. Radiometric normalization + final mosaics ----------------------
    logger.info("=== Step 11: Radiometric Normalization ===")
    band_results: dict[str, BandRadiometricResult] = {}
    mosaic_paths: dict[str, str] = {}
    coverage_by_band: dict[str, dict] = {}
    total_bagrn_time = 0.0
    total_volrn_time = 0.0

    from src.mosaic import create_mosaic

    for band in bands:
        logger.info("--- Processing band: %s ---", band)
        result = normalize_registered_band(
            scenes,
            G,
            band,
            radiometric_control_idx,
            registration_grid=mosaic_grid,
            block_size_pixels=block_size,
            lambda_param=lambda_param,
            rho=rho,
            max_iter=max_iter,
            tol=tol,
            cloud_masks=cloud_masks,
        )
        band_results[band] = result
        total_bagrn_time += result.bagrn_runtime_sec
        total_volrn_time += result.volrn_runtime_sec

        mosaic_path = out / (
            f"mosaic_{band}_{matcher_tag}_BAGRN_VOLRN_weighted.tif"
        )
        _, mosaic_diag = create_mosaic(
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
            return_diagnostics=True,
        )
        mosaic_paths[band] = str(mosaic_path)
        coverage_by_band[band] = {
            "union_valid_pixels": int(mosaic_diag["union_valid_pixels"]),
            "final_valid_pixels": int(mosaic_diag["final_valid_pixels"]),
            "union_but_final_invalid_pixels": int(
                mosaic_diag["union_but_final_invalid_pixels"]
            ),
            "coverage_hole_ratio": float(mosaic_diag["coverage_hole_ratio"]),
        }

        release_band_result(result)
        logger.info("--- Band %s complete ---", band)

    save_radiometric_metrics(band_results, out)
    save_normalization_info(
        band_results,
        radiometric_control_idx,
        radiometric_control_name,
        out,
    )

    if coverage_by_band:
        coverage_band = (
            registration_band
            if registration_band in coverage_by_band
            else next(iter(coverage_by_band))
        )
        mosaic_coverage_summary = {
            "band": coverage_band,
            **coverage_by_band[coverage_band],
            "per_band": coverage_by_band,
        }
        save_mosaic_coverage_summary(mosaic_coverage_summary, out)
    else:
        mosaic_coverage_summary = None

    # 12-13. Optional three-band product ---------------------------------
    rgb_path = None
    preview_path = None
    required_rgb = {"B14", "B8", "B5"}
    if required_rgb.issubset(mosaic_paths):
        logger.info("=== Step 12: B14-B8-B5 Three-Band GeoTIFF ===")
        rgb_path = out / (
            f"mosaic_RGB_B14_B8_B5_{matcher_tag}_BAGRN_VOLRN.tif"
        )
        stack_three_band_geotiff(
            {
                "R": mosaic_paths["B14"],
                "G": mosaic_paths["B8"],
                "B": mosaic_paths["B5"],
            },
            rgb_path,
        )

        logger.info("=== Step 13: RGB Preview PNG ===")
        preview_path = out / (
            f"mosaic_RGB_B14_B8_B5_{matcher_tag}_BAGRN_VOLRN_preview.png"
        )
        make_rgb_preview(
            rgb_path, preview_path, out / "rgb_preview_stretch.json"
        )
    else:
        logger.info(
            "Skipping RGB product because processed bands are %s", list(bands)
        )

    # 14. Optional plots --------------------------------------------------
    if save_diagnostics:
        logger.info("=== Step 14: Diagnostic Plots ===")
        diag_dir = out / "diagnostics"
        diag_dir.mkdir(exist_ok=True)
        try:
            draw_footprints(
                scenes,
                diag_dir / "scene_footprints_before.png",
                "Scene Footprints (Before Registration)",
            )
            draw_footprints(
                scenes,
                diag_dir / "scene_footprints_after.png",
                "Scene Footprints (After Registration)",
                corrected=True,
                G=G,
            )
            draw_overlap_graph(edges, scenes, diag_dir / "overlap_graph.png")
            # Legacy plotting helper name is SIFT-specific, but it accepts a title
            # and plots status only; downstream registration remains matcher-agnostic.
            draw_accepted_sift_graph(
                accepted,
                pairwise_results,
                scenes,
                diag_dir / f"accepted_{matcher}_graph.png",
                title=f"Accepted {matcher_tag} Graph",
            )
            draw_spanning_tree(
                tree_edges, scenes, diag_dir / "spanning_tree.png"
            )
            draw_global_consistency(
                consistency, diag_dir / "global_edge_consistency.png"
            )
        except Exception as exc:
            logger.warning("Diagnostic plot generation failed: %s", exc)

    # Final summaries -----------------------------------------------------
    elapsed = time.perf_counter() - t_total
    registration_summary = collect_registration_summary(
        matcher=matcher,
        random_seed=random_seed,
        geographic_edges=len(edges),
        pairwise_results=pairwise_results,
        consistency=consistency,
        bagrn_runtime_sec=total_bagrn_time,
        volrn_runtime_sec=total_volrn_time,
        total_runtime_sec=elapsed,
    )
    save_registration_summary(registration_summary, out)

    worst_p95 = max(
        (c["global_p95_px"] for c in consistency), default=0.0
    )
    outputs = {
        f"{band.lower()}_normalized": path
        for band, path in mosaic_paths.items()
    }
    if rgb_path is not None:
        outputs["rgb_geotiff"] = str(rgb_path)
    if preview_path is not None:
        outputs["rgb_preview"] = str(preview_path)

    summary = {
        "status": "OK",
        "matcher": matcher,
        "n_scenes": len(scenes),
        "geographic_edges": len(edges),
        "accepted_edges": n_ok,
        "failed_edges": n_fail,
        "random_seed": random_seed,
        "geometry_reference_index": geometry_ref_idx,
        "geometry_reference_name": geometry_ref_name,
        "radiometric_control_index": radiometric_control_idx,
        "radiometric_control_name": radiometric_control_name,
        "cloud_mask_enabled": True,
        "cloud_masks": cloud_mask_summary,
        "reference_index": geometry_ref_idx,
        "reference_name": geometry_ref_name,
        "spanning_tree_edges": len(tree_edges),
        "worst_global_p95_px": worst_p95,
        "matching_runtime_sec": registration_summary["matching_runtime_sec"],
        "geometry_runtime_sec": registration_summary["geometry_runtime_sec"],
        "pairwise_total_runtime_sec": registration_summary[
            "pairwise_total_runtime_sec"
        ],
        "bagrn_runtime_sec": round(total_bagrn_time, 6),
        "volrn_runtime_sec": round(total_volrn_time, 6),
        "radiometric_overlap_pairs": sum(
            len(r.overlaps) for r in band_results.values()
        ),
        "mosaic_coverage": mosaic_coverage_summary,
        "outputs": outputs,
        "runtime_sec": elapsed,
    }
    if matcher == "sift":
        summary["accepted_sift_edges"] = n_ok
        summary["failed_sift_edges"] = n_fail

    with open(out / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary)
    return summary


def _print_summary(summary: dict) -> None:
    """Print a compact matcher-independent run summary."""
    print("\n" + "=" * 60)
    print(
        f"Five-scene {summary['matcher'].upper()} + BAGRN + VOLRN mosaic complete"
    )
    print("=" * 60)
    print(f"\nScenes: {summary['n_scenes']}")
    print(
        "Geometry reference: "
        f"[{summary['geometry_reference_index']}] "
        f"{summary['geometry_reference_name']}"
    )
    print(
        "Radiometric control: "
        f"[{summary['radiometric_control_index']}] "
        f"{summary['radiometric_control_name']}"
    )
    print(f"Accepted edges: {summary['accepted_edges']}")
    print(f"Failed edges: {summary['failed_edges']}")
    print(
        "Matching runtime: "
        f"{summary.get('matching_runtime_sec', 'N/A')} s"
    )
    print(
        "RANSAC geometry runtime: "
        f"{summary.get('geometry_runtime_sec', 'N/A')} s"
    )
    print(f"BAGRN runtime: {summary.get('bagrn_runtime_sec', 'N/A')} s")
    print(f"VOLRN runtime: {summary.get('volrn_runtime_sec', 'N/A')} s")
    print(f"Total runtime: {summary['runtime_sec']:.1f} s")
    print("\nFinal outputs:")
    for key, value in summary["outputs"].items():
        print(f"  {key}: {value}")
    print("=" * 60)
