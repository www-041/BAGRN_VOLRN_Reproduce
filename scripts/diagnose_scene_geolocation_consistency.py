"""CLI: scene-geolocation consistency diagnosis for the 0-1/0-4/1-4 triangle.

Reads B14 bands, the native geotransforms and the saved pairwise artefacts;
builds the metadata-predicted and direct (frame-normalised) relations per
edge, samples the world correction field, measures the metadata-only local
phase displacement field, closes the world-correction triangle, tests the
scene-level constant-correction model, and classifies the geolocation
inconsistency.  No matcher/RANSAC/MST/BAGRN/VOLRN is run or modified.

Usage (PowerShell)::

    python -m scripts.diagnose_scene_geolocation_consistency `
        --run-dir "data/output/five_scene_sift_B14" `
        --frame-diagnostics-dir "data/output/five_scene_sift_B14/common_grid_anchor_diagnostics" `
        --loop-diagnostics-dir "data/output/five_scene_sift_B14/loop_diagnostics" `
        --output-dir "data/output/five_scene_sift_B14/scene_geolocation_consistency"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from src.multiscene_sift import geolocation_diagnostics as gd
from src.multiscene_sift.frame_diagnostics import (
    match_view_frame_params,
    rebuild_pair_common_grids,
    write_json,
)
from src.multiscene_sift.loop_diagnostics import (
    load_pairwise_results,
    load_run_config,
    load_scenes_for_run,
)
from src.multiscene_sift.models import Scene

EDGES = [(0, 1), (0, 4), (4, 1)]
PAIRS = [(0, 1), (0, 4), (1, 4)]


def _world_sample_points(
    scene_i: Scene, scene_j: Scene, band: str, fracs=(0.1, 0.5, 0.9)
) -> np.ndarray:
    """A few world points inside the two scenes' overlap."""
    b_i = scene_i.bounds[band]
    b_j = scene_j.bounds[band]
    left = max(b_i.left, b_j.left)
    bottom = max(b_i.bottom, b_j.bottom)
    right = min(b_i.right, b_j.right)
    top = min(b_i.top, b_j.top)
    pts = []
    for fy in fracs:
        for fx in fracs:
            pts.append((left + fx * (right - left), bottom + fy * (top - bottom)))
    return np.asarray(pts, dtype=np.float64)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--frame-diagnostics-dir", required=True)
    parser.add_argument("--loop-diagnostics-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tile-grid", type=int, default=4)
    parser.add_argument("--min-joint-valid-fraction", type=float, default=0.6)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    frame_dir = Path(args.frame_diagnostics_dir)
    loop_dir = Path(args.loop_diagnostics_dir) if args.loop_diagnostics_dir \
        else run_dir / "loop_diagnostics"
    out_dir = Path(args.output_dir) if args.output_dir else (
        run_dir / "scene_geolocation_consistency"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not frame_dir.is_dir():
        print(f"ERROR: frame diagnostics dir not found: {frame_dir}")
        return 1

    # --- 00 baseline -------------------------------------------------------
    baseline = gd.load_geolocation_baseline(run_dir, frame_dir)
    write_json(out_dir / "00_baseline.json", baseline)

    config = load_run_config(run_dir)
    band = str(config.get("registration_band", "B14"))
    scenes = load_scenes_for_run(run_dir)
    pair_rows = load_pairwise_results(run_dir / "pairwise_summary.json")

    # --- rebuild pair grids (frame diagnostics layer) ----------------------
    grids = rebuild_pair_common_grids(scenes, band, [(0, 1), (0, 4), (1, 4)])
    grid_map = {f"pair_{i}_{j}": grids[f"pair_{i}_{j}"] for i, j in PAIRS}
    match_views = {
        name: match_view_frame_params(
            tuple(grid_map[name]["overlap_window"]),
            int(config.get("match_max_side", 1600)),
        )
        for name in grid_map
    }

    # --- 01 metadata vs direct transforms ----------------------------------
    transforms = gd.edge_world_corrections(
        scenes, band, pair_rows, grid_map, match_views,
        [(0, 1), (0, 4), (4, 1)],
    )
    # Re-key so the returned names are edge_0_1 / edge_0_4 / edge_4_1.
    transforms = {f"edge_{i}_{j}": transforms[f"edge_{i}_{j}"] for i, j in EDGES}
    write_json(out_dir / "01_metadata_vs_direct_transforms.json", transforms)

    # --- 02/03 correction field sampling (matrix level) --------------------
    field_rows: list[dict] = []
    field_summaries: dict = {}
    for i, j in EDGES:
        t = transforms.get(f"edge_{i}_{j}")
        if not t or not t.get("available"):
            continue
        shape_j = scenes[j].shapes[band]
        samples = gd.sample_correction_field(
            np.asarray(t["T_meta_i_from_j"]),
            np.asarray(t["T_direct_i_from_j"]),
            shape_j,
        )
        for s in samples:
            row = {"edge_i": i, "edge_j": j, **s}
            field_rows.append(row)
        field_summaries[f"edge_{i}_{j}"] = gd.correction_field_summary(samples)
    _write_csv(
        out_dir / "02_correction_field_samples.csv",
        ["edge_i", "edge_j", "sample_id", "sample_fraction_x", "sample_fraction_y",
         "j_col", "j_row", "meta_i_col", "meta_i_row", "direct_i_col",
         "direct_i_row", "correction_dx_px", "correction_dy_px",
         "correction_dx_m", "correction_dy_m", "correction_mag_m", "valid"],
        field_rows,
    )
    write_json(out_dir / "03_correction_field_summary.json", field_summaries)

    # --- 04/05 metadata-only tile displacement fields (imagery) ------------
    tile_rows: list[dict] = []
    shift_summaries: dict = {}
    edge_tiles: dict[str, list[dict]] = {}
    for i, j in EDGES:
        tiles = gd.measure_metadata_only_tile_shifts(
            scenes[i], scenes[j], band,
            tile_grid=max(args.tile_grid, 2),
            min_joint_valid_fraction=args.min_joint_valid_fraction,
        )
        for t in tiles:
            row = {"edge_i": i, "edge_j": j, **t}
            tile_rows.append(row)
        edge_tiles[f"edge_{i}_{j}"] = tiles
        shift_summaries[f"edge_{i}_{j}"] = gd.metadata_only_shift_summary(tiles)
    _write_csv(
        out_dir / "04_metadata_only_tile_shifts.csv",
        ["edge_i", "edge_j", "tile_row", "tile_col", "center_world_x",
         "center_world_y", "joint_valid_fraction", "texture_i", "texture_j",
         "phase_dx_px", "phase_dy_px", "phase_dx_m", "phase_dy_m",
         "phase_magnitude_px", "phase_confidence", "accepted"],
        tile_rows,
    )
    write_json(out_dir / "05_metadata_only_shift_summary.json", shift_summaries)

    # --- 06 world-correction triangle closure -------------------------------
    triangle_closure = None
    sample_world = _world_sample_points(scenes[0], scenes[1], band)
    if all(transforms.get(f"edge_{i}_{j}", {}).get("available") for i, j in EDGES):
        C_0_1 = np.asarray(transforms["edge_0_1"]["C_world_i_from_j"])
        C_0_4 = np.asarray(transforms["edge_0_4"]["C_world_i_from_j"])
        C_4_1 = np.asarray(transforms["edge_4_1"]["C_world_i_from_j"])
        triangle_closure = gd.world_correction_triangle_closure(
            C_0_1, C_0_4, C_4_1, sample_world, pixel_size=14.0
        )
        # closure samples csv (per-point displacement)
        pts = np.asarray(sample_world, dtype=np.float64)
        h = np.hstack([pts, np.ones((len(pts), 1))])
        direct = (C_0_1 @ h.T).T[:, :2]
        composed = (C_0_4 @ (C_4_1 @ h.T)).T[:, :2]
        mag = np.linalg.norm((composed - direct) / 14.0, axis=1)
        closure_rows = [
            {"point_id": k, "world_x": pts[k, 0], "world_y": pts[k, 1],
             "closure_dx_px": (composed[k] - direct[k])[0] / 14.0,
             "closure_dy_px": (composed[k] - direct[k])[1] / 14.0,
             "closure_magnitude_px": float(mag[k])}
            for k in range(len(pts))
        ]
        _write_csv(out_dir / "06_world_correction_triangle_samples.csv",
                   ["point_id", "world_x", "world_y", "closure_dx_px",
                    "closure_dy_px", "closure_magnitude_px"], closure_rows)
        write_json(out_dir / "06_world_correction_triangle_closure.json",
                   triangle_closure)

    # --- 07 scene-level constant-correction model ---------------------------
    model = None
    if triangle_closure is not None:
        model = gd.scene_level_constant_correction_test(
            C_0_1, C_0_4, C_4_1, sample_world, pixel_size=14.0
        )
        write_json(out_dir / "07_scene_level_constant_correction_test.json", model)

    # --- 08 within-overlap spatial variation --------------------------------
    spatial = {name: gd.within_overlap_spatial_variation(tiles)
               for name, tiles in edge_tiles.items()}
    write_json(out_dir / "08_within_overlap_spatial_variation.json", spatial)

    # --- 09/10/11 figures ----------------------------------------------------
    try:
        gd.plot_metadata_only_shift_vectors(
            edge_tiles, out_dir / "09_metadata_only_shift_vectors.png"
        )
        edge_models = {
            f"edge_{i}_{j}": np.asarray(transforms[f"edge_{i}_{j}"][
                "T_direct_i_from_j"])
            for i, j in EDGES if transforms[f"edge_{i}_{j}"].get("available")
        }
        edge_scenes = {
            f"edge_{i}_{j}": (
                gd.get_native_pixel_to_world(scenes[i], band),
                gd.get_native_pixel_to_world(scenes[j], band),
            )
            for i, j in EDGES
        }
        gd.plot_metadata_vs_direct_corrections(
            edge_tiles, edge_models, edge_scenes,
            out_dir / "10_metadata_vs_direct_corrections.png",
        )
        edge_summaries = {}
        for i, j in EDGES:
            name = f"edge_{i}_{j}"
            tiles = [t for t in edge_tiles.get(name, []) if t.get("accepted")]
            center = (
                (np.median([t["center_world_x"] for t in tiles]),
                 np.median([t["center_world_y"] for t in tiles]))
                if tiles else (0.0, 0.0)
            )
            tra_meta = transforms.get(name, {}).get("translation_m", [0.0, 0.0])
            # mean correction seen in the metadata-only tiles (the world
            # correction implied by the content) is close to the correction
            # field; report the matrix translation plus tile spread.
            tight = shift_summaries.get(name, {})
            std_mag = (tight.get("std_dx_px") or 0.0) * 14.0
            edge_summaries[name] = {
                "overlap_center_world": center,
                "mean_correction_m": list(tra_meta),
                "std_mag_m": std_mag,
            }
        gd.plot_triangle_correction_map(
            edge_summaries, out_dir / "11_triangle_correction_map.png"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: figures skipped: {type(exc).__name__}: {exc}")

    # --- 10/11 classifier + summary -----------------------------------------
    spatial_flags = {
        k: v.get("classification", "?") for k, v in spatial.items()
    }
    spatial_supported = "SPATIALLY_VARYING" in spatial_flags.values()
    conflict_flag = False
    for i, j in EDGES:
        name = f"edge_{i}_{j}"
        t = transforms.get(name, {})
        shift = shift_summaries.get(name, {})
        if not t.get("available") or not shift.get("n_tiles_accepted"):
            continue
        corr = t.get("translation_equiv_px14", [0.0, 0.0])
        phase_mean = (shift.get("mean_dx_px", 0.0) or 0.0,
                      shift.get("mean_dy_px", 0.0) or 0.0)
        phase_mag = np.hypot(*phase_mean)
        direct_mag = np.hypot(corr[0], corr[1])
        if phase_mag > 10.0 and abs(phase_mag - direct_mag) > 8.0:
            conflict_flag = True

    evidence = {
        "triangle_closure_mean_px": (
            triangle_closure["sample_mean_px"] if triangle_closure else None
        ),
        "scene_level_constant_model": (
            model.get("status") if model else None
        ),
        "spatial_variation_supported": spatial_supported,
        "within_overlap_displacement_constant": all(
            field_summaries.get(f"edge_{i}_{j}", {}).get("classification")
            == "NEAR_CONSTANT_WITHIN_OVERLAP"
            for i, j in EDGES
        ),
        "direct_edge_phase_conflict_with_metadata": conflict_flag,
    }
    final_state = gd.classify_geolocation_inconsistency(evidence)
    summary = gd.build_geolocation_summary(
        transforms=transforms,
        correction_fields=field_summaries,
        triangle_closure=triangle_closure or {},
        scene_model=model or {},
        spatial_variation=spatial,
        metadata_shifts=shift_summaries,
        final_state=final_state,
        output_dir=str(out_dir),
    )
    write_json(out_dir / "12_geolocation_consistency_summary.json", summary)
    gd.write_geolocation_summary_text(summary, out_dir / "12_geolocation_consistency_summary.txt")

    print("\n" + "=" * 66)
    print(f"Geolocation consistency diagnosis complete -> {out_dir}")
    for i, j in EDGES:
        name = f"edge_{i}_{j}"
        t = transforms.get(name, {})
        corr = t.get("translation_equiv_px14")
        if corr:
            print(f"  {name} metadata->direct correction: ({corr[0]:+.2f}, "
                  f"{corr[1]:+.2f}) px14")
    if triangle_closure:
        print(
            f"Triangle closure: mean={triangle_closure['sample_mean_px']:.2f} "
            f"px  ({triangle_closure['classification']})"
        )
    if model:
        print(f"Scene-level constant model: {model['status']}")
    print(f"Within-overlap variation: {spatial_flags}")
    print(f"Final state: {final_state}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())