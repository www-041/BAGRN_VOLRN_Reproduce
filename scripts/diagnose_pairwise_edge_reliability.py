"""CLI: pairwise edge-reliability / false-good diagnosis for 7 edges.

Reuses the frozen five-scene SIFT baseline to recover real inlier coordinates,
computes inlier spatial-support metrics and nearest-control distance maps,
builds direct-warp overlaps, measures per-tile local phase residual fields and
their coupling to control support, and produces an evidence-based diagnostic
for each edge.  No SIFT/RANSAC/Affine/MST semantics change and no ML training.

Usage (PowerShell)::

    &.\\".venv-registration\\Scripts\\python.exe" `
        -m scripts.diagnose_pairwise_edge_reliability `
        --input-root "D:/科研/地质一号/文献/BAGRN_VOLRN_Reproduce/data/input/flat" `
        --five-scene-run-dir "data/output/five_scene_sift_B14" `
        --selected-pair-dir "data/output/nine_scene_selected_pair_consistency" `
        --dense-01-dir "data/output/five_scene_sift_B14/dense_01_phase_diagnostics" `
        --loop-diag-dir "data/output/five_scene_sift_B14/loop_diagnostics" `
        --output-dir "data/output/edge_reliability_diagnostics"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from rasterio.transform import Affine

from src.multiscene_sift import edge_reliability_diagnostics as erd
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
from src.multiscene_sift.nine_scene_overlap_diagnostic import SCENE_NAMES_9
from src.multiscene_sift.dataset import discover_five_scenes

BAND = "B14"
QUALITY_CONFIG = {"min_joint_valid_fraction": 0.60, "min_tile_size_px": 128,
                  "min_texture_std": 1.0, "max_phase_error_px": 1.0}


def _csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", required=True)
    p.add_argument("--five-scene-run-dir", required=True)
    p.add_argument("--selected-pair-dir", required=True)
    p.add_argument("--dense-01-dir", default=None)
    p.add_argument("--loop-diag-dir", default=None)
    p.add_argument("--output-dir", default="data/output/edge_reliability_diagnostics")
    args = p.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- 00 baseline ----------------
    baseline = erd.load_edge_reliability_baseline(
        args.selected_pair_dir, args.five_scene_run_dir, args.dense_01_dir)
    write_json(out / "00_edge_reliability_baseline.json",
               {"edges": baseline, "groups": erd.EDGE_GROUPS})

    # ---------------- scenes, config, artifacts ----------------
    config = load_run_config(args.five_scene_run_dir)
    scenes, _ = discover_five_scenes(args.input_root, list(SCENE_NAMES_9),
                                     bands=(BAND,))
    pairs_needed = [(i, j) for i, j in erd.DIAGNOSTIC_EDGES]
    grids = rebuild_pair_common_grids(scenes, BAND, pairs_needed)
    grid_map = {f"pair_{min(i, j)}_{max(i, j)}": grids[f"pair_{min(i, j)}_{max(i, j)}"]
                for i, j in pairs_needed}
    mv_map = {name: match_view_frame_params(
        tuple(grid_map[name]["overlap_window"]),
        int(config.get("match_max_side", 1600))) for name in grid_map}

    saved_rows = load_pairwise_results(Path(args.five_scene_run_dir)
                                       / "pairwise_summary.json")


    # 0-1 inliers from existing loop-diagnostics point table
    old_01_points = None
    if args.loop_diag_dir:
        p04 = Path(args.loop_diag_dir) / "04_point_residuals.csv"
        if p04.is_file():
            rows = list(csv.DictReader(open(p04, encoding="utf-8")))
            old_01_points = np.array([[float(r["ref_x"]), float(r["ref_y"])]
                                      for r in rows])

    # ---------------- per-edge analysis ----------------
    feature_rows = []
    diag_rows = []
    reproduction = []
    point_csv = []
    spatial_csv = []
    dist_csv = []
    res_tile_csv = []
    support_csv = []
    residual_summary = {}
    dist_summary = {}
    phase_summary = {}
    support_summary = {}

    for (i, j) in erd.DIAGNOSTIC_EDGES:
        name = f"pair_{min(i, j)}_{max(i, j)}"
        key = f"{i}-{j}"
        grid = grid_map[name]
        mv = mv_map[name]
        # recover inliers (replay new pairs; reuse 0-1 table)
        reg = None
        saved_row = next((r for r in saved_rows
                          if {int(r["idx_i"]), int(r["idx_j"])} == {i, j}), None)
        n_in = 0
        inlier_mv = None
        status_gate = "SKIPPED_REUSE"
        if (i, j) in ((0, 6), (4, 6), (1, 8), (2, 5), (0, 5), (5, 6)):
            from src.multiscene_sift.network_consistency_diagnostics import (
                load_baseline_registration_config, run_selected_pair_registration)
            bl = load_baseline_registration_config(args.five_scene_run_dir)
            reg = run_selected_pair_registration(scenes[i], scenes[j], bl)
            status_gate = erd.reproduce_gate_status(saved_row, reg)
            inlier_mv = np.asarray(reg.inlier_ref_xy, dtype=np.float64)
            n_in = reg.inliers
        else:
            if old_01_points is not None:
                inlier_mv = np.asarray(old_01_points, dtype=np.float64)
                n_in = len(inlier_mv)
                status_gate = "REUSED_LOOP_04_CSV"
        reproduction.append({"edge": key, "status": status_gate,
                             "n_inliers_recovered": n_in,
                             "n_inliers_saved": (int(saved_row["inliers"])
                                                 if saved_row else None)})

        from src.multiscene_sift.network_consistency_diagnostics import (
            registration_as_row, direct_edge_world_transform)
        t_world = None
        if reg is not None:
            t_world = direct_edge_world_transform(
                [registration_as_row(reg)], i, j, reg.pair_common_transform, mv)
        elif saved_row is not None:
            t_world = direct_edge_world_transform(
                [saved_row], i, j, Affine(*grid["transform"]), mv)
        if t_world is None:
            diag_rows.append({"edge": key,
                              "diagnosis": "MIXED_OR_UNDERDETERMINED",
                              "note": "missing canonical transform/inliers"})
            continue
        print(f"[{key}] direct transform: ok, inliers={n_in}")

        # direct-warp overlap + local residual field (shared frame first)
        overlap = erd.build_direct_warp_overlap(scenes[i], scenes[j], BAND,
                                                t_world, max_side=4096)
        res_field = erd.measure_direct_residual_field(overlap, (4, 6),
                                                      QUALITY_CONFIG)
        res_sum = erd.summarize_direct_residual_field(res_field)
        phase_summary[key] = res_sum

        # inliers -> direct-overlap grid pixels (same frame as residual tiles)
        # NOTE: reg.inlier_ref_xy is already in common-grid pixels (verified
        # range ~ overlap window), so no match-view -> common conversion.
        from rasterio.transform import Affine
        common_px = np.asarray(inlier_mv, dtype=np.float64)
        common_tf = reg.pair_common_transform if reg is not None \
            else Affine(*grid["transform"])
        world = np.array([common_tf * (px, py) for px, py in common_px])
        inv_overlap = ~overlap["transform"]
        grid_px = np.array([inv_overlap * (wx, wy) for wx, wy in world])

        support = erd.relate_residual_to_control_support(
            res_field["tiles"], grid_px, overlap["transform"])
        support_summary[key] = support

        valid_mask = overlap["joint_mask"]
        oshape = (overlap["height"], overlap["width"])
        spatial = erd.compute_inlier_spatial_metrics(grid_px, valid_mask, oshape)
        sample_pts = erd.sample_grid_points(oshape, 32)
        dist = erd.nearest_control_statistics(grid_px, sample_pts)
        dist_summary[key] = dist

        # residual pattern of RANSAC inliers (match-view common frame approx.)
        res = (None if reg is None or reg.inlier_ref_xy is None else
               np.linalg.norm(
                   np.asarray(reg.inlier_ref_xy) - np.asarray(reg.inlier_tgt_xy),
                   axis=1))
        rsp = (erd.summarize_residual_spatial_pattern(common_px, res,
                                                      (overlap["height"],
                                                       overlap["width"]))
               if res is not None else None)
        residual_summary[key] = rsp

        # feature + diagnosis
        diag = erd.classify_edge_reliability_pattern({
            "inliers": n_in,
            "occupancy_8x8": spatial["occupancy_8x8"]["occupancy_ratio"],
            "direct_phase_median_px": res_sum.get("median_px"),
            "fraction_tiles_gt5px": res_sum.get("fraction_tiles_gt5px", 0.0),
            "corr_spearman_phase_vs_distance": support.get("corr_spearman_phase_vs_distance"),
            "fraction_farther_than_128px": dist.get("fraction_farther_than_128px", 0.0),
        })
        base_row = next(b for b in baseline if b["edge"] == [i, j])
        feature_rows.append({
            "edge": key, "group": base_row["group"],
            "raw_matches": base_row["raw_matches"], "inliers": n_in,
            "inlier_ratio": base_row["inlier_ratio"], "rmse_px": base_row["rmse_px"],
            "p95_px": base_row["p95_px"], "old_coverage": base_row["coverage"],
            "occupancy_4x4": round(spatial["occupancy_4x4"]["occupancy_ratio"], 4),
            "occupancy_8x8": round(spatial["occupancy_8x8"]["occupancy_ratio"], 4),
            "hull_coverage": round(spatial["hull_coverage"], 4),
            "quadrant_entropy": round(spatial["quadrants"]["entropy"], 4)
            if spatial["quadrants"]["entropy"] is not None else None,
            "centroid_bias": round(spatial["centroid"]["centroid_offset_normalized"], 4),
            "p95_nearest_control_distance_px": (
                round(dist["p95_nearest_control_px"], 1)
                if dist["p95_nearest_control_px"] is not None else None),
            "fraction_farther_than_128px": round(dist.get("fraction_farther_than_128px", 0.0), 3),
            "full_overlap_phase_residual_px": base_row["full_overlap_phase_residual_px"],
            "local_phase_median_px": (round(res_sum["median_px"], 2)
                                      if res_sum.get("median_px") is not None else None),
            "local_phase_p95_px": (round(res_sum["p95_px"], 2)
                                   if res_sum.get("p95_px") is not None else None),
            "fraction_tiles_gt5px": round(res_sum.get("fraction_tiles_gt5px", 0.0), 3),
            "ncc": base_row["ncc"],
            "corr_phase_vs_control_distance": (
                round(support["corr_spearman_phase_vs_distance"], 3)
                if support.get("corr_spearman_phase_vs_distance") is not None else None),
            "far_minus_near_residual_px": (
                round(support["far_minus_near_median_px"], 2)
                if support.get("far_minus_near_median_px") is not None else None),
            "diagnosis": diag,
        })
        diag_rows.append({"edge": key, "diagnosis": diag,
                          "supporting": support, "residual_summary": res_sum,
                          "spatial": spatial, "dist_stats": dist})

        # point / csv rows
        if reg is not None and reg.inlier_tgt_xy is not None:
            tgt_pts = np.asarray(reg.inlier_tgt_xy, dtype=np.float64)
            resid = np.linalg.norm(inlier_mv - tgt_pts, axis=1)
        else:
            tgt_pts = np.full(inlier_mv.shape, np.nan)
            resid = np.zeros(len(inlier_mv))
        for pid, (ref, tgt) in enumerate(zip(inlier_mv, tgt_pts)):
            point_csv.append({"edge_i": i, "edge_j": j, "point_id": pid,
                              "ref_x": float(ref[0]), "ref_y": float(ref[1]),
                              "tgt_x": float(tgt[0]), "tgt_y": float(tgt[1]),
                              "residual_px": float(resid[pid]),
                              "reproduction_status": status_gate})
        spatial_csv.append({"edge": key, **{
            f"occ8x8": round(spatial["occupancy_8x8"]["occupancy_ratio"], 4),
            "hull_cov": round(spatial["hull_coverage"], 4),
            "q_entropy": round(spatial["quadrants"]["entropy"], 4)
            if spatial["quadrants"]["entropy"] is not None else None,
            "cent_bias": round(spatial["centroid"]["centroid_offset_normalized"], 4),
        }})
        dist_csv.append({"edge": key,
                         **{k: (round(v, 1) if isinstance(v, float) else v)
                            for k, v in dist.items()}})
        for t in res_field["tiles"]:
            res_tile_csv.append({"edge": key, "grid_n": t["grid_n"],
                                 "tile_row": t["tile_row"], "tile_col": t["tile_col"],
                                 "center_world_x": t["center_world_x"],
                                 "center_world_y": t["center_world_y"],
                                 "phase_mag_px": t["phase_mag_px"],
                                 "accepted": t["accepted"],
                                 "reject_reason": t["reject_reason"]})
        for t2 in res_field["tiles"]:
            if not t2.get("accepted"):
                continue
            gpx = ~overlap["transform"] * (t2["center_world_x"], t2["center_world_y"])
            d = float(np.min(np.linalg.norm(grid_px - np.array(gpx), axis=1))) \
                if len(grid_px) else 0.0
            support_csv.append({"edge": key, "tile_grid": t2["grid_n"],
                                "nearest_control_distance_px": d,
                                "phase_mag_px": t2["phase_mag_px"]})

    # ---------------- write products ----------------
    _csv(out / "01_inlier_points.csv",
         ["edge_i", "edge_j", "point_id", "ref_x", "ref_y", "tgt_x", "tgt_y",
          "residual_px", "reproduction_status"], point_csv)
    write_json(out / "01_inlier_reproduction_summary.json", {"edges": reproduction})
    _csv(out / "02_inlier_spatial_metrics.csv",
         ["edge", "occ8x8", "hull_cov", "q_entropy", "cent_bias"], spatial_csv)
    write_json(out / "02_inlier_spatial_metrics.json", {
        "edges": [{"edge": r["edge"], **r["spatial"]}
                  for r in diag_rows if "spatial" in r]})
    _csv(out / "03_control_distance_summary.csv",
         ["edge", "n_samples", "median_nearest_control_px",
          "p90_nearest_control_px", "p95_nearest_control_px",
          "max_nearest_control_px", "fraction_farther_than_64px",
          "fraction_farther_than_128px"], dist_csv)
    write_json(out / "03_control_distance_summary.json", dist_summary)
    write_json(out / "05_inlier_residual_spatial_summary.json", residual_summary)
    _csv(out / "07_direct_residual_tiles.csv",
         ["edge", "grid_n", "tile_row", "tile_col", "center_world_x",
          "center_world_y", "phase_mag_px", "accepted", "reject_reason"],
         res_tile_csv)
    write_json(out / "07_direct_residual_summary.json", phase_summary)
    _csv(out / "08_residual_vs_control_support.csv",
         ["edge", "tile_grid", "nearest_control_distance_px", "phase_mag_px"],
         support_csv)
    write_json(out / "08_residual_vs_control_support_summary.json", support_summary)

    _csv(out / "10_edge_reliability_features.csv", list(feature_rows[0].keys()),
         feature_rows)
    write_json(out / "10_edge_reliability_features.json", {"edges": feature_rows})

    diag_out = [{"edge": r["edge"], "diagnosis": r["diagnosis"]} for r in diag_rows]
    write_json(out / "12_edge_diagnoses.json", diag_out)
    (out / "12_edge_diagnoses.txt").write_text(
        "\n".join(f"{r['edge']}: {r['diagnosis']}" for r in diag_out) + "\n",
        encoding="utf-8")

    good = next((r for r in feature_rows if r["edge"] == "0-6"), None)
    false_good = next((r for r in feature_rows if r["edge"] == "2-5"), None)
    comp = {"good_0_6": good, "false_good_candidate_2_5": false_good,
            "metrics_compared": list(good.keys()) if good else []}
    write_json(out / "11_good_vs_false_good_comparison.json", comp)
    txt = ["0-6 vs 2-5 comparison",
           f"0-6: inliers={good['inliers']} rmse={good['rmse_px']:.2f} "
           f"cov={good['old_coverage']} occ8={good['occupancy_8x8']} "
           f"hull={good['hull_coverage']} p95_ctrl={good['p95_nearest_control_distance_px']} "
           f"phase_med={good['local_phase_median_px']} bad_frac={good['fraction_tiles_gt5px']} "
           f"diag={good['diagnosis']}",
           f"2-5: inliers={false_good['inliers']} rmse={false_good['rmse_px']:.2f} "
           f"cov={false_good['old_coverage']} occ8={false_good['occupancy_8x8']} "
           f"hull={false_good['hull_coverage']} p95_ctrl={false_good['p95_nearest_control_distance_px']} "
           f"phase_med={false_good['local_phase_median_px']} bad_frac={false_good['fraction_tiles_gt5px']} "
           f"corr={false_good['corr_phase_vs_control_distance']} diag={false_good['diagnosis']}"]
    (out / "11_good_vs_false_good_comparison.txt").write_text(
        "\n".join(txt) + "\n", encoding="utf-8")

    conclusion = {
        "1_0_6_why_reliable": next((r for r in diag_rows if r["edge"] == "0-6"),
                                   {}).get("diagnosis"),
        "2_2_5_false_good_candidate": next((r for r in diag_rows
                                            if r["edge"] == "2-5"),
                                           {}).get("diagnosis"),
        "3_2_5_spatial_explanation": (
            next(r for r in feature_rows if r["edge"] == "2-5").get(
                "corr_phase_vs_control_distance")),
        "edges": diag_out,
        "can_conclude": ["7-edge diagnostic set analysed; spatial support, "
                         "nearest-control distance, and direct local residual "
                         "reported; no ML training performed."],
        "cannot_conclude": ["No new reliability score is validated for general "
                            "use; phase is not absolute truth; Affine is not "
                            "declared universally insufficient; no matcher "
                            "change is implied."],
    }
    write_json(out / "13_edge_reliability_conclusion.json", conclusion)
    (out / "13_edge_reliability_conclusion.txt").write_text(
        "\n".join([f"{k}: {v}" for k, v in conclusion.items() if k != "edges"] +
                  ["", "diagnoses:"] + [f"  {r['edge']}: {r['diagnosis']}"
                                        for r in diag_out]) + "\n",
        encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Edge-reliability diagnosis complete -> {out}")
    for r in diag_rows:
        print(f"  {r['edge']:5s} {r['diagnosis']}")
    if good and false_good:
        print("\n0-6 vs 2-5:")
        for k in ("inliers", "occupancy_8x8", "hull_coverage",
                  "p95_nearest_control_distance_px", "local_phase_median_px",
                  "fraction_tiles_gt5px", "corr_phase_vs_control_distance"):
            print(f"  {k:34s} 0-6={good.get(k)}  2-5={false_good.get(k)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())