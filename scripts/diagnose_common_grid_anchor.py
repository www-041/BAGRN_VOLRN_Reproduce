"""CLI: isolate where the ~52 px 0-1 closure translation first enters the chain.

Runs the layer-by-layer frame probes: frame semantics, origin-invariance,
pair common grids, anchor deltas, scene-4 frame trace, production vs
reference pixel-to-world, production vs explicit global composition, a
unified-reference counterfactual, independent tree-edge phase/NCC checks, the
frame-chain figure, and an evidence-driven root-cause summary.

Diagnosis only — SIFT/LoFTR/RANSAC/Affine/MST and all production geometry are
never modified, and no complete five-scene run is started.

Usage (PowerShell)::

    python -m scripts.diagnose_common_grid_anchor `
        --run-dir "data/output/five_scene_sift_B14" `
        --loop-diagnostics-dir "data/output/five_scene_sift_B14/loop_diagnostics" `
        --output-dir "data/output/five_scene_sift_B14/common_grid_anchor_diagnostics"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.multiscene_sift import frame_diagnostics as fd
from src.multiscene_sift.loop_diagnostics import (
    load_global_transforms,
    load_pairwise_results,
    load_run_config,
    load_scenes_for_run,
    find_pair_row,
)

logger = logging.getLogger(__name__)

BAND_DEFAULT = "B14"


def _load_01_inliers(loop_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read the 0-1 inlier (ref, tgt) match-view points from 04 CSV."""
    path = loop_dir / "04_point_residuals.csv"
    if not path.is_file():
        return None
    import csv

    ref, tgt = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ref.append((float(row["ref_x"]), float(row["ref_y"])))
            tgt.append((float(row["tgt_x"]), float(row["tgt_y"])))
    if not ref or len(ref) != len(tgt):
        return None
    return np.asarray(ref, dtype=np.float64), np.asarray(tgt, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--loop-diagnostics-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    loop_dir = Path(args.loop_diagnostics_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (
        run_dir / "common_grid_anchor_diagnostics"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_run_config(run_dir)
    band = str(config.get("registration_band", BAND_DEFAULT))

    # --- Task 1: baseline + frame semantics -------------------------------
    if not loop_dir.is_dir():
        print(f"ERROR: loop diagnostics dir not found: {loop_dir}")
        return 1
    baseline = {
        "run_dir": str(run_dir),
        "loop_diagnostics_dir": str(loop_dir),
        "loop_baseline": fd.load_frame_diagnostic_baseline(loop_dir),
    }
    fd.write_json(out_dir / "00_baseline_snapshot.json", baseline)
    fd.write_json(out_dir / "01_frame_semantics.json", fd.frame_semantics_summary())

    # --- Task 3: origin invariance (synthetic) -----------------------------
    origin_probe = fd.origin_invariance_probe()
    fd.write_json(out_dir / "02_origin_invariance_probe.json", origin_probe)

    # --- scenes ------------------------------------------------------------
    try:
        scenes = load_scenes_for_run(run_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot load scenes: {exc}")
        return 1

    # --- Task 4: rebuild pair common grids ---------------------------------
    pairs = [(0, 4), (1, 4), (0, 1)]
    grids = fd.rebuild_pair_common_grids(scenes, band, pairs)
    fd.write_json(out_dir / "03_pair_common_grids.json", grids)

    match_views = {
        name: fd.match_view_frame_params(
            tuple(grid["overlap_window"]), int(config.get("match_max_side", 1600))
        )
        for name, grid in grids.items()
    }
    fd.write_json(out_dir / "04_anchor_delta.json", fd.anchor_deltas(grids))

    # --- Task 5: scene-4 frame trace ---------------------------------------
    scene4 = scenes[4]
    trace_rows, trace_summary = fd.trace_scene4_through_grids(
        scene4, band, grids["pair_0_4"], grids["pair_1_4"]
    )
    fd.write_scene4_trace_csv(trace_rows, out_dir / "05_scene4_frame_trace.csv")
    trace_summary["rows"] = trace_rows
    fd.write_json(out_dir / "05_scene4_frame_trace_summary.json", trace_summary)

    # --- run artifacts ------------------------------------------------------
    pair_rows = load_pairwise_results(run_dir / "pairwise_summary.json")
    G_prod = load_global_transforms(run_dir / "global_transforms.json")
    ref_index = None
    with open(run_dir / "global_transforms.json", encoding="utf-8") as f:
        ref_index = int(json.load(f)["reference_index"])

    # --- Task 6: production vs reference pixel-to-world --------------------
    edge_compare = fd.edge_world_transform_comparison(
        pair_rows, grids, match_views, pixel_size=abs(scenes[0].transforms[band].a)
    )
    fd.write_json(out_dir / "06_edge_world_transform_comparison.json", edge_compare)

    # --- Task 7/8: closure on the true 0-1 inliers -------------------------
    inliers = _load_01_inliers(loop_dir)
    pixel_size_x = abs(scenes[0].transforms[band].a)
    pixel_size_y = abs(scenes[0].transforms[band].e or pixel_size_x)
    comp = None
    if inliers is not None and ref_index is not None:
        ref_inl, tgt_inl = inliers
        if len(ref_inl) == 0:
            inliers = None
    if inliers is not None and ref_index is not None:
        ref_inl, tgt_inl = inliers
        comp = fd.global_composition_comparison(
            scenes, band, pair_rows, grids, match_views,
            G_prod, ref_index, ref_inl, tgt_inl,
            grids["pair_0_1"], match_views["pair_0_1"],
            pixel_size_x, pixel_size_y,
        )
        fd.write_json(out_dir / "07_global_composition_comparison.json", comp)

        # --- 08: unified-reference counterfactual table ---------------------
        row01 = find_pair_row(pair_rows, 0, 1)
        direct_p95 = row01[1].get("residual_p95") if row01 else None
        counterfactual = {
            "note": (
                "Unified-reference counterfactual: stored match-view pixel "
                "matrices are lifted into common-grid pixels before world "
                "conjugation, then composed with scene 4 as identity. "
                "Production artifacts are not overwritten."
            ),
            "reference": {"index": ref_index, "name": scenes[ref_index].name},
            "closure": comp["closure"],
            "direct_pair_p95_px": direct_p95,
        }
        fd.write_json(out_dir / "08_unified_reference_counterfactual.json",
                      counterfactual)
        txt = []
        txt.append("=== Unified-reference counterfactual (0-1 closure, px) ===")
        txt.append(
            "Direct (pairwise, saved): "
            + (f"P95 = {float(direct_p95):.2f}" if direct_p95 is not None else "unavailable")
        )
        for name, key in (("Production MST (G from artefacts)", "production"),
                          ("Unified-reference (frame-corrected)", "explicit_unified")):
            stats = comp["closure"].get(key)
            if stats is None:
                txt.append(f"{name}: unavailable")
            else:
                txt.append(
                    f"{name}: median={stats['median_px']:.2f} "
                    f"RMSE={stats['rmse_px']:.2f} P95={stats['p95_px']:.2f} "
                    f"max={stats['max_px']:.2f} "
                    f"dx={stats['dx_mean_px']:+.2f} dy={stats['dy_mean_px']:+.2f}"
                )
        (out_dir / "08_unified_reference_counterfactual.txt").write_text(
            "\n".join(txt) + "\n", encoding="utf-8"
        )

    # --- Task 9: independent tree-edge checks ------------------------------
    tree_checks = fd.tree_edge_independent_checks(
        scenes, band, pair_rows, grids, match_views, out_dir,
        pixel_size=pixel_size_x,
    )
    fd.write_json(out_dir / "09_tree_edge_independent_checks.json", tree_checks)

    # --- Task 10: frame-chain figure ---------------------------------------
    fd.draw_frame_chain(out_dir / "10_frame_chain.png")

    # --- Task 11: root-cause summary ---------------------------------------
    evidence = {
        "origin_invariance_pass": bool(origin_probe["passes"]),
        "pixel_to_world_frame_shift_confirmed": None,
        **({} if comp is None else {
            "production_closure_p95_px": comp["closure"]["production"]["p95_px"],
            "explicit_closure_p95_px": (
                comp["closure"]["explicit_unified"]["p95_px"]
                if comp["closure"]["explicit_unified"] else None
            ),
            "production_closure_median_px": comp["closure"]["production"]["median_px"],
        }),
        "production_equals_explicit": None,
        "direct_pair_p95_px": (find_pair_row(pair_rows, 0, 1) or (None, {}))[1].get(
            "residual_p95"
        ),
        "tree_edge_04_phase_mag_px": (
            tree_checks.get("pair_0_4", {}).get("phase_magnitude_px_14m")
        ),
        "tree_edge_14_phase_mag_px": (
            tree_checks.get("pair_1_4", {}).get("phase_magnitude_px_14m")
        ),
    }
    if comp is not None:
        prod = comp["closure"]["production"]["p95_px"]
        expl_c = comp["closure"]["explicit_unified"]
        if expl_c is not None:
            evidence["pixel_to_world_frame_shift_confirmed"] = (
                prod > 10.0 and expl_c["p95_px"] <= 3.0
            )
            evidence["production_equals_explicit"] = (
                prod <= 3.0 and expl_c["p95_px"] <= 3.0
            )
    verdict = fd.classify_frame_root_cause(evidence)
    verdict["answers"] = {
        "1_C04_origin": grids["pair_0_4"]["origin_world_x"],
        "2_C14_origin": grids["pair_1_4"]["origin_world_x"],
        "3_raw_anchor_delta_px_14m": None,
        "4_pixel_affine_to_world_origin_invariant": bool(origin_probe["passes"]),
        "5_production_equals_explicit_composition": evidence.get(
            "production_equals_explicit"
        ),
        "6_unified_reference_closure_p95_px": (
            (comp["closure"]["explicit_unified"]["p95_px"]
             if comp and comp["closure"]["explicit_unified"] else None)
        ),
    }
    fd.write_json(out_dir / "11_root_cause_summary.json", verdict)
    txt = [
        "=== Root-cause summary ===",
        f"State: {verdict['state']}",
        f"Reason: {verdict['reason']}",
        "",
        "Answers:",
    ]
    for key, value in verdict["answers"].items():
        txt.append(f"  {key}: {value}")
    (out_dir / "11_root_cause_summary.txt").write_text(
        "\n".join(txt) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 66)
    print(f"Common-grid anchor diagnosis complete -> {out_dir}")
    print(
        f"Origin invariance: {'PASS' if origin_probe['passes'] else 'FAIL'}"
    )
    if comp is not None:
        prod = comp["closure"]["production"]
        expl = comp["closure"]["explicit_unified"]
        print(
            f"0-1 closure P95: production={prod['p95_px']:.2f} px  "
            f"explicit={expl and expl['p95_px'] or float('nan'):.2f} px"
        )
    print(f"Root cause: {verdict['state']}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())