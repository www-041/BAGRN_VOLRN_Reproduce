"""CLI: root-cause diagnosis of a non-tree loop closure in a five-scene run.

Diagnosis only — this script never changes SIFT / LoFTR / RANSAC / MST /
global-registration behaviour, and it only re-runs ONE pair (the problem
edge) to recover inlier point coordinates the summary did not persist.

Usage (PowerShell)::

    python -m scripts.diagnose_loop_closure `
        --input-root "D:/.../data/input/flat" `
        --run-dir "data/output/five_scene_sift_B14" `
        --output-dir "data/output/five_scene_sift_B14/loop_diagnostics" `
        --problem-edge auto

Run this on the real DZ01V data yourself; the author does not execute it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.multiscene_sift import loop_diagnostics as diag

logger = logging.getLogger(__name__)

OUTPUT_FILES = [
    "01_problem_edge.json",
    "02_transform_comparison.json",
    "03_pair_reproduction.json",
    "04_point_residuals.csv",
    "05_error_pattern.json",
    "06_error_vectors.png",
    "07_dx_histogram.png",
    "08_dy_histogram.png",
    "09_direct_0_1_overlay.png",
    "10_mst_0_4_1_overlay.png",
    "11_crop_A_direct.png",
    "12_crop_A_mst.png",
    "13_crop_B_direct.png",
    "14_crop_B_mst.png",
    "15_crop_C_direct.png",
    "16_crop_C_mst.png",
    "17_loop_triangle.png",
    "18_diagnosis_summary.json",
    "18_diagnosis_summary.txt",
]


def _parse_problem_edge(value: str, n_scenes: int) -> tuple[int, int] | None:
    """Parse ``--problem-edge`` (``auto`` or ``i,j``) with bounds checking."""
    if value is None or str(value).strip().lower() == "auto":
        return None
    parts = [p.strip() for p in str(value).replace(";", ",").split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "--problem-edge must be 'auto' or '<i>,<j>'"
        )
    try:
        i, j = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--problem-edge: {value!r} is not ints") from exc
    if i < 0 or j < 0 or i >= n_scenes or j >= n_scenes:
        raise argparse.ArgumentTypeError(
            f"--problem-edge {i},{j} out of range [0, {n_scenes})"
        )
    return i, j


def _scene_pixel_size(scenes, band: str) -> tuple[float, float]:
    """Ordinarily the runner uses ``abs(transform.a)`` for both axes."""
    transform = scenes[0].transforms[band]
    return abs(transform.a), abs(transform.e) or abs(transform.a)


def _saved_pair_row(pair_rows, i: int, j: int) -> dict | None:
    found = diag.find_pair_row(pair_rows, i, j)
    return None if found is None else found[1]


def _q5_pairwise_sufficiency(saved_row: dict | None, edge_record: dict) -> str:
    """Answer Q5 strictly from the diagnostics."""
    global_p95 = float(edge_record["global_p95_px"])
    if saved_row is None:
        return (
            "Cannot answer: the problem edge has no saved direct pairwise row."
        )
    pairwise_p95 = float(saved_row.get("residual_p95") or float("nan"))
    pairwise_rmse = float(saved_row.get("residual_rmse") or float("nan"))
    from math import isfinite

    if not isfinite(pairwise_p95):
        return "Cannot answer: saved pairwise residual for the edge is unavailable."
    if global_p95 <= max(pairwise_p95 * 1.5, pairwise_p95 + 1.0):
        return (
            "The pairwise residuals alone ARE sufficient: the direct pair's P95 "
            f"({pairwise_p95:.2f} px) is comparable with the composed closure P95 "
            f"({global_p95:.2f} px), so the error is already visible in the pair."
        )
    return (
        "The pairwise residuals alone are NOT sufficient: the direct pair looks "
        f"locally accurate (RMSE {pairwise_rmse:.2f} px, P95 {pairwise_p95:.2f} px) "
        f"while the composed closure error is {global_p95:.2f} px. The discrepancy "
        "only appears when the MST path is composed (or the metric is mis-measured)."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True, help="Directory of a five-scene run")
    parser.add_argument(
        "--output-dir", default=None,
        help="Diagnostics output dir (default: <run-dir>/loop_diagnostics)",
    )
    parser.add_argument(
        "--input-root", default=None,
        help="Optional input root for re-running the problem pair (fresh discovery)",
    )
    parser.add_argument(
        "--problem-edge", default="auto",
        help="'auto' or '<i>,<j>' (default: auto = worst non-tree edge by P95)",
    )
    parser.add_argument("--crop-size", type=int, default=512, help="Overlay crop size in px")
    parser.add_argument("--n-crops", type=int, default=3, help="Number of crop pairs")
    parser.add_argument(
        "--arrow-scale", type=float, default=5.0, help="Visual error-arrow magnification"
    )
    parser.add_argument(
        "--overlay-max-side", type=int, default=2048,
        help="Max display size (px) of the overlay grids",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "loop_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Run config ---------------------------------------------------------
    try:
        config = diag.load_run_config(run_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    n_scenes = int(config.get("n_scenes", 0)) if config.get("n_scenes") else len(
        config.get("scene_names", [])
    )
    if not n_scenes:
        # Fall back to the consistency table size.
        try:
            rows0 = diag.load_consistency_rows(run_dir / "global_edge_consistency.csv")
            n_scenes = max(max(int(r["idx_i"]), int(r["idx_j"])) for r in rows0) + 1
        except Exception:
            n_scenes = 0

    try:
        edge = _parse_problem_edge(args.problem_edge, n_scenes)
    except argparse.ArgumentTypeError as exc:
        print(f"ERROR: {exc}")
        return 1

    # --- Scenes (used to re-run the pair) -----------------------------------
    scenes = None
    try:
        scenes = diag.load_scenes_for_run(run_dir, input_root=args.input_root)
        if n_scenes == 0:
            n_scenes = len(scenes)
    except Exception as exc:  # noqa: BLE001 -- graceful degradation is intended
        if edge is None:
            print(f"WARNING: could not load scenes ({exc}); only edge identification will run.")
        else:
            print(
                "WARNING: could not load scenes for pair re-running "
                f"({type(exc).__name__}: {exc}). Transform comparison and point "
                "residuals will be skipped; re-run with a correct --input-root."
            )

    # --- Task 1: problem edge + MST path -------------------------------------
    try:
        consistency_rows = diag.load_consistency_rows(run_dir / "global_edge_consistency.csv")
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    if edge is None:
        edge = diag.find_worst_non_tree_edge(consistency_rows)
    if edge is None:
        print("No non-tree edge available; loop closure cannot be evaluated.")
        return 0

    spanning = diag.load_spanning_tree(run_dir / "spanning_tree.json")
    if spanning["edges"]:
        tree_max = max(
            int(v) for e in spanning["edges"] for v in (e["parent"], e["child"])
        )
        n_scenes = max(n_scenes, tree_max + 1)

    mst_path = diag.find_tree_path(spanning["edges"], edge[0], edge[1])
    if mst_path is None:
        print("No path found between selected edge nodes in spanning tree.")
        return 1

    edge_record = diag.edge_consistency_record(consistency_rows, edge[0], edge[1])
    if edge_record is None:
        print(
            f"Selected non-tree edge {(edge[0], edge[1])} has no valid direct registration "
            "(missing consistency row)."
        )
        return 1

    diag.write_json(out_dir / "01_problem_edge.json", {
        "idx_i": edge[0],
        "idx_j": edge[1],
        "selected_by": args.problem_edge,
        "global_median_px": edge_record["global_median_px"],
        "global_rmse_px": edge_record["global_rmse_px"],
        "global_p90_px": edge_record["global_p90_px"],
        "global_p95_px": edge_record["global_p95_px"],
        "global_max_px": edge_record["global_max_px"],
        "n_points": edge_record["n_points"],
        "mst_path": mst_path,
    })

    # --- Global transforms ----------------------------------------------------
    try:
        G = diag.load_global_transforms(run_dir / "global_transforms.json")
    except FileNotFoundError:
        print(f"ERROR: global_transforms.json not found in {run_dir}")
        return 1

    pair_rows = diag.load_pairwise_results(run_dir / "pairwise_summary.json")
    saved_row = _saved_pair_row(pair_rows, edge[0], edge[1])
    if saved_row is None or saved_row.get("status") != "OK":
        print("Selected non-tree edge has no valid direct registration.")
        return 1

    q5_explanation = _q5_pairwise_sufficiency(saved_row, edge_record)
    reproduction = pattern = transform_compare = residual_stats = None
    reg = None
    visualization_files: dict = {}
    overlay_alignment: dict | None = None

    # --- Task 3: reproduce the problem pair -----------------------------------
    if scenes is not None:
        reg = None
        try:
            reg = diag.reproduce_pair(scenes, edge[0], edge[1], config)
        except Exception as exc:  # noqa: BLE001
            print(
                f"ERROR: re-running pair {edge} failed: {type(exc).__name__}: {exc}"
            )
            reg = None
        if reg is not None:
            reproduction = diag.compare_pair_to_saved(saved_row, reg)
            diag.write_json(out_dir / "03_pair_reproduction.json", reproduction)
            if reproduction.get("warning"):
                print("WARNING:", reproduction["warning"])

    # --- Task 2 + Task 4/5 need a reproduced common grid -----------------------
    if scenes is not None and reg is not None and reg.pair_common_transform is not None:
        band = str(config.get("registration_band", "B14"))
        pixel_size_x, pixel_size_y = _scene_pixel_size(scenes, band)

        common = reg.pair_common_transform
        T_direct_saved = diag.pair_direct_world_transform(
            pair_rows, edge[0], edge[1], common
        )
        T_direct_rep = (
            diag.pair_direct_world_transform(
                [diag.registration_as_pair_row(reg)], edge[0], edge[1], common
            )
            if reg.status == "OK"
            else None
        )
        T_mst = diag.mst_implied_transform(G, edge[0], edge[1])

        if T_direct_rep is not None:
            transform_compare = diag.compare_direct_and_mst_transforms(
                T_direct_rep, T_mst, pixel_size_x, pixel_size_y
            )
            transform_compare["direct_matrix_saved_world"] = (
                T_direct_saved.tolist() if T_direct_saved is not None else None
            )
            transform_compare["direct_saved_vs_reproduced_note"] = (
                "direct_matrix is the reproduced pair transform (T_direct_reproduced); "
                "direct_matrix_saved_world is the saved pixel_matrix transformed with "
                "the reproduced common grid."
            )
            diag.write_json(out_dir / "02_transform_comparison.json", transform_compare)

        # --- Task 4/5: point residuals --------------------------------------------
        try:
            dx_px, dy_px, err_px = diag.pair_point_residuals_px(
                reg, G, pixel_size_x, pixel_size_y
            )
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: point residuals unavailable: {exc}")
            dx_px = dy_px = err_px = None

        if dx_px is not None:
            diag.write_point_residuals_csv(
                reg, dx_px, dy_px, err_px, out_dir / "04_point_residuals.csv"
            )
            residual_stats = diag.residual_statistics(dx_px, dy_px, err_px)
            pattern = diag.classify_error_pattern(
                reg.inlier_ref_xy, dx_px, dy_px
            )
            pattern["residual_statistics"] = residual_stats
            diag.write_json(out_dir / "05_error_pattern.json", pattern)

            # --- Task 6: arrow plot + histograms ------------------------------------
            try:
                diag.plot_error_vectors(
                    scenes[edge[0]], scenes[edge[1]], band, reg,
                    dx_px, dy_px,
                    out_dir / "06_error_vectors.png",
                    arrow_scale=args.arrow_scale,
                )
                dx_h, dy_h = diag.plot_residual_histograms(
                    dx_px, dy_px, out_dir
                )
                visualization_files.update({
                    "06_error_vectors.png": str(out_dir / "06_error_vectors.png"),
                    "07_dx_histogram.png": str(dx_h),
                    "08_dy_histogram.png": str(dy_h),
                })
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: error-vector plots skipped: {type(exc).__name__}: {exc}")

            # --- Task 6 (C/D overlays) + Task 7 (crops) ------------------------------
            if T_direct_rep is not None:
                try:
                    overlay = diag.plot_overlay_comparison(
                        scenes[edge[0]], scenes[edge[1]], band,
                        T_direct_rep,
                        G,
                        out_dir,
                        max_side=args.overlay_max_side,
                    )
                    visualization_files.update({
                        "09_direct_0_1_overlay.png": overlay["direct_path"],
                        "10_mst_0_4_1_overlay.png": overlay["mst_path"],
                    })
                    overlay_alignment = {
                        "ncc_direct": float(overlay.get("ncc_direct")),
                        "ncc_mst": float(overlay.get("ncc_mst")),
                        "ncc_winner": overlay.get("ncc_winner"),
                    }
                    print(
                        "Overlay alignment NCC: "
                        f"direct={overlay_alignment['ncc_direct']:.4f}  "
                        f"mst={overlay_alignment['ncc_mst']:.4f}  "
                        f"(winner: {overlay_alignment['ncc_winner']})"
                    )
                    shift_report = {
                        "direct": overlay.get("block_shift_direct"),
                        "mst": overlay.get("block_shift_mst"),
                    }
                    diag.write_json(
                        out_dir / "overlay_block_shifts.json",
                        {"block": shift_report},
                    )
                    overlay_alignment["block_shifts"] = shift_report
                    for name, key in (("direct", "direct"), ("mst", "mst")):
                        bs = shift_report[key]
                        if bs and bs.get("n_shifts", 0):
                            print(
                                f"Phase-correlation residual shift ({name}): "
                                f"dx={bs['median_dx']:+.2f} px, dy={bs['median_dy']:+.2f} px "
                                f"({bs['n_shifts']} tiles)"
                            )
                    crops = diag.plot_overlay_crops(
                        overlay, reg, out_dir,
                        crop_size=args.crop_size, n_crops=args.n_crops,
                    )
                    for c in crops:
                        visualization_files[c["label"]] = c["path"]
                except Exception as exc:  # noqa: BLE001
                    print(f"WARNING: overlay/crop plots skipped: {type(exc).__name__}: {exc}")

        # --- Task 8: loop triangle ---------------------------------------------------
        try:
            quality = {}
            for r in pair_rows:
                q = float(r.get("inliers", 0)) * float(r.get("inlier_ratio", 0.0)) \
                    * float(r.get("coverage", 0.0))
                quality[(int(r["idx_i"]), int(r["idx_j"]))] = q
                quality[(int(r["idx_j"]), int(r["idx_i"]))] = q
            records = {}
            for r in consistency_rows:
                records[(int(r["idx_i"]), int(r["idx_j"]))] = r
            diag.plot_loop_triangle(
                edge, mst_path, records, quality,
                out_dir / "17_loop_triangle.png",
            )
            visualization_files["17_loop_triangle.png"] = str(
                out_dir / "17_loop_triangle.png"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: loop-triangle plot skipped: {type(exc).__name__}: {exc}")

    # --- Task 9: final summary ------------------------------------------------------
    final_state = diag.decide_final_state(
        transform_compare,
        pattern or {},
        float(edge_record["global_p95_px"]),
    )
    summary = diag.build_diagnosis_summary(
        run_dir=str(run_dir),
        output_dir=str(out_dir),
        edge=edge,
        mst_path=mst_path,
        edge_record=edge_record,
        transform_compare=transform_compare,
        pattern=pattern,
        residual_stats=residual_stats,
        reproduction=reproduction,
        final_state=final_state,
        q5_explanation=q5_explanation,
        visualization_files=visualization_files,
        overlay_alignment=overlay_alignment,
    )
    diag.write_json(out_dir / "18_diagnosis_summary.json", summary)
    diag.write_summary_text(summary, out_dir / "18_diagnosis_summary.txt")

    print("\n" + "=" * 66)
    print(f"Loop-closure diagnosis complete -> {out_dir}")
    print(f"Problem edge: {edge[0]}-{edge[1]}  MST path: {'->'.join(map(str, mst_path))}")
    print(f"Global P95: {edge_record['global_p95_px']} px")
    print(f"Residual pattern: {summary['questions']['Q3_error_type']}")
    print(f"Final state: {final_state['final_state']}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())