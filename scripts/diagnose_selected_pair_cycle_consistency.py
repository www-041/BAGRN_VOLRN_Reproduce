"""CLI: run the 8 selected new SIFT pairs and evaluate 4 new + 1 baseline cycle.

Uses the exact original five-scene SIFT configuration (frozen from its run
config + production constants), reuses the old 0-4/0-2/0-1/1-4 edges from the
saved artefacts, validates each new direct edge independently, normalises all
eligible edges to the canonical world frame, and computes cycle closures on
world sample points.  No full 9-scene MST, radiometric normalisation, or
mosaicking is run.

Usage (real, run by the user or the author per this round's scope)::

    cd D:\\科研\\地质一号\\文献\\BAGRN_VOLRN_Reproduce_two_image_flat
    &.\\".venv-registration\\Scripts\\python.exe" `
        -m scripts.diagnose_selected_pair_cycle_consistency `
        --input-root "D:/科研/地质一号/文献/BAGRN_VOLRN_Reproduce/data/input/flat" `
        --five-scene-run-dir "data/output/five_scene_sift_B14" `
        --nine-scene-overlap-dir "data/output/nine_scene_overlap_diagnostic" `
        --output-dir "data/output/nine_scene_selected_pair_consistency"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.multiscene_sift import network_consistency_diagnostics as ncd
from src.multiscene_sift.frame_diagnostics import (
    match_view_frame_params,
    rebuild_pair_common_grids,
    write_json,
)
from src.multiscene_sift.loop_diagnostics import (
    load_pairwise_results,
    plot_overlay_comparison,
)
from src.multiscene_sift.dataset import discover_five_scenes
from src.multiscene_sift.nine_scene_overlap_diagnostic import SCENE_NAMES_9

BAND = "B14"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--five-scene-run-dir", required=True)
    parser.add_argument("--nine-scene-overlap-dir", default=None)
    parser.add_argument("--output-dir", default="data/output/nine_scene_selected_pair_consistency")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Task 1: plan -----------------------------------------------------
    overlap_source = Path(args.nine_scene_overlap_dir) if args.nine_scene_overlap_dir \
        else None
    existed_in_bounds = {}
    if overlap_source is not None:
        src = overlap_source / "03_all_pair_geographic_overlap.json"
        if src.is_file():
            with open(src, encoding="utf-8") as f:
                rows = json.load(f)["rows"]
            by = {(r["idx_i"], r["idx_j"]): r for r in rows}
            for (i, j) in ncd.SELECTED_NEW_PAIRS:
                existed_in_bounds[f"{i}-{j}"] = bool(by[(i, j)]["intersection_exists"])
    plan = {
        "selected_new_pairs": [list(p) for p in ncd.SELECTED_NEW_PAIRS],
        "target_cycles": {k: list(v) for k, v in ncd.TARGET_CYCLES.items()},
        "existing_edges_reused": [list(p) for p in ncd.REUSED_OLD_PAIRS],
        "excluded_pairs": [[0, 3], [1, 2], [0, 7], [1, 3], [3, 7], [3, 8], [4, 5]],
        "source_overlap_artifacts": str(overlap_source) if overlap_source else None,
        "selected_pairs_overlap_exists": existed_in_bounds,
    }
    missing_bounds = [k for k, v in existed_in_bounds.items() if not v]
    if missing_bounds:
        print(f"STOP: selected pairs have no bounds intersection: {missing_bounds}")
        return 1
    write_json(out / "00_selected_pair_plan.json", plan)

    # --- Task 2: config snapshot -------------------------------------------
    config = ncd.load_baseline_registration_config(args.five_scene_run_dir)
    write_json(out / "01_registration_config_snapshot.json", config)

    # --- scenes & grids -----------------------------------------------------
    scenes, manifest = discover_five_scenes(args.input_root, list(SCENE_NAMES_9),
                                            bands=(BAND,))
    band = BAND
    pair_list = ncd.SELECTED_NEW_PAIRS + ncd.REUSED_OLD_PAIRS
    grids = rebuild_pair_common_grids(scenes, band, pair_list)
    grid_map = {
        f"pair_{min(i, j)}_{max(i, j)}": grids[f"pair_{min(i, j)}_{max(i, j)}"]
        for i, j in pair_list
    }
    match_views = {
        name: match_view_frame_params(
            tuple(grid_map[name]["overlap_window"]),
            int(config["match_max_side"]),
        )
        for name in grid_map
    }

    # --- Task 3: run the 8 selected new pairs ------------------------------
    new_results: list[dict] = []
    regs = {}
    for n, (i, j) in enumerate(ncd.SELECTED_NEW_PAIRS, start=1):
        print(f"Running {n}/8: {i}-{j} ...", flush=True)
        reg = ncd.run_selected_pair_registration(scenes[i], scenes[j], config)
        row = ncd.registration_as_row(reg)
        new_results.append(row)
        regs[(i, j)] = reg
    _write_csv(out / "02_new_pairwise_registration_results.csv", [
        "idx_i", "idx_j", "status", "raw_matches", "inliers",
        "inlier_ratio", "coverage", "residual_rmse", "residual_p95",
        "residual_median", "runtime_s",
    ], [{k: row[k] for k in ("idx_i", "idx_j", "status", "raw_matches",
                             "inliers", "inlier_ratio", "coverage",
                             "residual_rmse", "residual_p95",
                             "residual_median", "runtime_s")}
        for row in new_results])
    write_json(out / "02_new_pairwise_registration_results.json", {
        "results": new_results,
        "config_source": args.five_scene_run_dir,
    })

    # --- Task 4: direct geometry validation (new edges only) ----------------
    quality_rows = []
    for (i, j), reg in regs.items():
        row = {"idx_i": i, "idx_j": j, "status": reg.status}
        if reg.status != "OK":
            quality_rows.append({**row, "phase_mag_px": None, "ncc": None,
                                 "phase_tiles": 0, "validation_status":
                                 "REGISTRATION_REJECTED"})
            continue
        t_world = ncd.direct_edge_world_transform(
            [ncd.registration_as_row(reg)], i, j,
            reg.pair_common_transform, match_views[f"pair_{min(i, j)}_{max(i, j)}"],
        )
        try:
            overlay = plot_overlay_comparison(
                scenes[i], scenes[j], band, t_world,
                [np.eye(3) for _ in range(9)],
                out, max_side=2048,
            )
        except Exception as exc:  # noqa: BLE001
            quality_rows.append({**row, "phase_mag_px": None, "ncc": None,
                                 "phase_tiles": 0,
                                 "validation_status": "INSUFFICIENT_VALID_OVERLAP",
                                 "note": str(exc)[:120]})
            continue
        est = overlay.get("block_shift_direct") or {}
        n_tiles = int(est.get("n_shifts", 0) or 0)
        gw = max(int(overlay.get("grid_width_px", 1)), 1)
        scene_w = int(max(scenes[i].shapes[band][1], scenes[j].shapes[band][1]))
        grid_res = 14.0 * scene_w / max(gw, 1)
        if n_tiles:
            mag_px = float(np.hypot(est["median_dx"], est["median_dy"])) \
                * grid_res / 14.0
        else:
            mag_px = None
        if mag_px is not None and mag_px <= 2.0:
            vstatus = "VALIDATED"
        elif mag_px is not None and mag_px <= 5.0:
            vstatus = "PHASE_CHECK_SUSPECT"
        elif mag_px is not None:
            vstatus = "REGISTRATION_REJECTED"
        else:
            vstatus = "INSUFFICIENT_VALID_OVERLAP"
        quality_rows.append({
            **row, "phase_mag_px": mag_px, "ncc": overlay.get("ncc_direct"),
            "phase_tiles": n_tiles, "validation_status": vstatus,
        })
    _write_csv(out / "03_new_pair_quality_checks.csv", [
        "idx_i", "idx_j", "status", "phase_mag_px", "ncc", "phase_tiles",
        "validation_status",
    ], quality_rows)
    write_json(out / "03_new_pair_quality_checks.json", {
        "checks": quality_rows,
        "validation_threshold_px": 2.0,
        "suspect_threshold_px": 5.0,
    })

    # --- Task 5: eligibility ---------------------------------------------------
    eligible: set[tuple[int, int]] = set()
    eligibility_rows = []
    for q in quality_rows:
        i, j = q["idx_i"], q["idx_j"]
        ok = (q["status"] == "OK" and q["validation_status"] == "VALIDATED")
        eligibility_rows.append({
            "pair": f"{i}-{j}", "kind": "new",
            "eligible": bool(ok),
            "reason": "" if ok else q["validation_status"],
        })
        if ok:
            eligible.add((i, j))
    for (i, j) in ncd.REUSED_OLD_PAIRS:
        row = None
        saved_rows = load_pairwise_results(
            Path(args.five_scene_run_dir) / "pairwise_summary.json")
        for r in saved_rows:
            if {int(r["idx_i"]), int(r["idx_j"])} == {i, j}:
                row = r
                break
        ok = row is not None and row.get("status") == "OK"
        eligibility_rows.append({
            "pair": f"{i}-{j}", "kind": "existing",
            "eligible": bool(ok),
            "reason": "" if ok else ("missing" if row is None else row["status"]),
        })
        if ok:
            eligible.add((i, j))
    write_json(out / "05_cycle_edge_eligibility.json", {
        "edges": eligibility_rows, "eligible": [list(e) for e in sorted(eligible)],
    })

    # --- Task 6: canonical world transforms ------------------------------------
    saved_rows = load_pairwise_results(Path(args.five_scene_run_dir)
                                       / "pairwise_summary.json")
    world_edges: dict[tuple[int, int], np.ndarray] = {}
    edge_records = {}
    for (i, j) in sorted(eligible):
        key = f"pair_{min(i, j)}_{max(i, j)}"
        common = Affine_or_reg(i, j, regs, grids, key)
        if common is None:
            continue
        if (i, j) in regs:
            rows_for_edge = [ncd.registration_as_row(regs[(i, j)])]
        else:
            rows_for_edge = [r for r in saved_rows
                             if {int(r["idx_i"]), int(r["idx_j"])} == {i, j}]
            if not rows_for_edge:
                continue
        T = ncd.direct_edge_world_transform(
            rows_for_edge, i, j, common, match_views[key])
        if T is None:
            continue
        world_edges[(i, j)] = T
        dec = ncd.linear_components(T)
        edge_records[f"{i}-{j}"] = {
            "direction": f"{i} from {j}",
            "matrix": T.tolist(),
            "translation_m": [float(T[0, 2]), float(T[1, 2])],
            "translation_px14_equivalent": [float(T[0, 2] / 14.0),
                                            float(T[1, 2] / 14.0)],
            "rotation_deg": dec["rotation_deg"],
            "scale": [dec["scale_x"], dec["scale_y"]],
            "shear_deg": dec["shear_deg"],
            "source_artifact": ("new_registration" if (i, j) in regs
                                else "five_scene_saved_pairwise"),
        }
    write_json(out / "06_canonical_world_edge_transforms.json", {
        "edges": edge_records,
        "frame_conversion": "frame_diagnostics.world_transform_from_saved_pair "
                            "(match-view->common->world, adjust_frame=True)",
    })

    # --- Task 7/8/9: cycle closures ---------------------------------------------
    cycle_states: dict[str, tuple[str, dict | None]] = {}
    point_rows = []
    pixel_size = 14.0
    for name, cycle in ncd.TARGET_CYCLES.items():
        ok_edges, missing = ncd.cycle_edge_eligibility(cycle, set(world_edges))
        if not ok_edges:
            cycle_states[name] = ("UNAVAILABLE_DUE_TO_EDGE_FAILURE", {
                "missing_or_invalid_edges": missing})
            continue
        try:
            closure = ncd.compose_cycle_world(cycle, world_edges)
        except KeyError as exc:
            cycle_states[name] = ("UNAVAILABLE_DUE_TO_EDGE_FAILURE",
                                  {"error": str(exc)})
            continue
        pts = ncd.sample_cycle_world_points(scenes, cycle, band)
        stats = ncd.evaluate_cycle_on_points(closure, pts, pixel_size)
        if stats["n_points"] == 0:
            # fallback anchor-independent estimate from the matrix translation
            stats = {
                "n_points": 0,
                "median_px": float(np.hypot(closure[0, 2], closure[1, 2])
                                   / pixel_size),
                "rmse_px": float(np.hypot(closure[0, 2], closure[1, 2])
                                 / pixel_size),
                "p95_px": float(np.hypot(closure[0, 2], closure[1, 2])
                                / pixel_size),
                "max_px": float(np.hypot(closure[0, 2], closure[1, 2])
                                / pixel_size),
                "std_px": 0.0,
                "dx_mean_px": float(closure[0, 2] / pixel_size),
                "dy_mean_px": float(closure[1, 2] / pixel_size),
            }
        cls = ncd.classify_cycle(closure, stats)
        cycle_states[name] = (cls, stats)
        for k, pt in enumerate(pts):
            h = np.array([pt[0], pt[1], 1.0])
            moved = (closure @ h)[:2]
            d = (moved - pt) / pixel_size
            point_rows.append({
                "cycle": name, "nodes": "-".join(map(str, cycle)),
                "point_id": k, "world_x": pt[0], "world_y": pt[1],
                "dx_px14": float(d[0]), "dy_px14": float(d[1]),
                "magnitude_px14": float(np.hypot(*d)),
            })
    _write_csv(out / "07_cycle_point_residuals.csv", [
        "cycle", "nodes", "point_id", "world_x", "world_y", "dx_px14",
        "dy_px14", "magnitude_px14",
    ], point_rows)

    cycle_rows = []
    for name, (cls, stats) in cycle_states.items():
        s = stats if isinstance(stats, dict) and "median_px" in stats else {}
        cycle_rows.append({
            "cycle_id": name,
            "nodes": "-".join(map(str, ncd.TARGET_CYCLES[name])),
            "available": cls != "UNAVAILABLE_DUE_TO_EDGE_FAILURE",
            "edge_statuses": "eligible",
            "median_px": (None if not s else round(s["median_px"], 3)),
            "rmse_px": (None if not s else round(s["rmse_px"], 3)),
            "p95_px": (None if not s else round(s["p95_px"], 3)),
            "max_px": (None if not s else round(s["max_px"], 3)),
            "translation_dx_px": (None if not s else round(s["dx_mean_px"], 3)),
            "translation_dy_px": (None if not s else round(s["dy_mean_px"], 3)),
            "rotation_deg": None,
            "classification": cls,
        })
    _write_csv(out / "08_cycle_closure_summary.csv", [
        "cycle_id", "nodes", "available", "edge_statuses", "median_px",
        "rmse_px", "p95_px", "max_px", "translation_dx_px",
        "translation_dy_px", "rotation_deg", "classification",
    ], cycle_rows)
    write_json(out / "08_cycle_closure_summary.json", {
        "cycles": cycle_rows,
        "thresholds": ncd.CYCLE_CLASS_THRESHOLDS,
        "note": "thresholds are descriptive categories, not scientific truth",
    })

    # --- Task 11/12: evidence + network classifier ----------------------------
    ev = ncd.build_scene_evidence(
        {k: (v[0], None) for k, v in cycle_states.items()}, new_results)
    write_json(out / "09_scene_consistency_evidence.json", ev)
    n_new_ok = sum(1 for r in new_results if r["status"] == "OK")
    n_new_valid = sum(1 for q in quality_rows
                      if q["validation_status"] == "VALIDATED")
    evidence = {
        "c0_status": cycle_states.get("C0_baseline", ("UNAVAILABLE", None))[0],
        "c1_status": cycle_states.get("C1_scene0_ref4", ("UNAVAILABLE", None))[0],
        "c2_status": cycle_states.get("C2_scene0_ref2", ("UNAVAILABLE", None))[0],
        "c3_status": cycle_states.get("C3_scene0_new", ("UNAVAILABLE", None))[0],
        "c4_status": cycle_states.get("C4_scene1_new", ("UNAVAILABLE", None))[0],
        "new_pair_ok": n_new_ok,
        "new_pair_tried": len(new_results),
    }
    net_state = ncd.classify_network_inconsistency(evidence)
    (out / "09_scene_consistency_evidence.txt").write_text(
        "\n".join([
            "=== scene consistency evidence ===",
            f"network state: {net_state}",
            f"scene0 cycles: {ev['scene0_cycles']}",
            f"scene1 cycles: {ev['scene1_cycles']}",
            f"baseline contrast: {ev['baseline_contrast']}",
            f"scene0 new edges: {ev['scene0_new_edges']}",
            f"scene1 new edges: {ev['scene1_new_edges']}",
            f"new pairs OK: {n_new_ok}/{len(new_results)}; validated: {n_new_valid}",
        ]) + "\n", encoding="utf-8")

    # --- figures ---------------------------------------------------------------
    try:
        _plot_network_event(scenes, world_edges, eligible, out /
                            "10_registration_consistency_network.png")
        _plot_cycle_bars(cycle_rows, out / "11_cycle_closure_comparison.png")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: figures skipped: {type(exc).__name__}: {exc}")

    # --- Task 15: conclusion -----------------------------------------------------
    summary = {
        "1_new_pairs_registration_ok": n_new_ok,
        "2_new_pairs_direct_validated": n_new_valid,
        "3_failed_pairs": [
            {"pair": f"{r['idx_i']}-{r['idx_j']}", "status": r["status"]}
            for r in new_results if r["status"] != "OK"
        ] + [
            {"pair": f"{q['idx_i']}-{q['idx_j']}",
             "status": q["validation_status"]}
            for q in quality_rows if q["validation_status"] not in (
                "VALIDATED",)
        ],
        "4_c1_closure_p95": _cycle_p95(cycle_states, "C1_scene0_ref4"),
        "5_c2_closure_p95": _cycle_p95(cycle_states, "C2_scene0_ref2"),
        "6_c3_closure_p95": _cycle_p95(cycle_states, "C3_scene0_new"),
        "7_c4_closure_p95": _cycle_p95(cycle_states, "C4_scene1_new"),
        "8_c0_baseline_still_anomalous": (
            cycle_states.get("C0_baseline", (None, None))[0]
            not in (None, "UNAVAILABLE_DUE_TO_EDGE_FAILURE", "CLOSED")
        ),
        "9_scene0_cycles_consistent": any(
            cycle_states.get(k, ("", None))[0] in ("CLOSED", "SMALL_RESIDUAL")
            for k in ("C1_scene0_ref4", "C2_scene0_ref2", "C3_scene0_new")
        ),
        "10_scene1_cycle_consistent": cycle_states.get(
            "C4_scene1_new", ("", None))[0] in ("CLOSED", "SMALL_RESIDUAL"),
        "11_network_state": net_state,
        "12_next_step_reading": (
            "see decision tree: pair-0-1-specific vs scene-level vs broader; "
            "no global adjustment implemented this round"
        ),
        "can_conclude": [
            "Only the 8 selected new pairs were registered, with the frozen "
            "five-scene SIFT configuration.",
            "Cycle closures were computed in the canonical world frame.",
            "No global adjustment, MST, radiometric, or mosaic step was run.",
        ],
        "cannot_conclude": [
            "Absolute scene geolocation correctness.",
            "Whether bundle adjustment is required.",
            "Whether local warping is required.",
        ],
    }
    write_json(out / "12_network_consistency_conclusion.json", summary)
    lines = ["=== network consistency conclusion ==="]
    for n, (k, v) in enumerate(summary.items(), start=1):
        if k in ("can_conclude", "cannot_conclude"):
            continue
        lines.append(f"{n}. {v}")
    lines.append("")
    lines.append("CAN conclude:")
    lines += ["  - " + c for c in summary["can_conclude"]]
    lines.append("CANNOT conclude:")
    lines += ["  - " + c for c in summary["cannot_conclude"]]
    (out / "12_network_consistency_conclusion.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"Selected-pair cycle consistency complete -> {out}")
    for name, (cls, stats) in cycle_states.items():
        s = stats if isinstance(stats, dict) and "p95_px" in stats else None
        if s:
            print(f"  {name:14s} P95={s['p95_px']:6.2f}px  median="
                  f"{s['median_px']:6.2f}px  -> {cls}")
        else:
            print(f"  {name:14s} -> {cls}")
    print(f"new pairs OK={n_new_ok}/{len(new_results)}  "
          f"validated={n_new_valid}")
    print(f"network state: {net_state}")
    print("=" * 66)
    return 0


def _cycle_p95(cycle_states, name):
    cls, stats = cycle_states.get(name, (None, None))
    if not isinstance(stats, dict) or "p95_px" not in stats:
        return None
    return stats["p95_px"]


def Affine_or_reg(i, j, regs, grids, key):
    """Common transform: use the new reg's transform; else rebuild from grids."""
    from rasterio.transform import Affine

    if (i, j) in regs and regs[(i, j)].pair_common_transform is not None:
        return regs[(i, j)].pair_common_transform
    if key in grids:
        t = grids[key]["transform"]
        return Affine(*t)
    return None


def _plot_network_event(scenes, world_edges, eligible, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import re

    fig, ax = plt.subplots(figsize=(9, 8))
    pos = {}
    for s in scenes:
        m = re.search(r"_E([0-9.]+)_N([0-9.]+)_", s.name)
        pos[s.index] = (float(m.group(1)), float(m.group(2)))
    for i, (x, y) in pos.items():
        color = "steelblue" if i in (0, 1, 2, 3, 4) else "seagreen"
        ax.plot(x, y, "o", color=color, ms=15, mec="black")
        ax.text(x, y + 0.02, str(i), ha="center", fontsize=8)
    drawn = set()
    for (i, j), T in world_edges.items():
        if (i, j) in drawn or (j, i) in drawn:
            continue
        drawn.add((i, j))
        kind = "existing" if (i, j) in ncd.REUSED_OLD_PAIRS or \
            (j, i) in ncd.REUSED_OLD_PAIRS else "new"
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        if (0, 1) in ((i, j), (j, i)):
            ax.plot([x1, x2], [y1, y2], "--", color="orange", lw=1.6,
                    label="baseline anomalous 0-1")
        elif kind == "new":
            ax.plot([x1, x2], [y1, y2], "-", color="green", lw=1.6,
                    label="new validated edge")
        else:
            ax.plot([x1, x2], [y1, y2], "-", color="black", lw=1.2,
                    label="existing validated edge")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="best")
    ax.set_title("9-scene registration network (validated direct edges)")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_cycle_bars(cycle_rows, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [r["cycle_id"] for r in cycle_rows]
    values = [r["p95_px"] for r in cycle_rows]
    colors = []
    for r in cycle_rows:
        if not r["available"]:
            colors.append("lightgray")
        elif (r["classification"] or "") == "CLOSED":
            colors.append("green")
        elif (r["classification"] or "").startswith("SMALL"):
            colors.append("yellowgreen")
        else:
            colors.append("crimson")
    x = np.arange(len(names))
    ax.bar(x, [v if v is not None else 0 for v in values], color=colors)
    for xi, r in zip(x, cycle_rows):
        if not r["available"] or r["p95_px"] is None:
            ax.text(xi, 0.02, "N/A", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(xi, r["p95_px"] + 0.3, f"{r['p95_px']:.1f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("P95 closure residual (px)")
    ax.set_title("Cycle closure comparison (C0 = baseline 0-1-4)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())