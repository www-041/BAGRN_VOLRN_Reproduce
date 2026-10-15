"""CLI: nine-scene geographic overlap network pre-check (metadata only).

Reads the B14 footprints of the fixed nine scenes (5 original + 4 new) from
their GeoTIFF metadata, computes the true 36-pair geographic overlap graph
reusing ``overlap_graph`` semantics, groups scenes by acquisition time, lists
scene0 / scene1 neighbours, enumerates triangle candidates, recommends the
next registration pairs — and runs NO registration.

Usage (PowerShell)::

    python -m scripts.diagnose_nine_scene_overlap_network `
        --input-root "D:/科研/地质一号/文献/BAGRN_VOLRN_Reproduce/data/input/flat" `
        --output-dir "data/output/nine_scene_overlap_diagnostic"
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.multiscene_sift import nine_scene_overlap_diagnostic as nsd
from src.multiscene_sift.dataset import discover_five_scenes
from src.multiscene_sift.frame_diagnostics import write_json

logger = logging.getLogger(__name__)

METRIC_DEFINITION = (
    "overlap_ratio_i = intersection_area / area(scene_i); "
    "intersection from axis-aligned bounds boxes; identical formula to "
    "src/multiscene_sift/overlap_graph.py build_geographic_overlap_graph."
)


def _scene_meta(scene, band: str) -> dict:
    import rasterio

    path = scene.band_paths[band]
    with rasterio.open(path) as src:
        return {
            "scene_index": scene.index,
            "scene_name": scene.name,
            "b14_path": path,
            "crs": str(src.crs),
            "width": int(src.width),
            "height": int(src.height),
            "resolution_x": float(abs(src.transform.a)),
            "resolution_y": float(abs(src.transform.e)),
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
        }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", default="data/output/nine_scene_overlap_diagnostic")
    parser.add_argument("--band", default="B14")
    parser.add_argument(
        "--five-scene-run-dir", default=None,
        help="Optional dir of the previous five-scene sift run; its "
             "pairwise_status labels original-5 pairs that were empirically "
             "NO_OVERLAP / LOW_INLIER_RATIO.",
    )
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    band = args.band

    known_status: dict[tuple[int, int], str] = {}
    if args.five_scene_run_dir:
        run5 = Path(args.five_scene_run_dir)
        pw = run5 / "pairwise_summary.json"
        if pw.is_file():
            with open(pw, encoding="utf-8") as f:
                for r in json.load(f)["results"]:
                    i, j = int(r["idx_i"]), int(r["idx_j"])
                    if i > j:
                        i, j = j, i
                    if i in nsd.ORIGINAL_INDICES and j in nsd.ORIGINAL_INDICES:
                        known_status[(i, j)] = r["status"]
        else:
            print(f"WARNING: {pw} not found; known_status empty")

    if not nsd.validate_scene_names(list(nsd.SCENE_NAMES_9)):
        print("ERROR: scene-name list must contain exactly 9 unique entries.")
        return 1

    # --- load scenes (reuses tested discovery; fixed index order) -----------
    scenes, manifest = discover_five_scenes(
        args.input_root, list(nsd.SCENE_NAMES_9), bands=(band,)
    )

    # --- Task 3: manifest ----------------------------------------------------
    metadatas = [_scene_meta(s, band) for s in scenes]
    crs_list = {m["crs"] for m in metadatas}
    res_list = {(m["resolution_x"], m["resolution_y"]) for m in metadatas}
    problems = []
    for m in metadatas:
        if m["width"] <= 0 or m["height"] <= 0:
            problems.append(f"scene {m['scene_index']}: invalid shape")
        b = m["bounds"]
        if not (b["right"] > b["left"] and b["top"] > b["bottom"]):
            problems.append(f"scene {m['scene_index']}: invalid bounds")
    if not metadatas:
        problems.append("no scenes read")
    if len(crs_list) > 1:
        problems.append(f"CRS mismatch: {crs_list}")
    for rx, ry in res_list:
        base = metadatas[0]["resolution_x"]
        if abs(rx - base) / base > 0.01 or abs(ry - base) / base > 0.01:
            problems.append(f"resolution mismatch ({rx},{ry}) vs base {base}")
            break
    manifest_01 = {
        "n_scenes": len(metadatas),
        "scenes": metadatas,
        "checks": {
            "all_b14_present": all(m["b14_path"] for m in metadatas),
            "crs_consistent": len(crs_list) == 1,
            "resolution_consistent": len(res_list) == 1,
            "bounds_valid": not [p for p in problems if "bounds" in p],
            "problems": problems,
        },
        "discoveries_ok": True,
    }
    write_json(out / "01_nine_scene_manifest.json", manifest_01)
    _write_csv(out / "01_nine_scene_manifest.csv",
               ["scene_index", "scene_name", "b14_path", "crs", "width",
                "height", "resolution_x", "resolution_y", "nodata", "dtype"],
               [{k: m[k] for k in ("scene_index", "scene_name", "b14_path",
                                   "crs", "width", "height", "resolution_x",
                                   "resolution_y", "nodata", "dtype")}
                for m in metadatas])
    if problems:
        print("STOP: manifest problems:")
        for p in problems:
            print("  ", p)
        return 1

    # --- Task 4: acquisition groups ------------------------------------------
    groups = nsd.acquisition_groups(scenes)
    write_json(out / "02_acquisition_groups.json", groups)

    # --- Task 6: all 36 pairs ------------------------------------------------
    overlaps = nsd.all_pair_overlaps(scenes, band)
    pair_rows = []
    for r in overlaps:
        row = {
            "idx_i": r["idx_i"], "idx_j": r["idx_j"],
            "scene_i": r["scene_i"], "scene_j": r["scene_j"],
            "intersection_exists": int(r["intersection_exists"]),
            "intersection_width_m": (None if r["intersection_width_m"] is None
                                     else round(r["intersection_width_m"], 2)),
            "intersection_height_m": (None if r["intersection_height_m"] is None
                                      else round(r["intersection_height_m"], 2)),
            "intersection_area_m2": round(r["intersection_area_m2"], 3),
            "scene_i_area_m2": round(r["scene_i_area_m2"], 3),
            "scene_j_area_m2": round(r["scene_j_area_m2"], 3),
            "overlap_ratio_i": round(r["overlap_ratio_i"], 6),
            "overlap_ratio_j": round(r["overlap_ratio_j"], 6),
            "project_existing_overlap_metric": round(r["overlap_ratio_i"], 6),
        }
        pair_rows.append(row)
    _write_csv(out / "03_all_pair_geographic_overlap.csv", [
        "idx_i", "idx_j", "scene_i", "scene_j", "intersection_exists",
        "intersection_width_m", "intersection_height_m", "intersection_area_m2",
        "scene_i_area_m2", "scene_j_area_m2", "overlap_ratio_i",
        "overlap_ratio_j", "project_existing_overlap_metric",
    ], pair_rows)
    write_json(out / "03_all_pair_geographic_overlap.json", {
        "n_pairs": len(overlaps),
        "expected_pairs": 36,
        "overlap_metric_definition": METRIC_DEFINITION,
        "rows": pair_rows,
    })

    # --- Task 7: matrices ------------------------------------------------------
    ratio_mat, exists_mat = nsd.overlap_matrices(overlaps, len(scenes))
    names = [s.name for s in scenes]
    _write_csv(out / "04_overlap_ratio_matrix.csv",
               ["scene"] + [str(i) for i in range(len(scenes))],
               [{"scene": str(i), **{str(j): (round(ratio_mat[i, j], 6)
                                              if i != j else 1.0)
                                     for j in range(len(scenes))}}
                for i in range(len(scenes))])
    _write_csv(out / "04_overlap_exists_matrix.csv",
               ["scene"] + [str(i) for i in range(len(scenes))],
               [{"scene": str(i), **{str(j): int(exists_mat[i, j])
                                     for j in range(len(scenes))}}
                for i in range(len(scenes))])

    # --- Task 8/9: scene0 / scene1 neighbours ---------------------------------
    def _neigh_rows(nlist, anchor):
        rows = []
        for x in nlist:
            row = {k: (round(v, 6) if isinstance(v, float) else v)
                   for k, v in x.items()}
            key = tuple(sorted((anchor, x["neighbor_scene"])))
            known = known_status.get(key, "") if all(
                k in nsd.ORIGINAL_INDICES for k in key) else ""
            row["known_status"] = known
            rows.append(row)
        return rows

    neigh0 = nsd.neighbors_of(scenes, overlaps, 0)
    neigh1 = nsd.neighbors_of(scenes, overlaps, 1)
    _write_csv(out / "05_scene0_neighbors.csv", [
        "neighbor_scene", "neighbor_name", "overlap_ratio",
        "overlap_area_km2", "overlap_area_m2", "same_acquisition_date",
        "time_gap_seconds", "existing_or_new", "note", "known_status",
    ], _neigh_rows(neigh0, 0))
    _write_csv(out / "06_scene1_neighbors.csv", [
        "neighbor_scene", "neighbor_name", "overlap_ratio",
        "overlap_area_km2", "overlap_area_m2", "same_acquisition_date",
        "time_gap_seconds", "existing_or_new", "note", "known_status",
    ], _neigh_rows(neigh1, 1))

    # --- Task 10: triangles ----------------------------------------------------
    triangles = nsd.enumerate_triangles(overlaps)
    tri_rows = nsd.triangle_rows(triangles, scenes, overlaps)
    _write_csv(out / "07_triangle_candidates.csv", [
        "triangle_id", "scene_a", "scene_b", "scene_c",
        "overlap_ab", "overlap_bc", "overlap_ca",
        "min_overlap", "mean_overlap", "contains_scene0",
        "contains_scene1", "contains_problem_edge_0_1", "priority",
    ], tri_rows)

    # --- Task 11: figure ---------------------------------------------------------
    try:
        _plot_graph(scenes, overlaps, out / "08_nine_scene_overlap_graph.png")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: graph figure skipped: {type(exc).__name__}: {exc}")

    # --- Task 12: recommendations ------------------------------------------------
    recs = nsd.select_recommended_pairs(scenes, overlaps, triangles)
    for rec in recs:
        a, b = (int(x) for x in rec["pair"].split("-"))
        key = tuple(sorted((a, b)))
        status = known_status.get(key, "")
        rec["known_status"] = status or ""
        if status in ("NO_OVERLAP", "LOW_INLIER_RATIO",
                      "TOO_FEW_INLIERS", "FAILED"):
            rec["why_important"] += (
                f"; previous 5-scene run reported {status} for this pair "
                "(bounds-overlap may not imply useful imagery overlap)"
            )
    write_json(out / "09_recommended_registration_pairs.json", {
        "recommendations": recs,
        "note": "geographic-overlap pre-check only; no registration was run",
    })
    (out / "09_recommended_registration_pairs.txt").write_text(
        "\n".join(
            [f"{r['pair']}: {r['why_important']} (overlap_ratio={r['overlap_ratio']}, "
             f"area={r['overlap_area_km2']} km2) enables={r['enables_cycle']}"
             for r in recs]
        ) + "\n", encoding="utf-8",
    )

    # --- Task 13: summary ---------------------------------------------------------
    triangles_01 = [
        tr for tr in tri_rows
        if tr["contains_scene0"] and tr["contains_scene1"]
    ]
    summary = {
        "1_nine_scenes_read": len(metadatas) == 9 and not problems,
        "2_new_scene_overlaps": {
            str(k): [n["neighbor_scene"] for n in nsd.neighbors_of(scenes, overlaps, k)]
            for k in nsd.NEW_INDICES
        },
        "3_scene0_neighbor_count": len(neigh0),
        "4_scene1_neighbor_count": len(neigh1),
        "5_scene0_new_neighbors": [n["neighbor_scene"] for n in neigh0
                                   if n["existing_or_new"] == "new"],
        "6_scene1_new_neighbors": [n["neighbor_scene"] for n in neigh1
                                   if n["existing_or_new"] == "new"],
        "7_triangles_with_0_1": [
            (tr["scene_a"], tr["scene_b"], tr["scene_c"])
            for tr in triangles_01
        ],
        "7_triangle_ids_0_1": [tr["triangle_id"] for tr in triangles_01],
        "8_total_triangle_candidates": len(triangles),
        "9_recommended_pairs": [r["pair"] for r in recs],
        "10_ready_for_multi_scene_network": bool(triangles),
        "overlap_metric_definition": METRIC_DEFINITION,
        "note": "No registration was run.",
    }
    write_json(out / "10_overlap_network_summary.json", summary)
    lines = [
        "=== Nine-scene overlap network summary ===",
        f"1. 9 scenes read: {summary['1_nine_scenes_read']}",
        f"2. new-scene overlaps: {summary['2_new_scene_overlaps']}",
        f"3. scene0 neighbours: {summary['3_scene0_neighbor_count']} -> {[n['neighbor_scene'] for n in neigh0]}",
        f"4. scene1 neighbours: {summary['4_scene1_neighbor_count']} -> {[n['neighbor_scene'] for n in neigh1]}",
        f"5. scene0 new neighbours: {summary['5_scene0_new_neighbors']}",
        f"6. scene1 new neighbours: {summary['6_scene1_new_neighbors']}",
        f"7. triangles containing 0-1: {summary['7_triangle_ids_0_1']}",
        f"8. total triangle candidates: {summary['8_total_triangle_candidates']}",
        f"9. recommended pairs: {summary['9_recommended_pairs']}",
        f"10. ready for multi-scene network: {summary['10_ready_for_multi_scene_network']}",
        "",
        "No registration was run this round.",
    ]
    (out / "10_overlap_network_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )

    print("\n" + "=" * 66)
    print(f"Nine-scene overlap pre-check complete -> {out}")
    print(f"scenes read: {len(metadatas)}/9, problems: {problems or 'none'}")
    print(f"scene0 neighbours ({len(neigh0)}): "
          + ", ".join(f"{n['neighbor_scene']}({n['overlap_ratio']:.3f})"
                      for n in neigh0))
    print(f"scene1 neighbours ({len(neigh1)}): "
          + ", ".join(f"{n['neighbor_scene']}({n['overlap_ratio']:.3f})"
                      for n in neigh1))
    print(f"triangles: {len(triangles)} | recommended: {[r['pair'] for r in recs]}")
    print("=" * 66)
    return 0


def _plot_graph(scenes, overlaps, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    positions = {}
    for s in scenes:
        m = re_parse_center(s.name)
        positions[s.index] = m
    for i, (x, y) in positions.items():
        color = "steelblue" if i in nsd.ORIGINAL_INDICES else "seagreen"
        ax.plot(x, y, "o", color=color, ms=16, mec="black")
        ax.text(x, y - 0.012, f"{i}\n{i in (0, 1) and '★' or ''}",
                ha="center", va="top", fontsize=8)
        en = f"E{x:.1f} N{y:.1f}"
        datetime_str = re_dt_scene(s.name)
        ax.text(x + 0.02, y + 0.012, f"{en}\n{datetime_str}", fontsize=6,
                va="bottom")
    for r in overlaps:
        if not r["intersection_exists"]:
            continue
        i, j = r["idx_i"], r["idx_j"]
        x1, y1 = positions[i]
        x2, y2 = positions[j]
        style = ":g" if i == 0 and j == 1 else "-k"
        ax.plot([x1, x2], [y1, y2], style, lw=0.8, alpha=0.6)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, f"{r['overlap_ratio_i']:.2f}", fontsize=6,
                color="darkred", ha="center")
    ax.set_aspect("equal")
    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")
    ax.set_title("Nine-scene geographic overlap graph\n"
                 "(blue=original 0-4, green=new 5-8; 0-1 = previously diagnosed pair)")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


import re  # noqa: E402


def re_parse_center(name: str) -> tuple[float, float]:
    m = re.search(r"_E([0-9.]+)_N([0-9.]+)_", name)
    return float(m.group(1)), float(m.group(2))


def re_dt_scene(name: str) -> str:
    dt = nsd.parse_acquisition_datetime(name)
    return dt.strftime("%Y-%m-%d")


if __name__ == "__main__":
    sys.exit(main())