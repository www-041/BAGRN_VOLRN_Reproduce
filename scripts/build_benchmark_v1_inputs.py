"""Read-only, bounded-memory inventory builder for Benchmark V1.

Only B9 nodata masks are read; no matcher or radiometric algorithm is executed.
"""

from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_geom


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/benchmark/v1"
TASK15 = ROOT / "data/output/final_pipeline/b9_13scene_formal_v2"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, records: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def day_gap(time_i: str, time_j: str) -> float:
    return round(abs((datetime.fromisoformat(time_i) - datetime.fromisoformat(time_j)).total_seconds()) / 86400, 6)


def overlap_window(a, b):
    """Native aligned intersection, checked against both datasets' integer grid."""
    if a.crs != b.crs or a.res != b.res or a.transform.b != 0 or b.transform.b != 0:
        raise ValueError("B9 grids are not axis-aligned with common CRS/resolution")
    left, right = max(a.bounds.left, b.bounds.left), min(a.bounds.right, b.bounds.right)
    bottom, top = max(a.bounds.bottom, b.bounds.bottom), min(a.bounds.top, b.bounds.top)
    if right <= left or top <= bottom:
        return None
    windows = []
    for ds in (a, b):
        col = (left - ds.transform.c) / ds.res[0]
        row = (ds.transform.f - top) / ds.res[1]
        width, height = (right - left) / ds.res[0], (top - bottom) / ds.res[1]
        values = (col, row, width, height)
        if any(abs(v - round(v)) > 1e-6 for v in values):
            raise ValueError("B9 grids do not share integer pixel boundaries")
        windows.append(Window(*(int(round(v)) for v in values)))
    return tuple(windows)


def count_shared_valid(a, b, chunk_rows: int = 512) -> tuple[int, int]:
    pair = overlap_window(a, b)
    if pair is None:
        return 0, 0
    wa, wb = pair
    total = int(wa.width * wa.height)
    shared = 0
    for offset in range(0, int(wa.height), chunk_rows):
        height = min(chunk_rows, int(wa.height) - offset)
        ma = a.read_masks(1, window=Window(wa.col_off, wa.row_off + offset, wa.width, height))
        mb = b.read_masks(1, window=Window(wb.col_off, wb.row_off + offset, wb.width, height))
        shared += int(np.count_nonzero((ma != 0) & (mb != 0)))
    return shared, total


def choose_pairs(pairs: list[dict], count: int = 15) -> set[str]:
    """Coverage-first stratification; deliberately never reads matcher outcomes."""
    pool = [p for p in pairs if p["valid_overlap_pixels"] > 0]
    selected: list[dict] = []
    scene_uses: dict[int, int] = {}
    bins: dict[str, int] = {}
    while pool and len(selected) < count:
        def rank(p):
            ids = (p["idx_i"], p["idx_j"])
            bucket = p["time_bin"]
            risk = int(p["quality_i"] == "RISK" or p["quality_j"] == "RISK")
            return (sum(scene_uses.get(i, 0) for i in ids), bins.get(bucket, 0),
                    -risk if sum(x["quality_i"] == "RISK" or x["quality_j"] == "RISK" for x in selected) < 3 else risk,
                    -min(p["valid_overlap_pixels"], 1_000_000), p["pair_id"])
        winner = min(pool, key=rank)
        pool.remove(winner)
        selected.append(winner)
        for idx in (winner["idx_i"], winner["idx_j"]):
            scene_uses[idx] = scene_uses.get(idx, 0) + 1
        bins[winner["time_bin"]] = bins.get(winner["time_bin"], 0) + 1
    return {p["pair_id"] for p in selected}


def time_bin(gap: float, same_date: bool) -> str:
    if same_date:
        return "SAME_DATE"
    if gap < 7:
        return "CROSS_DATE_LT_7D"
    if gap < 30:
        return "CROSS_DATE_7_29D"
    return "CROSS_DATE_GE_30D"


def make_region(pair: dict, a, b, all_datasets: dict[str, object], index_by_id: dict[str, int]) -> dict:
    left, right = max(a.bounds.left, b.bounds.left), min(a.bounds.right, b.bounds.right)
    bottom, top = max(a.bounds.bottom, b.bounds.bottom), min(a.bounds.top, b.bounds.top)
    # A central, fixed 40%-by-40% bbox candidate; exact usable support is checked at evaluation time.
    dx, dy = (right - left) * .4, (top - bottom) * .4
    cx, cy = (left + right) / 2, (bottom + top) / 2
    ring = [[cx-dx/2, cy-dy/2], [cx+dx/2, cy-dy/2],
            [cx+dx/2, cy+dy/2], [cx-dx/2, cy+dy/2], [cx-dx/2, cy-dy/2]]
    geom = transform_geom(a.crs, "EPSG:4326", {"type": "Polygon", "coordinates": [ring]})
    participating = sorted(index_by_id[name] for name, ds in all_datasets.items()
                           if min(ds.bounds.right, cx+dx/2) > max(ds.bounds.left, cx-dx/2)
                           and min(ds.bounds.top, cy+dy/2) > max(ds.bounds.bottom, cy-dy/2))
    return {"type": "Feature", "geometry": geom, "properties": {
        "region_id": "REGION_" + pair["pair_id"], "pair_id": pair["pair_id"],
        "scene_i": pair["scene_i"], "scene_j": pair["scene_j"],
        "bbox_intersecting_scene_indices": participating, "bbox_coverage_scene_count": len(participating),
        "valid_coverage_scene_count": "UNKNOWN",
        "time_gap_days": pair["days_apart"], "quality_i": pair["quality_i"],
        "quality_j": pair["quality_j"], "terrain": "TERRAIN_UNKNOWN",
        "stable_landcover_truth": "UNKNOWN", "change_truth": "UNKNOWN",
        "review_status": "UNVERIFIED", "source_crs": "EPSG:32650",
        "selection_note": "center of native B9 bbox intersection; valid-pixel support within region unverified"}}


def main() -> None:
    scenes = rows(OUT / "scene_inventory.csv")
    bands = rows(OUT / "v_band_inventory.csv")
    manifest = rows(TASK15 / "scene_manifest.csv")
    edges = rows(TASK15 / "overlap_edges.csv")
    reg = {r["pair_id"]: r for r in rows(TASK15 / "registration_pair_metrics.csv")}
    if len(scenes) != 13 or len(manifest) != 13 or len(bands) != 208 or len(edges) != 78:
        raise RuntimeError("Inventory or frozen Task15 pair count differs from expected 13/208/78")
    by_id = {s["product_id"]: s for s in scenes}
    if set(by_id) != {m["scene_id"] for m in manifest}:
        raise RuntimeError("Scene inventory and Task15 manifest disagree")
    by_band = {(b["product_id"], b["band"]): b for b in bands}
    pairs = []
    with ExitStack() as stack:
        datasets = {m["scene_id"]: stack.enter_context(rasterio.open(m["b9_path"])) for m in manifest}
        for e in edges:
            i, j = int(e["idx_i"]), int(e["idx_j"])
            mi, mj = manifest[i], manifest[j]
            if e["scene_i"] != mi["scene_id"] or e["scene_j"] != mj["scene_id"]:
                raise RuntimeError("Overlap edge index is inconsistent with Task15 scene manifest")
            count, bbox_px = count_shared_valid(datasets[e["scene_i"]], datasets[e["scene_j"]])
            gap = day_gap(mi["acquisition_time"], mj["acquisition_time"])
            same = mi["acquisition_time"][:10] == mj["acquisition_time"][:10]
            available = [f"B{k}" for k in range(1, 17) if
                         (e["scene_i"], str(k)) in by_band and (e["scene_j"], str(k)) in by_band]
            pair_id = f"{i:02d}_{j:02d}"
            r = reg.get(pair_id)
            p = {"pair_id": pair_id, "idx_i": i, "idx_j": j,
                 "scene_i": e["scene_i"], "scene_j": e["scene_j"],
                 "scene_id_i": by_id[e["scene_i"]]["scene_id"],
                 "scene_id_j": by_id[e["scene_j"]]["scene_id"],
                 "acquisition_i": mi["acquisition_time"], "acquisition_j": mj["acquisition_time"],
                 "days_apart": gap, "same_date": same, "time_bin": time_bin(gap, same),
                 "bbox_overlap_area_m2": float(e["intersection_area"]),
                 "bbox_overlap_pixels": bbox_px, "valid_overlap_pixels": count,
                 "valid_overlap_definition": "native_B9_both_rasterio_masks_nonzero_EPSG32650_14m",
                 "quality_i": by_id[e["scene_i"]]["b9_quality_status_tag"],
                 "quality_j": by_id[e["scene_j"]]["b9_quality_status_tag"],
                 "common_v_bands": ";".join(available),
                 "registration_status_task15": r["status"] if r else "NOT_ATTEMPTED",
                 "registration_accepted_task15": r["accepted_for_global"] if r else "False",
                 "verified_checkpoint_count": 0,
                 "terrain": "TERRAIN_UNKNOWN", "stable_landcover_truth": "UNKNOWN",
                 "real_change_truth": "UNKNOWN", "radiometric_temporal_suitability":
                 "CANDIDATE_REQUIRES_STABLE_CHANGE_REVIEW" if count > 0 and not same else
                 "SAME_DATE_CONTROL" if count > 0 else "NO_VALID_B9_OVERLAP"}
            pairs.append(p)
            print(f"mask {pair_id}: {count} / {bbox_px}", flush=True)
        selected = choose_pairs(pairs)
        for p in pairs:
            p["checkpoint_round1_recommended"] = p["pair_id"] in selected
            p["selection_basis"] = "coverage_time_quality_overlap_strata_no_matcher_score" if p["pair_id"] in selected else "NOT_SELECTED"
        region_pairs = [p for p in pairs if p["pair_id"] in selected]
        index_by_id = {m["scene_id"]: int(m["scene_index"]) for m in manifest}
        features = [make_region(p, datasets[p["scene_i"]], datasets[p["scene_j"]], datasets, index_by_id)
                    for p in region_pairs]
    pair_fields = list(pairs[0])
    write_csv(OUT / "pair_candidates.csv", pairs, pair_fields)
    temporal_fields = [f for f in pair_fields if f not in ("checkpoint_round1_recommended", "selection_basis")]
    write_csv(OUT / "temporal_pair_inventory.csv", [p for p in pairs if p["valid_overlap_pixels"] > 0], temporal_fields)
    (OUT / "evaluation_region_candidates.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "name": "benchmark_v1_unverified_region_candidates",
                    "features": features}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checkpoint_fields = ["point_id", "scene_i", "scene_j", "band", "row_i", "col_i", "row_j", "col_j",
                         "coordinate_precision_note", "landcover_type", "terrain_class", "review_status",
                         "reviewer", "split", "review_date", "notes"]
    write_csv(OUT / "registration_checkpoints_template.csv", [], checkpoint_fields)
    summary = {"pairs_total": len(pairs), "bbox_overlap_pairs": sum(p["bbox_overlap_pixels"] > 0 for p in pairs),
               "valid_overlap_pairs": sum(p["valid_overlap_pixels"] > 0 for p in pairs),
               "time_bins": {bin_name: sum(p["time_bin"] == bin_name and p["valid_overlap_pixels"] > 0 for p in pairs)
                             for bin_name in ("SAME_DATE", "CROSS_DATE_LT_7D", "CROSS_DATE_7_29D", "CROSS_DATE_GE_30D")},
               "recommended": sorted(selected)}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
