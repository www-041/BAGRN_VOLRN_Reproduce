"""Create three mask-verified long-gap region candidates and human review rows."""

from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window
from rasterio.warp import transform_geom

from build_benchmark_v1_inputs import overlap_window


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "docs/benchmark/v1"
OUT = V1 / "phase2"
PAIRS = ("00_10", "02_10", "04_11")
TILE = 512


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def best_valid_tile(a, b, tile_size=TILE):
    pair = overlap_window(a, b)
    if pair is None:
        raise ValueError("No bounding intersection")
    wa, wb = pair
    best = None
    for y in range(0, int(wa.height), tile_size):
        for x in range(0, int(wa.width), tile_size):
            width = min(tile_size, int(wa.width) - x)
            height = min(tile_size, int(wa.height) - y)
            if width < tile_size // 2 or height < tile_size // 2:
                continue
            u = Window(wa.col_off + x, wa.row_off + y, width, height)
            v = Window(wb.col_off + x, wb.row_off + y, width, height)
            mask = (a.read_masks(1, window=u) != 0) & (b.read_masks(1, window=v) != 0)
            count = int(np.count_nonzero(mask))
            candidate = (count / (width * height), count, u, v)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None or best[1] == 0:
        raise ValueError("No mask-verified tile")
    return best


def stretch(data, mask):
    values = data[mask > 0]
    if not values.size:
        return np.zeros(data.shape, dtype=np.uint8)
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        hi = lo + 1
    result = np.clip((data.astype(np.float32) - lo) * 255 / (hi-lo), 0, 255).astype(np.uint8)
    result[mask == 0] = 0
    return result


def geometry_from_window(ds, window):
    left, bottom, right, top = rasterio.windows.bounds(window, ds.transform)
    ring = [[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]
    return transform_geom(ds.crs, "EPSG:4326", {"type": "Polygon", "coordinates": [ring]}), (left, bottom, right, top)


def main():
    pairs = {p["pair_id"]: p for p in read_rows(V1 / "pair_candidates.csv")}
    manifest = {int(m["scene_index"]): m for m in read_rows(
        ROOT / "data/output/final_pipeline/b9_13scene_formal_v2/scene_manifest.csv")}
    original = json.loads((V1 / "evaluation_region_candidates.geojson").read_text(encoding="utf-8"))
    if len(original["features"]) != 15:
        raise RuntimeError("Phase1 region count changed; refuse to extend")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "previews").mkdir(exist_ok=True)
    identity_fields = ["scene_index", "scene_id", "band", "source_path", "source_file_size_bytes",
                       "recorded_source_sha256_task15", "recorded_mtl_sha256_task15",
                       "current_hash_check", "input_version", "other_bands_hash_status"]
    with (OUT / "input_identity_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=identity_fields)
        writer.writeheader()
        for m in manifest.values():
            writer.writerow(dict(scene_index=m["scene_index"], scene_id=m["scene_id"], band="B9",
                                 source_path=m["b9_path"], source_file_size_bytes=m["source_file_size_bytes"],
                                 recorded_source_sha256_task15=m["source_file_sha256"],
                                 recorded_mtl_sha256_task15=m["mtl_file_sha256"],
                                 current_hash_check="NOT_RECHECKED_PHASE2",
                                 input_version="TASK15_FORMAL_V2_RECORDED_SOURCE",
                                 other_bands_hash_status="B1_B8_B10_B16_NOT_FROZEN"))
    extended, new_features = [], []
    for pair_id in PAIRS:
        pair = pairs[pair_id]
        i, j = int(pair["idx_i"]), int(pair["idx_j"])
        with ExitStack() as stack:
            a = stack.enter_context(rasterio.open(manifest[i]["b9_path"]))
            b = stack.enter_context(rasterio.open(manifest[j]["b9_path"]))
            fraction, valid_count, wa, wb = best_valid_tile(a, b)
            geom, bbox = geometry_from_window(a, wa)
            if int(pair["valid_overlap_pixels"]) <= 0:
                raise RuntimeError("Long-gap pair has no valid B9 overlap")
            x = {"pair_id": pair_id, "scene_i": pair["scene_i"], "scene_j": pair["scene_j"],
                 "scene_index_i": i, "scene_index_j": j,
                 "acquisition_i": pair["acquisition_i"], "acquisition_j": pair["acquisition_j"],
                 "days_apart": pair["days_apart"], "quality_i": pair["quality_i"],
                 "quality_j": pair["quality_j"],
                 "pair_total_valid_overlap_pixels": pair["valid_overlap_pixels"],
                 "candidate_window_valid_pixels": valid_count,
                 "candidate_window_total_pixels": int(wa.width * wa.height),
                 "candidate_window_valid_fraction": round(fraction, 8),
                 "candidate_crs": "EPSG:32650", "candidate_left_m": bbox[0],
                 "candidate_bottom_m": bbox[1], "candidate_right_m": bbox[2],
                 "candidate_top_m": bbox[3],
                 "source_sha256_i": manifest[i]["source_file_sha256"],
                 "source_sha256_j": manifest[j]["source_file_sha256"],
                 "candidate_label": "UNKNOWN", "review_status": "UNVERIFIED",
                 "role": "UNASSIGNED", "terrain": "TERRAIN_UNKNOWN"}
            extended.append(x)
            new_features.append({"type": "Feature", "geometry": geom,
                                 "properties": {"region_id": "LONG_" + pair_id, **x,
                                                "mask_verification": "native_B9_both_masks_nonzero_in_exact_window"}})
            left = stretch(a.read(1, window=wa), a.read_masks(1, window=wa))
            right = stretch(b.read(1, window=wb), b.read_masks(1, window=wb))
            width, height = int(wa.width), int(wa.height)
            canvas = Image.new("RGB", (width * 2, height + 40), "white")
            canvas.paste(Image.fromarray(left).convert("RGB"), (0, 40))
            canvas.paste(Image.fromarray(right).convert("RGB"), (width, 40))
            ImageDraw.Draw(canvas).text((6, 8), f"{pair_id} | {pair['days_apart']} d | UNVERIFIED | B9", fill="black")
            canvas.save(OUT / "previews" / f"long_{pair_id}_unverified.png")
    with (OUT / "extended_temporal_pair_candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(extended[0]))
        writer.writeheader()
        writer.writerows(extended)
    (OUT / "extended_temporal_regions.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "name": "unverified_long_gap_candidates", "features": new_features},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    review_fields = ["region_id", "pair_id", "source", "scene_i", "scene_j", "days_apart",
                     "label", "review_status", "role", "terrain", "landcover",
                     "reviewer", "review_time_utc", "evidence_notes", "geometry_source",
                     "spatial_buffer_m", "leakage_check_status"]
    review_rows = []
    for feature in original["features"] + new_features:
        prop = feature["properties"]
        review_rows.append({"region_id": prop["region_id"], "pair_id": prop["pair_id"],
                            "source": "PHASE2_MASK_VERIFIED" if prop["region_id"].startswith("LONG_") else "PHASE1_BBOX_ONLY",
                            "scene_i": prop["scene_i"], "scene_j": prop["scene_j"],
                            "days_apart": prop.get("days_apart", prop.get("time_gap_days", "")),
                            "label": "UNKNOWN", "review_status": "UNVERIFIED", "role": "UNASSIGNED",
                            "terrain": "TERRAIN_UNKNOWN", "landcover": "UNKNOWN", "reviewer": "",
                            "review_time_utc": "", "evidence_notes": "",
                            "geometry_source": "extended_temporal_regions.geojson" if prop["region_id"].startswith("LONG_") else "../evaluation_region_candidates.geojson",
                            "spatial_buffer_m": 140, "leakage_check_status": "NOT_CHECKED"})
    with (OUT / "radiometric_region_review_template.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_fields)
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"Generated {len(extended)} mask-verified long-gap candidates, {len(review_rows)} UNVERIFIED review rows")
    for row in extended:
        print(row["pair_id"], row["days_apart"], row["candidate_window_valid_pixels"],
              row["candidate_window_total_pixels"], row["candidate_window_valid_fraction"])


if __name__ == "__main__":
    main()
