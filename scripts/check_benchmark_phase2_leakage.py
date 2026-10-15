"""Read-only spatial DEV/TEST overlap and adjacency checker for reviewed regions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rasterio.warp import transform_geom
from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "docs/benchmark/v1"
PHASE2 = V1 / "phase2"


def inspect(review_path: Path, old_geojson: Path, new_geojson: Path, buffer_m: float = 140) -> dict:
    if buffer_m < 0:
        raise ValueError("buffer_m must be nonnegative")
    with review_path.open(newline="", encoding="utf-8-sig") as stream:
        reviews = list(csv.DictReader(stream))
    geometries = {}
    for path in (old_geojson, new_geojson):
        fc = json.loads(path.read_text(encoding="utf-8"))
        for feature in fc["features"]:
            key = feature["properties"]["region_id"]
            if key in geometries:
                raise ValueError(f"Duplicate region ID: {key}")
            geometries[key] = shape(transform_geom("EPSG:4326", "EPSG:32650", feature["geometry"]))
    seen = set()
    development, test, unassigned = [], [], []
    for row in reviews:
        key = row["region_id"]
        if key in seen or key not in geometries:
            raise ValueError(f"Duplicate or missing geometry for {key}")
        seen.add(key)
        role = row["role"]
        if role not in ("DEVELOPMENT", "TEST", "UNASSIGNED"):
            raise ValueError(f"Unknown role for {key}: {role}")
        if role == "UNASSIGNED":
            unassigned.append(key)
        elif row["review_status"] != "APPROVED":
            unassigned.append(key)  # role alone never authorizes formal test
        elif role == "DEVELOPMENT":
            development.append(key)
        else:
            test.append(key)
    if seen != set(geometries):
        raise ValueError(f"Review CSV is missing region IDs: {sorted(set(geometries) - seen)}")
    conflicts = []
    for dev in development:
        for tst in test:
            gap = geometries[dev].distance(geometries[tst])
            if gap <= buffer_m:
                conflicts.append({"development": dev, "test": tst, "distance_m": gap})
    return {"regions_in_review": len(reviews), "unassigned_or_unapproved": unassigned,
            "development_count": len(development), "test_count": len(test),
            "buffer_m": buffer_m, "conflicts": conflicts,
            "ready_for_spatial_split": not unassigned and bool(development) and bool(test) and not conflicts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=PHASE2 / "radiometric_region_review_template.csv")
    parser.add_argument("--buffer-m", type=float, default=140)
    args = parser.parse_args()
    result = inspect(args.review, V1 / "evaluation_region_candidates.geojson",
                     PHASE2 / "extended_temporal_regions.geojson", args.buffer_m)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["conflicts"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
