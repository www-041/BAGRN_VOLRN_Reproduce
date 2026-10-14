from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark_v1_radiometric_r1.inventory import _read_csv, select_representative_pairs  # noqa: E402
from src.benchmark_v1_radiometric_r1.roi_candidates import build_r1_roi_candidates  # noqa: E402
from src.benchmark_v1_radiometric_r1.roi_store import ROI_FIELDS  # noqa: E402
from src.benchmark_v1_radiometric_r1.roi import to_projected  # noqa: E402
from shapely.geometry import shape  # noqa: E402


def main() -> int:
    docs = ROOT / "docs/benchmark/v1"
    out = docs / "radiometric_r1"
    pairs = _read_csv(out / "radiometric_pair_inventory.csv")
    selected = select_representative_pairs(pairs, target=10)
    scenes = _read_csv(docs / "scene_inventory.csv")
    manifest = _read_csv(ROOT / "data/output/final_pipeline/b9_13scene_formal_v2/scene_manifest.csv")
    hashes = {row["scene_id"]: row["source_file_sha256"] for row in manifest if row.get("band") == "B9"}
    base = json.loads((docs / "evaluation_region_candidates.geojson").read_text(encoding="utf-8"))
    extended = json.loads((docs / "phase2/extended_temporal_regions.geojson").read_text(encoding="utf-8"))
    collection = build_r1_roi_candidates(selected, [base], [extended], scenes, hashes)
    candidate_path = out / "radiometric_roi_candidates.geojson"
    candidate_path.write_text(json.dumps(collection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = list(ROI_FIELDS) + ["quality_i", "quality_j", "temporal_bin", "candidate_geometry_source"]
    template_path = out / "radiometric_roi_review_template.csv"
    with template_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for feature in collection["features"]:
            props = feature["properties"]
            row = {key: props.get(key, "") for key in fields}
            row.update({
                "geometry": to_projected(shape(feature["geometry"]), "EPSG:4326").wkt,
                "geometry_crs": "EPSG:32650", "role": props.get("spatial_role", "UNASSIGNED"),
                "grid_cell_id": props.get("grid_cell_id", ""), "spatial_status": props.get("spatial_status", ""),
                "purpose": "UNASSIGNED", "label": "UNKNOWN", "review_status": "UNVERIFIED",
                "reviewer": "", "review_time_utc": "", "second_reviewer": "",
                "second_review_status": "UNVERIFIED", "second_review_time_utc": "",
                "candidate_source": props.get("source_region_id", ""),
                "spatial_overlap_audit": "PENDING_PAIRWISE_AUDIT",
            })
            writer.writerow(row)
    print(f"R1_ROI_CANDIDATES_READY count={len(collection['features'])} all=UNVERIFIED")
    print(f"GeoJSON: {candidate_path.relative_to(ROOT)}")
    print(f"Template: {template_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
