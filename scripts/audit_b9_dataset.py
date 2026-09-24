"""Generate a metadata-only manifest for the external B9 reference dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.multiscene_sift.b9_dataset import discover_b9_scenes


DEFAULT_B9_ROOT = Path(
    r"D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce"
    r"\data\input\B9_reference_13scenes_full"
)
DEFAULT_OUTPUT_DIR = Path("data/output/b9_five_scene_validation/dataset_audit")
MANIFEST_FIELDS = [
    "scene_index",
    "scene_id",
    "acquisition_time",
    "time_dir",
    "scene_dir",
    "b9_path",
    "mtl_path",
    "crs",
    "pixel_size_x_m",
    "pixel_size_y_m",
    "width",
    "height",
    "left",
    "bottom",
    "right",
    "top",
    "cloud_cover",
]


def build_manifest_rows(records: list[dict]) -> list[dict]:
    """Convert adapter records into the stable audit schema."""
    aliases = {
        "pixel_size_x_m": "resolution_x_m",
        "pixel_size_y_m": "resolution_y_m",
    }
    rows = []
    for index, record in enumerate(records):
        row = {
            field: record.get(aliases.get(field, field))
            for field in MANIFEST_FIELDS
        }
        row["scene_index"] = index
        rows.append(row)
    return rows


def write_manifest(
    records: list[dict],
    output_dir: str | Path,
    *,
    expected_count: int = 13,
) -> dict:
    """Write CSV/JSON audit artifacts and return the audit summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_manifest_rows(records)
    status = "COMPLETE" if len(rows) == expected_count else "AUDIT_SCENE_COUNT_MISMATCH"
    summary = {
        "status": status,
        "expected_scene_count": expected_count,
        "n_scenes": len(rows),
        "scenes": rows,
    }

    with open(output_dir / "01_scene_manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "01_scene_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_B9_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    records = discover_b9_scenes(args.input_root)
    summary = write_manifest(records, args.output_dir)
    print(f"status={summary['status']}")
    print(f"scenes={summary['n_scenes']}")
    print(f"output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
