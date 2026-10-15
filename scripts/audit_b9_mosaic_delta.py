"""Compare MST and Translation-L2 mosaic diagnostics without ranking them."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DELTA_FIELDS = (
    "delta_overlap_ZNCC",
    "delta_gradient_magnitude_NCC",
    "delta_gradient_orientation_cosine",
    "delta_seam_gradient_NCC",
    "delta_aux_intensity_MAE",
)


def _delta(translation_value, mst_value):
    if translation_value is None or mst_value is None:
        return None
    return round(float(translation_value) - float(mst_value), 12)


def pair_delta(mst: dict, translation: dict) -> dict:
    return {
        "delta_overlap_ZNCC": _delta(translation.get("ZNCC_intensity"), mst.get("ZNCC_intensity")),
        "delta_gradient_magnitude_NCC": _delta(translation.get("gradient_magnitude_NCC"), mst.get("gradient_magnitude_NCC")),
        "delta_gradient_orientation_cosine": _delta(translation.get("gradient_orientation_cosine"), mst.get("gradient_orientation_cosine")),
        "delta_aux_intensity_MAE": _delta(translation.get("intensity_MAE"), mst.get("intensity_MAE")),
        "common_valid_pixels": min(int(mst["valid_overlap_pixels"]), int(translation["valid_overlap_pixels"])),
    }


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _mosaic_difference(mst_path: Path, translation_path: Path) -> tuple[int, float, float]:
    with rasterio.open(mst_path) as first, rasterio.open(translation_path) as second:
        if first.width != second.width or first.height != second.height or first.transform != second.transform or first.crs != second.crs:
            raise ValueError("MST and Translation mosaics do not share the canonical grid")
        first_data = first.read(1).astype(np.float32)
        second_data = second.read(1).astype(np.float32)
        common = (first_data != first.nodata) & (second_data != second.nodata)
    delta = np.abs(first_data[common] - second_data[common])
    return int(delta.size), float(delta.mean()) if delta.size else 0.0, float(np.percentile(delta, 95)) if delta.size else 0.0


def run_delta_audit(
    mosaic_root: str | Path,
    overlap_csv: str | Path,
    seam_csv: str | Path,
    output_csv: str | Path,
) -> list[dict]:
    overlap_rows = _read_csv(Path(overlap_csv))
    seam_rows = _read_csv(Path(seam_csv))
    output_rows: list[dict] = []
    for matcher in ("sift", "loftr", "efficient_loftr", "lightglue_disk"):
        overlap_by_pair = {
            (row["global_method"], row["pair"]): row
            for row in overlap_rows if row["matcher"] == matcher
        }
        seam_by_pair = {
            (row["global_method"], row["pair"]): row
            for row in seam_rows if row["matcher"] == matcher
        }
        diff_pixels, diff_mean, diff_p95 = _mosaic_difference(
            Path(mosaic_root) / matcher / "mst" / "mosaic.tif",
            Path(mosaic_root) / matcher / "translation_l2" / "mosaic.tif",
        )
        edges = []
        for pair in sorted({pair for method, pair in overlap_by_pair if method == "MST"}):
            mst = overlap_by_pair[("MST", pair)]
            translation = overlap_by_pair[("Translation-L2", pair)]
            seam_mst = seam_by_pair[("MST", pair)]
            seam_translation = seam_by_pair[("Translation-L2", pair)]
            delta = pair_delta(mst, translation)
            delta["delta_seam_gradient_NCC"] = _delta(
                seam_translation.get("gradient_magnitude_NCC"),
                seam_mst.get("gradient_magnitude_NCC"),
            )
            row = {
                "row_type": "edge",
                "matcher": matcher,
                "pair": pair,
                **delta,
                "mosaic_abs_diff_common_pixels": diff_pixels,
                "mosaic_abs_diff_mean": diff_mean,
                "mosaic_abs_diff_p95": diff_p95,
                "sign_convention": "correlation delta=Translation-L2 minus MST; negative MAE means lower Translation-L2 mismatch",
                "common_valid_pixels_definition": "minimum of the two separately warped pair overlap counts",
            }
            output_rows.append(row)
            edges.append(row)
        summary = {field: float(np.mean([row[field] for row in edges if row[field] is not None])) for field in DELTA_FIELDS}
        summary["common_valid_pixels"] = int(sum(row["common_valid_pixels"] for row in edges))
        output_rows.append({
            "row_type": "matcher_summary",
            "matcher": matcher,
            "pair": "ALL_10",
            **summary,
            "mosaic_abs_diff_common_pixels": diff_pixels,
            "mosaic_abs_diff_mean": diff_mean,
            "mosaic_abs_diff_p95": diff_p95,
            "sign_convention": "correlation delta=Translation-L2 minus MST; negative MAE means lower Translation-L2 mismatch",
            "common_valid_pixels_definition": "sum of conservative per-edge minima",
        })
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_type", "matcher", "pair", *DELTA_FIELDS, "common_valid_pixels", "mosaic_abs_diff_common_pixels", "mosaic_abs_diff_mean", "mosaic_abs_diff_p95", "sign_convention", "common_valid_pixels_definition"]
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mosaic-root", required=True, type=Path)
    parser.add_argument("--overlap-csv", required=True, type=Path)
    parser.add_argument("--seam-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = run_delta_audit(args.mosaic_root, args.overlap_csv, args.seam_csv, args.output_csv)
    print(f"delta rows={len(rows)} (4 summaries + 40 edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
