"""Recommend B9 tie-point candidates; never create approved benchmark truth.

Only four phase-2 trial pairs are allowed. This script reads source B9 GeoTIFFs
and their native masks, not Task15 matcher outputs or RANSAC inliers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import rasterio
from rasterio.windows import transform as window_transform

from benchmark_candidate_core import CandidateReviewStore, generate_candidates_arrays
from build_benchmark_v1_inputs import overlap_window


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "docs/benchmark/v1/phase2/input_identity_manifest.csv"
DEFAULT_DIR = ROOT / "docs/benchmark/v1/phase2/auto_candidates"
TRIAL_PAIRS = ("00_01", "02_03", "05_07", "08_10")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_rows() -> dict[int, dict]:
    with IDENTITY.open(newline="", encoding="utf-8-sig") as stream:
        return {int(row["scene_index"]): row for row in csv.DictReader(stream)}


def generate(pair_id: str, output: Path, *, max_candidates: int = 40,
             grid_side: int = 8, search_radius: int = 12) -> dict:
    if pair_id not in TRIAL_PAIRS:
        raise ValueError("Only four frozen phase-2 trial pairs may be generated")
    if output.exists():
        raise FileExistsError(f"Candidate file exists; refusing to replace reviews: {output}")
    i, j = (int(part) for part in pair_id.split("_"))
    sources = _identity_rows()
    left, right = sources[i], sources[j]
    path_i, path_j = Path(left["source_path"]), Path(right["source_path"])
    for path, row in ((path_i, left), (path_j, right)):
        if not path.is_file() or path.stat().st_size != int(row["source_file_size_bytes"]):
            raise RuntimeError(f"Source file identity/size changed: {path}")
        if _sha256(path) != row["recorded_source_sha256_task15"]:
            raise RuntimeError(f"Source SHA256 differs from frozen Task15 manifest: {path}")
    with rasterio.open(path_i) as a, rasterio.open(path_j) as b:
        pair = overlap_window(a, b)
        if pair is None:
            raise RuntimeError("No geospatial B9 bounding overlap")
        wa, wb = pair
        ai = a.read(1, window=wa)
        bj = b.read(1, window=wb)
        mask_i = a.read_masks(1, window=wa) != 0
        mask_j = b.read_masks(1, window=wb) != 0
        if not (mask_i & mask_j).any():
            raise RuntimeError("No actual shared-valid B9 support")
        rows = generate_candidates_arrays(ai, bj, mask_i, mask_j,
                                          window_transform(wa, a.transform),
                                          window_transform(wb, b.transform),
                                          pair_id=pair_id, scene_i=left["scene_id"],
                                          scene_j=right["scene_id"],
                                          max_candidates=max_candidates,
                                          grid_side=grid_side, search_radius=search_radius)
        for row in rows:
            row["row_i"] += wa.row_off
            row["col_i"] += wa.col_off
            row["row_j"] += wb.row_off
            row["col_j"] += wb.col_off
        overlap_pixels = int((mask_i & mask_j).sum())
    store = CandidateReviewStore(output, pair_id, left["scene_id"], right["scene_id"])
    store.replace_suggestions(rows)
    store.save()
    report = {"pair_id": pair_id, "source_path_i": str(path_i), "source_path_j": str(path_j),
              "verified_sha256_i": left["recorded_source_sha256_task15"],
              "verified_sha256_j": right["recorded_source_sha256_task15"],
              "recommendation_method": "ShiTomasi_gradientNCC_bidirectional",
              "max_requested": max_candidates, "generated": len(rows),
              "grid_side": grid_side, "occupied_grid_cells": len({r["grid_cell_id"] for r in rows}),
              "shared_valid_overlap_pixels": overlap_pixels,
              "automatic_approved": 0,
              "note": "suggestions only; matcher score is not geometric truth or human precision"}
    output.with_suffix(".manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, choices=TRIAL_PAIRS)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--grid-side", type=int, default=8)
    parser.add_argument("--search-radius", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = generate(args.pair, args.output or DEFAULT_DIR / f"{args.pair}.csv",
                      max_candidates=args.max_candidates,
                      grid_side=args.grid_side, search_radius=args.search_radius)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
