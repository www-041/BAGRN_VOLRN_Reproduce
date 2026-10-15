"""Compute structural metrics on separately warped source-scene overlaps."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_b9_weighted_mosaic import _load_global_transforms, _load_sources, _project_scene
from src.multiscene_sift.band_geometry import apply_world_correction_to_transform
from src.multiscene_sift.mosaic_protocol import RUN_KEYS, grid_transform
from src.registration_benchmark.mosaic_diagnostics import compute_overlap_metrics, write_overlap_metrics_csv


def run_overlap_audit(
    source_config: str | Path,
    global_root: str | Path,
    output_grid: str | Path,
    output_csv: str | Path,
    *,
    min_valid_pixels: int = 100,
) -> list[dict]:
    grid = json.loads(Path(output_grid).read_text(encoding="utf-8"))
    width, height = int(grid["width"]), int(grid["height"])
    destination_transform = grid_transform(grid)
    crs = str(grid["crs"])
    rows: list[dict] = []
    for run_key in RUN_KEYS:
        matcher, method = run_key.split("/")
        arrays, source_transforms, nodata_values, scene_ids = _load_sources(Path(source_config))
        global_transforms = _load_global_transforms(Path(global_root) / matcher / method, len(arrays))
        corrected_transforms = [
            apply_world_correction_to_transform(source_transforms[index], global_transforms[index])
            for index in range(len(arrays))
        ]
        projected = []
        for array, transform, nodata in zip(arrays, corrected_transforms, nodata_values):
            image, valid = _project_scene(
                array, transform, nodata, crs, destination_transform, width, height
            )
            projected.append((image.astype(np.float32), valid))
        del arrays

        for i, j in itertools.combinations(range(len(projected)), 2):
            image_i, valid_i = projected[i]
            image_j, valid_j = projected[j]
            metrics = compute_overlap_metrics(
                image_i,
                image_j,
                valid_i & valid_j,
                min_valid_pixels=min_valid_pixels,
            )
            rows.append({
                "matcher": matcher,
                "global_method": "MST" if method == "mst" else "Translation-L2",
                "run": run_key,
                "pair": f"{i}-{j}",
                "scene_i": scene_ids[i],
                "scene_j": scene_ids[j],
                "accepted_pair": True,
                **metrics,
            })
        del projected

    write_overlap_metrics_csv(rows, output_csv)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--global-root", required=True, type=Path)
    parser.add_argument("--output-grid", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = run_overlap_audit(args.source_config, args.global_root, args.output_grid, args.output_csv)
    print(f"overlap metric rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
