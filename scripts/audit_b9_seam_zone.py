"""Measure structural quality inside the fixed weighted-feather seam zone."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_b9_weighted_mosaic import _load_global_transforms, _load_sources, _project_scene
from src.multiscene_sift.band_geometry import apply_world_correction_to_transform
from src.multiscene_sift.mosaic_protocol import RUN_KEYS, grid_transform
from src.registration_benchmark.mosaic_diagnostics import (
    build_seam_zone_mask,
    compute_overlap_metrics,
    write_overlap_metrics_csv,
)


SEAM_BALANCE_THRESHOLD = 0.25


def run_seam_zone_audit(
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
            weight = distance_transform_edt(valid).astype(np.float32)
            weight[valid] += 1e-6
            projected.append((image.astype(np.float32), valid, weight))
        del arrays

        for i, j in itertools.combinations(range(len(projected)), 2):
            image_i, valid_i, weight_i = projected[i]
            image_j, valid_j, weight_j = projected[j]
            seam_zone = build_seam_zone_mask(
                valid_i,
                valid_j,
                weight_i,
                weight_j,
                balance_threshold=SEAM_BALANCE_THRESHOLD,
            )
            metrics = compute_overlap_metrics(
                image_i,
                image_j,
                seam_zone,
                min_valid_pixels=min_valid_pixels,
            )
            rows.append({
                "matcher": matcher,
                "global_method": "MST" if method == "mst" else "Translation-L2",
                "run": run_key,
                "pair": f"{i}-{j}",
                "scene_i": scene_ids[i],
                "scene_j": scene_ids[j],
                "seam_zone_rule": "pairwise normalized feather weights; min(w_i_norm,w_j_norm)>=0.25",
                "seam_balance_threshold": SEAM_BALANCE_THRESHOLD,
                "seam_zone_pixels": int(seam_zone.sum()),
                "gradient_magnitude_NCC": metrics["gradient_magnitude_NCC"],
                "gradient_orientation_cosine": metrics["gradient_orientation_cosine"],
                "aux_intensity_MAE": metrics["intensity_MAE"],
                "aux_intensity_RMSE": metrics["intensity_RMSE"],
                "aux_mean_bias": metrics["mean_bias"],
                "auxiliary_metric_label": "RADIOMETRY_SENSITIVE",
                "status": metrics["status"],
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
    rows = run_seam_zone_audit(args.source_config, args.global_root, args.output_grid, args.output_csv)
    print(f"seam metric rows={len(rows)} threshold={SEAM_BALANCE_THRESHOLD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
