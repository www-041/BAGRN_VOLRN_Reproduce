"""Generate standardized visual diagnostics for the eight B9 mosaics."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_b9_weighted_mosaic import _load_global_transforms, _load_sources, _project_scene
from src.multiscene_sift.band_geometry import apply_world_correction_to_transform
from src.multiscene_sift.mosaic_protocol import RUN_KEYS, grid_transform
from src.registration_benchmark.mosaic_diagnostics import build_seam_zone_mask


def make_checkerboard(first: np.ndarray, second: np.ndarray, *, tiles: int = 8) -> np.ndarray:
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("checkerboard images must be same-shaped 2-D arrays")
    if tiles <= 0:
        raise ValueError("tiles must be positive")
    height, width = first.shape
    rows = np.arange(height)[:, None] // max(1, height // tiles)
    cols = np.arange(width)[None, :] // max(1, width // tiles)
    choose_first = (rows + cols) % 2 == 0
    return np.where(choose_first, first, second)


def gradient_overlay(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.nan_to_num(np.asarray(first, dtype=np.float64), nan=0.0)
    second = np.nan_to_num(np.asarray(second, dtype=np.float64), nan=0.0)
    grad_first = np.hypot(*np.gradient(first))
    grad_second = np.hypot(*np.gradient(second))
    scale = max(float(np.percentile(np.concatenate([grad_first.ravel(), grad_second.ravel()]), 98)), 1e-12)
    red = np.clip(grad_first / scale * 255.0, 0, 255)
    green = np.clip(grad_second / scale * 255.0, 0, 255)
    blue = np.clip(np.abs(grad_first - grad_second) / scale * 255.0, 0, 255)
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def _stretch(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = image[mask & np.isfinite(image)]
    low, high = (np.percentile(values, [2, 98]) if values.size else (0.0, 1.0))
    scale = max(float(high - low), 1e-12)
    normalized = np.zeros(image.shape, dtype=np.float64)
    np.divide(image - low, scale, out=normalized, where=np.isfinite(image))
    result = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    result[~mask] = 0
    return result


def _crop_bounds(mask: np.ndarray, padding: int = 8) -> list[int] | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    r0 = max(0, int(rows.min()) - padding)
    r1 = min(mask.shape[0], int(rows.max()) + padding + 1)
    c0 = max(0, int(cols.min()) - padding)
    c1 = min(mask.shape[1], int(cols.max()) + padding + 1)
    return [r0, r1, c0, c1]


def _resize_gray(image: np.ndarray, max_side: int = 1200) -> np.ndarray:
    height, width = image.shape
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image.astype(np.uint8)
    return np.asarray(Image.fromarray(image).resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR))


def _resize_rgb(image: np.ndarray, max_side: int = 1200) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image.astype(np.uint8)
    return np.asarray(Image.fromarray(image).resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR))


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _select_regions(geometry_rows: list[dict], seam_rows: list[dict]) -> dict:
    selection = {"edge_regions": {}, "seam_regions": {}}
    for matcher in ("sift", "loftr", "efficient_loftr", "lightglue_disk"):
        edges = [row for row in geometry_rows if row["matcher"] == matcher]
        worst = max(edges, key=lambda row: float(row["p95_pixel"]))
        stable = min(edges, key=lambda row: float(row["p95_pixel"]))
        pairs = {}
        for row in edges:
            pairs.setdefault(row["pair"], []).append(float(row["p95_pixel"]))
        edge_pairs = [worst["pair"], stable["pair"]]
        if matcher == "lightglue_disk":
            edge_pairs.append("1-3")
        selection["edge_regions"][matcher] = {
            "worst_pair": worst["pair"],
            "worst_p95_pixel": float(worst["p95_pixel"]),
            "stable_pair": stable["pair"],
            "stable_p95_pixel": float(stable["p95_pixel"]),
            "pairs_for_figures": sorted(set(edge_pairs)),
        }
        seam_by_pair = {}
        for row in seam_rows:
            if row["matcher"] == matcher:
                seam_by_pair.setdefault(row["pair"], []).append(float(row["gradient_magnitude_NCC"]))
        ranked = sorted((sum(values) / len(values), pair) for pair, values in seam_by_pair.items())
        selection["seam_regions"][matcher] = {
            "highest_mismatch_pair": ranked[0][1],
            "highest_mismatch_gradient_magnitude_NCC": ranked[0][0],
            "median_pair": ranked[len(ranked) // 2][1],
            "criterion": "lowest mean seam gradient_magnitude_NCC across MST and Translation-L2",
        }
    return selection


def _load_projected_run(source_config: Path, global_run: Path, grid: dict):
    arrays, source_transforms, nodata_values, scene_ids = _load_sources(source_config)
    transforms = _load_global_transforms(global_run, len(arrays))
    corrected = [
        apply_world_correction_to_transform(source_transforms[i], transforms[i])
        for i in range(len(arrays))
    ]
    destination = grid_transform(grid)
    projected = []
    for array, transform, nodata in zip(arrays, corrected, nodata_values):
        image, mask = _project_scene(
            array, transform, nodata, str(grid["crs"]), destination, int(grid["width"]), int(grid["height"])
        )
        weight = distance_transform_edt(mask).astype(np.float32)
        weight[mask] += 1e-6
        projected.append((image.astype(np.float32), mask, weight))
    return projected, scene_ids


def _save_pair_figures(figures_dir: Path, matcher: str, method: str, pair: str, projected) -> dict:
    i, j = (int(value) for value in pair.split("-"))
    first, valid_i, _ = projected[i]
    second, valid_j, _ = projected[j]
    overlap = valid_i & valid_j
    bounds = _crop_bounds(overlap)
    if bounds is None:
        return {"pair": pair, "status": "NO_OVERLAP"}
    r0, r1, c0, c1 = bounds
    first_crop, second_crop = first[r0:r1, c0:c1], second[r0:r1, c0:c1]
    mask_crop = overlap[r0:r1, c0:c1]
    first_display = _stretch(first_crop, mask_crop)
    second_display = _stretch(second_crop, mask_crop)
    checker = _resize_gray(make_checkerboard(first_display, second_display), 1200)
    checker_path = figures_dir / f"checkerboard_{matcher}_{method}_{pair}.png"
    Image.fromarray(checker, mode="L").save(checker_path)
    gradient_path = figures_dir / f"gradient_{matcher}_{method}_{pair}.png"
    Image.fromarray(_resize_rgb(gradient_overlay(first_crop, second_crop), 1200), mode="RGB").save(gradient_path)
    return {"pair": pair, "bounds": bounds, "checkerboard": str(checker_path), "gradient_overlay": str(gradient_path)}


def render_visual_audit(
    source_config: str | Path,
    global_root: str | Path,
    mosaic_root: str | Path,
    audit_dir: str | Path,
    geometry_csv: str | Path,
    seam_csv: str | Path,
) -> dict:
    source_config, global_root, mosaic_root, audit_dir = map(Path, (source_config, global_root, mosaic_root, audit_dir))
    figures_dir = audit_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    grid = json.loads((mosaic_root / "protocol" / "canonical_output_grid.json").read_text(encoding="utf-8"))
    selection = _select_regions(_read_csv(Path(geometry_csv)), _read_csv(Path(seam_csv)))
    rendered = {"official_mosaics": [], "contributor_maps": [], "pair_figures": [], "seam_zooms": [], "mosaic_differences": []}
    seam_crop_bounds = {}

    for run_key in RUN_KEYS:
        matcher, method = run_key.split("/")
        run_dir = mosaic_root / matcher / method
        mosaic_preview = figures_dir / f"mosaic_{matcher}_{method}.png"
        contributor = figures_dir / f"contributor_{matcher}_{method}.png"
        shutil.copy2(run_dir / "preview.png", mosaic_preview)
        shutil.copy2(audit_dir / "contributor_maps" / f"{matcher}_{method}.png", contributor)
        rendered["official_mosaics"].append(str(mosaic_preview))
        rendered["contributor_maps"].append(str(contributor))

        projected, _scene_ids = _load_projected_run(source_config, global_root / matcher / method, grid)
        edge_pairs = selection["edge_regions"][matcher]["pairs_for_figures"]
        for pair in edge_pairs:
            rendered["pair_figures"].append(_save_pair_figures(figures_dir, matcher, method, pair, projected))

        for region_name, pair in (
            ("highest_mismatch", selection["seam_regions"][matcher]["highest_mismatch_pair"]),
            ("median", selection["seam_regions"][matcher]["median_pair"]),
        ):
            if matcher not in seam_crop_bounds or region_name not in seam_crop_bounds[matcher]:
                i, j = (int(value) for value in pair.split("-"))
                _, valid_i, weight_i = projected[i]
                _, valid_j, weight_j = projected[j]
                seam = build_seam_zone_mask(valid_i, valid_j, weight_i, weight_j, balance_threshold=0.25)
                seam_crop_bounds.setdefault(matcher, {})[region_name] = _crop_bounds(seam)
            bounds = seam_crop_bounds[matcher][region_name]
            if bounds:
                i, j = (int(value) for value in pair.split("-"))
                first, _, _ = projected[i]
                second, _, _ = projected[j]
                r0, r1, c0, c1 = bounds
                seam = _stretch(first[r0:r1, c0:c1], np.ones((r1 - r0, c1 - c0), dtype=bool))
                seam_target = _stretch(second[r0:r1, c0:c1], np.ones((r1 - r0, c1 - c0), dtype=bool))
                path = figures_dir / f"seam_zoom_{matcher}_{method}_{region_name}_{pair}.png"
                Image.fromarray(_resize_gray(make_checkerboard(seam, seam_target, tiles=4), 1000), mode="L").save(path)
                rendered["seam_zooms"].append(str(path))
        del projected

    for matcher in ("sift", "loftr", "efficient_loftr", "lightglue_disk"):
        mst = mosaic_root / matcher / "mst" / "mosaic.tif"
        translation = mosaic_root / matcher / "translation_l2" / "mosaic.tif"
        with rasterio.open(mst) as first, rasterio.open(translation) as second:
            first_data, second_data = first.read(1).astype(np.float32), second.read(1).astype(np.float32)
            first_valid = first_data != first.nodata
            second_valid = second_data != second.nodata
        common = first_valid & second_valid
        delta = np.abs(first_data - second_data)
        display = _stretch(delta, common)
        delta_path = figures_dir / f"mosaic_delta_{matcher}_mst_vs_translation_l2.png"
        Image.fromarray(_resize_gray(display, 1200), mode="L").save(delta_path)
        rendered["mosaic_differences"].append(str(delta_path))

    metadata = {
        "schema_version": 1,
        "canonical_grid": grid.get("grid_identity", grid),
        "preview_policy": "copy of runner percentile-stretched preview.png",
        "contributor_map_policy": "same extent and shared display maximum from Task 5",
        "checkerboard_source": "separately warped source scenes, not blended mosaic",
        "seam_zone_rule": "pairwise normalized feather weights; min(w_i_norm,w_j_norm)>=0.25",
        "mosaic_difference_note": "absolute difference is radiometry/warp-interpolation sensitive and is not geometric ground truth",
        "selection": selection,
        "seam_crop_bounds": seam_crop_bounds,
        "rendered": rendered,
    }
    path = figures_dir / "07_figure_selection.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--global-root", required=True, type=Path)
    parser.add_argument("--mosaic-root", required=True, type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--geometry-csv", required=True, type=Path)
    parser.add_argument("--seam-csv", required=True, type=Path)
    args = parser.parse_args(argv)
    metadata = render_visual_audit(
        args.source_config,
        args.global_root,
        args.mosaic_root,
        args.audit_dir,
        args.geometry_csv,
        args.seam_csv,
    )
    print(f"visual diagnostics generated: {len(metadata['rendered']['official_mosaics'])} mosaics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
