"""Frozen output-grid and provenance helpers for the B9 mosaic stage."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
from rasterio.transform import Affine


RUN_KEYS = tuple(
    f"{matcher}/{method}"
    for matcher in ("sift", "loftr", "efficient_loftr", "lightglue_disk")
    for method in ("mst", "translation_l2")
)


def _matrix_to_affine(matrix: np.ndarray) -> Affine:
    return Affine(*np.asarray(matrix, dtype=float).reshape(3, 3).flat[:6])


def scene_affine(record: Mapping) -> Affine:
    values = record.get("transform") or [14.0, 0.0, record["left"], 0.0, -14.0, record["top"]]
    if len(values) != 6:
        raise ValueError("source inventory transform must contain six coefficients")
    return Affine(*map(float, values))


def transformed_scene_bounds(record: Mapping, global_matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Return the envelope of the four source corners after ``G * T``."""
    transform = scene_affine(record)
    corrected = _matrix_to_affine(np.asarray(global_matrix, dtype=float) @ np.array([
        [transform.a, transform.b, transform.c],
        [transform.d, transform.e, transform.f],
        [0.0, 0.0, 1.0],
    ]))
    height, width = map(int, record["shape"])
    points = [corrected @ (0, 0), corrected @ (width, 0), corrected @ (0, height), corrected @ (width, height)]
    xs, ys = zip(*points)
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def build_canonical_output_grid(
    source_inventory: list[Mapping],
    transform_sets: Mapping[str, Mapping[int, np.ndarray]],
    resolution: float,
    crs: str,
) -> dict:
    """Build one north-up grid containing all eight transformed footprints."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if set(transform_sets) != set(RUN_KEYS):
        raise ValueError("canonical grid requires exactly the eight frozen Global runs")
    raw_bounds = []
    transformed_bounds: dict[str, list[dict]] = {}
    for record in source_inventory:
        raw_bounds.append(transformed_scene_bounds(record, np.eye(3)))
    for run_key in RUN_KEYS:
        matrices = transform_sets[run_key]
        if set(matrices) != set(range(len(source_inventory))):
            raise ValueError(f"{run_key}: transform scene indices do not match source inventory")
        transformed_bounds[run_key] = []
        for scene_index, record in enumerate(source_inventory):
            bounds = transformed_scene_bounds(record, matrices[scene_index])
            transformed_bounds[run_key].append({"scene_index": scene_index, "bounds": list(bounds)})
    all_bounds = [item["bounds"] for values in transformed_bounds.values() for item in values]
    source_union = [min(b[0] for b in raw_bounds), min(b[1] for b in raw_bounds), max(b[2] for b in raw_bounds), max(b[3] for b in raw_bounds)]
    transformed_union = [min(b[0] for b in all_bounds), min(b[1] for b in all_bounds), max(b[2] for b in all_bounds), max(b[3] for b in all_bounds)]
    left = math.floor((transformed_union[0] + 1e-9) / resolution) * resolution
    bottom = math.floor((transformed_union[1] + 1e-9) / resolution) * resolution
    right = math.ceil((transformed_union[2] - 1e-9) / resolution) * resolution
    top = math.ceil((transformed_union[3] - 1e-9) / resolution) * resolution
    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError("canonical output grid has non-positive dimensions")
    transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)
    return {
        "schema_version": 1,
        "crs": str(crs), "resolution": float(resolution),
        "pixel_size": float(resolution), "pixel_size_m": float(resolution),
        "origin": [float(left), float(top)], "width": width, "height": height,
        "transform": [float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)],
        "bounds": [float(left), float(bottom), float(right), float(top)],
        "source_union_bounds": source_union,
        "transformed_union_bounds": transformed_union,
        "run_scene_bounds": transformed_bounds,
        "grid_identity": {"crs": str(crs), "resolution": float(resolution),
                           "pixel_size": float(resolution),
                           "origin": [float(left), float(top)], "width": width,
                           "height": height,
                           "bounds": [float(left), float(bottom), float(right), float(top)]},
    }


def write_canonical_grid(grid: Mapping, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(grid, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def grid_transform(grid: Mapping) -> Affine:
    return Affine(*map(float, grid["transform"]))


def same_grid(left: Mapping, right: Mapping, atol: float = 1e-9) -> bool:
    return left["crs"] == right["crs"] and left["width"] == right["width"] and left["height"] == right["height"] and np.allclose(left["transform"], right["transform"], atol=atol) and np.allclose(left["bounds"], right["bounds"], atol=atol)
