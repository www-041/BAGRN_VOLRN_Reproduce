"""Analysis-only diagnostics for persisted weighted-feather mosaics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from PIL import Image

from src.multiscene_sift.mosaic_protocol import RUN_KEYS


def compute_contributor_metrics(
    valid_mask: np.ndarray,
    contributor_count: np.ndarray,
    weight_sum: np.ndarray,
) -> dict:
    valid_mask = np.asarray(valid_mask, dtype=bool)
    contributor_count = np.asarray(contributor_count)
    weight_sum = np.asarray(weight_sum)
    if valid_mask.shape != contributor_count.shape or valid_mask.shape != weight_sum.shape:
        raise ValueError("coverage diagnostic arrays must share one shape")
    valid_counts = contributor_count[valid_mask]
    return {
        "valid_mosaic_pixels": int(valid_mask.sum()),
        "nodata_pixels": int((~valid_mask).sum()),
        "one_contributor_pixels": int((valid_mask & (contributor_count == 1)).sum()),
        "two_contributor_pixels": int((valid_mask & (contributor_count == 2)).sum()),
        "three_plus_contributor_pixels": int((valid_mask & (contributor_count >= 3)).sum()),
        "max_contributor_count": int(valid_counts.max()) if valid_counts.size else 0,
        "mean_contributor_count_valid": float(valid_counts.mean()) if valid_counts.size else 0.0,
        "invalid_weight_sum_pixels": int((valid_mask & (weight_sum <= 0.0)).sum()),
        "total_pixels": int(valid_mask.size),
    }


def write_contributor_png(
    path: str | Path,
    contributor_count: np.ndarray,
    valid_mask: np.ndarray,
    *,
    display_max_count: int,
) -> Path:
    if display_max_count <= 0:
        raise ValueError("display_max_count must be positive")
    contributor_count = np.asarray(contributor_count)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    scaled = np.zeros(contributor_count.shape, dtype=np.uint8)
    scaled[valid_mask] = np.clip(
        np.rint(contributor_count[valid_mask] / display_max_count * 255.0), 0, 255
    ).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(scaled, mode="L").save(path)
    return path


def _load_run_arrays(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    paths = [run_dir / name for name in ("valid_mask.tif", "contributor_count.tif", "weight_sum.tif")]
    with rasterio.open(paths[0]) as valid, rasterio.open(paths[1]) as contributors, rasterio.open(paths[2]) as weights:
        identity = {
            "width": valid.width,
            "height": valid.height,
            "crs": valid.crs.to_string() if valid.crs else None,
            "transform": list(valid.transform)[:6],
        }
        return valid.read(1).astype(bool), contributors.read(1), weights.read(1), identity


def run_coverage_audit(mosaic_root: str | Path, audit_dir: str | Path) -> list[dict]:
    mosaic_root = Path(mosaic_root)
    audit_dir = Path(audit_dir)
    rows: list[dict] = []
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict]] = {}
    common_identity = None
    for run_key in RUN_KEYS:
        matcher, method = run_key.split("/")
        run_dir = mosaic_root / matcher / method
        arrays[run_key] = _load_run_arrays(run_dir)
        identity = arrays[run_key][3]
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise ValueError(f"run {run_key} does not share the canonical output grid")
        metrics = compute_contributor_metrics(*arrays[run_key][:3])
        rows.append({"run": run_key, "matcher": matcher, "global_method": method, **metrics})

    display_max_count = max(row["max_contributor_count"] for row in rows)
    map_dir = audit_dir / "contributor_maps"
    for run_key, (valid, contributors, _weights, _identity) in arrays.items():
        matcher, method = run_key.split("/")
        write_contributor_png(
            map_dir / f"{matcher}_{method}.png",
            contributors,
            valid,
            display_max_count=display_max_count,
        )
    audit_dir.mkdir(parents=True, exist_ok=True)
    csv_path = audit_dir / "01_run_inventory.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "grid_identity": common_identity,
        "display_max_contributor_count": display_max_count,
        "total_pixels_by_run": sorted({row["total_pixels"] for row in rows}),
        "runs": len(rows),
        "csv": str(csv_path),
    }
    (audit_dir / "01_coverage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return rows
