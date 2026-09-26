"""Run one deterministic weighted-feather mosaic from persisted Global transforms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mosaic import create_mosaic
from src.multiscene_sift.band_geometry import apply_world_correction_to_transform
from src.multiscene_sift.mosaic_protocol import grid_transform


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _scene_path(config_path: Path, record: Mapping) -> Path:
    value = record.get("b9_path") or record.get("path")
    if not value:
        raise ValueError("source scene record must contain b9_path or path")
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path)


def _load_sources(config_path: Path) -> tuple[list[np.ndarray], list, list, list[str]]:
    config = _read_json(config_path)
    scenes = config.get("scenes")
    if not scenes:
        raise ValueError("source config has no scenes")

    arrays: list[np.ndarray] = []
    transforms = []
    nodata_values: list = []
    scene_ids: list[str] = []
    source_crs = None
    for index, record in enumerate(scenes):
        path = _scene_path(config_path, record)
        with rasterio.open(path) as src:
            if src.count < 1:
                raise ValueError(f"scene {index} has no raster bands: {path}")
            crs = src.crs.to_string() if src.crs else None
            if source_crs is None:
                source_crs = crs
            elif crs != source_crs:
                raise ValueError(f"source CRS mismatch at scene {index}: {crs} != {source_crs}")
            arrays.append(src.read(1))
            transforms.append(src.transform)
            nodata_values.append(src.nodata if src.nodata is not None else record.get("nodata"))
            scene_ids.append(str(record.get("scene_id", path.stem)))
    return arrays, transforms, nodata_values, scene_ids


def _load_global_transforms(global_run_dir: Path, scene_count: int) -> dict[int, np.ndarray]:
    path = global_run_dir / "global_transforms.json"
    payload = _read_json(path)
    transforms = {
        int(item["scene"]): np.asarray(item["matrix"], dtype=np.float64)
        for item in payload.get("transforms", [])
    }
    expected = set(range(scene_count))
    if set(transforms) != expected:
        raise ValueError(f"global transform scene indices {sorted(transforms)} != {sorted(expected)}")
    for scene, matrix in transforms.items():
        if matrix.shape != (3, 3):
            raise ValueError(f"global transform {scene} is not 3x3: {matrix.shape}")
    return transforms


def _project_scene(
    array: np.ndarray,
    transform,
    nodata,
    crs: str,
    destination_transform,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    projected = np.full((height, width), np.nan, dtype=np.float64)
    reproject(
        source=array.astype(np.float64, copy=False),
        destination=projected,
        src_transform=transform,
        src_crs=crs,
        dst_transform=destination_transform,
        dst_crs=crs,
        src_nodata=nodata,
        dst_nodata=np.nan,
        init_dest_nodata=True,
        resampling=Resampling.bilinear,
    )
    return projected, np.isfinite(projected)


def _write_single_band(path: Path, data: np.ndarray, transform, crs: str, nodata, dtype: str) -> None:
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype, copy=False), 1)


def _write_preview(path: Path, mosaic_path: Path, valid_mask: np.ndarray) -> dict:
    with rasterio.open(mosaic_path) as src:
        image = src.read(1).astype(np.float32)
    values = image[valid_mask]
    if values.size:
        p2, p98 = np.percentile(values, [2.0, 98.0])
        if p98 <= p2:
            p2, p98 = float(values.min()), float(values.max())
    else:
        p2, p98 = 0.0, 1.0
    scale = max(float(p98 - p2), 1e-12)
    stretched = np.clip((image - p2) / scale * 255.0, 0.0, 255.0).astype(np.uint8)
    stretched[~valid_mask] = 0
    Image.fromarray(stretched, mode="L").save(path)
    return {"method": "percentile", "lower": float(p2), "upper": float(p98)}


def run_weighted_mosaic(
    source_config: str | Path,
    global_run_dir: str | Path,
    output_grid: str | Path,
    output_dir: str | Path,
    *,
    run_name: str | None = None,
) -> dict:
    """Warp, blend, and audit one persisted Global solution; never infer a matcher."""
    source_config = Path(source_config)
    global_run_dir = Path(global_run_dir)
    output_grid = Path(output_grid)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = _read_json(output_grid)
    width, height = int(grid["width"]), int(grid["height"])
    resolution = float(grid.get("resolution", grid["pixel_size"]))
    crs = str(grid["crs"])
    destination_transform = grid_transform(grid)
    arrays, source_transforms, nodata_values, scene_ids = _load_sources(source_config)
    global_transforms = _load_global_transforms(global_run_dir, len(arrays))
    corrected_transforms = [
        apply_world_correction_to_transform(source_transforms[index], global_transforms[index])
        for index in range(len(arrays))
    ]

    mosaic_path = output_dir / "mosaic.tif"
    create_mosaic(
        arrays=[array[np.newaxis, :, :] for array in arrays],
        transforms=corrected_transforms,
        crs=crs,
        nodata_values=nodata_values,
        output_path=str(mosaic_path),
        resolution=resolution,
        mode="weighted",
        output_transform=destination_transform,
        output_width=width,
        output_height=height,
    )

    valid_mask = np.zeros((height, width), dtype=bool)
    contributor_count = np.zeros((height, width), dtype=np.uint8)
    weight_sum = np.zeros((height, width), dtype=np.float64)
    for array, transform, nodata in zip(arrays, corrected_transforms, nodata_values):
        _, mask = _project_scene(
            array, transform, nodata, crs, destination_transform, width, height
        )
        valid_mask |= mask
        contributor_count += mask.astype(np.uint8)
        weight = distance_transform_edt(mask).astype(np.float64)
        weight[mask] += 1e-6
        weight_sum += weight

    _write_single_band(output_dir / "valid_mask.tif", valid_mask.astype(np.uint8), destination_transform, crs, 0, "uint8")
    _write_single_band(output_dir / "contributor_count.tif", contributor_count, destination_transform, crs, 0, "uint8")
    _write_single_band(output_dir / "weight_sum.tif", weight_sum.astype(np.float32), destination_transform, crs, 0.0, "float32")
    preview_stretch = _write_preview(output_dir / "preview.png", mosaic_path, valid_mask)

    with rasterio.open(mosaic_path) as mosaic:
        mosaic_valid = mosaic.read(1) != mosaic.nodata
        mosaic_profile = {
            "width": mosaic.width,
            "height": mosaic.height,
            "transform": list(mosaic.transform)[:6],
            "crs": mosaic.crs.to_string() if mosaic.crs else None,
            "dtype": mosaic.dtypes[0],
            "nodata": mosaic.nodata,
        }
    summary = {
        "schema_version": 1,
        "run_name": run_name,
        "blend_mode": "weighted",
        "radiometric_normalization": "NONE",
        "source_config": str(source_config),
        "global_run_dir": str(global_run_dir),
        "scene_ids": scene_ids,
        "output_grid": grid.get("grid_identity", grid),
        "mosaic_profile": mosaic_profile,
        "valid_pixels": int(valid_mask.sum()),
        "mosaic_valid_pixels": int(mosaic_valid.sum()),
        "contributor_count": {
            "max": int(contributor_count.max(initial=0)),
            "mean_valid": float(contributor_count[valid_mask].mean()) if valid_mask.any() else 0.0,
            "one": int((contributor_count == 1).sum()),
            "two": int((contributor_count == 2).sum()),
            "three_or_more": int((contributor_count >= 3).sum()),
        },
        "weight_sum_invalid_pixels": int((weight_sum <= 0.0).sum()),
        "preview_stretch": preview_stretch,
        "outputs": {name: name for name in ("mosaic.tif", "valid_mask.tif", "contributor_count.tif", "weight_sum.tif", "preview.png")},
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"output_dir": output_dir, "mosaic_path": mosaic_path, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--global-run-dir", required=True, type=Path)
    parser.add_argument("--output-grid", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)
    run_weighted_mosaic(
        args.source_config,
        args.global_run_dir,
        args.output_grid,
        args.output_dir,
        run_name=args.run_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
