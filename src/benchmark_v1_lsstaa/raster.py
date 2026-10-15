"""Read-only DZ01V four-band adapter for 10x LSSTAA estimation."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import json
import re

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

from src.benchmark_v1_unified.aligned import corrected_affine


def masked_block_mean(values: np.ndarray, valid: np.ndarray, factor: int = 10) -> np.ndarray:
    if values.shape != valid.shape or values.ndim != 2 or factor < 1:
        raise ValueError("INVALID_REDUCE_INPUT")
    h, w = values.shape
    hp, wp = ((h+factor-1)//factor)*factor, ((w+factor-1)//factor)*factor
    clean = np.zeros((hp, wp), dtype=np.float64)
    weight = np.zeros((hp, wp), dtype=np.float64)
    good = valid & np.isfinite(values)
    clean[:h, :w][good] = values[good]
    weight[:h, :w][good] = 1
    sums = clean.reshape(hp//factor, factor, wp//factor, factor).sum(axis=(1, 3))
    counts = weight.reshape(hp//factor, factor, wp//factor, factor).sum(axis=(1, 3))
    out = np.zeros_like(sums, dtype=np.float32)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def verify_colocation(profiles: dict[str, dict]) -> None:
    if set(profiles) != {"B9", "B4", "B7", "B14"}:
        raise ValueError("MISSING_REQUIRED_BAND")
    ref = profiles["B9"]
    for band, item in profiles.items():
        if any(item[k] != ref[k] for k in ("crs", "transform", "width", "height")):
            raise ValueError(f"BAND_GRID_MISMATCH:{band}")


def proxy_ndvi(red: np.ndarray, nir: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if red.shape != nir.shape or red.shape != valid.shape:
        raise ValueError("INVALID_PROXY_SHAPE")
    # Demo_LSSTAA_v1 reads its stack as single before the ratio calculation.
    a = np.asarray(red, dtype=np.float32)
    b = np.asarray(nir, dtype=np.float32)
    denom = a+b
    good = valid & np.isfinite(a) & np.isfinite(b) & (denom != 0)
    out = np.zeros(a.shape, dtype=np.float32)
    out[good] = (b[good]-a[good])/denom[good]*np.float32(10000)
    return out


def source_scene_paths(source_root: Path, scene_id: str) -> dict[str, Path]:
    stamp = re.search(r"_(\d{14})_", scene_id)
    if stamp is None:
        raise ValueError("INVALID_SCENE_ID")
    folder = source_root / stamp.group(1) / scene_id
    return {band: folder / f"{scene_id}_{band}.TIF" for band in ("B9", "B4", "B7", "B14")}


def verified_scene_date(mtl_path: Path) -> str:
    match = re.search(r"START_IMAGING_TIME\s*=\s*(\d{4}-\d{2}-\d{2})T", mtl_path.read_text(encoding="utf-8", errors="replace"))
    if match is None:
        raise ValueError(f"MISSING_ACQUISITION_DATE:{mtl_path}")
    date.fromisoformat(match.group(1))
    return match.group(1)


def _coarse_frozen_b9(data_path: Path, mask_path: Path, factor: int = 10) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(data_path) as data, rasterio.open(mask_path) as mask:
        if (data.width, data.height, data.transform, data.crs) != (mask.width, mask.height, mask.transform, mask.crs):
            raise ValueError("FROZEN_B9_MASK_GRID_MISMATCH")
        width = (data.width+factor-1)//factor
        output = np.zeros(((data.height+factor-1)//factor, width), dtype=np.float32)
        fraction = np.zeros_like(output)
        for row in range(0, data.height, factor):
            h = min(factor, data.height-row)
            win = Window(0, row, data.width, h)
            values = data.read(1, window=win)
            valid = mask.read(1, window=win) != 0
            output[row//factor, :] = masked_block_mean(values, valid, factor)[0]
            fraction[row//factor, :] = masked_block_mean(valid.astype(np.float32), np.ones_like(valid, dtype=bool), factor)[0]
    return output, fraction


def _coarse_source_band(path: Path, world_matrix: np.ndarray, coarse_shape: tuple[int, int],
                        coarse_transform: Affine) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros(coarse_shape, dtype=np.float32)
    fraction = np.zeros(coarse_shape, dtype=np.float32)
    with rasterio.open(path) as src:
        corrected = corrected_affine(src.transform, world_matrix)
        reproject(source=rasterio.band(src, 1), destination=out,
                  src_transform=corrected, src_crs=src.crs,
                  dst_transform=coarse_transform, dst_crs=src.crs,
                  src_nodata=src.nodata if src.nodata is not None else 0,
                  dst_nodata=0, resampling=Resampling.average)
        source_valid = (src.read(1) != (src.nodata if src.nodata is not None else 0)).astype(np.float32)
        reproject(source=source_valid, destination=fraction,
                  src_transform=corrected, src_crs=src.crs,
                  dst_transform=coarse_transform, dst_crs=src.crs,
                  src_nodata=None, dst_nodata=0, resampling=Resampling.average)
    return out, fraction


def load_three_scene_estimation(freeze_root: Path, source_root: Path,
                                task15_root: Path) -> tuple[np.ndarray, list[int], dict]:
    frozen = json.loads((freeze_root / "frozen_manifest.json").read_text(encoding="utf-8"))
    source_cfg = json.loads((task15_root / "stages/04_canonical_warp/task14_source_config.json").read_text(encoding="utf-8"))
    transforms = json.loads((task15_root / "stages/03_global_adjustment/global/translation_l2/global_transforms.json").read_text(encoding="utf-8"))
    grid = frozen["grid"]
    factor = 10
    shape = ((grid["height"]+factor-1)//factor, (grid["width"]+factor-1)//factor)
    coarse_transform = Affine(*grid["transform"]) * Affine.scale(factor)
    stack = np.zeros((*shape, 3, 4), dtype=np.float32)
    dates, details = [], []
    for index in (0, 1, 2):
        scene = source_cfg["scenes"][index]
        scene_id = scene["scene_id"]
        paths = source_scene_paths(source_root, scene_id)
        if any(not p.is_file() for p in paths.values()):
            raise FileNotFoundError(f"MISSING_SOURCE_BAND:{scene_id}")
        profiles = {}
        for band, path in paths.items():
            with rasterio.open(path) as ds:
                profiles[band] = {"crs": ds.crs, "transform": ds.transform,
                                  "width": ds.width, "height": ds.height}
                if ds.dtypes[0] != "uint16" or ds.nodata != 0:
                    raise ValueError(f"UNEXPECTED_SOURCE_TYPE_OR_NODATA:{path}")
        verify_colocation(profiles)
        mtl = next(iter(paths.values())).parent / f"{scene_id}_MTL.txt"
        dates.append(verified_scene_date(mtl))
        aligned_data = freeze_root / "aligned_inputs" / f"scene_{index:03d}_B9_input.tif"
        aligned_mask = freeze_root / "aligned_inputs" / f"scene_{index:03d}_input_mask.tif"
        b9, b9_fraction = _coarse_frozen_b9(aligned_data, aligned_mask)
        stack[:, :, index, 0] = b9
        fractions = [b9_fraction]
        world = np.asarray(transforms["transforms"][index]["matrix"], dtype=np.float64)
        for band_position, band in enumerate(("B4", "B7", "B14"), start=1):
            values, fraction = _coarse_source_band(paths[band], world, shape, coarse_transform)
            stack[:, :, index, band_position] = values
            fractions.append(fraction)
        valid = np.logical_and.reduce([f > 0 for f in fractions])
        stack[:, :, index, :][~valid] = 0
        details.append({"scene_index": index, "scene_id": scene_id, "date": dates[-1],
                        "coarse_all_band_valid_pixels": int(valid.sum()),
                        "coarse_band_valid_pixels": {b: int((f > 0).sum()) for b, f in zip(("B9", "B4", "B7", "B14"), fractions)},
                        "source_paths": {b: str(p) for b, p in paths.items()}})
    from .core import absolute_day_offsets
    return stack, absolute_day_offsets(dates), {"grid": grid, "factor": factor,
                                              "unit": "UNKNOWN", "scenes": details,
                                              "band_order": ["B9", "B4", "B7", "B14"]}
