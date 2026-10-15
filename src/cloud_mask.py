"""Conservative cloud-mask generation for DZ01V VNIR scenes.

The mask is used only to exclude cloud pixels from radiometric parameter
estimation and quality metrics. It is not a cloud-removal/reconstruction
algorithm, and cloud pixels remain in the final normalized imagery.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re
from typing import Mapping, Optional

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation, label

logger = logging.getLogger(__name__)

CLOUD_BANDS = ("B2", "B5", "B7", "B14")
DEFAULT_CLOUD_MASK_PARAMS = {
    "visible_percentile": 97.0,
    "nir_percentile": 90.0,
    "max_whiteness": 0.35,
    "max_ndvi": 0.35,
    "min_component_size": 9,
    "dilation_iterations": 1,
}


@dataclass
class CloudMaskRecord:
    """Source-grid cloud mask and its geometry."""

    mask: np.ndarray
    transform: object
    crs: object
    cloud_fraction: float
    metadata_cloud_cover: Optional[float]


def parse_dz01_mtl(path: str | Path) -> dict:
    """Parse radiometric calibration and cloud-cover fields from DZ01 MTL."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    mult: dict[str, float] = {}
    add: dict[str, float] = {}

    for m in re.finditer(r"RADIANCE_MULT_BAND_(\d+)\s*=\s*([+\-0-9.Ee]+)", text):
        mult[f"B{int(m.group(1))}"] = float(m.group(2))
    for m in re.finditer(r"RADIANCE_ADD_BAND_(\d+)\s*=\s*([+\-0-9.Ee]+)", text):
        add[f"B{int(m.group(1))}"] = float(m.group(2))

    cloud_cover = None
    m = re.search(r"\bCLOUD_COVER\s*=\s*([+\-0-9.Ee]+)", text)
    if m:
        cloud_cover = float(m.group(1))

    return {
        "radiance_mult": mult,
        "radiance_add": add,
        "cloud_cover": cloud_cover,
    }


def _robust_scale(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vals = data[valid]
    if vals.size == 0:
        return np.zeros_like(data, dtype=np.float64)
    lo = float(np.percentile(vals, 2.0))
    hi = float(np.percentile(vals, 98.0))
    if hi - lo < 1e-12:
        hi = lo + 1.0
    scaled = (data.astype(np.float32, copy=False) - np.float32(lo)) / np.float32(hi - lo)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1 or not mask.any():
        return mask.astype(bool, copy=True)
    labels, n = label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    keep = counts >= int(min_size)
    keep[0] = False
    return keep[labels]


def detect_cloud_mask(
    bands: Mapping[str, np.ndarray],
    radiance_mult: Mapping[str, float],
    radiance_add: Optional[Mapping[str, float]] = None,
    nodata_values: Optional[Mapping[str, Optional[float]]] = None,
    *,
    visible_percentile: float = 97.0,
    nir_percentile: float = 90.0,
    max_whiteness: float = 0.35,
    max_ndvi: float = 0.35,
    min_component_size: int = 9,
    dilation_iterations: int = 1,
) -> np.ndarray:
    """Detect conservative cloud candidates from B2/B5/B7/B14.

    The detector is intentionally conservative and scene-adaptive. DN values are
    first converted to radiance-like units using MTL calibration, then each band
    is robustly scaled within the scene. Cloud candidates must be bright in the
    visible and NIR, spectrally neutral in the visible, and not strongly
    vegetation-like.
    """
    radiance_add = radiance_add or {}
    nodata_values = nodata_values or {}

    missing = [b for b in CLOUD_BANDS if b not in bands]
    if missing:
        raise ValueError(f"Missing cloud-detection bands: {missing}")

    shape = np.asarray(bands[CLOUD_BANDS[0]]).shape
    if len(shape) != 2:
        raise ValueError("Cloud-detection bands must be 2-D")

    valid = np.ones(shape, dtype=bool)
    for band in CLOUD_BANDS:
        arr = np.asarray(bands[band])
        if arr.shape != shape:
            raise ValueError(f"Cloud band {band} shape {arr.shape} != {shape}")
        if band not in radiance_mult:
            raise ValueError(f"Missing radiance multiplier for {band}")
        nd = nodata_values.get(band)
        band_valid = np.isfinite(arr)
        if nd is not None:
            band_valid &= arr != nd
        valid &= band_valid

    if not valid.any():
        return np.zeros(shape, dtype=bool)

    # Scale one band at a time so four additional full-size radiance arrays are
    # not kept in memory simultaneously.
    scaled: dict[str, np.ndarray] = {}
    for band in CLOUD_BANDS:
        arr = np.asarray(bands[band])
        rad = (
            arr.astype(np.float32, copy=False) * np.float32(radiance_mult[band])
            + np.float32(radiance_add.get(band, 0.0))
        )
        scaled[band] = _robust_scale(rad, valid)
    blue = scaled["B2"]
    green = scaled["B5"]
    red = scaled["B7"]
    nir = scaled["B14"]

    vis = (blue + green + red) / 3.0
    whiteness = (
        np.abs(blue - vis) + np.abs(green - vis) + np.abs(red - vis)
    ) / (3.0 * np.maximum(vis, 1e-6))
    ndvi_like = (nir - red) / np.maximum(nir + red, 1e-6)

    vis_thr = float(np.percentile(vis[valid], visible_percentile))
    nir_thr = float(np.percentile(nir[valid], nir_percentile))

    mask = (
        valid
        & (vis >= vis_thr)
        & (nir >= nir_thr)
        & (whiteness <= max_whiteness)
        & (ndvi_like <= max_ndvi)
    )

    mask = _remove_small_components(mask, min_component_size)
    if dilation_iterations > 0 and mask.any():
        mask = binary_dilation(mask, iterations=int(dilation_iterations))
        mask &= valid
    return mask.astype(bool)


def _find_single(scene_dir: Path, pattern: str, what: str) -> Path:
    matches = sorted(scene_dir.glob(pattern))
    if not matches:
        # Be tolerant of lowercase tif extensions.
        if pattern.endswith(".TIF"):
            matches = sorted(scene_dir.glob(pattern[:-4] + ".tif"))
    if not matches:
        raise FileNotFoundError(f"{what} not found in {scene_dir}: {pattern}")
    if len(matches) > 1:
        logger.warning("Multiple %s files in %s; using %s", what, scene_dir, matches[0].name)
    return matches[0]


def _read_on_reference_grid(path: Path, ref_profile: dict) -> tuple[np.ndarray, Optional[float]]:
    with rasterio.open(path) as src:
        data = src.read(1)
        nd = src.nodata
        same_grid = (
            src.crs == ref_profile["crs"]
            and src.transform == ref_profile["transform"]
            and src.width == ref_profile["width"]
            and src.height == ref_profile["height"]
        )
        if same_grid:
            return data, nd

        dst = np.full((ref_profile["height"], ref_profile["width"]), np.nan, dtype=np.float32)
        reproject(
            source=data.astype(np.float32, copy=False),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            src_nodata=nd,
            dst_nodata=np.nan,
            init_dest_nodata=True,
            resampling=Resampling.bilinear,
        )
        return dst, np.nan


def generate_scene_cloud_mask(
    scene,
    output_dir: str | Path | None = None,
    **detector_kwargs,
) -> CloudMaskRecord:
    """Generate one source-grid cloud mask for a Scene."""
    scene_dir = Path(scene.directory)
    mtl_path = _find_single(scene_dir, "*_MTL.txt", "MTL metadata")
    meta = parse_dz01_mtl(mtl_path)

    band_paths = {b: _find_single(scene_dir, f"*_{b}.TIF", b) for b in CLOUD_BANDS}
    with rasterio.open(band_paths["B14"]) as ref:
        ref_profile = {
            "crs": ref.crs,
            "transform": ref.transform,
            "width": ref.width,
            "height": ref.height,
        }
        ref_nodata = ref.nodata

    arrays: dict[str, np.ndarray] = {}
    nodata: dict[str, Optional[float]] = {}
    for band, path in band_paths.items():
        arr, nd = _read_on_reference_grid(path, ref_profile)
        arrays[band] = arr
        nodata[band] = ref_nodata if band == "B14" else nd

    mask = detect_cloud_mask(
        arrays,
        meta["radiance_mult"],
        meta["radiance_add"],
        nodata,
        **detector_kwargs,
    )
    valid = np.ones(mask.shape, dtype=bool)
    b14 = arrays["B14"]
    valid &= np.isfinite(b14)
    if ref_nodata is not None:
        valid &= b14 != ref_nodata
    denom = int(valid.sum())
    frac = float((mask & valid).sum() / denom) if denom else 0.0

    record = CloudMaskRecord(
        mask=mask,
        transform=ref_profile["transform"],
        crs=ref_profile["crs"],
        cloud_fraction=frac,
        metadata_cloud_cover=meta.get("cloud_cover"),
    )

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scene.name}_cloud_mask.tif"
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            width=mask.shape[1],
            height=mask.shape[0],
            count=1,
            dtype="uint8",
            crs=record.crs,
            transform=record.transform,
            compress="lzw",
        ) as dst:
            dst.write(mask.astype(np.uint8), 1)
        logger.info(
            "Cloud mask [%s]: detected=%.4f metadata=%s -> %s",
            scene.name,
            frac,
            record.metadata_cloud_cover,
            out_path,
        )

    return record


def generate_cloud_masks(scenes, output_dir: str | Path | None = None, **detector_kwargs):
    """Generate one source-grid cloud mask per scene."""
    return [generate_scene_cloud_mask(s, output_dir=output_dir, **detector_kwargs) for s in scenes]
