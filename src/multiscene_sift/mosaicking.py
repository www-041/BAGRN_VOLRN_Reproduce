"""Multi-scene mosaic production via existing src.mosaic.create_mosaic."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from src.mosaic import create_mosaic
from src.multiscene_sift.models import Scene, MosaicGrid
from src.multiscene_sift.band_geometry import apply_world_correction_to_transform

logger = logging.getLogger(__name__)


def compute_shared_mosaic_grid(
    scenes: list[Scene],
    G: list[np.ndarray],
    band: str = "B14",
    resolution: float | None = None,
) -> MosaicGrid:
    """Compute shared output mosaic grid from corrected scene footprints.

    Args:
        scenes: Scene list.
        G: Global world-correction matrices per scene.
        band: Band to compute grid from.
        resolution: Output resolution (default: finest input resolution).

    Returns:
        :class:`MosaicGrid` with CRS, transform, width, height.
    """
    # Apply corrections and collect bounds
    corrected_transforms = []
    for s, g in zip(scenes, G):
        orig_tf = s.transforms[band]
        corrected_tf = apply_world_correction_to_transform(orig_tf, g)
        corrected_transforms.append(corrected_tf)

    # Read one scene for CRS and shape
    ref_path = scenes[0].band_paths[band]
    with rasterio.open(ref_path) as src:
        crs = src.crs

    # Union bounds
    lefts, bottoms, rights, tops = [], [], [], []
    for s, tf in zip(scenes, corrected_transforms):
        h, w = s.shapes[band]
        left = tf.c
        top = tf.f
        right = tf.c + w * tf.a
        bottom = tf.f + h * tf.e  # e is negative
        lefts.append(left)
        bottoms.append(bottom)
        rights.append(right)
        tops.append(top)

    left = min(lefts)
    bottom = min(bottoms)
    right = max(rights)
    top = max(tops)

    # Resolution
    if resolution is None:
        resolution = min(abs(tf.a) for tf in corrected_transforms)

    width = int(np.ceil((right - left) / resolution))
    height = int(np.ceil((top - bottom) / resolution))

    mosaic_transform = Affine(resolution, 0, left, 0, -resolution, top)

    logger.info(
        "Mosaic grid: %d×%d px, resolution=%.1f m, bounds=(%.0f, %.0f, %.0f, %.0f)",
        width, height, resolution, left, bottom, right, top,
    )
    return MosaicGrid(crs=crs, transform=mosaic_transform,
                      width=width, height=height, resolution=resolution)


def make_mosaic(
    scenes: list[Scene],
    G: list[np.ndarray],
    band: str,
    mosaic_grid: MosaicGrid,
    output_path: str | Path,
    mode: str = "source_selection",
) -> str:
    """Create a mosaic for a single band.

    Args:
        scenes: Scene list.
        G: Global world-correction matrices.
        band: Band name (B14, B8, B5).
        mosaic_grid: Shared output grid.
        output_path: Output GeoTIFF path.
        mode: Mosaic mode (``"source_selection"`` or ``"weighted"``).

    Returns:
        Output file path.
    """
    arrays = []
    transforms = []
    nodata_vals = []

    for s, g in zip(scenes, G):
        path = s.band_paths[band]
        with rasterio.open(path) as src:
            data = src.read(1)  # single band
            orig_tf = src.transform
            nodata_vals.append(src.nodata)

        corrected_tf = apply_world_correction_to_transform(orig_tf, g)
        arrays.append(data[np.newaxis, :, :])  # add band dim
        transforms.append(corrected_tf)

    return create_mosaic(
        arrays=arrays,
        transforms=transforms,
        crs=str(mosaic_grid.crs),
        nodata_values=nodata_vals,
        output_path=str(output_path),
        resolution=mosaic_grid.resolution,
        mode=mode,
    )


def make_mosaic_original_transforms(
    scenes: list[Scene],
    band: str,
    mosaic_grid: MosaicGrid,
    output_path: str | Path,
    mode: str = "source_selection",
) -> str:
    """Create a mosaic using original (uncorrected) transforms for baseline.

    Args:
        scenes: Scene list.
        band: Band name.
        mosaic_grid: Shared output grid (computed from corrected transforms).
        output_path: Output GeoTIFF path.
        mode: Mosaic mode.

    Returns:
        Output file path.
    """
    arrays = []
    transforms = []
    nodata_vals = []

    for s in scenes:
        path = s.band_paths[band]
        with rasterio.open(path) as src:
            data = src.read(1)
            orig_tf = src.transform
            nodata_vals.append(src.nodata)

        arrays.append(data[np.newaxis, :, :])
        transforms.append(orig_tf)

    return create_mosaic(
        arrays=arrays,
        transforms=transforms,
        crs=str(mosaic_grid.crs),
        nodata_values=nodata_vals,
        output_path=str(output_path),
        resolution=mosaic_grid.resolution,
        mode=mode,
    )