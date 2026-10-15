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


def raster_bounds_from_transform(transform, height: int, width: int):
    """Compute geographic (left, bottom, right, top) from four corners.

    Handles arbitrary affine transforms (rotation, shear) by transforming
    all four image corners and taking the bounding envelope.

    Args:
        transform: Rasterio ``Affine``.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        ``(left, bottom, right, top)`` in CRS units.
    """
    corners = [
        transform * (0, 0),
        transform * (width, 0),
        transform * (0, height),
        transform * (width, height),
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs), min(ys), max(xs), max(ys)


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

    # Union bounds using four-corner method for affine safety
    lefts, bottoms, rights, tops = [], [], [], []
    for s, tf in zip(scenes, corrected_transforms):
        h, w = s.shapes[band]
        l, b, r, t = raster_bounds_from_transform(tf, h, w)
        lefts.append(l)
        bottoms.append(b)
        rights.append(r)
        tops.append(t)

    left = min(lefts)
    bottom = min(bottoms)
    right = max(rights)
    top = max(tops)

    # Resolution: use the full affine pixel-vector lengths.  abs(tf.a) alone
    # underestimates resolution when the corrected transform has rotation/shear.
    if resolution is None:
        pixel_sizes = []
        for tf in corrected_transforms:
            pixel_sizes.append(float(np.hypot(tf.a, tf.d)))
            pixel_sizes.append(float(np.hypot(tf.b, tf.e)))
        resolution = min(v for v in pixel_sizes if v > 0)

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
        output_transform=mosaic_grid.transform,
        output_width=mosaic_grid.width,
        output_height=mosaic_grid.height,
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
        output_transform=mosaic_grid.transform,
        output_width=mosaic_grid.width,
        output_height=mosaic_grid.height,
    )