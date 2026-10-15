"""Shared fixtures for registration-benchmark tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine


@pytest.fixture
def tmp_dir():
    """Temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def _write_geotiff(
    path: str,
    arr: np.ndarray,
    transform: Affine,
    crs: str = "EPSG:4326",
    nodata: float | None = None,
) -> str:
    """Write a 2D array as a single-band GeoTIFF."""
    if arr.ndim != 2:
        raise ValueError("arr must be 2-D")
    h, w = arr.shape
    profile = {
        "driver": "GTiff",
        "dtype": arr.dtype.name,
        "count": 1,
        "height": h,
        "width": w,
        "transform": transform,
        "crs": crs,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


def make_synthetic_pair(
    tmp_dir: Path,
    ref_size: int = 512,
    tgt_size: int = 512,
    ref_origin: tuple[float, float] = (0.0, 512.0),
    tgt_origin: tuple[float, float] = (128.0, 512.0),
    resolution: float = 1.0,
    crs: str = "EPSG:4326",
) -> tuple[str, str]:
    """Create two synthetic GeoTIFFs with a known overlap.

    Returns ``(ref_path, tgt_path)``.
    """
    rng = np.random.default_rng(42)

    # Reference: origin at (0, 512), so top edge is row 0 at y=512
    ref_transform = Affine(resolution, 0.0, ref_origin[0], 0.0, -resolution, ref_origin[1])
    ref_data = (rng.uniform(0, 255, size=(ref_size, ref_size))).astype(np.float32)

    # Target: shifted 128 px to the right
    tgt_transform = Affine(resolution, 0.0, tgt_origin[0], 0.0, -resolution, tgt_origin[1])
    tgt_data = (rng.uniform(0, 255, size=(tgt_size, tgt_size))).astype(np.float32)

    ref_path = str(tmp_dir / "ref_synth.tif")
    tgt_path = str(tmp_dir / "tgt_synth.tif")

    _write_geotiff(ref_path, ref_data, ref_transform, crs=crs)
    _write_geotiff(tgt_path, tgt_data, tgt_transform, crs=crs)

    return ref_path, tgt_path


def make_translated_texture_pair(
    tmp_dir: Path,
    size: int = 256,
    dx: float = 7.0,
    dy: float = -5.0,
    resolution: float = 1.0,
    crs: str = "EPSG:4326",
) -> tuple[str, str]:
    """Create two GeoTIFFs where the target is a pure translation of the reference.

    The reference contains a structured texture (gradient + circles).
    The target has the **same** texture but the target image is shifted
    by ``(dx, dy)`` **geographically** (i.e. target origin offset).

    Returns ``(ref_path, tgt_path)``.
    """
    y, x = np.mgrid[0:size, 0:size]

    # Structured texture: gradient + radial pattern
    texture = (
        0.5 * np.sin(x * 0.1) * np.cos(y * 0.1)
        + 0.3 * np.sin((x - size / 2) ** 2 + (y - size / 2) ** 2) / 150.0
        + 0.2 * np.random.default_rng(123).uniform(-1, 1, size=(size, size))
    )
    texture = ((texture - texture.min()) / (texture.max() - texture.min()) * 255).astype(np.float32)

    ref_transform = Affine(resolution, 0.0, 0.0, 0.0, -resolution, float(size))
    # Shift target origin by (-dx, -dy) so that the *same* geographic point
    # lands at a different pixel in the target image.
    tgt_transform = Affine(
        resolution, 0.0, -dx * resolution,
        0.0, -resolution, float(size) + dy * resolution,
    )

    ref_path = str(tmp_dir / "ref_tex.tif")
    tgt_path = str(tmp_dir / "tgt_tex.tif")

    _write_geotiff(ref_path, texture.copy(), ref_transform, crs=crs)
    _write_geotiff(tgt_path, texture.copy(), tgt_transform, crs=crs)

    return ref_path, tgt_path