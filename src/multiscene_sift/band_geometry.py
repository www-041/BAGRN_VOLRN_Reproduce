"""Pixel-to-world affine conversion and band geometry application."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling


def pixel_affine_to_world(
    pixel_matrix: np.ndarray,
    common_transform,
) -> np.ndarray:
    """Convert a pair-wise pixel affine to world coordinates.

    Given an affine matrix ``M_pixel`` in the pair's common-grid pixel space
    and the rasterio geotransform ``T`` of that common grid:

        A_world = T * M_pixel * T^{-1}

    Args:
        pixel_matrix: 3×3 affine matrix in common-grid pixel coordinates.
        common_transform: Rasterio ``Affine`` of the pair's common grid.

    Returns:
        3×3 affine matrix in world coordinates.
    """
    M = np.asarray(pixel_matrix, dtype=np.float64)
    if M.shape != (3, 3):
        raise ValueError(f"Expected 3×3 pixel matrix, got {M.shape}")

    T = _transform_to_matrix(common_transform)
    T_inv = np.linalg.inv(T)
    A = T @ M @ T_inv
    return A


def apply_world_correction_to_transform(
    original_transform,
    world_matrix: np.ndarray,
) -> Affine:
    """Apply a world-coordinate correction to a rasterio geotransform.

    Given scene *i*'s world correction matrix ``G_i`` and the original
    raster transform ``T_{i,b}`` for band *b*:

        T'_{i,b} = G_i * T_{i,b}

    Args:
        original_transform: Original rasterio ``Affine`` for the band.
        world_matrix: 3×3 world correction matrix ``G_i``.

    Returns:
        Corrected rasterio ``Affine``.
    """
    G = np.asarray(world_matrix, dtype=np.float64)
    T = _transform_to_matrix(original_transform)
    T_prime = G @ T
    return Affine(*T_prime.flat[:6])


def reproject_to_shared_north_up_grid(
    data: np.ndarray,
    corrected_transform: Affine,
    src_crs,
    src_nodata,
    shared_transform: Affine,
    shared_width: int,
    shared_height: int,
    dst_crs=None,
    resampling=Resampling.bilinear,
):
    """Reproject one registered scene onto a local window of a shared north-up grid.

    ``corrected_transform`` already contains the scene-level geometric
    correction ``G_i``.  The returned array is actually resampled; it is not
    merely paired with a rotated/sheared transform.  All returned transforms
    are translation-only subwindows of ``shared_transform`` and therefore
    share one pixel lattice.

    Returns
    -------
    data_aligned : np.ndarray
        2-D float64 array on the shared north-up lattice.
    local_transform : rasterio.Affine
        North-up transform for the local scene window.
    local_bounds : tuple
        ``(left, bottom, right, top)`` of the local output window.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D band array, got shape {data.shape}")
    if shared_width <= 0 or shared_height <= 0:
        raise ValueError("Shared grid width/height must be positive")
    if abs(shared_transform.b) > 1e-12 or abs(shared_transform.d) > 1e-12:
        raise ValueError("Shared registration grid must be north-up (b=d=0)")
    if shared_transform.a <= 0 or shared_transform.e >= 0:
        raise ValueError("Shared registration grid must use positive x and negative y pixel size")

    if dst_crs is None:
        dst_crs = src_crs

    rows, cols = data.shape
    world_corners = [
        corrected_transform * (0, 0),
        corrected_transform * (cols, 0),
        corrected_transform * (0, rows),
        corrected_transform * (cols, rows),
    ]
    inv_shared = ~shared_transform
    pix_corners = [inv_shared * p for p in world_corners]
    col_vals = [p[0] for p in pix_corners]
    row_vals = [p[1] for p in pix_corners]

    # Tiny epsilon prevents exact integer boundaries from expanding by one
    # pixel because of floating-point roundoff in affine inversion.
    eps = 1e-9
    c0 = max(0, int(np.floor(min(col_vals) + eps)))
    c1 = min(shared_width, int(np.ceil(max(col_vals) - eps)))
    r0 = max(0, int(np.floor(min(row_vals) + eps)))
    r1 = min(shared_height, int(np.ceil(max(row_vals) - eps)))

    if c1 <= c0 or r1 <= r0:
        raise ValueError("Corrected scene footprint does not intersect the shared registration grid")

    local_transform = shared_transform * Affine.translation(c0, r0)
    out_h = r1 - r0
    out_w = c1 - c0
    destination = np.full((out_h, out_w), np.nan, dtype=np.float64)

    effective_src_nodata = src_nodata
    if effective_src_nodata is None and np.issubdtype(data.dtype, np.floating):
        if np.isnan(data).any():
            effective_src_nodata = np.nan

    reproject(
        source=data.astype(np.float64, copy=False),
        destination=destination,
        src_transform=corrected_transform,
        src_crs=src_crs,
        dst_transform=local_transform,
        dst_crs=dst_crs,
        src_nodata=effective_src_nodata,
        dst_nodata=np.nan,
        init_dest_nodata=True,
        resampling=resampling,
    )

    left, top = local_transform * (0, 0)
    right, bottom = local_transform * (out_w, out_h)
    local_bounds = (min(left, right), min(bottom, top), max(left, right), max(bottom, top))
    return destination, local_transform, local_bounds


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _transform_to_matrix(transform) -> np.ndarray:
    """Convert a rasterio Affine to a 3×3 numpy matrix."""
    return np.array([
        [transform.a, transform.b, transform.c],
        [transform.d, transform.e, transform.f],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

def reproject_mask_to_grid(
    mask: np.ndarray,
    src_transform: Affine,
    src_crs,
    dst_transform: Affine,
    dst_crs,
    dst_shape: tuple[int, int],
) -> np.ndarray:
    """Reproject a boolean mask onto an exact target grid using nearest-neighbor.

    ``True`` means masked/cloud. The destination is always returned as bool.
    Pixels outside the source footprint default to False; the corresponding
    image-validity mask is handled separately by the radiometric pipeline.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got shape {arr.shape}")
    out_h, out_w = map(int, dst_shape)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid destination shape: {dst_shape}")

    destination = np.zeros((out_h, out_w), dtype=np.uint8)
    reproject(
        source=arr.astype(np.uint8, copy=False),
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=None,
        dst_nodata=0,
        init_dest_nodata=True,
        resampling=Resampling.nearest,
    )
    return destination.astype(bool)
