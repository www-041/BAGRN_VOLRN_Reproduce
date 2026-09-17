"""Pixel-to-world affine conversion and band geometry application."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import Affine


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