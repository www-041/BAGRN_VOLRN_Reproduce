"""Radiometric normalization adapter for five-scene SIFT pipeline.

Integrates existing BAGRN, VOLRN, and metrics modules into the
multi-scene workflow. Processes one band at a time for memory safety.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass

import numpy as np
import rasterio

from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.metrics import compute_all, compute_per_pair
from src.overlap import detect_multi_overlap
from src.multiscene_sift.band_geometry import apply_world_correction_to_transform
from src.multiscene_sift.mosaicking import raster_bounds_from_transform

logger = logging.getLogger(__name__)


@dataclass
class BandRadiometricResult:
    """Radiometric normalization result for one band."""
    band: str
    original_metrics: dict
    bagrn_metrics: dict
    volrn_metrics: dict
    theta_mu: np.ndarray
    theta_sigma: np.ndarray
    bagrn_runtime_sec: float
    volrn_runtime_sec: float
    normalized_arrays: list[np.ndarray]
    nodata_values: list[float | None]
    corrected_transforms: list
    corrected_bounds: list[tuple[float, float, float, float]]
    overlaps: list[dict]
    registered_metrics: dict


def normalize_registered_band(
    scenes,
    world_transforms: list[np.ndarray],
    band: str,
    reference_idx: int,
    block_size_pixels: int = 200,
    lambda_param: float = 0.1,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> BandRadiometricResult:
    """Run full radiometric normalization pipeline for one band.

    Steps:
        1. Read registered arrays (original DN) from disk.
        2. Apply world corrections to transforms and bounds.
        3. Compute radiometric overlaps after registration.
        4. Compute Registered-stage metrics.
        5. Run BAGRN + BAGRN metrics.
        6. Run VOLRN + VOLRN metrics.
        7. Release intermediate arrays.

    Args:
        scenes: Scene list.
        world_transforms: Global G_i per scene.
        band: Band name.
        reference_idx: Index of reference scene for BAGRN.
        block_size_pixels: VOLRN block size.
        lambda_param: VOLRN lambda.
        rho: VOLRN rho.
        max_iter: VOLRN max iterations.
        tol: VOLRN tolerance.

    Returns:
        :class:`BandRadiometricResult`.
    """
    n = len(scenes)

    # ---- 1. Read source arrays ----------------------------------------------
    arrays_registered = []
    nodata_values = []
    corrected_transforms = []
    corrected_bounds = []

    for s, g in zip(scenes, world_transforms):
        path = s.band_paths[band]
        with rasterio.open(path) as src:
            data = src.read(1)
            orig_tf = src.transform
            nd = src.nodata
        # Apply world correction
        corrected_tf = apply_world_correction_to_transform(orig_tf, g)
        h, w = s.shapes[band]
        bounds = raster_bounds_from_transform(corrected_tf, h, w)

        arrays_registered.append(data[np.newaxis, :, :])
        nodata_values.append(nd)
        corrected_transforms.append(corrected_tf)
        corrected_bounds.append(bounds)

    # ---- 2. Radiometric overlaps --------------------------------------------
    overlaps = detect_multi_overlap(
        corrected_bounds, corrected_transforms, min_pixels=100,
    )

    # Verify reference connectivity
    if not _is_connected_to_reference(overlaps, n, reference_idx):
        components = _rad_components(overlaps, n)
        raise RuntimeError(
            f"Radiometric overlap graph disconnected from reference "
            f"[{reference_idx}]. Components: {components}"
        )

    # ---- 3. Registered-stage metrics ----------------------------------------
    registered_metrics = compute_all(
        arrays_registered, arrays_registered,
        nodata_values, overlaps, bands=[0],
    )

    # ---- 4. BAGRN -----------------------------------------------------------
    t0 = time.perf_counter()
    bagrn_result, theta_mu, theta_sigma = bagrn_normalize(
        arrays_registered, nodata_values, overlaps,
        control_idx=reference_idx,
    )
    bagrn_runtime = time.perf_counter() - t0

    # BAGRN metrics
    bagrn_metrics = compute_all(
        arrays_registered, bagrn_result,
        nodata_values, overlaps, bands=[0],
    )

    # ---- 5. VOLRN -----------------------------------------------------------
    t0 = time.perf_counter()
    volrn_result, volrn_coeffs = volrn_normalize(
        bagrn_result,
        corrected_transforms,
        corrected_bounds,
        nodata_values,
        block_size_pixels=block_size_pixels,
        lambda_param=lambda_param,
        rho=rho,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )
    volrn_runtime = time.perf_counter() - t0

    # VOLRN metrics
    volrn_metrics = compute_all(
        bagrn_result, volrn_result,
        nodata_values, overlaps, bands=[0],
    )

    # ---- 6. Result ----------------------------------------------------------
    return BandRadiometricResult(
        band=band,
        original_metrics={},
        bagrn_metrics=bagrn_metrics,
        volrn_metrics=volrn_metrics,
        theta_mu=theta_mu,
        theta_sigma=theta_sigma,
        bagrn_runtime_sec=bagrn_runtime,
        volrn_runtime_sec=volrn_runtime,
        normalized_arrays=volrn_result,
        nodata_values=nodata_values,
        corrected_transforms=corrected_transforms,
        corrected_bounds=corrected_bounds,
        overlaps=overlaps,
        registered_metrics=registered_metrics,
    )


def release_band_result(result: BandRadiometricResult) -> None:
    """Explicitly release large arrays from a band result."""
    result.normalized_arrays.clear()
    gc.collect()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_connected_to_reference(
    overlaps: list[dict], n: int, ref_idx: int,
) -> bool:
    """Check that every scene is connected to the reference via overlaps."""
    adj = {i: set() for i in range(n)}
    for ov in overlaps:
        i, j = ov["idx_i"], ov["idx_j"]
        adj[i].add(j)
        adj[j].add(i)
    visited = set()
    stack = [ref_idx]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        stack.extend(adj[v] - visited)
    return len(visited) == n


def _rad_components(
    overlaps: list[dict], n: int,
) -> list[list[int]]:
    """Return connected components of the radiometric overlap graph."""
    adj = {i: set() for i in range(n)}
    for ov in overlaps:
        i, j = ov["idx_i"], ov["idx_j"]
        adj[i].add(j)
        adj[j].add(i)
    visited: set[int] = set()
    components = []
    for start in range(n):
        if start in visited:
            continue
        stack = [start]
        comp = []
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            comp.append(v)
            stack.extend(adj[v] - visited)
        if comp:
            components.append(sorted(comp))
    return components