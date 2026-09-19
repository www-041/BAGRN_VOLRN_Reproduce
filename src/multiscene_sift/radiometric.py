"""Radiometric normalization adapter for five-scene SIFT pipeline.

Integrates existing BAGRN, VOLRN, and metrics modules into the
multi-scene workflow. Processes one band at a time for memory safety.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import rasterio
from rasterio.transform import Affine

from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.metrics import compute_all, compute_per_pair
from src.overlap import detect_multi_overlap
from src.multiscene_sift.band_geometry import (
    apply_world_correction_to_transform,
    reproject_to_shared_north_up_grid,
    reproject_mask_to_grid,
)

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
    cloud_masks_registered: list[np.ndarray] = field(default_factory=list)


def normalize_registered_band(
    scenes,
    world_transforms: list[np.ndarray],
    band: str,
    radiometric_control_idx: int,
    registration_grid=None,
    block_size_pixels: int = 200,
    lambda_param: float = 0.1,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-4,
    cloud_masks=None,
) -> BandRadiometricResult:
    """Run full radiometric normalization pipeline for one band.

    Steps:
        1. Read source arrays from disk and apply scene-level world corrections.
        2. Reproject every corrected scene to a local window of one shared
           north-up pixel lattice (Scheme A).
        3. Compute radiometric overlaps on the actually registered rasters.
        4. Compute Registered-stage metrics.
        5. Run BAGRN + BAGRN metrics.
        6. Run VOLRN + VOLRN metrics.
        7. Release intermediate arrays.

    Args:
        scenes: Scene list.
        world_transforms: Global G_i per scene.
        band: Band name.
        radiometric_control_idx: Index of the BAGRN radiometric control scene.
            This is independent from the geometry reference scene.
        registration_grid: Shared north-up grid (normally the B14 mosaic grid).
            If omitted, a compatible grid is derived from the corrected scenes.
        block_size_pixels: VOLRN block size.
        lambda_param: VOLRN lambda.
        rho: VOLRN rho.
        max_iter: VOLRN max iterations.
        tol: VOLRN tolerance.

    Returns:
        :class:`BandRadiometricResult`.
    """
    n = len(scenes)

    # ---- 1. Geometric warp to one shared north-up pixel lattice -------------
    arrays_registered = []
    nodata_values = []
    corrected_transforms = []
    corrected_bounds = []
    cloud_masks_registered = []

    # First collect the corrected source geometry.  G_i is still expressed in
    # the scene CRS, so all bands must use the same CRS as the registration grid.
    sources = []
    for s, g in zip(scenes, world_transforms):
        path = s.band_paths[band]
        with rasterio.open(path) as src:
            data = src.read(1)
            orig_tf = src.transform
            nd = src.nodata
            src_crs = src.crs

        corrected_tf = apply_world_correction_to_transform(orig_tf, g)
        sources.append((data, corrected_tf, nd, src_crs))

    if not sources:
        raise ValueError("No scenes available for radiometric normalization")
    if cloud_masks is not None and len(cloud_masks) != n:
        raise ValueError("cloud_masks length must match scenes length")

    # The normal full pipeline passes the B14-derived shared mosaic grid so all
    # bands use exactly the same resolution and pixel-lattice origin.  Deriving
    # here keeps direct/unit-test callers backward compatible.
    if registration_grid is None:
        crs0 = sources[0][3]
        for idx, (_, _, _, src_crs) in enumerate(sources):
            if src_crs != crs0:
                raise ValueError(
                    f"Scene {idx} CRS {src_crs} differs from scene 0 CRS {crs0}; "
                    "world corrections cannot be shared across CRSs"
                )

        bounds_all = []
        pixel_sizes = []
        for data, corrected_tf, _, _ in sources:
            h, w = data.shape
            corners = [
                corrected_tf * (0, 0),
                corrected_tf * (w, 0),
                corrected_tf * (0, h),
                corrected_tf * (w, h),
            ]
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            bounds_all.append((min(xs), min(ys), max(xs), max(ys)))
            pixel_sizes.extend([
                float(np.hypot(corrected_tf.a, corrected_tf.d)),
                float(np.hypot(corrected_tf.b, corrected_tf.e)),
            ])

        resolution = min(v for v in pixel_sizes if v > 0)
        left = min(b[0] for b in bounds_all)
        bottom = min(b[1] for b in bounds_all)
        right = max(b[2] for b in bounds_all)
        top = max(b[3] for b in bounds_all)
        grid_transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)
        grid_width = max(1, int(np.ceil((right - left) / resolution)))
        grid_height = max(1, int(np.ceil((top - bottom) / resolution)))
        grid_crs = crs0
    else:
        grid_transform = registration_grid.transform
        grid_width = int(registration_grid.width)
        grid_height = int(registration_grid.height)
        grid_crs = registration_grid.crs

    if abs(grid_transform.b) > 1e-12 or abs(grid_transform.d) > 1e-12:
        raise ValueError("registration_grid must be north-up (b=d=0)")

    for img_idx, (data, corrected_tf, nd, src_crs) in enumerate(sources):
        if src_crs != grid_crs:
            raise ValueError(
                f"Scene {img_idx} CRS {src_crs} differs from registration grid CRS "
                f"{grid_crs}; G_i is not valid across different CRSs"
            )

        aligned, aligned_tf, bounds = reproject_to_shared_north_up_grid(
            data,
            corrected_transform=corrected_tf,
            src_crs=src_crs,
            src_nodata=nd,
            shared_transform=grid_transform,
            shared_width=grid_width,
            shared_height=grid_height,
            dst_crs=grid_crs,
        )

        arrays_registered.append(aligned[np.newaxis, :, :])
        nodata_values.append(nd)
        corrected_transforms.append(aligned_tf)
        corrected_bounds.append(bounds)

        if cloud_masks is not None:
            record = cloud_masks[img_idx]
            corrected_mask_tf = apply_world_correction_to_transform(
                record.transform, world_transforms[img_idx]
            )
            aligned_cloud = reproject_mask_to_grid(
                record.mask,
                src_transform=corrected_mask_tf,
                src_crs=record.crs,
                dst_transform=aligned_tf,
                dst_crs=grid_crs,
                dst_shape=aligned.shape,
            )
        else:
            aligned_cloud = np.zeros(aligned.shape, dtype=bool)
        cloud_masks_registered.append(aligned_cloud)

        logger.info(
            "Registered scene %d on north-up grid: %dx%d, origin=(%.3f, %.3f)",
            img_idx, aligned.shape[1], aligned.shape[0], aligned_tf.c, aligned_tf.f,
        )

    # ---- 2. Radiometric overlaps --------------------------------------------
    overlaps = detect_multi_overlap(
        corrected_bounds, corrected_transforms, min_pixels=100,
    )

    # Verify radiometric-control connectivity
    if not _is_connected_to_control(overlaps, n, radiometric_control_idx):
        components = _rad_components(overlaps, n)
        raise RuntimeError(
            f"Radiometric overlap graph disconnected from control "
            f"[{radiometric_control_idx}]. Components: {components}"
        )

    # ---- 3. Registered-stage metrics ----------------------------------------
    registered_metrics = compute_all(
        arrays_registered, arrays_registered,
        nodata_values, overlaps, bands=[0],
        cloud_masks=cloud_masks_registered,
    )

    # ---- 4. BAGRN -----------------------------------------------------------
    t0 = time.perf_counter()
    bagrn_result, theta_mu, theta_sigma = bagrn_normalize(
        arrays_registered, nodata_values, overlaps,
        control_idx=radiometric_control_idx,
        cloud_masks=cloud_masks_registered,
    )
    bagrn_runtime = time.perf_counter() - t0

    # BAGRN metrics
    bagrn_metrics = compute_all(
        arrays_registered, bagrn_result,
        nodata_values, overlaps, bands=[0],
        cloud_masks=cloud_masks_registered,
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
        cloud_masks=cloud_masks_registered,
    )
    volrn_runtime = time.perf_counter() - t0

    # VOLRN metrics
    # Paper Eq.(39) defines GL between the source image I_i and the
    # radiometrically normalized image I_hat_i.  For the final two-stage
    # result, the source baseline is therefore the registered original, not
    # the intermediate BAGRN output.  ADM/ADSD/CD still depend only on
    # ``volrn_result`` through compute_all().
    volrn_metrics = compute_all(
        arrays_registered, volrn_result,
        nodata_values, overlaps, bands=[0],
        cloud_masks=cloud_masks_registered,
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
        cloud_masks_registered=cloud_masks_registered,
    )


def release_band_result(result: BandRadiometricResult) -> None:
    """Explicitly release large arrays from a band result."""
    result.normalized_arrays.clear()
    result.cloud_masks_registered.clear()
    gc.collect()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_connected_to_control(
    overlaps: list[dict], n: int, control_idx: int,
) -> bool:
    """Check that every scene is connected to the radiometric control via overlaps."""
    adj = {i: set() for i in range(n)}
    for ov in overlaps:
        i, j = ov["idx_i"], ov["idx_j"]
        adj[i].add(j)
        adj[j].add(i)
    visited = set()
    stack = [control_idx]
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