"""Common-grid preparation for fair matcher comparison.

Every matcher must see exactly the same overlapping pixel region
at the same resolution.  This module handles:

1. Loading two GeoTIFFs onto a shared geographic grid.
2. Building a preprocessed, down-sampled *MatchView* from the overlap.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import rasterio
from rasterio import warp as rio_warp
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

from src.registration_benchmark.models import CommonGridPair, MatchView

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_pair_to_common_grid(
    ref_path: str,
    tgt_path: str,
    band: int = 1,
) -> CommonGridPair:
    """Load two GeoTIFF band images onto a shared geographic grid.

    The common grid uses the **finer** pixel resolution of the two inputs
    and covers their union extent.  Both images are reprojected via bilinear
    resampling; valid masks are propagated via nearest-neighbour.

    Returns:
        A :class:`CommonGridPair` whose ``ref_raw`` / ``tgt_raw`` have the
        same shape and pixel-grid alignment.
    """
    t0 = time.perf_counter()

    # 1. Open datasets ----------------------------------------------------------
    with rasterio.open(ref_path) as ref_ds, rasterio.open(tgt_path) as tgt_ds:
        ref_crs = ref_ds.crs
        tgt_crs = tgt_ds.crs

        # --- CRS guard (first version: mismatched CRS → error) -----------------
        if ref_crs != tgt_crs:
            raise ValueError(
                f"CRS mismatch: ref={ref_crs}, tgt={tgt_crs}. "
                f"Automatic reprojection is not supported yet."
            )

        ref_transform = ref_ds.transform
        tgt_transform = tgt_ds.transform

        # 2. Common resolution (finer wins) ------------------------------------
        resolution = min(
            abs(ref_transform.a),
            abs(tgt_transform.a),
        )

        # 3. Union extent in geographic coordinates -----------------------------
        # ref
        ref_left, ref_bottom, ref_right, ref_top = _bounds(ref_ds)
        # tgt
        tgt_left, tgt_bottom, tgt_right, tgt_top = _bounds(tgt_ds)

        left = min(ref_left, tgt_left)
        bottom = min(ref_bottom, tgt_bottom)
        right = max(ref_right, tgt_right)
        top = max(ref_top, tgt_top)

        # 4. Common-grid shape and transform ------------------------------------
        common_transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)

        cols = int(np.ceil((right - left) / resolution))
        rows = int(np.ceil((top - bottom) / resolution))

        # Read full-band data
        ref_data = ref_ds.read(band).astype(np.float64)
        tgt_data = tgt_ds.read(band).astype(np.float64)

        ref_nodata = ref_ds.nodata
        tgt_nodata = tgt_ds.nodata

        # Build valid masks (use nodata if set, else mark all valid)
        ref_valid_orig = _valid_mask(ref_data, ref_nodata)
        tgt_valid_orig = _valid_mask(tgt_data, tgt_nodata)

    # 5. Reproject onto common grid -------------------------------------------
    common_shape = (rows, cols)

    ref_common = np.full(common_shape, np.nan, dtype=np.float64)
    tgt_common = np.full(common_shape, np.nan, dtype=np.float64)
    ref_valid = np.zeros(common_shape, dtype=bool)
    tgt_valid = np.zeros(common_shape, dtype=bool)

    # Reproject reference
    reproject(
        source=ref_data,
        destination=ref_common,
        src_transform=ref_transform,
        src_crs=ref_crs,
        dst_transform=common_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear,
        src_nodata=ref_nodata,
        dst_nodata=np.nan,
    )
    reproject(
        source=ref_valid_orig.astype(np.uint8),
        destination=ref_valid.view(np.uint8),
        src_transform=ref_transform,
        src_crs=ref_crs,
        dst_transform=common_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest,
        src_nodata=None,
    )

    # Reproject target
    reproject(
        source=tgt_data,
        destination=tgt_common,
        src_transform=tgt_transform,
        src_crs=tgt_crs,
        dst_transform=common_transform,
        dst_crs=tgt_crs,
        resampling=Resampling.bilinear,
        src_nodata=tgt_nodata,
        dst_nodata=np.nan,
    )
    reproject(
        source=tgt_valid_orig.astype(np.uint8),
        destination=tgt_valid.view(np.uint8),
        src_transform=tgt_transform,
        src_crs=tgt_crs,
        dst_transform=common_transform,
        dst_crs=tgt_crs,
        resampling=Resampling.nearest,
        src_nodata=None,
    )

    # Ensure NaNs are marked invalid in masks
    ref_valid = ref_valid & np.isfinite(ref_common)
    tgt_valid = tgt_valid & np.isfinite(tgt_common)

    # 6. Overlap bounding rectangle -------------------------------------------
    joint_valid = ref_valid & tgt_valid
    overlap_window = _valid_bounds(joint_valid)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Common grid: %s px, overlap window=%s (%.2f s)",
        common_shape,
        overlap_window,
        elapsed,
    )

    return CommonGridPair(
        ref_raw=ref_common,
        tgt_raw=tgt_common,
        ref_valid=ref_valid,
        tgt_valid=tgt_valid,
        transform=common_transform,
        crs=ref_crs,
        overlap_window=overlap_window,
    )


def build_match_view(
    pair: CommonGridPair,
    max_side: int = 1600,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> MatchView:
    """Build a preprocessed, down-sampled view of the overlap region.

    Processing steps applied **only** to the overlap crop of each image:

    1. Percentile clip → ``[p_low, p_high]``.
    2. Linear stretch → ``[0, 1]``.
    3. Down-scale to ``max_side`` while preserving aspect ratio.

    Reference and target use **identical** scale factors.

    Args:
        pair: A :class:`CommonGridPair` from :func:`load_pair_to_common_grid`.
        max_side: Longest side of the output match-view image in pixels.
        percentile_low: Lower percentile for contrast stretch.
        percentile_high: Upper percentile for contrast stretch.

    Returns:
        A :class:`MatchView` ready to be consumed by any matcher.
    """
    row_start, row_end, col_start, col_end = pair.overlap_window

    # --- crop overlap ----------------------------------------------------------
    ref_crop = pair.ref_raw[row_start:row_end, col_start:col_end].copy()
    tgt_crop = pair.tgt_raw[row_start:row_end, col_start:col_end].copy()
    valid_ref = pair.ref_valid[row_start:row_end, col_start:col_end]
    valid_tgt = pair.tgt_valid[row_start:row_end, col_start:col_end]

    # --- percentile stretch ----------------------------------------------------
    def _stretch(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
        vals = img[valid]
        if len(vals) < 10:
            # Degenerate case: return zeros
            return np.zeros_like(img, dtype=np.float32)

        p_low = np.percentile(vals, percentile_low)
        p_high = np.percentile(vals, percentile_high)

        if p_high - p_low < 1e-8:
            return np.zeros_like(img, dtype=np.float32)

        out = np.clip(img, p_low, p_high)
        out = (out - p_low) / (p_high - p_low)
        out[~valid] = 0.0
        return out.astype(np.float32)

    ref_norm = _stretch(ref_crop, valid_ref)
    tgt_norm = _stretch(tgt_crop, valid_tgt)

    # --- resize to max_side (same scale for both) ------------------------------
    import cv2

    h, w = ref_norm.shape
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)

    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    ref_resized = cv2.resize(ref_norm, (new_w, new_h), interpolation=cv2.INTER_AREA)
    tgt_resized = cv2.resize(tgt_norm, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ref_valid_resized = cv2.resize(
        valid_ref.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    tgt_valid_resized = cv2.resize(
        valid_tgt.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # Scale factors: match-view pixel → common-grid pixel
    # origin_x/y = col_start / row_start in common-grid coordinates
    logger.info(
        "Match view: %d×%d (scale=%.4f), overlap origin=(%d, %d)",
        new_w,
        new_h,
        scale,
        col_start,
        row_start,
    )

    return MatchView(
        ref=ref_resized,
        tgt=tgt_resized,
        ref_valid=ref_valid_resized,
        tgt_valid=tgt_valid_resized,
        origin_x=float(col_start),
        origin_y=float(row_start),
        scale_x=scale,
        scale_y=scale,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bounds(ds: rasterio.io.DatasetReader) -> tuple[float, float, float, float]:
    """Return ``(left, bottom, right, top)`` of a rasterio dataset."""
    left = ds.bounds.left
    bottom = ds.bounds.bottom
    right = ds.bounds.right
    top = ds.bounds.top
    return left, bottom, right, top


def _valid_mask(data: np.ndarray, nodata: float | None) -> np.ndarray:
    """Build a boolean valid-pixel mask from an array and its nodata value."""
    if nodata is not None:
        return data != nodata
    return np.ones(data.shape, dtype=bool)


def _valid_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(row_start, row_end, col_start, col_end)`` bounding box of
    ``True`` pixels in *mask*.

    If no pixel is valid, returns ``(0, 0, 0, 0)``.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():
        return (0, 0, 0, 0)

    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]

    return (
        int(r_idx[0]),
        int(r_idx[-1]) + 1,
        int(c_idx[0]),
        int(c_idx[-1]) + 1,
    )