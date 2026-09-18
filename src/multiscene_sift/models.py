"""Data models for five-scene SIFT multi-image registration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REQUIRED_BANDS = ("B14", "B8", "B5")


@dataclass
class Scene:
    """A single DZ01V scene with B14/B8/B5 bands."""

    index: int
    name: str
    directory: str
    band_paths: dict[str, str]
    crs: object  # rasterio CRS
    transforms: dict[str, object]  # band -> rasterio Affine
    shapes: dict[str, tuple[int, int]]
    nodata: dict[str, float | None]
    bounds: dict[str, tuple[float, float, float, float]]


@dataclass
class OverlapEdge:
    """Geographic overlap between two scenes (B14 band)."""

    idx_i: int
    idx_j: int
    intersection_area: float
    overlap_ratio_i: float
    overlap_ratio_j: float


@dataclass
class PairwiseRegistration:
    """Result of pairwise SIFT+RANSAC registration on B14."""

    idx_i: int
    idx_j: int
    status: str
    raw_matches: int
    inliers: int
    inlier_ratio: float
    coverage: float
    residual_median: float
    residual_rmse: float
    residual_p95: float
    pair_pixel_matrix: list[list[float]]
    pair_common_transform: object  # rasterio Affine for the pair's common grid
    runtime_sec: float
    # Inlier point coordinates in pair common-grid pixel space (for global consistency)
    inlier_ref_xy: np.ndarray | None = None  # (N,2) ref inlier points
    inlier_tgt_xy: np.ndarray | None = None  # (N,2) tgt inlier points


@dataclass
class MosaicGrid:
    """Shared output mosaic grid definition."""

    crs: object
    transform: object
    width: int
    height: int
    resolution: float