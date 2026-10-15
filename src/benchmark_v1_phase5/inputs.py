"""Windowed B9 reprojection onto one overlap grid and one matcher view."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, bounds as window_bounds, from_bounds, transform as window_transform

from .protocol import PairManifest, Phase5Config, verify_manifest_bytes


@dataclass(frozen=True)
class ViewProvenance:
    """Pixel-centre maps make view coordinates reversible to both native rasters."""

    native_affine_i: Affine
    native_affine_j: Affine
    overlap_affine: Affine
    resize_x: float
    resize_y: float
    stretch_percentiles: tuple[int, int]
    stretch_bounds_i: tuple[float, float]
    stretch_bounds_j: tuple[float, float]
    data_resampling: str
    mask_resampling: str

    def _view_to_native(self, x: float, y: float, native_affine: Affine) -> tuple[float, float]:
        col = (x + 0.5) / self.resize_x - 0.5
        row = (y + 0.5) / self.resize_y - 0.5
        world_x, world_y = self.overlap_affine * (col + 0.5, row + 0.5)
        native_col, native_row = (~native_affine) * (world_x, world_y)
        return native_row - 0.5, native_col - 0.5

    def view_to_native_i(self, x: float, y: float) -> tuple[float, float]:
        return self._view_to_native(x, y, self.native_affine_i)

    def view_to_native_j(self, x: float, y: float) -> tuple[float, float]:
        return self._view_to_native(x, y, self.native_affine_j)


@dataclass(frozen=True)
class MatchView:
    left: np.ndarray
    right: np.ndarray
    mask_i: np.ndarray
    mask_j: np.ndarray
    joint_mask: np.ndarray
    provenance: ViewProvenance


@dataclass(frozen=True)
class PairInput:
    pair_id: str
    status: str
    view: MatchView | None
    native_affine_i: Affine
    native_affine_j: Affine
    native_joint_valid_count: int
    view_joint_valid_count: int
    reason: str | None = None


def _validate_raster(src: rasterio.io.DatasetReader, config: Phase5Config) -> None:
    affine = src.transform
    if (src.crs is None or src.crs.to_string() != config.crs or src.count != 1
            or not all(math.isfinite(value) for value in affine[:6])
            or affine.a != config.native_pixel_m or affine.e != -config.native_pixel_m
            or affine.b != 0 or affine.d != 0):
        raise ValueError("Unsupported CRS/grid or nonfinite affine")


def _overlap_grid(left, right, pixel_m: int) -> tuple[Affine, int, int] | None:
    west = max(left.bounds.left, right.bounds.left)
    east = min(left.bounds.right, right.bounds.right)
    south = max(left.bounds.bottom, right.bounds.bottom)
    north = min(left.bounds.top, right.bounds.top)
    if west >= east or south >= north:
        return None
    # Select only reference-grid pixels whose centres lie inside both bounds.
    col0 = max(0, math.ceil((west - left.transform.c) / pixel_m - 0.5))
    col1 = min(left.width, math.ceil((east - left.transform.c) / pixel_m - 0.5))
    row0 = max(0, math.ceil((left.transform.f - north) / pixel_m - 0.5))
    row1 = min(left.height, math.ceil((left.transform.f - south) / pixel_m - 0.5))
    if col1 <= col0 or row1 <= row0:
        return None
    return left.transform * Affine.translation(col0, row0), col1 - col0, row1 - row0


def _read_source_tile(src, window: Window) -> tuple[np.ndarray, np.ndarray]:
    """The sole raster read point; always requires a bounded source window."""
    data = src.read(1, window=window).astype("float32")
    mask = src.dataset_mask(window=window) > 0
    return data, mask


def _source_window(src, destination_affine: Affine, height: int, width: int) -> Window:
    left, bottom, right, top = window_bounds(Window(0, 0, width, height), destination_affine)
    floating = from_bounds(left, bottom, right, top, transform=src.transform)
    col0 = max(0, math.floor(floating.col_off) - 1)
    row0 = max(0, math.floor(floating.row_off) - 1)
    col1 = min(src.width, math.ceil(floating.col_off + floating.width) + 1)
    row1 = min(src.height, math.ceil(floating.row_off + floating.height) + 1)
    return Window(col0, row0, max(0, col1 - col0), max(0, row1 - row0))


def _resample_tile(src, destination_affine: Affine, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    window = _source_window(src, destination_affine, *shape)
    data = np.zeros(shape, dtype="float32")
    valid = np.zeros(shape, dtype="uint8")
    if window.width == 0 or window.height == 0:
        return data, valid.astype(bool)
    source, mask = _read_source_tile(src, window)
    source[~mask | ~np.isfinite(source)] = np.nan
    source_affine = window_transform(window, src.transform)
    reproject(source, data, src_transform=source_affine, src_crs=src.crs,
              dst_transform=destination_affine, dst_crs=src.crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
    reproject(mask.astype("uint8"), valid, src_transform=source_affine, src_crs=src.crs,
              dst_transform=destination_affine, dst_crs=src.crs,
              src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
    return data, (valid > 0) & np.isfinite(data)


def _tile_slices(height: int, width: int, side: int):
    for row in range(0, height, side):
        for col in range(0, width, side):
            yield row, col, min(side, height - row), min(side, width - col)


def _count_joint_native_support(left, right, affine: Affine, height: int, width: int) -> int:
    count = 0
    for row, col, h, w in _tile_slices(height, width, 256):
        tile_affine = affine * Affine.translation(col, row)
        _, left_valid = _resample_tile(left, tile_affine, (h, w))
        _, right_valid = _resample_tile(right, tile_affine, (h, w))
        count += int(np.count_nonzero(left_valid & right_valid))
    return count


def _stretch(data: np.ndarray, valid: np.ndarray, low: int, high: int) -> tuple[np.ndarray, tuple[float, float]]:
    values = data[valid]
    lo, hi = np.percentile(values, (low, high))
    output = np.zeros_like(data, dtype="float32")
    if hi > lo:
        # Frozen 2nd/98th percentile clipping maps valid overlap radiance to [0, 1].
        output[valid] = np.clip((values - lo) / (hi - lo), 0, 1)
    return output, (float(lo), float(hi))


def prepare_pair(manifest: PairManifest, config: Phase5Config) -> PairInput:
    """Return a shared B9 view or a distinct native/view preprocessing status."""
    if manifest.pair_id not in config.pair_ids:
        raise ValueError("Pair is outside the Phase5 protocol")
    verify_manifest_bytes(manifest)
    with rasterio.open(manifest.path_i) as left, rasterio.open(manifest.path_j) as right:
        _validate_raster(left, config)
        _validate_raster(right, config)
        grid = _overlap_grid(left, right, config.native_pixel_m)
        if grid is None:
            return PairInput(manifest.pair_id, "NO_VALID_OVERLAP", None,
                             left.transform, right.transform, 0, 0)
        overlap_affine, native_width, native_height = grid
        native_count = _count_joint_native_support(left, right, overlap_affine,
                                                   native_height, native_width)
        if native_count == 0:
            return PairInput(manifest.pair_id, "NO_VALID_OVERLAP", None,
                             left.transform, right.transform, 0, 0)
        scale = min(1.0, config.max_view_side / max(native_width, native_height))
        view_width = max(1, int(round(native_width * scale)))
        view_height = max(1, int(round(native_height * scale)))
        resize_x, resize_y = view_width / native_width, view_height / native_height
        view_affine = overlap_affine * Affine.scale(1 / resize_x, 1 / resize_y)
        a = np.zeros((view_height, view_width), dtype="float32")
        b = np.zeros_like(a)
        mask_i = np.zeros(a.shape, dtype=bool)
        mask_j = np.zeros(a.shape, dtype=bool)
        joint = np.zeros(a.shape, dtype=bool)
        # Each destination tile maps to at most ~256 native pixels per side.
        tile_side = max(1, min(256, int(256 * min(resize_x, resize_y))))
        for row, col, h, w in _tile_slices(view_height, view_width, tile_side):
            tile_affine = view_affine * Affine.translation(col, row)
            tile_a, mask_a = _resample_tile(left, tile_affine, (h, w))
            tile_b, mask_b = _resample_tile(right, tile_affine, (h, w))
            a[row:row+h, col:col+w] = tile_a
            b[row:row+h, col:col+w] = tile_b
            mask_i[row:row+h, col:col+w] = mask_a
            mask_j[row:row+h, col:col+w] = mask_b
            joint[row:row+h, col:col+w] = mask_a & mask_b
        view_count = int(np.count_nonzero(joint))
        if view_count == 0:
            return PairInput(manifest.pair_id, "PREPROCESSING_FAIL", None,
                             left.transform, right.transform, native_count, 0,
                             "NO_USABLE_VIEW_AFTER_RESIZE")
        low, high = config.stretch_percentiles
        stretched_i, bounds_i = _stretch(a, joint, low, high)
        stretched_j, bounds_j = _stretch(b, joint, low, high)
        provenance = ViewProvenance(left.transform, right.transform, overlap_affine,
                                    resize_x, resize_y, (low, high), bounds_i, bounds_j,
                                    "bilinear", "nearest")
        view = MatchView(stretched_i, stretched_j, mask_i, mask_j, joint, provenance)
        return PairInput(manifest.pair_id, "READY", view, left.transform, right.transform,
                         native_count, view_count)
