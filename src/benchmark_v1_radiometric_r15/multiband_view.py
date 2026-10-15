from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from affine import Affine
from rasterio.features import geometry_mask, geometry_window
from rasterio.windows import Window
from shapely.geometry import mapping


# Band choices are derived from the frozen V-band wavelength inventory.
COMPOSITES = {
    "TRUE_COLOR": (7, 4, 3),       # red, green, blue: 633.5, 530.5, 490 nm
    "NIR_FALSE_COLOR": (14, 7, 4), # NIR, red, green: 850.5, 633.5, 530.5 nm
}


def _valid(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    mask = np.asarray(mask, dtype=bool)
    if data.shape != mask.shape:
        raise ValueError("data and valid mask must have the same shape")
    return mask & np.isfinite(data)


def stretch_pair(left: np.ndarray, right: np.ndarray, valid_left: np.ndarray,
                 valid_right: np.ndarray, *, mode: str, low_pct: float = 2.0,
                 high_pct: float = 98.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return preview-only percentile ranges; never changes source arrays."""
    if mode not in {"independent", "shared"}:
        raise ValueError("stretch mode must be 'independent' or 'shared'")
    if not 0 <= low_pct < high_pct <= 100:
        raise ValueError("stretch percentiles must satisfy 0 <= low < high <= 100")
    a, b = np.asarray(left), np.asarray(right)
    va, vb = _valid(a, valid_left), _valid(b, valid_right)

    def limits(values: np.ndarray) -> tuple[float, float]:
        if values.size == 0:
            return (0.0, 1.0)
        lo, hi = np.percentile(values.astype(np.float64, copy=False), (low_pct, high_pct))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = float(lo) + 1.0
        return float(lo), float(hi)

    if mode == "independent":
        return limits(a[va]), limits(b[vb])
    joint = np.concatenate((a[va], b[vb]))
    shared = limits(joint)
    return shared, shared


def render_composite(arrays: Mapping[int, np.ndarray], masks: Mapping[int, np.ndarray],
                     bands: Sequence[int], ranges: Sequence[tuple[float, float]]) -> np.ndarray:
    """Render an RGB uint8 preview from named R/G/B bands and fixed stretches."""
    if len(bands) != 3 or len(ranges) != 3:
        raise ValueError("RGB preview requires three bands and three stretches")
    first = np.asarray(arrays[bands[0]]) if bands[0] in arrays else None
    if first is None:
        raise ValueError(f"missing band B{bands[0]}")
    height, width = first.shape
    output = np.zeros((height, width, 3), dtype=np.uint8)
    all_valid = np.ones((height, width), dtype=bool)
    prepared = []
    for channel, (band, limits) in enumerate(zip(bands, ranges)):
        if band not in arrays or band not in masks:
            raise ValueError(f"missing band B{band}")
        data = np.asarray(arrays[band], dtype=np.float32)
        valid = _valid(data, masks[band])
        if data.shape != (height, width):
            raise ValueError("all composite bands must have the same window shape")
        low, high = (float(v) for v in limits)
        if not np.isfinite(low + high) or high <= low:
            raise ValueError("each stretch must have finite increasing bounds")
        scaled = np.clip((data - low) / (high - low), 0.0, 1.0)
        prepared.append((channel, scaled, valid))
        all_valid &= valid
    for channel, scaled, _valid_mask in prepared:
        output[:, :, channel] = np.round(scaled * 255.0).astype(np.uint8)
    output[~all_valid] = 0
    return output


def difference_preview(left: np.ndarray, right: np.ndarray,
                       valid_left: np.ndarray, valid_right: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Return raw signed scene_i-scene_j differences on the common valid mask."""
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("difference inputs must use the same mapped window")
    valid = _valid(a, valid_left) & _valid(b, valid_right)
    out = np.full(a.shape, np.nan, dtype=np.float32)
    out[valid] = a[valid] - b[valid]
    return out, valid


def signed_difference_rgb(difference: np.ndarray, valid: np.ndarray,
                          limit: float | None = None) -> np.ndarray:
    """Render negative values blue, zero white, and positive values red for preview only."""
    diff = np.asarray(difference, dtype=np.float32)
    mask = _valid(diff, valid)
    if limit is None:
        limit = float(np.percentile(np.abs(diff[mask]), 98)) if mask.any() else 1.0
    limit = max(float(limit), 1e-12)
    safe_diff = np.where(mask, diff, 0.0)
    t = np.clip(np.abs(safe_diff) / limit, 0.0, 1.0)
    result = np.zeros((*diff.shape, 3), dtype=np.uint8)
    neutral = np.round((1.0 - t) * 255).astype(np.uint8)
    red = safe_diff >= 0
    result[:, :, 0] = np.where(red, 255, neutral)
    result[:, :, 1] = neutral
    result[:, :, 2] = np.where(red, neutral, 255)
    result[~mask] = 0
    return result


def read_window_padded(dataset, window: Window) -> tuple[np.ndarray, np.ndarray]:
    """Read only a requested raster window, padding off-image areas as invalid."""
    window = window.round_offsets().round_lengths()
    height, width = int(window.height), int(window.width)
    if height <= 0 or width <= 0:
        raise ValueError("window must have positive dimensions")
    data = np.zeros((height, width), dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)
    c0, r0 = max(0, int(window.col_off)), max(0, int(window.row_off))
    c1 = min(dataset.width, int(window.col_off) + width)
    r1 = min(dataset.height, int(window.row_off) + height)
    if c1 <= c0 or r1 <= r0:
        return data, valid
    source = Window(c0, r0, c1-c0, r1-r0)
    dst_r, dst_c = r0-int(window.row_off), c0-int(window.col_off)
    target = (slice(dst_r, dst_r + r1-r0), slice(dst_c, dst_c + c1-c0))
    data[target] = dataset.read(1, window=source)
    valid[target] = dataset.read_masks(1, window=source) > 0
    valid[target] &= np.isfinite(data[target])
    return data, valid


def roi_band_stats(data: np.ndarray, valid: np.ndarray, transform: Affine, geometry
                   ) -> dict[str, float | int | None]:
    """Compute ROI support and raw-DN summary statistics on a bounded array window."""
    geom = mapping(geometry) if hasattr(geometry, "geom_type") else geometry
    inside = geometry_mask([geom], out_shape=np.asarray(data).shape, transform=transform,
                           invert=True, all_touched=False)
    usable = inside & _valid(data, valid)
    total = int(inside.sum())
    values = np.asarray(data)[usable].astype(np.float64, copy=False)
    if total == 0 or values.size == 0:
        return {"valid_fraction": 0.0 if total else None, "valid_count": 0,
                "roi_pixel_count": total, "median": None, "mad": None, "std": None}
    median = float(np.median(values))
    return {"valid_fraction": float(values.size / total), "valid_count": int(values.size),
            "roi_pixel_count": total, "median": median,
            "mad": float(np.median(np.abs(values - median))), "std": float(np.std(values))}


def read_raster_roi_stats(dataset, geometry, *, max_pixels: int = 2_000_000
                          ) -> dict[str, float | int | None | str]:
    """Read only the ROI's bounded raster window; pixels outside the scene count invalid."""
    try:
        window = geometry_window(dataset, [mapping(geometry) if hasattr(geometry, "geom_type") else geometry],
                                 boundless=True).round_offsets().round_lengths()
    except Exception as exc:
        return {"valid_fraction": 0.0, "valid_count": 0, "roi_pixel_count": 0,
                "median": None, "mad": None, "std": None,
                "status": f"NO_SPATIAL_SUPPORT:{type(exc).__name__}"}
    if window.width <= 0 or window.height <= 0:
        return {"valid_fraction": 0.0, "valid_count": 0, "roi_pixel_count": 0,
                "median": None, "mad": None, "std": None, "status": "EMPTY_ROI_WINDOW"}
    if window.width * window.height > max_pixels:
        return {"valid_fraction": None, "valid_count": 0,
                "roi_pixel_count": int(window.width * window.height),
                "median": None, "mad": None, "std": None,
                "status": "ROI_WINDOW_EXCEEDS_READ_LIMIT"}
    data = dataset.read(1, window=window, boundless=True, fill_value=0)
    valid = dataset.read_masks(1, window=window, boundless=True) > 0
    result = roi_band_stats(data, valid, dataset.window_transform(window), geometry)
    result["status"] = "OK" if result["valid_count"] else "NO_VALID_PIXELS"
    return result
