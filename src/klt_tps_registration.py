"""Pure-array KLT/TPS registration primitives.

The implementation preserves the senior-provided KLT matching semantics while
leaving raster I/O and pipeline orchestration to the project modules.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates

from src.coregistration import (
    build_pair_overlap_context,
    map_pixel_centers_between_grids,
)


def normalize_klt_image(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Apply the source script's robust local contrast normalization."""
    a = np.asarray(data).astype(np.float32)
    mask = np.asarray(valid, dtype=bool)
    if a.ndim != 2 or mask.shape != a.shape:
        raise ValueError("KLT normalization requires same-shape 2D data/mask")
    mask &= np.isfinite(a)
    if np.count_nonzero(mask) < 100:
        raise ValueError("KLT normalization requires at least 100 valid pixels")
    low, high = np.percentile(a[mask], [1, 99])
    if not np.isfinite(high - low) or high <= low:
        raise ValueError("valid image lacks usable intensity range")
    image = np.clip((a - low) / (high - low), 0, 1)
    image[~mask] = np.median(image[mask])
    mean = gaussian_filter(image, 7)
    std = np.sqrt(
        np.maximum(gaussian_filter(image * image, 7) - mean * mean, 0)
        + 0.0025
    )
    return np.uint8(np.clip((image - mean) / std * 0.18 + 0.5, 0, 1) * 255)


def build_klt_interior_mask(valid: np.ndarray, margin: int) -> np.ndarray:
    """Erode a validity mask with the source square interior kernel."""
    mask = np.asarray(valid, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("KLT interior mask requires a 2D validity mask")
    margin = int(margin)
    if margin < 0:
        raise ValueError("KLT interior margin must be non-negative")
    if margin == 0:
        return (mask != 0).astype(np.uint8)
    kernel = np.ones((2 * margin + 1, 2 * margin + 1), dtype=np.uint8)
    return cv2.erode(
        (mask != 0).astype(np.uint8), kernel,
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )


def _require_same_2d_images(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference)
    mov = np.asarray(moving)
    if ref.ndim != 2 or mov.ndim != 2 or ref.shape != mov.shape:
        raise ValueError("KLT matching requires same-shape 2D overlap images")
    if min(ref.shape) < 32 or max(ref.shape) >= 32767:
        raise ValueError("insufficient KLT overlap dimensions")
    return ref, mov


def match_bidirectional_klt(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_valid: np.ndarray,
    moving_valid: np.ndarray,
    params: dict[str, Any] | None = None,
    *,
    training_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Find source-to-target controls with forward/backward Pyramid-LK."""
    params = params or {}
    ref, mov = _require_same_2d_images(reference, moving)
    ref_valid = np.asarray(reference_valid, dtype=bool)
    mov_valid = np.asarray(moving_valid, dtype=bool)
    if ref_valid.shape != ref.shape or mov_valid.shape != mov.shape:
        raise ValueError("KLT matching requires same-shape validity masks")
    if training_mask is not None:
        training_mask = np.asarray(training_mask, dtype=bool)
        if training_mask.shape != ref.shape:
            raise ValueError("KLT training mask must match overlap image shape")
        ref_training_valid = ref_valid & training_mask
        moving_training_valid = mov_valid & training_mask
    else:
        ref_training_valid = ref_valid
        moving_training_valid = mov_valid

    cv2.setNumThreads(int(params.get("klt_tps_threads", 4)))
    try:
        ref_norm = normalize_klt_image(ref, ref_training_valid)
        mov_norm = normalize_klt_image(mov, moving_training_valid)
    except ValueError as exc:
        raise ValueError(f"insufficient KLT image support: {exc}") from exc

    window = int(params.get("klt_tps_window", 9))
    pyramid_level = int(params.get("klt_tps_pyramid_level", 3))
    ref_interior = build_klt_interior_mask(
        ref_training_valid, max(20, window // 2 + 2),
    )
    corners = cv2.goodFeaturesToTrack(
        ref_norm,
        maxCorners=int(params.get("klt_tps_max_corners", 4000)),
        qualityLevel=0.003,
        minDistance=float(params.get("klt_tps_min_corner_distance", 5.0)),
        mask=ref_interior,
        blockSize=5,
    )
    initial_count = 0 if corners is None else int(len(corners))
    if corners is None or initial_count < int(params.get("klt_tps_min_points", 30)):
        raise ValueError(
            f"insufficient KLT corners: {initial_count} < "
            f"{int(params.get('klt_tps_min_points', 30))}"
        )
    p = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 0.001)
    lk_kwargs = {
        "winSize": (window, window),
        "maxLevel": pyramid_level,
        "criteria": criteria,
    }
    q, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        ref_norm, mov_norm, p.reshape(-1, 1, 2), None, **lk_kwargs
    )
    if q is None or status_forward is None:
        raise ValueError("insufficient valid forward KLT tracks")
    q = np.asarray(q, dtype=np.float32).reshape(-1, 2)
    sf = np.asarray(status_forward).reshape(-1).astype(bool)
    finite = sf & np.isfinite(q).all(axis=1)
    p_forward = p[finite]
    q_forward = q[finite]
    if len(q_forward) < int(params.get("klt_tps_min_points", 30)):
        raise ValueError("insufficient valid forward KLT tracks")

    p_back, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        mov_norm, ref_norm, q_forward.reshape(-1, 1, 2), None, **lk_kwargs
    )
    if p_back is None or status_backward is None:
        raise ValueError("insufficient valid backward KLT tracks")
    p_back = np.asarray(p_back, dtype=np.float32).reshape(-1, 2)
    sb = np.asarray(status_backward).reshape(-1).astype(bool)
    if len(p_back) != len(p_forward):
        raise ValueError("insufficient valid backward KLT tracks")
    height, width = mov.shape
    q_rows = np.rint(q_forward[:, 1]).astype(int)
    q_cols = np.rint(q_forward[:, 0]).astype(int)
    in_bounds = (
        (q_forward[:, 0] > 10) & (q_forward[:, 0] < width - 10)
        & (q_forward[:, 1] > 10) & (q_forward[:, 1] < height - 10)
    )
    safe_rows = np.clip(q_rows, 0, height - 1)
    safe_cols = np.clip(q_cols, 0, width - 1)
    moving_interior = build_klt_interior_mask(
        moving_training_valid, max(10, window // 2 + 2),
    )
    interior = moving_interior[safe_rows, safe_cols] != 0
    fb = np.linalg.norm(p_forward - p_back, axis=1)
    accepted = sb & np.isfinite(fb) & in_bounds & interior
    accepted &= fb < float(params.get("klt_tps_fb_threshold", 0.5))
    p_acc = p_forward[accepted].astype(np.float64)
    q_acc = q_forward[accepted].astype(np.float64)
    fb_acc = fb[accepted].astype(np.float64)
    minimum = int(params.get("klt_tps_min_points", 30))
    if len(p_acc) < minimum:
        raise ValueError(f"insufficient accepted KLT matches: {len(p_acc)} < {minimum}")
    centered = p_acc - p_acc.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError("collinear KLT controls")
    return {
        "initial_corner_count": initial_count,
        "accepted_point_count": int(len(p_acc)),
        "reference_points_xy": p_acc,
        "moving_points_xy": q_acc,
        "displacement_xy": q_acc - p_acc,
        "forward_backward_error": fb_acc,
    }


def map_klt_matches_to_moving_grid(
    reference_points_overlap_xy: np.ndarray,
    moving_points_overlap_xy: np.ndarray,
    overlap_transform: Affine,
    moving_transform: Affine,
) -> dict[str, np.ndarray]:
    """Map overlap-grid output/source controls into moving-native pixels."""
    output_xy = map_pixel_centers_between_grids(
        reference_points_overlap_xy, overlap_transform, moving_transform,
    )
    source_xy = map_pixel_centers_between_grids(
        moving_points_overlap_xy, overlap_transform, moving_transform,
    )
    return {
        "control_points_xy": output_xy,
        "source_points_xy": source_xy,
        "displacement_xy": source_xy - output_xy,
    }


def _unavailable_pair_result(
    reason: str,
    *,
    context: dict[str, Any] | None = None,
    match: dict[str, Any] | None = None,
    mapped: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    result = {
        "available": False,
        "failure_reason": str(reason),
        "overlap_context": context,
        "initial_corner_count": int((match or {}).get("initial_corner_count", 0)),
        "accepted_point_count": int((match or {}).get("accepted_point_count", 0)),
        "reference_points_overlap_xy": np.empty((0, 2), dtype=float),
        "moving_points_overlap_xy": np.empty((0, 2), dtype=float),
        "forward_backward_error": np.empty((0,), dtype=float),
        "control_points_moving_xy": np.empty((0, 2), dtype=float),
        "source_points_moving_xy": np.empty((0, 2), dtype=float),
        "displacement_xy": np.empty((0, 2), dtype=float),
        "flow": None,
        "geometry": None,
    }
    if match is not None:
        result.update({
            "reference_points_overlap_xy": np.asarray(
                match.get("reference_points_xy", []), dtype=float).reshape((-1, 2)),
            "moving_points_overlap_xy": np.asarray(
                match.get("moving_points_xy", []), dtype=float).reshape((-1, 2)),
            "forward_backward_error": np.asarray(
                match.get("forward_backward_error", []), dtype=float).reshape((-1,)),
        })
    if mapped is not None:
        result.update({
            "control_points_moving_xy": np.asarray(
                mapped.get("control_points_xy", []), dtype=float).reshape((-1, 2)),
            "source_points_moving_xy": np.asarray(
                mapped.get("source_points_xy", []), dtype=float).reshape((-1, 2)),
            "displacement_xy": np.asarray(
                mapped.get("displacement_xy", []), dtype=float).reshape((-1, 2)),
        })
    return result


def estimate_klt_tps_pair(
    reference_band: np.ndarray,
    reference_transform: Affine,
    moving_band: np.ndarray,
    moving_transform: Affine,
    reference_nodata: float | None,
    moving_nodata: float | None,
    params: dict[str, Any] | None,
    *,
    training_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate overlap-grid KLT controls and map them to moving-native pixels."""
    params = params or {}
    context = build_pair_overlap_context(
        reference_band, reference_transform,
        moving_band, moving_transform,
        reference_nodata, moving_nodata,
    )
    if not context.get("available"):
        return _unavailable_pair_result(context.get("failure_reason", "No geographic overlap"))
    if training_mask is not None:
        training_mask = np.asarray(training_mask, dtype=bool)
        if training_mask.shape != context["ref_overlap"].shape:
            raise ValueError("KLT training mask must match overlap image shape")
        context["training_mask_semantics"] = "reference overlap grid mask"
    try:
        match_kwargs = {} if training_mask is None else {"training_mask": training_mask}
        match = match_bidirectional_klt(
            context["ref_overlap"], context["tgt_overlap"],
            context["ref_valid"], context["tgt_valid"], params,
            **match_kwargs,
        )
        mapped = map_klt_matches_to_moving_grid(
            match["reference_points_xy"], match["moving_points_xy"],
            context["overlap_transform"], moving_transform,
        )
    except (ValueError, RuntimeError) as exc:
        result = _unavailable_pair_result(str(exc))
        result["overlap_context"] = context
        return result
    try:
        flow = build_tps_dense_flow(
            mapped["control_points_xy"], mapped["displacement_xy"],
            np.asarray(moving_band).shape, params,
        )
        geometry = inspect_tps_dense_flow(
            flow, float(params.get("klt_tps_max_shift", 50.0)),
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return _unavailable_pair_result(
            f"KLT/TPS geometry rejected: {exc}",
            context=context, match=match, mapped=mapped,
        )
    controls = mapped["control_points_xy"]
    xmin, ymin = np.min(controls, axis=0)
    xmax, ymax = np.max(controls, axis=0)
    try:
        from scipy.spatial import ConvexHull
        hull_area = float(ConvexHull(controls).volume)
        image_area = float(np.asarray(moving_band).shape[0] * np.asarray(moving_band).shape[1])
        control_hull_fraction = hull_area / image_area if image_area > 0 else None
    except Exception:
        control_hull_fraction = None
    magnitude = np.linalg.norm(flow.astype(float), axis=2)
    geometry.update({
        "control_bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
        "control_hull_fraction": control_hull_fraction,
        "flow_p50_magnitude": float(np.percentile(magnitude, 50)),
        "flow_p95_magnitude": float(np.percentile(magnitude, 95)),
    })
    return {
        "available": True,
        "failure_reason": None,
        "overlap_context": context,
        "initial_corner_count": int(match["initial_corner_count"]),
        "accepted_point_count": int(match["accepted_point_count"]),
        "reference_points_overlap_xy": np.asarray(match["reference_points_xy"], dtype=float),
        "moving_points_overlap_xy": np.asarray(match["moving_points_xy"], dtype=float),
        "forward_backward_error": np.asarray(match["forward_backward_error"], dtype=float),
        "control_points_moving_xy": mapped["control_points_xy"],
        "source_points_moving_xy": mapped["source_points_xy"],
        "displacement_xy": mapped["displacement_xy"],
        "flow": flow,
        "geometry": geometry,
    }


def build_tps_dense_flow(
    control_points_xy: np.ndarray,
    displacement_xy: np.ndarray,
    shape: tuple[int, int],
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Fit the source thin-plate spline and expand it to a dense field."""
    params = params or {}
    points = np.asarray(control_points_xy, dtype=float)
    displacement = np.asarray(displacement_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("TPS control points must have shape (N, 2)")
    if displacement.shape != points.shape or len(points) < 3:
        raise ValueError("insufficient TPS controls")
    if not np.isfinite(points).all() or not np.isfinite(displacement).all():
        raise ValueError("TPS controls must be finite")
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError("collinear TPS controls")
    h, w = (int(shape[0]), int(shape[1]))
    if h <= 0 or w <= 0:
        raise ValueError("TPS field shape must be positive")
    step = int(params.get("klt_tps_field_step", 4))
    if step < 1:
        raise ValueError("TPS field step must be positive")
    model = RBFInterpolator(
        points,
        displacement,
        kernel="thin_plate_spline",
        smoothing=float(params.get("klt_tps_smoothing", 3.0)),
        neighbors=min(int(params.get("klt_tps_neighbors", 80)), len(points)),
    )
    gy, gx = np.mgrid[0:h + step:step, 0:w + step:step]
    positions = np.column_stack([gx.ravel(), gy.ravel()])
    coarse = np.empty((len(positions), 2), dtype=np.float64)
    for start in range(0, len(positions), 20000):
        coarse[start:start + 20000] = model(positions[start:start + 20000])
    coarse = coarse.reshape(gy.shape + (2,))
    rows = np.arange(h, dtype=np.float64)
    cols = np.arange(w, dtype=np.float64)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    flow = np.empty((h, w, 2), dtype=np.float32)
    for row_start in range(0, h, 256):
        row_end = min(row_start + 256, h)
        coords = [
            row_grid[row_start:row_end] / step,
            col_grid[row_start:row_end] / step,
        ]
        for component in range(2):
            flow[row_start:row_end, :, component] = map_coordinates(
                coarse[..., component], coords, order=3, mode="nearest",
            ).astype(np.float32)
    return flow


def inspect_tps_dense_flow(flow: np.ndarray, max_shift: float) -> dict[str, Any]:
    """Apply the source Jacobian folding and maximum-shift safety gate."""
    field = np.asarray(flow)
    if field.ndim != 3 or field.shape[2] != 2:
        raise ValueError("TPS flow must have shape (H, W, 2)")
    if not np.isfinite(field).all():
        raise ValueError("TPS flow must contain only finite values")
    dx_y, dx_x = np.gradient(field[..., 0])
    dy_y, dy_x = np.gradient(field[..., 1])
    determinant = (1 + dx_x) * (1 + dy_y) - dx_y * dy_x
    fold_pixels = int(np.count_nonzero(determinant <= 0))
    max_displacement = float(np.max(np.linalg.norm(field.astype(float), axis=2)))
    result = {
        "jacobian_min": float(np.min(determinant)),
        "jacobian_max": float(np.max(determinant)),
        "fold_pixels": fold_pixels,
        "max_displacement_pixels": max_displacement,
    }
    if fold_pixels:
        raise ValueError(f"TPS flow contains fold pixels: {fold_pixels}")
    if not np.isfinite(max_shift) or max_shift <= 0:
        raise ValueError("TPS max shift must be positive and finite")
    if max_displacement > float(max_shift):
        raise ValueError(
            f"TPS flow max shift {max_displacement:.3f} exceeds {float(max_shift):.3f}"
        )
    return result


def warp_multiband_with_tps_flow(
    moving: np.ndarray,
    flow: np.ndarray,
    nodata: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp every original moving band once using output-to-source flow."""
    source = np.asarray(moving)
    field = np.asarray(flow)
    if source.ndim != 3:
        raise ValueError("multiband TPS warp requires a 3D (bands, H, W) array")
    bands, height, width = source.shape
    if field.shape != (height, width, 2):
        raise ValueError("TPS flow shape must match the moving scene")
    if min(height, width) < 4:
        raise ValueError("TPS warp requires images at least 4 pixels wide and high")
    if max(height, width) >= 32767:
        raise ValueError("OpenCV remap backend requires each dimension < 32767 in Phase I")
    if not np.isfinite(field).all():
        raise ValueError("TPS flow must contain only finite values")
    y, x = np.indices((height, width), dtype=np.float32)
    sx = x + field[..., 0].astype(np.float32)
    sy = y + field[..., 1].astype(np.float32)
    source_valid = np.all(np.isfinite(source), axis=0)
    if nodata is not None:
        if isinstance(nodata, float) and np.isnan(nodata):
            source_valid &= np.all(~np.isnan(source), axis=0)
        else:
            source_valid &= np.all(source != nodata, axis=0)
    valid = (
        (sx >= 1) & (sx <= width - 3)
        & (sy >= 1) & (sy <= height - 3)
    )
    ix = np.clip(np.floor(sx).astype(np.int32), 1, width - 3)
    iy = np.clip(np.floor(sy).astype(np.int32), 1, height - 3)
    for oy in (-1, 0, 1, 2):
        for ox in (-1, 0, 1, 2):
            valid &= source_valid[iy + oy, ix + ox]

    work_type = np.float64 if source.dtype.itemsize > 2 else np.float32
    warped_work = np.empty((bands, height, width), dtype=work_type)
    work = source.astype(work_type, copy=True)
    work[:, ~source_valid] = 0
    for band in range(bands):
        warped_work[band] = cv2.remap(
            work[band], sx, sy, cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    valid &= np.all(np.isfinite(warped_work), axis=0)
    if not np.any(valid):
        raise ValueError("no valid source samples after TPS warp")
    warped = warped_work
    if source.dtype.kind in "ui":
        limits = np.iinfo(source.dtype)
        warped = np.clip(np.rint(warped), limits.min, limits.max)
    warped = warped.astype(source.dtype)
    fill = nodata if nodata is not None else 0
    warped[:, ~valid] = fill
    return warped, valid
