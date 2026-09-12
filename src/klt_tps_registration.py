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
    failure_diagnostics: dict[str, Any] | None = None,
    diagnostic_arrays: dict[str, np.ndarray] | None = None,
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
    if failure_diagnostics is not None:
        result["failure_diagnostics"] = failure_diagnostics
    if diagnostic_arrays is not None:
        result["_failure_diagnostic_arrays"] = diagnostic_arrays
    return result


def _build_tps_control_support_diagnostics(
    mapped: dict[str, np.ndarray],
    context: dict[str, Any],
    shape: tuple[int, int],
) -> dict[str, Any]:
    """Build compact control/hull/overlap evidence for a TPS gate failure."""
    controls = np.asarray(mapped["control_points_xy"], dtype=float)
    hull = build_control_hull_mask(controls, shape)
    overlap_bbox = build_target_overlap_bbox_mask(context, shape)
    if len(controls):
        control_bbox = [
            float(np.min(controls[:, 0])), float(np.min(controls[:, 1])),
            float(np.max(controls[:, 0])), float(np.max(controls[:, 1])),
        ]
    else:
        control_bbox = []
    return {
        "control_bbox": control_bbox,
        "control_hull_available": bool(hull["available"]),
        "control_hull_vertices_xy": np.asarray(
            hull["vertices_xy"], dtype=float).tolist(),
        "control_hull_pixels": int(hull["pixel_count"]),
        "control_hull_fraction": float(hull["fraction"]),
        "target_overlap_window": (
            list(overlap_bbox["window"])
            if overlap_bbox["window"] is not None else None
        ),
        "target_overlap_bbox_pixels": int(overlap_bbox["pixel_count"]),
        "target_overlap_bbox_fraction": float(overlap_bbox["fraction"]),
        "_hull_mask": hull["mask"],
        "_overlap_bbox_mask": overlap_bbox["mask"],
    }


def _build_tps_failure_diagnostics(
    stage: str,
    match: dict[str, Any],
    mapped: dict[str, np.ndarray],
    context: dict[str, Any],
    shape: tuple[int, int],
    *,
    analysis: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return JSON-safe pre-gate evidence and private arrays for artifact writers."""
    support = _build_tps_control_support_diagnostics(mapped, context, shape)
    hull_mask = support.pop("_hull_mask")
    overlap_bbox_mask = support.pop("_overlap_bbox_mask")
    diagnostics = {
        "stage": str(stage),
        "accepted_point_count": int(match.get("accepted_point_count", 0)),
        "control_displacement": summarize_control_displacements(
            mapped["displacement_xy"]
        ),
        "control_support": support,
        "field_displacement": None,
        "jacobian": None,
        "fold_support": None,
    }
    arrays: dict[str, np.ndarray] = {}
    if analysis is not None:
        summary = analysis["summary"]
        diagnostics["field_displacement"] = {
            "min": summary["displacement_min"],
            "p01": summary["displacement_p01"],
            "p05": summary["displacement_p05"],
            "median": summary["displacement_median"],
            "p95": summary["displacement_p95"],
            "p99": summary["displacement_p99"],
            "max": summary["displacement_max"],
        }
        diagnostics["jacobian"] = {
            "min": summary["jacobian_min"],
            "p01": summary["jacobian_p01"],
            "p05": summary["jacobian_p05"],
            "median": summary["jacobian_median"],
            "p95": summary["jacobian_p95"],
            "p99": summary["jacobian_p99"],
            "max": summary["jacobian_max"],
        }
        diagnostics["fold_support"] = summarize_fold_support(
            analysis["fold_mask"], hull_mask, overlap_bbox_mask,
        )
        arrays.update({
            "jacobian_determinant": analysis["jacobian_determinant"],
            "fold_mask": analysis["fold_mask"],
        })
    return diagnostics, arrays


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
    moving_shape = tuple(np.asarray(moving_band).shape)
    try:
        flow = build_tps_dense_flow(
            mapped["control_points_xy"], mapped["displacement_xy"],
            moving_shape, params,
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        failure_diagnostics, diagnostic_arrays = _build_tps_failure_diagnostics(
            "tps_fit", match, mapped, context, moving_shape,
        )
        return _unavailable_pair_result(
            f"KLT/TPS geometry rejected: {exc}",
            context=context, match=match, mapped=mapped,
            failure_diagnostics=failure_diagnostics,
            diagnostic_arrays=diagnostic_arrays,
        )
    analysis = analyze_tps_dense_flow(flow)
    failure_diagnostics, diagnostic_arrays = _build_tps_failure_diagnostics(
        "tps_geometry_gate", match, mapped, context, moving_shape,
        analysis=analysis,
    )
    diagnostic_arrays["flow"] = flow
    try:
        geometry = inspect_tps_dense_flow(
            flow, float(params.get("klt_tps_max_shift", 50.0)),
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return _unavailable_pair_result(
            f"KLT/TPS geometry rejected: {exc}",
            context=context, match=match, mapped=mapped,
            failure_diagnostics=failure_diagnostics,
            diagnostic_arrays=diagnostic_arrays,
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


def summarize_finite_values(values: np.ndarray) -> dict[str, float | int | None]:
    """Summarize finite values without filtering or altering the source array."""
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "p01": None,
            "p05": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def summarize_control_displacements(
    displacement_xy: np.ndarray,
) -> dict[str, Any]:
    """Return compact displacement statistics without rejecting any controls."""
    displacement = np.asarray(displacement_xy, dtype=float)
    if displacement.ndim != 2 or displacement.shape[1] != 2:
        raise ValueError("control displacement must have shape (N, 2)")
    dx = displacement[:, 0]
    dy = displacement[:, 1]
    magnitude = np.hypot(dx, dy)
    return {
        "count": int(len(displacement)),
        "dx": summarize_finite_values(dx),
        "dy": summarize_finite_values(dy),
        "magnitude": summarize_finite_values(magnitude),
    }


def analyze_tps_dense_flow(flow: np.ndarray) -> dict[str, Any]:
    """Analyze a dense TPS field before applying the safety gate."""
    field = np.asarray(flow)
    if field.ndim != 3 or field.shape[2] != 2:
        raise ValueError("TPS flow must have shape (H, W, 2)")
    if field.shape[0] == 0 or field.shape[1] == 0:
        raise ValueError("TPS flow must have a non-empty spatial shape")
    if not np.isfinite(field).all():
        raise ValueError("TPS flow must contain only finite values")
    dx_y, dx_x = np.gradient(field[..., 0])
    dy_y, dy_x = np.gradient(field[..., 1])
    determinant = (1 + dx_x) * (1 + dy_y) - dx_y * dy_x
    fold_mask = determinant <= 0
    magnitude = np.linalg.norm(field.astype(float), axis=2)
    jacobian = summarize_finite_values(determinant)
    displacement = summarize_finite_values(magnitude)
    summary = {
        "jacobian_min": jacobian["min"],
        "jacobian_p01": jacobian["p01"],
        "jacobian_p05": jacobian["p05"],
        "jacobian_median": jacobian["median"],
        "jacobian_p95": jacobian["p95"],
        "jacobian_p99": jacobian["p99"],
        "jacobian_max": jacobian["max"],
        "fold_pixels": int(np.count_nonzero(fold_mask)),
        "fold_fraction": float(np.mean(fold_mask)),
        "displacement_min": displacement["min"],
        "displacement_p01": displacement["p01"],
        "displacement_p05": displacement["p05"],
        "displacement_median": displacement["median"],
        "displacement_p95": displacement["p95"],
        "displacement_p99": displacement["p99"],
        "displacement_max": displacement["max"],
    }
    return {
        "jacobian_determinant": determinant,
        "fold_mask": fold_mask,
        "magnitude": magnitude,
        "summary": summary,
    }


def build_control_hull_mask(
    control_points_xy: np.ndarray,
    shape: tuple[int, int],
) -> dict[str, Any]:
    """Rasterize the moving-native control convex hull for diagnostics only."""
    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("control hull shape must be positive")
    points = np.asarray(control_points_xy, dtype=float)
    empty = np.zeros((height, width), dtype=bool)
    if (
        points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
        or not np.isfinite(points).all()
        or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2
    ):
        return {
            "available": False,
            "mask": empty,
            "vertices_xy": np.empty((0, 2), dtype=float),
            "pixel_count": 0,
            "fraction": 0.0,
        }
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    vertices = np.rint(hull).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, vertices, 1)
    mask = mask.astype(bool)
    pixel_count = int(np.count_nonzero(mask))
    return {
        "available": True,
        "mask": mask,
        "vertices_xy": hull.astype(float),
        "pixel_count": pixel_count,
        "fraction": float(pixel_count / (height * width)),
    }


def build_target_overlap_bbox_mask(
    overlap_context: dict[str, Any],
    shape: tuple[int, int],
) -> dict[str, Any]:
    """Rasterize the geographic target overlap window, without valid-data semantics."""
    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("target overlap shape must be positive")
    window = overlap_context.get("tgt_window")
    empty = np.zeros((height, width), dtype=bool)
    if window is None or len(window) != 4:
        return {
            "available": False, "mask": empty, "window": None,
            "pixel_count": 0, "fraction": 0.0,
        }
    row_start, row_end, col_start, col_end = (int(value) for value in window)
    clipped_start = max(0, row_start)
    clipped_end = min(height, row_end)
    clipped_col_start = max(0, col_start)
    clipped_col_end = min(width, col_end)
    if clipped_end <= clipped_start or clipped_col_end <= clipped_col_start:
        return {
            "available": False, "mask": empty,
            "window": (row_start, row_end, col_start, col_end),
            "pixel_count": 0, "fraction": 0.0,
        }
    empty[clipped_start:clipped_end, clipped_col_start:clipped_col_end] = True
    pixel_count = int(np.count_nonzero(empty))
    return {
        "available": True,
        "mask": empty,
        "window": (row_start, row_end, col_start, col_end),
        "pixel_count": pixel_count,
        "fraction": float(pixel_count / (height * width)),
    }


def summarize_fold_support(
    fold_mask: np.ndarray,
    control_hull_mask: np.ndarray | None,
    overlap_bbox_mask: np.ndarray | None,
) -> dict[str, Any]:
    """Summarize where fold pixels fall relative to diagnostic support regions."""
    folds = np.asarray(fold_mask, dtype=bool)
    if folds.ndim != 2:
        raise ValueError("fold mask must be a 2D array")
    result = {
        "fold_pixels_total": int(np.count_nonzero(folds)),
        "fold_fraction_total": float(np.mean(folds)),
    }
    for prefix, region in (
        ("control_hull", control_hull_mask),
        ("overlap_bbox", overlap_bbox_mask),
    ):
        if region is None:
            continue
        support = np.asarray(region, dtype=bool)
        if support.shape != folds.shape:
            raise ValueError(f"{prefix} mask must match fold mask shape")
        inside = folds & support
        outside = folds & ~support
        support_pixels = int(np.count_nonzero(support))
        outside_pixels = int(support.size - support_pixels)
        result.update({
            f"{prefix}_pixels": support_pixels,
            f"fold_pixels_inside_{prefix}": int(np.count_nonzero(inside)),
            f"fold_pixels_outside_{prefix}": int(np.count_nonzero(outside)),
            f"fold_fraction_inside_{prefix}": (
                float(np.count_nonzero(inside) / support_pixels)
                if support_pixels else None
            ),
            f"fold_fraction_outside_{prefix}": (
                float(np.count_nonzero(outside) / outside_pixels)
                if outside_pixels else None
            ),
            f"fold_share_inside_{prefix}": (
                float(np.count_nonzero(inside) / np.count_nonzero(folds))
                if np.count_nonzero(folds) else None
            ),
        })
    return result


def inspect_tps_dense_flow(flow: np.ndarray, max_shift: float) -> dict[str, Any]:
    """Apply the source Jacobian folding and maximum-shift safety gate."""
    analysis = analyze_tps_dense_flow(flow)
    summary = analysis["summary"]
    fold_pixels = int(summary["fold_pixels"])
    max_displacement = float(summary["displacement_max"])
    result = {
        "jacobian_min": float(summary["jacobian_min"]),
        "jacobian_max": float(summary["jacobian_max"]),
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
