"""Geometry-only ownership resolution for a saved monotonic seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from .seam import SeamResult


@dataclass(frozen=True)
class SourceSideResult:
    status: str
    side_1_source: str | None
    side_2_source: str | None
    method: str
    confidence: float
    diagnostics: dict[str, Any]


def _side_masks(seam: SeamResult, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if seam.status != "OK" or seam.orientation not in {"vertical", "horizontal"}:
        raise ValueError("a successful vertical or horizontal seam is required")
    path = np.asarray(seam.row_col_path)
    length = shape[0] if seam.orientation == "vertical" else shape[1]
    if path.shape != (length, 2):
        raise ValueError("seam path must span every image line")
    line_axis = 0 if seam.orientation == "vertical" else 1
    transverse_axis = 1 - line_axis
    if not np.array_equal(path[:, line_axis], np.arange(length)):
        raise ValueError("seam path line coordinates must be ordered")
    centers = path[:, transverse_axis].astype(np.int64, copy=False)
    limit = shape[transverse_axis]
    if ((centers < 0) | (centers >= limit)).any():
        raise ValueError("seam path is outside image bounds")
    if seam.orientation == "vertical":
        coordinates = np.arange(shape[1])[None, :]
        return coordinates < centers[:, None], coordinates > centers[:, None]
    coordinates = np.arange(shape[0])[:, None]
    return coordinates < centers[None, :], coordinates > centers[None, :]


def _centroid_values(footprint: Any) -> tuple[float, float]:
    centroid = footprint.centroid
    return float(centroid.x), float(centroid.y)


def _centroid_axis(footprint: Any, orientation: str) -> float:
    x, y = _centroid_values(footprint)
    # Raster rows increase downward while projected world y increases upward.
    return x if orientation == "vertical" else -y


def resolve_source_sides(
    seam: SeamResult,
    overlap_mask: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    scene_a_footprint: Any,
    scene_b_footprint: Any,
) -> SourceSideResult:
    """Resolve side ownership from seam geometry and registered valid support.

    Side 1 is LEFT for a vertical seam and TOP for a horizontal seam; side 2
    is the corresponding RIGHT/BOTTOM region.  No radiometric metric or scene
    identifier is consulted.
    """
    overlap = np.asarray(overlap_mask, dtype=bool)
    va = np.asarray(valid_a, dtype=bool)
    vb = np.asarray(valid_b, dtype=bool)
    if overlap.ndim != 2 or any(x.shape != overlap.shape for x in (va, vb)):
        raise ValueError("overlap and validity masks must be same-shape 2D arrays")
    side1, side2 = _side_masks(seam, overlap.shape)
    # Contact evidence must be adjacent to the shared-valid seam corridor;
    # disconnected exclusive islands elsewhere in the crop are irrelevant.
    side1 &= overlap
    side2 &= overlap
    # The overlap mask identifies the shared corridor; exclusive support is
    # deliberately taken from the registered footprints outside that corridor.
    exclusive_a = va & ~vb
    exclusive_b = vb & ~va
    try:
        raw_centroid_a = _centroid_values(scene_a_footprint)
        raw_centroid_b = _centroid_values(scene_b_footprint)
        centroid_a = _centroid_axis(scene_a_footprint, seam.orientation)
        centroid_b = _centroid_axis(scene_b_footprint, seam.orientation)
    except (AttributeError, TypeError, ValueError):
        raw_centroid_a = raw_centroid_b = (float("nan"), float("nan"))
        centroid_a = centroid_b = float("nan")
    separation = abs(centroid_a - centroid_b) if np.isfinite([centroid_a, centroid_b]).all() else float("nan")
    # Count all one-pixel contacts between exclusive support boundaries and
    # the seam-side shared corridor; a stray crop-edge pixel must not suppress
    # stronger interior boundary evidence.
    structure = np.ones((3, 3), dtype=bool)
    dilated_side1 = binary_dilation(side1, structure)
    dilated_side2 = binary_dilation(side2, structure)
    exclusive_boundary_a = exclusive_a & ~binary_erosion(exclusive_a, structure, border_value=0)
    exclusive_boundary_b = exclusive_b & ~binary_erosion(exclusive_b, structure, border_value=0)
    contact_a_side1 = int(np.count_nonzero(exclusive_boundary_a & dilated_side1))
    contact_a_side2 = int(np.count_nonzero(exclusive_boundary_a & dilated_side2))
    contact_b_side1 = int(np.count_nonzero(exclusive_boundary_b & dilated_side1))
    contact_b_side2 = int(np.count_nonzero(exclusive_boundary_b & dilated_side2))
    contact_method = "EXCLUSIVE_BOUNDARY_CONTACT"
    score_ab = contact_a_side1 + contact_b_side2
    score_ba = contact_b_side1 + contact_a_side2
    diagnostics: dict[str, Any] = {
        "side_1_pixel_count": int(np.count_nonzero(side1)),
        "side_2_pixel_count": int(np.count_nonzero(side2)),
        "exclusive_contact_a_side1": contact_a_side1,
        "exclusive_contact_a_side2": contact_a_side2,
        "exclusive_contact_b_side1": contact_b_side1,
        "exclusive_contact_b_side2": contact_b_side2,
        "assignment_score_ab": int(score_ab),
        "assignment_score_ba": int(score_ba),
        "contact_method": contact_method,
        "centroid_a": raw_centroid_a,
        "centroid_b": raw_centroid_b,
        "centroid_projection_a": centroid_a,
        "centroid_projection_b": centroid_b,
        "projection_separation": separation,
    }
    try:
        area_a = float(scene_a_footprint.area)
        area_b = float(scene_b_footprint.area)
        intersection_area = float(scene_a_footprint.intersection(scene_b_footprint).area)
        containment_ratio = intersection_area / max(min(area_a, area_b), 1e-12)
        identical = bool(scene_a_footprint.equals(scene_b_footprint))
        contains = bool(
            scene_a_footprint.contains(scene_b_footprint)
            or scene_b_footprint.contains(scene_a_footprint)
            or containment_ratio >= 0.99
        )
    except (AttributeError, TypeError, ValueError):
        containment_ratio = float("nan")
        identical = False
        contains = False
    diagnostics["nested_or_contained_footprints"] = contains
    diagnostics["identical_footprints"] = identical
    diagnostics["containment_ratio"] = containment_ratio
    if identical:
        diagnostics["selected_assignment"] = None
        return SourceSideResult("REQUIRES_MULTISCENE_LABELING", None, None, "UNRESOLVED_GEOMETRY", 0.0, diagnostics)
    if contains:
        diagnostics["selected_assignment"] = None
        return SourceSideResult("REQUIRES_MULTISCENE_LABELING", None, None, "CONTAINMENT_OR_NESTED_FOOTPRINT", 0.0, diagnostics)
    if score_ab != score_ba and max(score_ab, score_ba) > 0:
        selected = "AB" if score_ab > score_ba else "BA"
        margin = abs(score_ab - score_ba) / max(score_ab, score_ba)
        diagnostics["selected_assignment"] = selected
        return SourceSideResult(
            "EXCLUSIVE_CONTACT_RESOLVABLE",
            "A" if selected == "AB" else "B",
            "B" if selected == "AB" else "A",
            "EXCLUSIVE_BOUNDARY_CONTACT",
            float(margin),
            diagnostics,
        )

    try:
        bounds_a = scene_a_footprint.bounds
        bounds_b = scene_b_footprint.bounds
        span = max(
            (bounds_a[2] - bounds_a[0]) if seam.orientation == "vertical" else (bounds_a[3] - bounds_a[1]),
            (bounds_b[2] - bounds_b[0]) if seam.orientation == "vertical" else (bounds_b[3] - bounds_b[1]),
            1.0,
        )
    except (AttributeError, TypeError, ValueError):
        span = 1.0
    projection_tolerance = max(1e-6, 0.01 * float(span))
    diagnostics["projection_tolerance"] = projection_tolerance
    if np.isfinite(separation) and separation > projection_tolerance:
        side_centers = np.where(side1)[1 if seam.orientation == "vertical" else 0], np.where(side2)[1 if seam.orientation == "vertical" else 0]
        if all(values.size for values in side_centers):
            side1_axis = float(np.mean(side_centers[0]))
            side2_axis = float(np.mean(side_centers[1]))
            if (centroid_a < centroid_b) == (side1_axis < side2_axis):
                diagnostics["selected_assignment"] = "AB"
                return SourceSideResult("CENTROID_RESOLVABLE", "A", "B", "FOOTPRINT_CENTROID_PROJECTION", 1.0, diagnostics)
            if (centroid_b < centroid_a) == (side1_axis < side2_axis):
                diagnostics["selected_assignment"] = "BA"
                return SourceSideResult("CENTROID_RESOLVABLE", "B", "A", "FOOTPRINT_CENTROID_PROJECTION", 1.0, diagnostics)

    diagnostics["selected_assignment"] = None
    return SourceSideResult("REQUIRES_MULTISCENE_LABELING", None, None, "UNRESOLVED_GEOMETRY", 0.0, diagnostics)
