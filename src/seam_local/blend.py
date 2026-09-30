"""Geometry-directed cosine blending across one monotonic pairwise seam."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .seam import SeamResult
from .source_side import SourceSideResult


@dataclass(frozen=True)
class BlendResult:
    status: str
    image: np.ndarray | None
    valid_mask: np.ndarray | None
    source_side: str | None
    blend_pixels: int
    exclusive_counts: tuple[int, int, int, int]


def blend_across_seam(
    image_a: np.ndarray,
    image_b: np.ndarray,
    seam: SeamResult,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    blend_half_width: int = 64,
    source_side: SourceSideResult | None = None,
) -> BlendResult:
    """Blend sources using exclusive support or an explicit geometry assignment.

    ``LOW`` is left of a vertical seam or above a horizontal seam. Each
    orientation requires exclusive support of both sources on opposite sides;
    conflicting or absent support is ambiguous without an explicit assignment.
    """
    a = np.asarray(image_a)
    b = np.asarray(image_b)
    va = np.asarray(valid_a, dtype=bool)
    vb = np.asarray(valid_b, dtype=bool)
    if a.ndim != 2 or 0 in a.shape or any(x.shape != a.shape for x in (b, va, vb)):
        raise ValueError("sources and masks must be same-shape nonempty 2D arrays")
    if blend_half_width < 1:
        raise ValueError("blend_half_width must be positive")
    if seam.status != "OK" or seam.orientation not in {"vertical", "horizontal"}:
        raise ValueError("a successful vertical or horizontal seam is required")
    length = a.shape[0] if seam.orientation == "vertical" else a.shape[1]
    limit = a.shape[1] if seam.orientation == "vertical" else a.shape[0]
    path = np.asarray(seam.row_col_path)
    line_axis = 0 if seam.orientation == "vertical" else 1
    transverse_axis = 1 - line_axis
    if path.shape != (length, 2) or not np.array_equal(path[:, line_axis], np.arange(length)):
        raise ValueError("seam must span image lines in order")
    centers = np.asarray(path[:, transverse_axis], dtype=np.int64)
    if (centers < 0).any() or (centers >= limit).any():
        raise ValueError("seam is outside image bounds")

    va = va & np.isfinite(a)
    vb = vb & np.isfinite(b)
    counts = np.zeros(4, dtype=np.int64)  # A-low, A-high, B-low, B-high.
    coordinates = np.arange(limit)
    for line, center in enumerate(centers):
        slot = (line, slice(None)) if seam.orientation == "vertical" else (slice(None), line)
        only_a = va[slot] & ~vb[slot]
        only_b = vb[slot] & ~va[slot]
        low, high = coordinates < center, coordinates > center
        counts += (
            np.count_nonzero(only_a & low), np.count_nonzero(only_a & high),
            np.count_nonzero(only_b & low), np.count_nonzero(only_b & high),
        )
    a_low_b_high = counts[0] > 0 and counts[3] > 0
    b_low_a_high = counts[2] > 0 and counts[1] > 0
    count_tuple = tuple(int(value) for value in counts)
    if source_side is None:
        if a_low_b_high == b_low_a_high:
            return BlendResult("AMBIGUOUS_SOURCE_SIDE", None, None, None, 0, count_tuple)
        a_on_low = bool(a_low_b_high)
    else:
        if (source_side.status not in {"EXCLUSIVE_CONTACT_RESOLVABLE", "CENTROID_RESOLVABLE"}
                or (source_side.side_1_source, source_side.side_2_source)
                not in {("A", "B"), ("B", "A")}):
            raise ValueError("explicit source-side assignment must be resolved with opposite A/B sources")
        a_on_low = source_side.side_1_source == "A"
    source_side = "A_LOW_B_HIGH" if a_on_low else "B_LOW_A_HIGH"

    result = np.full(a.shape, np.nan, dtype=np.float64)
    blend_pixels = 0
    for line, center in enumerate(centers):
        slot = (line, slice(None)) if seam.orientation == "vertical" else (slice(None), line)
        a_line = np.asarray(a[slot], dtype=np.float64)
        b_line = np.asarray(b[slot], dtype=np.float64)
        a_valid, b_valid = va[slot], vb[slot]
        output = result[slot]
        only_a, only_b = a_valid & ~b_valid, b_valid & ~a_valid
        output[only_a] = a_line[only_a]
        output[only_b] = b_line[only_b]
        both = a_valid & b_valid
        displacement = coordinates - center
        low = both & (displacement <= -blend_half_width)
        high = both & (displacement >= blend_half_width)
        output[low] = a_line[low] if a_on_low else b_line[low]
        output[high] = b_line[high] if a_on_low else a_line[high]
        band = both & (np.abs(displacement) < blend_half_width)
        if band.any():
            # Raised-cosine transition is 0 at -width and 1 at +width.
            weight_high = .5 * (1.0 - np.cos(np.pi * (displacement[band] + blend_half_width) / (2 * blend_half_width)))
            low_source, high_source = (a_line, b_line) if a_on_low else (b_line, a_line)
            with np.errstate(over="ignore", invalid="ignore"):
                output[band] = (1.0 - weight_high) * low_source[band] + weight_high * high_source[band]
            blend_pixels += int(np.count_nonzero(band))
    valid = va | vb
    if not np.isfinite(result[valid]).all():
        return BlendResult("NUMERICAL_INVALID", None, valid, source_side, blend_pixels, count_tuple)
    return BlendResult("OK", result, valid, source_side, blend_pixels, count_tuple)
