"""Symmetric local moment correction in the fixed Task13A seam corridor.

The two sources move toward the same segment mean and standard deviation.
Coefficients are interpolated along seam arc length; a cosine weight returns
both sources to their original BAGRN values at the corridor boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .seam import SeamResult


@dataclass(frozen=True)
class LocalMomentSegment:
    index: int
    center_arc: float
    valid_pair_pixels: int
    a_a: float
    b_a: float
    a_b: float
    b_b: float
    fallback_from: int | None = None


@dataclass(frozen=True)
class LocalMomentEstimate:
    status: str
    segments: tuple[LocalMomentSegment, ...]
    gain_min: float | None
    gain_max: float | None
    b_min: float | None
    b_max: float | None
    line_arcs: np.ndarray


@dataclass(frozen=True)
class LocalMomentResult:
    status: str
    corrected_a: np.ndarray
    corrected_b: np.ndarray
    segments: tuple[LocalMomentSegment, ...]
    gain_min: float | None
    gain_max: float | None
    b_min: float | None
    b_max: float | None


def _seam_lines(seam: SeamResult, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return transverse coordinates and cumulative arc length for every line."""
    if seam.status != "OK" or seam.orientation not in {"vertical", "horizontal"}:
        raise ValueError("a successful vertical or horizontal seam is required")
    path = np.asarray(seam.row_col_path)
    length = shape[0] if seam.orientation == "vertical" else shape[1]
    transverse_limit = shape[1] if seam.orientation == "vertical" else shape[0]
    if path.shape != (length, 2):
        raise ValueError("seam path must contain one coordinate per image line")
    line_axis = 0 if seam.orientation == "vertical" else 1
    transverse_axis = 1 - line_axis
    if not np.array_equal(path[:, line_axis], np.arange(length)):
        raise ValueError("seam path must span image lines in order")
    transverse = np.asarray(path[:, transverse_axis], dtype=np.int64)
    if (transverse < 0).any() or (transverse >= transverse_limit).any():
        raise ValueError("seam path is outside image bounds")
    arcs = np.zeros(length, dtype=np.float64)
    if length > 1:
        arcs[1:] = np.cumsum(np.hypot(1.0, np.diff(transverse)))
    return transverse, arcs


def _clipped_moments(values: np.ndarray, eps: float) -> tuple[float, float] | None:
    low, high = np.percentile(values, (1, 99))
    clipped = values[(values >= low) & (values <= high)]
    if clipped.size == 0:
        return None
    mean = float(np.mean(clipped))
    sigma = float(np.std(clipped))
    if not np.isfinite(mean) or not np.isfinite(sigma) or sigma <= eps:
        return None
    return mean, sigma


def estimate_seam_segment_moments(
    a: np.ndarray,
    b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    seam: SeamResult,
    *,
    half_width: int = 128,
    segment_length: int = 256,
    min_valid_pairs: int = 4096,
    eps: float = 1e-6,
) -> LocalMomentEstimate:
    """Estimate independent source gains toward symmetric clipped moments.

    Only jointly valid pixels within the initial seam corridor inform a
    segment. Missing or degenerate segments borrow the nearest solvable
    segment's coefficients; a completely unsolvable seam is explicit.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    valid_a = np.asarray(valid_a, dtype=bool)
    valid_b = np.asarray(valid_b, dtype=bool)
    if a.ndim != 2 or 0 in a.shape or any(x.shape != a.shape for x in (b, valid_a, valid_b)):
        raise ValueError("sources and validity masks must be same-shape nonempty 2D arrays")
    if half_width < 1 or segment_length < 1 or min_valid_pairs < 1 or eps <= 0:
        raise ValueError("corridor, segment, sample count, and epsilon must be positive")
    transverse, arcs = _seam_lines(seam, a.shape)
    indices = np.floor(arcs / segment_length).astype(np.int64)
    count = int(indices[-1]) + 1
    samples_a: list[list[np.ndarray]] = [[] for _ in range(count)]
    samples_b: list[list[np.ndarray]] = [[] for _ in range(count)]
    positions: list[list[float]] = [[] for _ in range(count)]
    for line, center in enumerate(transverse):
        segment = int(indices[line])
        positions[segment].append(float(arcs[line]))
        left = max(0, int(center) - half_width)
        right = min(a.shape[1 if seam.orientation == "vertical" else 0], int(center) + half_width + 1)
        if seam.orientation == "vertical":
            piece_a, piece_b = a[line, left:right], b[line, left:right]
            joint = valid_a[line, left:right] & valid_b[line, left:right]
        else:
            piece_a, piece_b = a[left:right, line], b[left:right, line]
            joint = valid_a[left:right, line] & valid_b[left:right, line]
        joint &= np.isfinite(piece_a) & np.isfinite(piece_b)
        if joint.any():
            samples_a[segment].append(np.asarray(piece_a[joint], dtype=np.float64))
            samples_b[segment].append(np.asarray(piece_b[joint], dtype=np.float64))

    segments: list[LocalMomentSegment] = []
    solvable: list[int] = []
    for index in range(count):
        center_arc = float(np.mean(positions[index])) if positions[index] else float((index + .5) * segment_length)
        source_a = np.concatenate(samples_a[index]) if samples_a[index] else np.empty(0)
        source_b = np.concatenate(samples_b[index]) if samples_b[index] else np.empty(0)
        n = int(source_a.size)
        coefficients: tuple[float, float, float, float] | None = None
        if n >= min_valid_pairs:
            moments_a = _clipped_moments(source_a, eps)
            moments_b = _clipped_moments(source_b, eps)
            if moments_a is not None and moments_b is not None:
                mean_a, sigma_a = moments_a
                mean_b, sigma_b = moments_b
                target_mean = .5 * (mean_a + mean_b)
                target_sigma = .5 * (sigma_a + sigma_b)
                gain_a, gain_b = target_sigma / sigma_a, target_sigma / sigma_b
                coefficients = (
                    gain_a, target_mean - gain_a * mean_a,
                    gain_b, target_mean - gain_b * mean_b,
                )
                if not np.isfinite(coefficients).all():
                    coefficients = None
        if coefficients is None:
            segments.append(LocalMomentSegment(index, center_arc, n, np.nan, np.nan, np.nan, np.nan))
        else:
            segments.append(LocalMomentSegment(index, center_arc, n, *coefficients))
            solvable.append(index)

    if not solvable:
        return LocalMomentEstimate("LOCAL_MOMENT_UNAVAILABLE", tuple(), None, None, None, None, arcs)
    for index, segment in enumerate(segments):
        if index in solvable:
            continue
        nearest = min(solvable, key=lambda candidate: (abs(segment.center_arc - segments[candidate].center_arc), candidate))
        donor = segments[nearest]
        segments[index] = replace(
            segment, a_a=donor.a_a, b_a=donor.b_a,
            a_b=donor.a_b, b_b=donor.b_b, fallback_from=nearest,
        )
    gains = np.array([(segment.a_a, segment.a_b) for segment in segments])
    offsets = np.array([(segment.b_a, segment.b_b) for segment in segments])
    gain_min, gain_max = float(gains.min()), float(gains.max())
    b_min, b_max = float(offsets.min()), float(offsets.max())
    status = "UNSTABLE_LOCAL_GAIN" if gain_min < .5 or gain_max > 2.0 else "OK"
    return LocalMomentEstimate(status, tuple(segments), gain_min, gain_max, b_min, b_max, arcs)


def apply_seam_local_correction(
    a: np.ndarray,
    b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    seam: SeamResult,
    *,
    half_width: int = 128,
    segment_length: int = 256,
    min_valid_pairs: int = 4096,
    eps: float = 1e-6,
) -> LocalMomentResult:
    """Apply line-interpolated local affine corrections with a cosine taper."""
    estimate = estimate_seam_segment_moments(
        a, b, valid_a, valid_b, seam,
        half_width=half_width, segment_length=segment_length,
        min_valid_pairs=min_valid_pairs, eps=eps,
    )
    corrected_a = np.asarray(a, dtype=np.float64).copy()
    corrected_b = np.asarray(b, dtype=np.float64).copy()
    if estimate.status == "LOCAL_MOMENT_UNAVAILABLE":
        return LocalMomentResult(estimate.status, corrected_a, corrected_b, estimate.segments, None, None, None, None)
    transverse, arcs = _seam_lines(seam, corrected_a.shape)
    centers = np.array([segment.center_arc for segment in estimate.segments])
    coefficients = np.array([
        (segment.a_a, segment.b_a, segment.a_b, segment.b_b)
        for segment in estimate.segments
    ])
    interpolated = np.column_stack([
        np.interp(arcs, centers, coefficients[:, column]) for column in range(4)
    ])
    valid_a = np.asarray(valid_a, dtype=bool)
    valid_b = np.asarray(valid_b, dtype=bool)
    for line, center in enumerate(transverse):
        left = max(0, int(center) - half_width)
        right = min(corrected_a.shape[1 if seam.orientation == "vertical" else 0], int(center) + half_width + 1)
        transverse_pixels = np.arange(left, right)
        weights = .5 * (1.0 + np.cos(np.pi * np.abs(transverse_pixels - center) / half_width))
        if seam.orientation == "vertical":
            slot = (line, slice(left, right))
        else:
            slot = (slice(left, right), line)
        for original, corrected, valid, gain, offset in (
            (a, corrected_a, valid_a, interpolated[line, 0], interpolated[line, 1]),
            (b, corrected_b, valid_b, interpolated[line, 2], interpolated[line, 3]),
        ):
            source = np.asarray(original[slot], dtype=np.float64)
            active = valid[slot] & np.isfinite(source)
            if active.any():
                with np.errstate(over="ignore", invalid="ignore"):
                    candidate = source[active] + weights[active] * ((gain - 1.0) * source[active] + offset)
                if not np.isfinite(candidate).all():
                    return LocalMomentResult(
                        "NUMERICAL_INVALID",
                        np.asarray(a, dtype=np.float64).copy(),
                        np.asarray(b, dtype=np.float64).copy(),
                        estimate.segments, estimate.gain_min, estimate.gain_max,
                        estimate.b_min, estimate.b_max,
                    )
                corrected[slot][active] = candidate
    return LocalMomentResult(
        estimate.status, corrected_a, corrected_b,
        estimate.segments, estimate.gain_min, estimate.gain_max,
        estimate.b_min, estimate.b_max,
    )
