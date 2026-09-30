"""Geometry-only source labeling for the frozen Task13B protocol.

The functions in this module deliberately operate on arrays rather than on
files.  A replay script can therefore build fields a window at a time while
the small synthetic tests exercise exactly the same arithmetic.  Pairwise
fields contain signed votes (positive means scene ``a`` and negative means
scene ``b``); the global resolver normalizes those votes by the number of
available resolved comparisons for *each* candidate scene.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt


LABEL_UNRESOLVED = -1
TIE_EPSILON = 1.0e-6
SEAM_CONFIDENCE_HALF_WIDTH = 64.0


@dataclass(frozen=True)
class PairwisePreferenceField:
    """Signed pairwise preference on a common raster grid.

    ``vote`` is positive when scene ``scene_a`` is preferred, negative when
    ``scene_b`` is preferred, and zero where the pair has no usable resolved
    preference.  ``available`` marks jointly valid pixels from a resolved
    pair, including a zero vote at the seam center.  ``confidence`` is the
    specified ``min(distance_to_seam / 64, 1)`` factor.
    """

    scene_a: int
    scene_b: int
    vote: np.ndarray
    available: np.ndarray
    confidence: np.ndarray
    orientation: str

    @property
    def preferred(self) -> np.ndarray:
        """Return +1 for ``scene_a``, -1 for ``scene_b``, 0 for no vote."""

        return np.sign(self.vote).astype(np.int8, copy=False)

    @property
    def valid(self) -> np.ndarray:
        """Alias retained for callers that call pair support ``valid``."""

        return self.available


@dataclass(frozen=True)
class MultiSceneLabelResult:
    """Result and diagnostics of geometry-only multiscene label selection."""

    labels: np.ndarray
    scores: np.ndarray
    comparison_counts: np.ndarray
    normalized_interiority: np.ndarray
    diagnostics: dict[str, Any]

    @property
    def label_map(self) -> np.ndarray:
        return self.labels


def _as_path_and_orientation(seam: Any, orientation: str | None) -> tuple[np.ndarray, str]:
    if hasattr(seam, "row_col_path"):
        path = np.asarray(seam.row_col_path)
        seam_orientation = str(seam.orientation)
        if getattr(seam, "status", "OK") != "OK":
            raise ValueError("a successful seam is required")
        if orientation is not None and orientation != seam_orientation:
            raise ValueError("orientation disagrees with seam result")
        orientation = seam_orientation
    else:
        path = np.asarray(seam)
        # Synthetic callers commonly provide only the row/column path.  Infer
        # its orientation from the axis that spans the raster; real seam
        # records still carry an explicit orientation and are validated below.
        if orientation is None and path.ndim == 2 and path.shape[1] == 2:
            rows = path[:, 0].astype(np.int64, copy=False)
            cols = path[:, 1].astype(np.int64, copy=False)
            if np.array_equal(rows, np.arange(len(rows))):
                orientation = "vertical"
            elif np.array_equal(cols, np.arange(len(cols))):
                orientation = "horizontal"
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be vertical or horizontal")
    if path.ndim != 2 or path.shape[1] != 2 or path.size == 0:
        raise ValueError("seam path must be an (N, 2) array")
    return path, orientation


def _side_assignment(side_1_source: Any) -> bool:
    """Return whether side 1 (left/top) belongs to scene ``a``."""

    if isinstance(side_1_source, (tuple, list)) and len(side_1_source) == 2:
        value = str(side_1_source[0]).upper()
    else:
        value = str(side_1_source).upper()
        if value in {"A_LOW_B_HIGH", "A_LEFT_B_RIGHT", "A_TOP_B_BOTTOM", "A"}:
            return True
        if value in {"B_LOW_A_HIGH", "B_LEFT_A_RIGHT", "B_TOP_A_BOTTOM", "B"}:
            return False
        # A side label such as "side_1_source=A" is convenient for CSV rows.
        if "A" in value and "B" in value:
            return value.startswith("A") or "A_LOW" in value or "A_LEFT" in value or "A_TOP" in value
        value = value[:1]
    if value == "A":
        return True
    if value == "B":
        return False
    raise ValueError("side_1_source must resolve to A or B")


def build_pairwise_preference_field(
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    seam: Any,
    side_1_source: Any = None,
    *,
    orientation: str | None = None,
    scene_a: int = 0,
    scene_b: int = 1,
    seam_half_width: float = SEAM_CONFIDENCE_HALF_WIDTH,
    source_side: Any = None,
) -> PairwisePreferenceField:
    """Build one signed preference field from a resolved saved seam.

    The confidence factor follows the Task13B rule exactly: ``m =
    min(abs(distance_to_seam) / 64, 1)``.  The seam center itself has no side
    preference (``m == 0``), and pixels outside the joint valid support are
    unavailable.  Passing ``side_1_source=None`` is an explicit unresolved
    pair and produces an all-zero, unavailable field.
    """

    if side_1_source is None and source_side is not None:
        side_1_source = source_side
    va = np.asarray(valid_a, dtype=bool)
    vb = np.asarray(valid_b, dtype=bool)
    if va.ndim != 2 or vb.shape != va.shape:
        raise ValueError("valid_a and valid_b must be same-shape 2D arrays")
    path, orientation = _as_path_and_orientation(seam, orientation)
    length = va.shape[0] if orientation == "vertical" else va.shape[1]
    limit = va.shape[1] if orientation == "vertical" else va.shape[0]
    if path.shape != (length, 2):
        raise ValueError("seam path must span every line of the input grid")
    line_axis = 0 if orientation == "vertical" else 1
    transverse_axis = 1 - line_axis
    if not np.array_equal(path[:, line_axis].astype(np.int64), np.arange(length)):
        raise ValueError("seam path line coordinates must be ordered")
    # The frozen Task13B replay computes seam-center distances in float32.
    # Keeping this dtype explicit is part of replay equivalence: a float64
    # intermediate can move pixels exactly at the 1e-6 tie threshold.
    centers = path[:, transverse_axis].astype(np.float32)
    if np.any((centers < 0) | (centers >= limit)):
        raise ValueError("seam path is outside input grid")
    # Match the frozen raster replay, which subtracts float32 seam centers
    # from NumPy's default integer coordinate array before the final float32
    # vote cast.
    coordinates = np.arange(limit)
    distance = np.abs(coordinates[None, :] - centers[:, None])
    confidence_oriented = np.minimum(distance / float(seam_half_width), 1.0).astype(np.float32)
    side1_oriented = coordinates[None, :] < centers[:, None]
    side2_oriented = coordinates[None, :] > centers[:, None]
    side_a_is_side1 = _side_assignment(side_1_source) if side_1_source is not None else True
    sign_oriented = np.where(side1_oriented, 1.0 if side_a_is_side1 else -1.0,
                             np.where(side2_oriented, -1.0 if side_a_is_side1 else 1.0, 0.0))
    if orientation == "vertical":
        confidence = confidence_oriented
        signed = sign_oriented
    else:
        confidence = confidence_oriented.T
        signed = sign_oriented.T
    available = va & vb & np.isfinite(signed)
    if side_1_source is None:
        available[:] = False
        signed = np.zeros_like(signed)
        confidence = np.zeros_like(confidence)
    vote = np.where(available, signed * confidence, 0.0).astype(np.float32)
    return PairwisePreferenceField(
        scene_a=int(scene_a),
        scene_b=int(scene_b),
        vote=vote,
        available=available,
        confidence=np.where(available, confidence, 0.0).astype(np.float32),
        orientation=orientation,
    )


def _coerce_field(key: tuple[int, int], value: Any, shape: tuple[int, int]) -> PairwisePreferenceField:
    """Accept the dataclass plus simple mapping/array forms for replay code."""

    a, b = (int(key[0]), int(key[1]))
    if isinstance(value, PairwisePreferenceField):
        field = value
        if (field.scene_a, field.scene_b) != (a, b):
            raise ValueError("pair key disagrees with PairwisePreferenceField scene ids")
        if field.vote.shape != shape or field.available.shape != shape:
            raise ValueError("pairwise field shape mismatch")
        return field
    if isinstance(value, Mapping):
        vote = np.asarray(value.get("vote", value.get("signed_vote")), dtype=np.float32)
        available_value = value.get("available", value.get("valid"))
        available = np.ones(shape, dtype=bool) if available_value is None else np.asarray(available_value, dtype=bool)
        confidence = np.abs(vote) if value.get("confidence") is None else np.asarray(value["confidence"], dtype=np.float32)
        orientation = str(value.get("orientation", "unknown"))
    else:
        vote = np.asarray(value, dtype=np.float32)
        available = np.ones(shape, dtype=bool)
        confidence = np.abs(vote)
        orientation = "unknown"
    if vote.shape != shape or available.shape != shape or confidence.shape != shape:
        raise ValueError("pairwise field shape mismatch")
    return PairwisePreferenceField(a, b, vote, available, confidence, orientation)


def _normalised_interiority(valid_masks: np.ndarray, provided: np.ndarray | None, p95_values: np.ndarray | None = None) -> np.ndarray:
    if provided is not None:
        interiority = np.asarray(provided)
        if interiority.shape != valid_masks.shape:
            raise ValueError("interiority must have shape (scenes, rows, cols)")
        # Preserve the input precision while calculating P95, then serialize
        # the normalized geometry to the frozen float32 label field.
        result = np.zeros(interiority.shape, dtype=np.float32)
        # Explicit arrays represent EDT-like geometry values.  Apply the
        # protocol's per-scene normalization independently; using a common
        # percentile would make the tie-break depend on the set/order of
        # scenes rather than on each footprint's own interior depth.
        for index, mask in enumerate(valid_masks):
            values = np.where(mask, np.maximum(interiority[index], 0.0), 0.0)
            positive = values[mask]
            p95 = (float(np.asarray(p95_values)[index]) if p95_values is not None
                   else (float(np.percentile(positive, 95)) if positive.size else 1.0))
            if not np.isfinite(p95) or p95 <= 0:
                p95 = 1.0
            result[index] = np.where(mask, np.minimum(values / max(p95, 1.0), 1.0), 0.0)
        return result
    result = np.zeros(valid_masks.shape, dtype=np.float32)
    for index, mask in enumerate(valid_masks):
        distances = distance_transform_edt(mask)
        positive = distances[mask]
        p95 = float(np.percentile(positive, 95)) if positive.size else 1.0
        if not np.isfinite(p95) or p95 <= 0:
            p95 = 1.0
        result[index] = np.where(mask, np.minimum(distances / p95, 1.0), 0.0).astype(np.float32)
    return result


def _directed_edge(vote: np.ndarray, available: np.ndarray, positive_a: bool) -> np.ndarray:
    # A pair is directed only if its preference is nonzero.  Zero at a seam
    # center is not evidence and must not manufacture a cycle.
    sign = vote if positive_a else -vote
    return available & (sign > TIE_EPSILON)


def aggregate_multiscene_labels(
    valid_masks: np.ndarray | Sequence[np.ndarray],
    pairwise_fields: Mapping[tuple[int, int], Any] | Sequence[Any],
    *,
    interiority: np.ndarray | None = None,
    interiority_p95: np.ndarray | None = None,
    tie_epsilon: float = TIE_EPSILON,
) -> MultiSceneLabelResult:
    """Resolve source labels using normalized geometry-only pair votes.

    ``labels`` contains zero-based scene indices.  Pixels with no valid scene
    or with an unresolved geometry tie are ``LABEL_UNRESOLVED``.  Pairwise
    fields are keyed by ``(scene_a, scene_b)`` and must use the same signed
    convention as :class:`PairwisePreferenceField`.
    """

    masks = np.asarray(valid_masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[0] < 1 or 0 in masks.shape[1:]:
        raise ValueError("valid_masks must be (scene, rows, cols) and nonempty")
    n_scenes, height, width = masks.shape
    shape = (height, width)
    if isinstance(pairwise_fields, Mapping):
        raw_fields = pairwise_fields.items()
    else:
        raw_fields = []
        for item in pairwise_fields:
            if isinstance(item, PairwisePreferenceField):
                raw_fields.append(((item.scene_a, item.scene_b), item))
            elif isinstance(item, Mapping) and "pair" in item:
                raw_fields.append((tuple(item["pair"]), item))
            else:
                raise ValueError("sequence pairwise fields must carry scene pair ids")
    fields: dict[tuple[int, int], PairwisePreferenceField] = {}
    for key, value in raw_fields:
        pair = (int(key[0]), int(key[1]))
        if pair[0] == pair[1] or min(pair) < 0 or max(pair) >= n_scenes:
            raise ValueError("invalid scene pair")
        fields[pair] = _coerce_field(pair, value, shape)

    raw_scores = np.zeros((n_scenes, height, width), dtype=np.float32)
    comparison_counts = np.zeros_like(raw_scores)
    for (a, b), field in fields.items():
        both = field.available & masks[a] & masks[b]
        vote = np.where(both, np.asarray(field.vote, dtype=np.float32), 0.0)
        raw_scores[a] += vote
        raw_scores[b] -= vote
        comparison_counts[a] += both
        comparison_counts[b] += both
    scores = np.divide(raw_scores, comparison_counts, out=np.zeros_like(raw_scores), where=comparison_counts > 0)
    geometry = _normalised_interiority(masks, interiority, interiority_p95)
    labels = np.full(shape, LABEL_UNRESOLVED, dtype=np.int16)
    coverage = np.count_nonzero(masks, axis=0).astype(np.int16)
    candidate = np.zeros(shape, dtype=bool)
    tie_mask = np.zeros(shape, dtype=bool)
    fallback_mask = np.zeros(shape, dtype=bool)
    unresolved_mask = np.zeros(shape, dtype=bool)
    margin = np.full(shape, np.nan, dtype=np.float32)

    # Resolve all pixels with array operations; the canonical grid is far too
    # large for a Python row/column loop.
    masked_scores = np.where(masks, scores, -np.inf)
    top = np.max(masked_scores, axis=0)
    candidate = np.any(masks, axis=0)
    top_ties = masks & (np.abs(scores - top[None, ...]) <= tie_epsilon)
    tie_count = np.count_nonzero(top_ties, axis=0)
    tie_mask = candidate & (tie_count > 1)
    best = np.argmax(masked_scores, axis=0).astype(np.int16)
    labels = np.where(candidate & ~tie_mask, best, LABEL_UNRESOLVED).astype(np.int16)

    # Compute the second-best score without sorting a huge 3-D array.
    first = top
    second = np.full(shape, -np.inf, dtype=np.float32)
    for scene_index in range(n_scenes):
        values = masked_scores[scene_index]
        second = np.maximum(second, np.where(values < first, values, -np.inf))
    margin = np.full(shape, np.inf, dtype=np.float32)
    finite_second = candidate & np.isfinite(second)
    margin[finite_second] = (first[finite_second] - second[finite_second]).astype(np.float32)

    interior_masked = np.where(top_ties, geometry, -np.inf)
    best_interior = np.max(interior_masked, axis=0)
    interior_ties = top_ties & (np.abs(geometry - best_interior[None, ...]) <= tie_epsilon)
    interior_count = np.count_nonzero(interior_ties, axis=0)
    fallback_mask = tie_mask & (interior_count == 1)
    interior_best = np.argmax(interior_masked, axis=0).astype(np.int16)
    labels = np.where(fallback_mask, interior_best, labels).astype(np.int16)
    unresolved_mask = tie_mask & (interior_count != 1)

    # Detect directed three-cycles using only available nonzero pair votes.
    cycle_mask = np.zeros(shape, dtype=bool)
    edge_lookup: dict[tuple[int, int], tuple[np.ndarray, bool]] = {}
    for (a, b), field in fields.items():
        edge_lookup[(a, b)] = (field.vote, True)
        edge_lookup[(b, a)] = (field.vote, False)
    for a in range(n_scenes):
        for b in range(a + 1, n_scenes):
            for c in range(b + 1, n_scenes):
                if (a, b) not in edge_lookup or (a, c) not in edge_lookup or (b, c) not in edge_lookup:
                    continue
                ab_vote, ab_a = edge_lookup[(a, b)]
                ac_vote, ac_a = edge_lookup[(a, c)]
                bc_vote, bc_a = edge_lookup[(b, c)]
                ab = _directed_edge(ab_vote, fields[(a, b)].available, ab_a)
                ac = _directed_edge(ac_vote, fields[(a, c)].available, ac_a)
                bc = _directed_edge(bc_vote, fields[(b, c)].available, bc_a)
                # a -> b -> c -> a, or the reverse orientation.
                reverse_ac = _directed_edge(ac_vote, fields[(a, c)].available, not ac_a)
                reverse_ab = _directed_edge(ab_vote, fields[(a, b)].available, not ab_a)
                reverse_bc = _directed_edge(bc_vote, fields[(b, c)].available, not bc_a)
                # Cycle diagnostics are meaningful only where all three
                # candidate scenes are jointly valid.  Saved pair fields are
                # normally already masked by their pair support, but callers
                # may provide a broader ``available`` array; intersecting the
                # scene masks prevents false cycles outside triple coverage.
                triple_support = masks[a] & masks[b] & masks[c]
                cycle_mask |= triple_support & ((ab & bc & reverse_ac) | (reverse_ab & reverse_bc & ac))

    two_scene_disagreement = np.zeros(shape, dtype=bool)
    for (a, b), field in fields.items():
        both_only = masks[a] & masks[b] & (coverage == 2)
        vote = np.asarray(field.vote)
        preferred_a = both_only & field.available & (vote > tie_epsilon)
        preferred_b = both_only & field.available & (vote < -tie_epsilon)
        two_scene_disagreement |= preferred_a & (labels != a)
        two_scene_disagreement |= preferred_b & (labels != b)

    coverage_histogram = {str(int(k)): int(np.count_nonzero(coverage == k)) for k in range(n_scenes + 1)}
    multiscene_pixels = int(np.count_nonzero(coverage >= 3))
    cycle_pixels = int(np.count_nonzero(cycle_mask))
    diagnostics: dict[str, Any] = {
        "coverage_count": coverage,
        "coverage_histogram": coverage_histogram,
        "candidate_pixels": int(np.count_nonzero(candidate)),
        "invalid_pixels": int(np.count_nonzero(coverage == 0)),
        "multiscene_pixels": multiscene_pixels,
        "cycle_pixels": cycle_pixels,
        "cycle_fraction": float(cycle_pixels / multiscene_pixels) if multiscene_pixels else 0.0,
        "cycle_mask": cycle_mask,
        "top_score_tie_pixels": int(np.count_nonzero(tie_mask)),
        "interiority_fallback_pixels": int(np.count_nonzero(fallback_mask)),
        "unresolved_pixels": int(np.count_nonzero(unresolved_mask)),
        "unresolved_mask": unresolved_mask,
        "score_margin": margin,
        "score_margin_min": float(np.nanmin(margin)) if np.isfinite(margin).any() else float("nan"),
        "score_margin_median": float(np.nanmedian(margin)) if np.isfinite(margin).any() else float("nan"),
        "two_scene_disagreement_pixels": int(np.count_nonzero(two_scene_disagreement)),
        "two_scene_disagreement_mask": two_scene_disagreement,
        "resolved_pair_count": int(len(fields)),
    }
    return MultiSceneLabelResult(labels, scores.astype(np.float32), comparison_counts.astype(np.float32), geometry, diagnostics)


__all__ = [
    "LABEL_UNRESOLVED",
    "TIE_EPSILON",
    "SEAM_CONFIDENCE_HALF_WIDTH",
    "PairwisePreferenceField",
    "MultiSceneLabelResult",
    "build_pairwise_preference_field",
    "aggregate_multiscene_labels",
]
