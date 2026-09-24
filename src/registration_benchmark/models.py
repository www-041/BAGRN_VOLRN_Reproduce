"""Unified data models for the registration benchmark.

All matchers must produce a :class:`MatchSet` so that downstream RANSAC,
metrics, and diagnostics operate on identical structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


PAIR_COMMON_GRID_FRAME = "pair_common_grid"
CONFIDENCE_SEMANTICS = "method_internal_only"

# Capability metadata is descriptive only.  It must never be used to alter
# the shared geometry thresholds or to compare confidence magnitudes across
# matcher families.
MATCHER_INFO = {
    "phase": {
        "name": "phase",
        "family": "sparse_classical_block",
        "detector_free": False,
        "uses_learned_features": False,
        "confidence_semantics": CONFIDENCE_SEMANTICS,
    },
    "sift": {
        "name": "sift",
        "family": "sparse_classical",
        "detector_free": False,
        "uses_learned_features": False,
        "confidence_semantics": CONFIDENCE_SEMANTICS,
    },
    "loftr": {
        "name": "loftr",
        "family": "semi_dense_detector_free",
        "detector_free": True,
        "uses_learned_features": True,
        "confidence_semantics": CONFIDENCE_SEMANTICS,
    },
    "efficient_loftr": {
        "name": "efficient_loftr",
        "family": "semi_dense_detector_free_sparse_like",
        "detector_free": True,
        "uses_learned_features": True,
        "confidence_semantics": CONFIDENCE_SEMANTICS,
    },
    "lightglue": {
        "name": "lightglue",
        "family": "sparse_learned",
        "detector_free": False,
        "uses_learned_features": True,
        "confidence_semantics": CONFIDENCE_SEMANTICS,
    },
    "lightglue_disk": {
        "name": "lightglue_disk",
        "family": "sparse_learned_local_features",
        "detector_free": False,
        "uses_learned_features": True,
        "confidence_semantics": CONFIDENCE_SEMANTICS,
    },
}


@dataclass
class MatchSet:
    """Tie-point matches produced by a single matcher.

    Coordinates follow the convention:

    * ``x`` = column
    * ``y`` = row
    * ``model(tgt_xy) ≈ ref_xy`` (i.e. the target is warped *onto* the reference)

    Attributes:
        method: Short name of the matcher (``"phase"``, ``"sift"``, …).
        ref_xy: Reference-image point coordinates, shape ``(N, 2)``.
        tgt_xy: Target-image point coordinates, shape ``(N, 2)``.
        confidence: Per-match confidence in ``[0, 1]``, shape ``(N,)``.
            Used for ranking / display only; not comparable across methods.
        runtime_sec: Wall-clock time spent inside the matcher.
        metadata: Arbitrary extra info (e.g. matcher parameters).
    """

    method: str
    ref_xy: np.ndarray
    tgt_xy: np.ndarray
    confidence: np.ndarray
    runtime_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, *, require_coordinate_frame: bool = False) -> None:
        """Raise :class:`ValueError` if any invariant is violated."""
        if self.ref_xy.ndim != 2 or self.ref_xy.shape[1] != 2:
            raise ValueError("ref_xy must have shape (N, 2)")

        if self.tgt_xy.shape != self.ref_xy.shape:
            raise ValueError("tgt_xy must match ref_xy shape")

        if self.confidence.shape != (len(self.ref_xy),):
            raise ValueError("confidence must have shape (N,)")

        if not np.isfinite(self.ref_xy).all():
            raise ValueError("ref_xy contains non-finite values")

        if not np.isfinite(self.tgt_xy).all():
            raise ValueError("tgt_xy contains non-finite values")

        if require_coordinate_frame and self.metadata.get("coordinate_frame") != PAIR_COMMON_GRID_FRAME:
            raise ValueError(
                "matcher coordinates must use the pair_common_grid coordinate frame"
            )

    def validate_for_geometry(self) -> None:
        """Validate the stricter contract required before shared RANSAC."""
        self.validate(require_coordinate_frame=True)
        if self.metadata.get("confidence_semantics") != CONFIDENCE_SEMANTICS:
            raise ValueError(
                "confidence must be marked as method_internal_only; "
                "cross-method confidence comparison is not allowed"
            )

        breakdown = self.metadata.get("runtime_breakdown")
        if breakdown is not None:
            required = {"feature_runtime_sec", "matcher_runtime_sec", "total_runtime_sec"}
            if not required.issubset(breakdown):
                raise ValueError("runtime_breakdown is missing required timing fields")


@dataclass
class CommonGridPair:
    """Two images resampled onto a shared geographic grid.

    Attributes:
        ref_raw: Reference image data (original DN), 2-D ``float64``.
        tgt_raw: Target image data (original DN), 2-D ``float64``.
        ref_valid: Valid-pixel mask for *ref_raw*.
        tgt_valid: Valid-pixel mask for *tgt_raw*.
        transform: Affine geotransform (``rasterio.transform``).
        crs: Coordinate reference system.
        overlap_window: Bounding rectangle of the valid overlap region
            as ``(row_start, row_end, col_start, col_end)``.
    """

    ref_raw: np.ndarray
    tgt_raw: np.ndarray

    ref_valid: np.ndarray
    tgt_valid: np.ndarray

    transform: Any
    crs: Any

    overlap_window: tuple[int, int, int, int]


@dataclass
class MatchView:
    """A preprocessed, down-sampled view of the overlap region for matching.

    Both *ref* and *tgt* are float32 arrays in ``[0, 1]`` that cover the
    same geographic area (the overlap).  Pixel-to-world coordinate mapping
    is provided by :meth:`to_canvas`.

    Attributes:
        ref: Reference image (float32, [0, 1]).
        tgt: Target image (float32, [0, 1]).
        ref_valid: Valid mask for *ref*.
        tgt_valid: Valid mask for *tgt*.
        origin_x: X-coordinate (column) of pixel ``(0, 0)`` in the original
            common-grid space.
        origin_y: Y-coordinate (row) of pixel ``(0, 0)`` in the original
            common-grid space.
        scale_x: Scale factor from match-view column → common-grid column.
        scale_y: Scale factor from match-view row → common-grid row.
    """

    ref: np.ndarray
    tgt: np.ndarray

    ref_valid: np.ndarray
    tgt_valid: np.ndarray

    origin_x: float
    origin_y: float

    scale_x: float
    scale_y: float

    def to_common_grid(self, xy: np.ndarray) -> np.ndarray:
        """Undo resize and overlap-crop offsets into pair-common-grid pixels.

        Args:
            xy: Array of shape ``(N, 2)`` with columns ``[x, y]`` in match-view
                pixel space.

        Returns:
            Array of shape ``(N, 2)`` with columns ``[x, y]`` in common-grid
            (canvas) pixel space.
        """
        xy = np.asarray(xy, dtype=np.float64)

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2)")

        out = xy.copy()

        out[:, 0] = self.origin_x + out[:, 0] / self.scale_x
        out[:, 1] = self.origin_y + out[:, 1] / self.scale_y

        return out

    def to_canvas(self, xy: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for :meth:`to_common_grid`."""
        return self.to_common_grid(xy)


def make_matchset_from_view(
    *,
    method: str,
    view: MatchView,
    ref_xy_view: np.ndarray,
    tgt_xy_view: np.ndarray,
    confidence: np.ndarray,
    runtime_sec: float,
    metadata: dict[str, Any] | None = None,
    runtime_breakdown: dict[str, float] | None = None,
) -> MatchSet:
    """Build a ``MatchSet`` after one canonical coordinate conversion.

    Matcher adapters may use any internal resized/cropped coordinates, but the
    object entering shared RANSAC is always in ``pair_common_grid``.
    """
    ref_xy_view = np.asarray(ref_xy_view, dtype=np.float64)
    tgt_xy_view = np.asarray(tgt_xy_view, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    meta = dict(metadata or {})
    meta["coordinate_frame"] = PAIR_COMMON_GRID_FRAME
    meta["confidence_semantics"] = CONFIDENCE_SEMANTICS
    if runtime_breakdown is None:
        runtime_breakdown = {
            "feature_runtime_sec": 0.0,
            "matcher_runtime_sec": float(runtime_sec),
        }
    meta["runtime_breakdown"] = {
        "feature_runtime_sec": float(runtime_breakdown.get("feature_runtime_sec", 0.0)),
        "matcher_runtime_sec": float(runtime_breakdown.get("matcher_runtime_sec", 0.0)),
        "total_runtime_sec": float(runtime_sec),
    }
    return MatchSet(
        method=method,
        ref_xy=view.to_common_grid(ref_xy_view),
        tgt_xy=view.to_common_grid(tgt_xy_view),
        confidence=confidence,
        runtime_sec=float(runtime_sec),
        metadata=meta,
    )
