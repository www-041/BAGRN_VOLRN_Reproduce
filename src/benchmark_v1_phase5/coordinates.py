"""Explicit Phase5 view, legacy common-grid, and native pixel-centre maps.

``phase5_view`` coordinates are x/y on PairInput.view. Legacy
``pair_common_grid`` coordinates have already passed through the legacy
matcher view's ``to_common_grid``; callers must attest that this grid is
the Phase5 overlap grid via ``phase5_common_grid='overlap_grid'``. Never
apply the legacy crop/scale a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from affine import Affine
import numpy as np

from .inputs import PairInput
from registration_benchmark.models import MatchSet


class UnsupportedGridError(ValueError):
    """A single native-pixel translation cannot represent these grids."""


@dataclass(frozen=True)
class NativeMatches:
    """Native row/col pairs and original Phase5 view x/y for coverage."""

    i_rc: np.ndarray
    j_rc: np.ndarray
    i_view_xy: np.ndarray
    j_view_xy: np.ndarray
    confidence: np.ndarray
    method: str
    view_shape: tuple[int, int]


def _check_linear_parts(i_affine: Affine, j_affine: Affine) -> None:
    coeffs = np.array([[i_affine.a, i_affine.b, i_affine.d, i_affine.e],
                       [j_affine.a, j_affine.b, j_affine.d, j_affine.e]], dtype=float)
    if (not np.isfinite(coeffs).all()
            or abs(i_affine.a * i_affine.e - i_affine.b * i_affine.d) < 1e-12
            or abs(j_affine.a * j_affine.e - j_affine.b * j_affine.d) < 1e-12
            or not np.allclose(coeffs[0], coeffs[1], rtol=0, atol=1e-9)):
        raise UnsupportedGridError("UNSUPPORTED_GRID: unequal or singular affine linear parts")


def georef_only_translation(i_affine: Affine, j_affine: Affine) -> tuple[float, float]:
    """Return the unranked GEOREF_ONLY native i->j (delta_row, delta_col).

    Equal linear parts make the offset independent of pixel position.
    The same world point at the i origin has j coordinate
    ``A_j^-1(t_i - t_j)``; native pixel-centre halves cancel.
    """
    _check_linear_parts(i_affine, j_affine)
    if not np.isfinite([i_affine.c, i_affine.f, j_affine.c, j_affine.f]).all():
        raise UnsupportedGridError("UNSUPPORTED_GRID: nonfinite affine origin")
    col, row = (~j_affine) * (i_affine.c, i_affine.f)
    return float(row), float(col)


def _view_xy(xy: np.ndarray, pair: PairInput, side: Literal["i", "j"]) -> np.ndarray:
    if pair.status != "READY" or pair.view is None:
        raise ValueError("Pair has no ready view")
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("view xy must be finite with shape (N, 2)")
    height, width = pair.view.joint_mask.shape
    if (np.any(xy[:, 0] < -.5) or np.any(xy[:, 0] >= width - .5)
            or np.any(xy[:, 1] < -.5) or np.any(xy[:, 1] >= height - .5)):
        raise ValueError("coordinate outside Phase5 view")
    # Nearest pixel centre determines eligibility on the shared true mask.
    cols = np.floor(xy[:, 0] + .5).astype(int)
    rows = np.floor(xy[:, 1] + .5).astype(int)
    side_mask = pair.view.mask_i if side == "i" else pair.view.mask_j
    if not side_mask[rows, cols].all():
        raise ValueError(f"coordinate outside scene-{side} valid view mask")
    return xy


def view_xy_to_native_rc(
    xy: np.ndarray, pair: PairInput, side: Literal["i", "j"]
) -> np.ndarray:
    """Map view x/y pixel centres through overlap world to native row/col."""
    if side not in ("i", "j"):
        raise ValueError("side must be i or j")
    xy = _view_xy(xy, pair, side)
    provenance = pair.view.provenance
    _check_linear_parts(pair.native_affine_i, pair.native_affine_j)
    if not np.isfinite([provenance.resize_x, provenance.resize_y]).all() or min(
            provenance.resize_x, provenance.resize_y) <= 0:
        raise ValueError("invalid Phase5 view resize")
    native_affine = pair.native_affine_i if side == "i" else pair.native_affine_j
    # View centre (x+.5,y+.5) maps back through resize to overlap
    # centre, then Affine maps that centre to projected world coordinates.
    overlap_col = (xy[:, 0] + .5) / provenance.resize_x
    overlap_row = (xy[:, 1] + .5) / provenance.resize_y
    world_x, world_y = provenance.overlap_affine * (overlap_col, overlap_row)
    native_col_edge, native_row_edge = (~native_affine) * (world_x, world_y)
    rc = np.column_stack((native_row_edge - .5, native_col_edge - .5))
    if not np.isfinite(rc).all() or np.any(rc < -.5 - 1e-9):
        raise ValueError("invalid native grid coordinate")
    return rc


def to_native_correspondences(matchset: MatchSet, pair: PairInput) -> NativeMatches:
    """Validate one declared frame and retain view positions for Task 3."""
    matchset.validate()
    if pair.status != "READY" or pair.view is None:
        raise ValueError("Pair has no ready view")
    frame = matchset.metadata.get("coordinate_frame")
    if frame == "pair_common_grid":
        matchset.validate_for_geometry()
        if matchset.metadata.get("phase5_common_grid") != "overlap_grid":
            raise ValueError("pair_common_grid requires phase5_common_grid='overlap_grid'")
        origin = np.asarray(matchset.metadata.get("legacy_view_origin_xy"), dtype=float)
        scale = np.asarray(matchset.metadata.get("legacy_view_scale_xy"), dtype=float)
        if (origin.shape != (2,) or scale.shape != (2,) or not np.isfinite(origin).all()
                or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError("pair_common_grid requires finite legacy_view_origin_xy and positive legacy_view_scale_xy")
        # The existing MatchView.to_common_grid contract is exactly
        # common_xy = origin_xy + view_xy / scale_xy.
        # Invert that recorded mapping once; Phase5 resize is already reflected
        # by the caller's explicit MatchView-to-overlap-grid metadata.
        i_view = (np.asarray(matchset.ref_xy, dtype=float) - origin) * scale
        j_view = (np.asarray(matchset.tgt_xy, dtype=float) - origin) * scale
    elif frame == "phase5_view":
        if matchset.metadata.get("confidence_semantics") != "method_internal_only":
            raise ValueError("confidence semantics must be method_internal_only")
        i_view = np.asarray(matchset.ref_xy, dtype=float)
        j_view = np.asarray(matchset.tgt_xy, dtype=float)
    else:
        raise ValueError("unsupported coordinate_frame")
    i_rc = view_xy_to_native_rc(i_view, pair, "i")
    j_rc = view_xy_to_native_rc(j_view, pair, "j")
    return NativeMatches(i_rc, j_rc, i_view.copy(), j_view.copy(),
                         np.asarray(matchset.confidence, dtype=float).copy(), matchset.method,
                         pair.view.joint_mask.shape)
