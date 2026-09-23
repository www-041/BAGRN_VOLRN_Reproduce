"""Phase-correlation block-matching adapter.

Wraps the existing ``collect_block_matches`` function so that it
produces a standard :class:`MatchSet`.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from rasterio.transform import from_origin

from src.coregistration import collect_block_matches
from src.registration_benchmark.models import MatchView, make_matchset_from_view

logger = logging.getLogger(__name__)


def match_phase(
    view: MatchView,
    block_size: int = 256,
    confidence_threshold: float = 0.5,
    max_shift: float = 80.0,
) -> MatchSet:
    """Run phase-correlation block matching on a :class:`MatchView`.

    Constructs a virtual pixel transform so that the existing
    ``collect_block_matches`` operates on the match-view images as if they
    were independent GeoTIFFs, then maps the resulting coordinates back to
    the common grid via :meth:`MatchView.to_canvas`.

    Args:
        view: Preprocessed overlap view from :func:`~common_grid.build_match_view`.
        block_size: Side length of each matching block in pixels.
        confidence_threshold: Minimum phase-correlation NCC confidence to
            accept a match (range ``[0, 1]``).
        max_shift: Maximum allowed block displacement in pixels.

    Returns:
        A :class:`MatchSet` with coordinates in the common-grid space.
    """
    t0 = time.perf_counter()

    h, w = view.ref.shape

    # Virtual pixel-is-area transform: pixel (r, c) → (c + 0.5, r + 0.5)
    tr = from_origin(0.0, float(h), 1.0, 1.0)

    matches, screening = collect_block_matches(
        arr_ref=view.ref,
        tr_ref=tr,
        arr_tgt=view.tgt,
        tr_tgt=tr,
        nodata_ref=np.nan,
        nodata_tgt=np.nan,
        block_size=block_size,
        max_global_shift=max_shift,
        confidence_threshold=confidence_threshold,
    )

    if not matches:
        elapsed = time.perf_counter() - t0
        logger.warning("Phase: zero matches from %d blocks", screening.get("total", 0))
        return make_matchset_from_view(
            method="phase",
            view=view,
            ref_xy_view=np.empty((0, 2)),
            tgt_xy_view=np.empty((0, 2)),
            confidence=np.empty(0),
            runtime_sec=elapsed,
            metadata={
                "screening": screening,
                "block_size": block_size,
                "confidence_threshold": confidence_threshold,
                "max_shift": max_shift,
            },
        )

    # Construct (N, 2) arrays in *view* pixel space, then map to canvas
    n = len(matches)
    ref_xy_view = np.empty((n, 2), dtype=np.float64)
    tgt_xy_view = np.empty((n, 2), dtype=np.float64)
    confidence = np.empty(n, dtype=np.float64)

    for i, m in enumerate(matches):
        # ref 位置 = block center + measured shift (shift aligns tgt→ref)
        ref_xy_view[i, 0] = m["ref_x"] + m["shift_dx"]
        ref_xy_view[i, 1] = m["ref_y"] + m["shift_dy"]
        # tgt 位置 = block center in target image
        tgt_xy_view[i, 0] = m["tgt_x"]
        tgt_xy_view[i, 1] = m["tgt_y"]
        confidence[i] = m["confidence"]

    # Map both from view pixel space → common-grid pixel space
    elapsed = time.perf_counter() - t0

    return make_matchset_from_view(
        method="phase",
        view=view,
        ref_xy_view=ref_xy_view,
        tgt_xy_view=tgt_xy_view,
        confidence=confidence,
        runtime_sec=elapsed,
        metadata={
            "screening": screening,
            "block_size": block_size,
            "confidence_threshold": confidence_threshold,
            "max_shift": max_shift,
        },
    )
