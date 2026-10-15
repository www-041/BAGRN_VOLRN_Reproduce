"""Monotonic overlap seam search for the frozen Task13A protocol."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeamResult:
    """A seam spanning every row (vertical) or column (horizontal)."""

    orientation: str
    row_col_path: np.ndarray
    total_cost: float | None
    mean_cost: float | None
    p95_cost: float | None
    status: str
    search_mode: str = "unknown"


def _unsupported(orientation: str) -> SeamResult:
    return SeamResult(
        orientation=orientation,
        row_col_path=np.empty((0, 2), dtype=np.int32),
        total_cost=None,
        mean_cost=None,
        p95_cost=None,
        status="UNSUPPORTED_TOPOLOGY",
    )


def _dynamic_path(
    cost: np.ndarray, valid: np.ndarray, centers: np.ndarray | None = None,
    half_width: int | None = None,
) -> np.ndarray | None:
    """Find the minimum cost top-to-bottom path using transverse steps -1, 0, 1.

    The caller transposes horizontal overlaps before entering this routine.
    Only previous-line costs are retained; signed backsteps recover the path.
    """
    length, width = cost.shape
    previous = np.full(width, np.inf, dtype=np.float64)
    backsteps = np.zeros((length, width), dtype=np.int8)
    columns = np.arange(width)

    for line in range(length):
        available = valid[line]
        if centers is not None:
            available = available & (np.abs(columns - centers[line]) <= half_width)
        if line == 0:
            previous = np.where(available, cost[line], np.inf)
            continue
        # For a current transverse coordinate x, predecessors are x-1,x,x+1.
        from_left = np.empty(width, dtype=np.float64)
        from_left[0] = np.inf
        from_left[1:] = previous[:-1]
        from_right = np.empty(width, dtype=np.float64)
        from_right[-1] = np.inf
        from_right[:-1] = previous[1:]
        candidates = np.stack((from_left, previous, from_right), axis=0)
        predecessor = np.argmin(candidates, axis=0)
        best = np.take_along_axis(candidates, predecessor[None, :], axis=0)[0]
        previous = np.where(available & np.isfinite(best), cost[line] + best, np.inf)
        backsteps[line] = predecessor.astype(np.int8) - 1
        if not np.isfinite(previous).any():
            return None

    endpoint = int(np.argmin(previous))
    if not np.isfinite(previous[endpoint]):
        return None
    path = np.empty(length, dtype=np.int32)
    path[-1] = endpoint
    for line in range(length - 1, 0, -1):
        path[line - 1] = path[line] + backsteps[line, path[line]]
    return path


def _downsample_cost(cost: np.ndarray, valid: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce each block to its cheapest jointly valid node, without fill values."""
    height, width = cost.shape
    coarse_shape = ((height + factor - 1) // factor, (width + factor - 1) // factor)
    reduced = np.full(coarse_shape, np.inf, dtype=np.float64)
    for row_offset in range(factor):
        for col_offset in range(factor):
            tile = cost[row_offset::factor, col_offset::factor]
            tile_valid = valid[row_offset::factor, col_offset::factor]
            view = reduced[: tile.shape[0], : tile.shape[1]]
            np.minimum(view, np.where(tile_valid, tile, np.inf), out=view)
    return reduced, np.isfinite(reduced)


def find_monotonic_seam(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    coarse_factor: int = 4,
    refine_half_width: int = 64,
) -> SeamResult:
    """Find the fixed Task13A monotonic seam, refining a coarse path if large.

    Vertical spans rows when overlap height >= width; horizontal spans columns
    otherwise. A path must span the entire input overlap and never use an
    invalid or nonfinite-cost pixel.
    """
    cost = np.asarray(cost, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if cost.ndim != 2 or cost.shape != valid_mask.shape or 0 in cost.shape:
        raise ValueError("cost and valid_mask must be same-shape nonempty 2D arrays")
    if coarse_factor < 1 or refine_half_width < 0:
        raise ValueError("coarse_factor must be positive and refine_half_width nonnegative")
    orientation = "vertical" if cost.shape[0] >= cost.shape[1] else "horizontal"
    transposed = orientation == "horizontal"
    oriented_cost = cost.T if transposed else cost
    oriented_valid = (valid_mask & np.isfinite(cost)).T if transposed else (valid_mask & np.isfinite(cost))

    if coarse_factor == 1 or min(cost.shape) < 256:
        transverse = _dynamic_path(oriented_cost, oriented_valid)
        search_mode = "full_resolution_small" if min(cost.shape) < 256 else "full_resolution_requested"
    else:
        search_mode = "coarse_refined"
        coarse_cost, coarse_valid = _downsample_cost(oriented_cost, oriented_valid, coarse_factor)
        coarse_path = _dynamic_path(coarse_cost, coarse_valid)
        if coarse_path is None:
            return _unsupported(orientation)
        coarse_lines = np.arange(len(coarse_path)) * coarse_factor + coarse_factor // 2
        full_lines = np.arange(oriented_cost.shape[0])
        centers = np.interp(full_lines, coarse_lines, coarse_path * coarse_factor + coarse_factor // 2)
        transverse = _dynamic_path(oriented_cost, oriented_valid, centers, refine_half_width)
        if transverse is None:
            # Block minima can connect on the coarse grid even when their
            # full-resolution pixels form isolated stubs. Preserve the fixed
            # monotonic topology by retrying without the coarse corridor.
            transverse = _dynamic_path(oriented_cost, oriented_valid)
            search_mode = "full_resolution_fallback"

    if transverse is None:
        return _unsupported(orientation)
    lines = np.arange(len(transverse), dtype=np.int32)
    row_col_path = np.column_stack((transverse, lines) if transposed else (lines, transverse))
    values = cost[row_col_path[:, 0], row_col_path[:, 1]]
    return SeamResult(
        orientation=orientation,
        row_col_path=row_col_path,
        total_cost=float(np.sum(values)),
        mean_cost=float(np.mean(values)),
        p95_cost=float(np.percentile(values, 95)),
        status="OK",
        search_mode=search_mode,
    )
