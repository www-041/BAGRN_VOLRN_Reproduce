"""Frozen deterministic one-point RANSAC for native-grid 2-D translations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coordinates import NativeMatches
from .protocol import Phase5Config


@dataclass(frozen=True)
class TranslationFit:
    status: str
    failure_code: str | None
    delta_row_col: tuple[float, float] | None
    inlier_mask: np.ndarray
    match_count: int
    inlier_count: int
    inlier_fraction: float
    coverage_cells: int
    coverage_rows: int
    coverage_columns: int
    seed: int
    hypotheses_tested: int
    median_residual_px: float | None


def _validate(matches: NativeMatches) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i_rc = np.asarray(matches.i_rc, dtype=float)
    j_rc = np.asarray(matches.j_rc, dtype=float)
    i_xy = np.asarray(matches.i_view_xy, dtype=float)
    j_xy = np.asarray(matches.j_view_xy, dtype=float)
    if (i_rc.ndim != 2 or i_rc.shape[1:] != (2,) or j_rc.shape != i_rc.shape
            or i_xy.shape != i_rc.shape or j_xy.shape != i_rc.shape):
        raise ValueError("correspondences must have matching shape (N, 2)")
    if (not np.isfinite(i_rc).all() or not np.isfinite(j_rc).all()
            or not np.isfinite(i_xy).all()
            or not np.isfinite(j_xy).all()):
        raise ValueError("correspondences must be finite")
    if (len(matches.view_shape) != 2 or min(matches.view_shape) <= 0
            or np.any(i_xy < -.5)
            or np.any(i_xy[:, 0] >= matches.view_shape[1] - .5)
            or np.any(i_xy[:, 1] >= matches.view_shape[0] - .5)
            or np.any(j_xy < -.5)
            or np.any(j_xy[:, 0] >= matches.view_shape[1] - .5)
            or np.any(j_xy[:, 1] >= matches.view_shape[0] - .5)):
        raise ValueError("invalid Phase5 view shape or coordinates")
    return i_rc, j_rc, i_xy


def _failure(code: str, n: int, config: Phase5Config, *, tested: int = 0,
             inlier_mask: np.ndarray | None = None, inlier_count: int = 0,
             fraction: float = 0.0, cells: int = 0, rows: int = 0,
             columns: int = 0, median: float | None = None) -> TranslationFit:
    mask = np.zeros(n, dtype=bool) if inlier_mask is None else inlier_mask
    return TranslationFit("ALGORITHM_FAIL", code, None, mask, n, inlier_count,
                          float(fraction), cells, rows, columns, config.seed,
                          tested, median)


def estimate_translation(matches: NativeMatches, config: Phase5Config) -> TranslationFit:
    """Estimate ``scene_i -> scene_j`` as median ``j_rc - i_rc``.

    Confidence is intentionally not used. Each seeded hypothesis is one
    observed displacement; consensus uses the frozen Euclidean radius in
    native B9 pixels. The final displacement is the componentwise median of
    the selected consensus and coverage uses the fixed full-view 4x4 grid.
    """
    i_rc, j_rc, i_xy = _validate(matches)
    n = len(i_rc)
    if n < config.min_matches:
        return _failure("INSUFFICIENT_MATCHES", n, config)
    delta = j_rc - i_rc
    rng = np.random.default_rng(config.seed)
    best: tuple[int, float, int, np.ndarray] | None = None
    tested = 0
    for index in range(config.ransac_hypotheses):
        hypothesis = delta[int(rng.integers(0, n))]
        residual = np.linalg.norm(delta - hypothesis, axis=1)
        inliers = residual <= config.ransac_radius_native_px
        count = int(inliers.sum())
        if count == 0:
            median = float("inf")
        else:
            median = float(np.median(residual[inliers]))
        key = (-count, median, index)
        if best is None or key < best[:3]:
            best = (key[0], key[1], key[2], inliers)
        tested += 1
    if best is None:
        return _failure("NO_CONSENSUS", n, config, tested=tested)
    chosen = best[3]
    inlier_count = int(chosen.sum())
    fraction = inlier_count / n
    if inlier_count < config.min_inliers:
        return _failure("INSUFFICIENT_INLIERS", n, config, tested=tested,
                        inlier_mask=chosen, inlier_count=inlier_count, fraction=fraction,
                        median=best[1])
    if fraction < config.min_inlier_fraction:
        return _failure("LOW_INLIER_FRACTION", n, config, tested=tested,
                        inlier_mask=chosen, inlier_count=inlier_count, fraction=fraction,
                        median=best[1])
    final_delta = np.median(delta[chosen], axis=0)
    residual = np.linalg.norm(delta - final_delta, axis=1)
    final_mask = residual <= config.ransac_radius_native_px
    final_count = int(final_mask.sum())
    final_fraction = final_count / n
    height, width = matches.view_shape
    grid_cols, grid_rows = config.coverage_grid
    col = np.floor(np.clip(i_xy[:, 0], 0, width - 1) / width * grid_cols).astype(int)
    row = np.floor(np.clip(i_xy[:, 1], 0, height - 1) / height * grid_rows).astype(int)
    col = np.minimum(col, grid_cols - 1)
    row = np.minimum(row, grid_rows - 1)
    occupied = {(int(row[k]), int(col[k])) for k in np.flatnonzero(final_mask)}
    cells = len(occupied)
    rows = len({r for r, _ in occupied})
    columns = len({c for _, c in occupied})
    if final_count < config.min_inliers:
        return _failure("INSUFFICIENT_INLIERS", n, config, tested=tested,
                        inlier_mask=final_mask, inlier_count=final_count,
                        fraction=final_fraction, cells=cells, rows=rows,
                        columns=columns, median=float(np.median(residual[final_mask])))
    if final_fraction < config.min_inlier_fraction:
        return _failure("LOW_INLIER_FRACTION", n, config, tested=tested,
                        inlier_mask=final_mask, inlier_count=final_count,
                        fraction=final_fraction, cells=cells, rows=rows,
                        columns=columns, median=float(np.median(residual[final_mask])))
    if (cells < config.min_coverage_cells or rows < config.min_coverage_rows
            or columns < config.min_coverage_columns):
        return _failure("INSUFFICIENT_SPATIAL_COVERAGE", n, config, tested=tested,
                        inlier_mask=final_mask, inlier_count=final_count,
                        fraction=final_fraction, cells=cells, rows=rows,
                        columns=columns, median=float(np.median(residual[final_mask])))
    return TranslationFit("SUCCESS", None,
                          (float(final_delta[0]), float(final_delta[1])), final_mask,
                          n, final_count, float(final_fraction), cells, rows, columns,
                          config.seed, tested, float(np.median(residual[final_mask])))
