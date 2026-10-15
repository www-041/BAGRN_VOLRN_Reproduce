"""Numerical translation of the pinned LSSTAA MATLAB code.

Zeros retain the official missing-observation semantics. Non-finite, empty,
disconnected and rank-deficient cases fail explicitly rather than yielding
plausible but unsupported coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np


def matlab_quantile(values: np.ndarray, q):
    """MATLAB quantile default: midpoint ranks (k-.5)/n (NumPy Hazen)."""
    return np.quantile(values, q, method="hazen")


def absolute_day_offsets(iso_dates: list[str]) -> list[int]:
    days = [date.fromisoformat(value) for value in iso_dates]
    return [(day - days[0]).days for day in days]


def _clean_top(image: np.ndarray, ratio: float = .01) -> np.ndarray:
    out = np.asarray(image).copy()
    valid = np.isfinite(out) & (out != 0)
    if not np.any(valid):
        raise ValueError("EMPTY_OBSERVATIONS")
    high = matlab_quantile(out[valid], 1 - ratio)
    out[~valid | (out > high)] = 0
    return out


def internal_bias_compensation(series: np.ndarray, days: list[int],
                               quantile_cut: float = .01,
                               day_threshold: int = 31) -> tuple[np.ndarray, np.ndarray]:
    if series.ndim != 3 or series.shape[2] != len(days):
        raise ValueError("INVALID_SERIES_SHAPE")
    count = len(days)
    diff = np.zeros((count, count), dtype=np.float64)
    clean = [_clean_top(series[:, :, i], quantile_cut) for i in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            if abs(days[i] - days[j]) > day_threshold:
                continue
            mask = (clean[i] != 0) & (clean[j] != 0)
            if np.count_nonzero(mask) < 1000:
                continue
            delta = clean[i][mask] - clean[j][mask]
            delta = delta[delta != 0]  # MATLAB median(diff_i(diff_i~=0))
            if not len(delta):
                continue
            diff[i, j] = np.median(delta)
            diff[j, i] = -diff[i, j]
    medians = np.empty(count, dtype=np.float64)
    for i in range(count):
        row = diff[i][diff[i] != 0]
        if not len(row):
            raise ValueError(f"EMPTY_BIAS_CONNECTION:{i}")
        medians[i] = np.median(row)
    # I_bias_corr = I*0 in the pinned MATLAB source retains I's single dtype.
    corrected = np.asarray(series).copy()
    for i in range(count):
        plane = corrected[:, :, i]
        valid = np.isfinite(plane) & (plane != 0)
        plane[valid] -= medians[i] / 2
        plane[~valid] = 0
    return corrected, medians


@dataclass(frozen=True)
class PifResult:
    pif: np.ndarray
    temporal_difference: np.ndarray
    observation_count: np.ndarray
    threshold: float
    bias_medians: np.ndarray
    bias_corrected: np.ndarray


def _matlab_int16(values: np.ndarray) -> np.ndarray:
    rounded = np.sign(values) * np.floor(np.abs(values) + .5)
    return np.clip(rounded, -32768, 32767).astype(np.int16)


def _matlab_hist_counts(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """MATLAB hist: a value on an interior right edge belongs to the left bin."""
    midpoints = (centers[:-1] + centers[1:]) / 2
    return np.bincount(np.searchsorted(midpoints, values, side="left"),
                       minlength=len(centers))


def pif_from_series(ndvi: np.ndarray, days: list[int],
                    quantile_cut: float = .01) -> PifResult:
    corrected, medians = internal_bias_compensation(ndvi, days, quantile_cut, 31)
    clean = [_clean_top(corrected[:, :, i], quantile_cut) for i in range(len(days))]
    obs = np.sum(corrected != 0, axis=2).astype(np.int16)
    numerator = np.zeros(ndvi.shape[:2], dtype=np.float64)
    denominator = np.zeros_like(numerator)
    for i in range(len(days)):
        for j in range(i + 1, len(days)):
            gap = abs(days[i] - days[j])
            if gap > 31:
                continue
            valid = (clean[i] != 0) & (clean[j] != 0)
            if np.count_nonzero(valid) < 1000:
                continue
            weight = 1 / (min(gap, 31) + 1)
            numerator[valid] += np.abs(clean[i][valid] - clean[j][valid]) * weight
            denominator[valid] += weight
    temporal = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=temporal, where=denominator > 0)
    temporal = _matlab_int16(temporal).astype(np.float64)
    positive_obs = obs[obs > 0]
    if not len(positive_obs):
        raise ValueError("EMPTY_OBSERVATIONS")
    obs_threshold = matlab_quantile(positive_obs, .05)
    temporal[obs <= obs_threshold] = 0
    values = temporal[temporal != 0]
    if not len(values):
        raise ValueError("EMPTY_PIF_TEMPORAL_DIFFERENCE")
    lo, hi = matlab_quantile(values, [.005, .995])
    values = values[(values >= lo) & (values <= hi)]
    if not len(values):
        raise ValueError("EMPTY_PIF_HISTOGRAM")
    centers = np.linspace(values.min(), values.max(), 50)
    if centers[0] == centers[-1]:
        raise ValueError("DEGENERATE_PIF_HISTOGRAM")
    counts = _matlab_hist_counts(values, centers)
    threshold = (centers[0] + centers[np.argmax(counts)]) / 2
    pif = (temporal > 0) & (temporal < threshold)
    if not np.any(pif):
        raise ValueError("EMPTY_PIF")
    return PifResult(pif, temporal, obs, float(threshold), medians, corrected)


def _regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.var(x) == 0 or np.var(y) == 0:
        raise ValueError("INVALID_REGRESSION")
    gain, offset = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (gain*x + offset))**2) / np.sum((y-np.mean(y))**2)
    if not np.isfinite([gain, offset, r2]).all():
        raise ValueError("INVALID_REGRESSION")
    return float(gain), float(offset), float(r2)


def choose_control(rsq: np.ndarray, gains: np.ndarray, offsets: np.ndarray,
                   overlap_ratio: np.ndarray) -> int:
    eligible = (rsq >= .8) & (np.abs(gains - 1) <= .3) & (np.abs(offsets) <= 600)
    scores = np.sum(np.where(eligible, rsq * overlap_ratio, 0), axis=1)
    if not np.any(scores > 0):
        raise ValueError("CONTROL_SELECTION_FAILED")
    return int(np.argmax(scores))


def control_statistics(corrected: np.ndarray, temporal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = corrected.shape[2]
    rsq = np.zeros((count, count))
    gains = np.ones_like(rsq)
    offsets = np.zeros_like(rsq)
    overlap = np.zeros_like(rsq)
    for i in range(count):
        for j in range(i + 1, count):
            first, second = corrected[:, :, i], corrected[:, :, j]
            mask = (first != 0) & (second != 0) & (temporal > 0)
            if np.count_nonzero(mask) < 100:
                continue
            try:
                gain, offset, r2 = _regression(first[mask], second[mask])
            except ValueError:
                continue
            if r2 < .5:
                continue
            rsq[i, j] = rsq[j, i] = r2
            gains[i, j] = gains[j, i] = gain  # official stores same fitted direction
            offsets[i, j] = offsets[j, i] = offset
            overlap[i, j] = overlap[j, i] = np.count_nonzero((first != 0) & (second != 0)) / first.size
    return rsq, gains, offsets, overlap


def select_pif_subset(x: np.ndarray, y: np.ndarray, bins: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 3:
        raise ValueError("EMPTY_PIF_SUBSET")
    # Select_PIF_subset.m computes feature ranges before removing zero pairs.
    xmin, xmax = matlab_quantile(x, [.01, .99])
    ymin, ymax = matlab_quantile(y, [.01, .99])
    good = (x != 0) & (y != 0)
    x, y = x[good], y[good]
    if len(x) < 3:
        raise ValueError("EMPTY_PIF_SUBSET")
    if xmin == xmax or ymin == ymax:
        raise ValueError("DEGENERATE_PIF_RANGE")
    col = np.clip(np.ceil((x-xmin)/((xmax-xmin)/bins)), 0, bins+1).astype(int)
    row = np.clip(np.ceil((y-ymin)/((ymax-ymin)/bins)), 0, bins+1).astype(int)
    cell = row*(bins+2)+col
    size = (bins+2)**2
    count = np.bincount(cell, minlength=size).reshape(bins+2, bins+2)
    gx = np.bincount(cell, weights=x, minlength=size).reshape(count.shape)
    gy = np.bincount(cell, weights=y, minlength=size).reshape(count.shape)
    np.divide(gx, count, out=gx, where=count > 0)
    np.divide(gy, count, out=gy, where=count > 0)
    picked = []
    for col_id in range(bins+2):
        total = count[:, col_id].sum()
        if total:
            row_id = int(np.searchsorted(np.cumsum(count[:, col_id]), total/2, side="right"))
            picked.append((row_id, col_id))
    for row_id in range(bins+2):
        total = count[row_id, :].sum()
        if total:
            col_id = int(np.searchsorted(np.cumsum(count[row_id, :]), total/2, side="right"))
            picked.append((row_id, col_id))
    xs = np.array([gx[r, c] for r, c in picked])
    ys = np.array([gy[r, c] for r, c in picked])
    keep = (xs != 0) & (ys != 0)
    xs, ys = xs[keep], ys[keep]
    _, _, r2 = _regression(xs, ys)
    return xs, ys, r2


def connection_graph(data: np.ndarray, days: list[int], pif: np.ndarray,
                     day_threshold: int = 14) -> tuple[list[tuple[int, int, np.ndarray, np.ndarray]], np.ndarray]:
    count = data.shape[2]
    edges = []
    graph = np.zeros((count, count), dtype=bool)
    for i in range(count):
        for j in range(i + 1, count):
            if abs(days[i]-days[j]) > day_threshold:
                continue
            a, b = data[:, :, i], data[:, :, j]
            mask = pif & np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
            if np.count_nonzero(mask) < 50:
                continue
            try:
                x, y, r2 = select_pif_subset(a[mask], b[mask], 100)
            except ValueError:
                continue
            if r2 < 0:
                continue
            edges.append((i, j, x, y))
            graph[i, j] = True
    return edges, graph


@dataclass(frozen=True)
class JointResult:
    gains: np.ndarray
    offsets: np.ndarray
    rank: int
    condition: float
    residual_rmse: float
    tie_count: int


def solve_joint(scene_count: int, control: int,
                edges: list[tuple[int, int, np.ndarray, np.ndarray]]) -> JointResult:
    if not 0 <= control < scene_count:
        raise ValueError("CONTROL_SELECTION_FAILED")
    adjacency = [set() for _ in range(scene_count)]
    for i, j, x, y in edges:
        if len(x) != len(y) or not len(x):
            raise ValueError("EMPTY_EDGE_TIES")
        adjacency[i].add(j)
        adjacency[j].add(i)
    reached = {control}
    pending = [control]
    while pending:
        for neighbor in adjacency[pending.pop()]:
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    if len(reached) != scene_count:
        raise ValueError("DISCONNECTED_FROM_CONTROL")
    unknowns = [i for i in range(scene_count) if i != control]
    index = {scene: 2*k for k, scene in enumerate(unknowns)}
    n = 2*len(unknowns)
    normal = np.zeros((n, n), dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    ties = 0
    residual_sum = 0.0
    for i, j, x, y in edges:
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("INVALID_EDGE_TIES")
        a = np.zeros((len(x), n), dtype=np.float64)
        if i != control:
            a[:, index[i]] = x
            a[:, index[i]+1] = 1
        if j != control:
            a[:, index[j]] = -y
            a[:, index[j]+1] = -1
        l = y-x  # official observation equation at gain=1, offset=0
        normal += a.T @ a
        rhs += a.T @ l
        ties += len(x)
        residual_sum += float(l @ l)
    rank = int(np.linalg.matrix_rank(normal))
    if rank != n or ties <= n:
        raise ValueError(f"RANK_DEFICIENT:{rank}/{n}")
    condition = float(np.linalg.cond(normal))
    if not np.isfinite(condition):
        raise ValueError("INVALID_NORMAL_MATRIX")
    delta = np.linalg.solve(normal, rhs)
    gains = np.ones(scene_count)
    offsets = np.zeros(scene_count)
    for scene, k in index.items():
        gains[scene] += delta[k]
        offsets[scene] = delta[k+1]
    if not np.isfinite(gains).all() or not np.isfinite(offsets).all():
        raise ValueError("INVALID_COEFFICIENTS")
    return JointResult(gains, offsets, rank, condition,
                       float(np.sqrt(residual_sum/(ties-n))), ties)


def apply_coefficients(values: np.ndarray, valid_mask: np.ndarray,
                       gain: float, offset: float) -> np.ndarray:
    if not np.isfinite([gain, offset]).all():
        raise ValueError("INVALID_COEFFICIENTS")
    out = np.full(values.shape, np.nan, dtype=np.float32)
    valid = valid_mask & np.isfinite(values)
    out[valid] = (np.asarray(values[valid], dtype=np.float64)*gain+offset).astype(np.float32)
    return out
