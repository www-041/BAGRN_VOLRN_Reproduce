"""Bounded-memory, exact implementations of frozen Unified V1 metrics.

The array-based functions remain the authority.  This module reads through
window iterators and uses disk-backed sorted runs for exact order statistics;
it never substitutes histograms or random samples.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import tempfile

import numpy as np
from rasterio.windows import Window
from scipy.ndimage import convolve

from .radiometric import PairDirectMetrics, PairDistributionMetrics
from .structure import KX, KY, MetricValue, StructureStats


_UINT64_MASK = (1 << 64) - 1
_UINT64_SIGN = 1 << 63


def _float_order_key(value: float) -> int:
    bits = int(np.asarray([value], dtype=np.float64).view(np.uint64)[0])
    return (~bits & _UINT64_MASK) if bits & _UINT64_SIGN else (bits ^ _UINT64_SIGN)


def _key_float(key: int) -> float:
    bits = (key ^ _UINT64_SIGN) if key & _UINT64_SIGN else (~key & _UINT64_MASK)
    return float(np.asarray([bits], dtype=np.uint64).view(np.float64)[0])


class ExternalOrderStatistics:
    """Exact linear quantiles with bounded RAM and one disk-backed run file."""

    def __init__(self, scratch_dir: str | Path, *, chunk_size: int = 1_000_000,
                 prefix: str = "order"):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.scratch_dir / f"{prefix}-{next(tempfile._get_candidate_names())}.f64"
        self.chunk_size = int(chunk_size)
        self._buffer = np.empty(self.chunk_size, dtype=np.float64)
        self._used = 0
        self.count = 0
        self._writer = self.path.open("wb")
        self._map = None
        self._runs: list[tuple[int, int]] = []
        self._selected: dict[int, float] = {}
        self.peak_file_bytes = 0
        self.closed = False

    def add(self, values) -> None:
        if self.closed or self._writer is None:
            raise RuntimeError("order-statistic store is closed")
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.isfinite(flat).all():
            raise ValueError("order-statistic values must be finite")
        offset = 0
        while offset < flat.size:
            take = min(self.chunk_size - self._used, flat.size - offset)
            self._buffer[self._used:self._used + take] = flat[offset:offset + take]
            self._used += take
            offset += take
            if self._used == self.chunk_size:
                self._flush_run()

    def _flush_run(self) -> None:
        if not self._used:
            return
        run = self._buffer[:self._used]
        run.sort(kind="quicksort")
        start = self.count
        run.tofile(self._writer)
        self._runs.append((start, self._used))
        self.count += self._used
        self._used = 0
        self._writer.flush()
        self.peak_file_bytes = max(self.peak_file_bytes, self.count * 8)

    def _finish(self) -> None:
        if self._map is not None:
            return
        self._flush_run()
        self._writer.flush()
        self._writer.close()
        self._writer = None
        if self.count:
            self._map = np.memmap(self.path, mode="r", dtype=np.float64, shape=(self.count,))

    def _run_views(self):
        self._finish()
        return [self._map[start:start + length] for start, length in self._runs]

    def kth(self, index: int) -> float:
        self._finish()
        if not 0 <= index < self.count:
            raise IndexError(index)
        if index in self._selected:
            return self._selected[index]
        views = self._run_views()
        low = min(_float_order_key(float(run[0])) for run in views)
        high = max(_float_order_key(float(run[-1])) for run in views)
        while low < high:
            middle = (low + high) // 2
            pivot = _key_float(middle)
            count_le = sum(int(np.searchsorted(run, pivot, side="right")) for run in views)
            if count_le > index:
                high = middle
            else:
                low = middle + 1
        value = _key_float(low)
        self._selected[index] = value
        return value

    def quantile(self, q: float) -> float:
        if not 0 <= q <= 1:
            raise ValueError("q must be in [0, 1]")
        self._finish()
        if not self.count:
            raise ValueError("cannot compute a quantile of empty support")
        h = (self.count - 1) * float(q)
        low = math.floor(h)
        high = math.ceil(h)
        a = self.kth(low)
        if low == high:
            return a
        b = self.kth(high)
        return float(a + (h - low) * (b - a))

    def close(self) -> None:
        if self.closed:
            return
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._map is not None:
            self._map._mmap.close()
            self._map = None
        self.closed = True
        self.path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def compute_pair_metrics_streaming(blocks, expected_count: int, scratch_dir: str | Path) -> dict:
    """Compute the frozen seven B1 metrics from (x, y, support) blocks."""
    expected_count = int(expected_count)
    if expected_count < 0:
        raise ValueError("expected_count cannot be negative")
    if expected_count == 0:
        return {
            "direct": asdict(PairDirectMetrics("NO_VALID_OVERLAP", 0, None, None, None, None)),
            "distribution": asdict(PairDistributionMetrics("NO_VALID_OVERLAP", 0, None, None, None, None, None, None, None)),
            "scratch_peak_bytes": 0,
        }

    x_all = np.empty(expected_count, dtype=np.float64)
    y_all = np.empty(expected_count, dtype=np.float64)
    offset = 0
    sum_abs = sum_sq = sum_signed = 0.0
    with ExternalOrderStatistics(scratch_dir, prefix="abs-error") as abs_stats:
        for x, y, support in blocks:
            a = np.asarray(x, dtype=np.float64)
            b = np.asarray(y, dtype=np.float64)
            mask = np.asarray(support, dtype=bool)
            if a.shape != b.shape or a.shape != mask.shape:
                raise ValueError("pair metric block shapes differ")
            if np.any(mask & (~np.isfinite(a) | ~np.isfinite(b))):
                raise ValueError("nonfinite pair value on frozen support")
            xa, ya = a[mask], b[mask]
            end = offset + xa.size
            if end > expected_count:
                raise ValueError("observed pair support exceeds frozen count")
            x_all[offset:end] = xa
            y_all[offset:end] = ya
            delta = xa - ya
            abs_delta = np.abs(delta)
            sum_abs += float(abs_delta.sum(dtype=np.float64))
            sum_sq += float(np.square(delta).sum(dtype=np.float64))
            sum_signed += float(delta.sum(dtype=np.float64))
            abs_stats.add(abs_delta)
            offset = end
        if offset != expected_count:
            raise ValueError(f"PAIR_SUPPORT_COUNT_MISMATCH:{offset}!={expected_count}")
        p95ae = abs_stats.quantile(0.95)
        scratch_peak = abs_stats.peak_file_bytes

    n = expected_count
    direct = PairDirectMetrics("EVALUABLE", n, sum_abs / n, math.sqrt(sum_sq / n), sum_signed / n, p95ae)
    mean_x = float(np.mean(x_all))
    mean_y = float(np.mean(y_all))
    ss_x = ss_y = 0.0
    for start in range(0, n, 1_000_000):
        stop = min(n, start + 1_000_000)
        ss_x += float(np.square(x_all[start:stop] - mean_x).sum(dtype=np.float64))
        ss_y += float(np.square(y_all[start:stop] - mean_y).sum(dtype=np.float64))
    std_x = math.sqrt(ss_x / n)
    std_y = math.sqrt(ss_y / n)
    x_all.sort(kind="quicksort")
    y_all.sort(kind="quicksort")
    w1_sum = 0.0
    for start in range(0, n, 1_000_000):
        stop = min(n, start + 1_000_000)
        w1_sum += float(np.abs(x_all[start:stop] - y_all[start:stop]).sum(dtype=np.float64))
    distribution = PairDistributionMetrics(
        "EVALUABLE", n, mean_x, mean_y, std_x, std_y,
        abs(mean_x - mean_y), abs(std_x - std_y), w1_sum / n,
    )
    return {"direct": asdict(direct), "distribution": asdict(distribution),
            "scratch_peak_bytes": scratch_peak}


def compute_pair_metrics_from_arrays(x, y, scratch_dir: str | Path, *, sort_in_place: bool = False,
                                     check=None) -> dict:
    """Compute all seven frozen pair metrics from equal-length finite arrays.

    ``sort_in_place`` is intended for disposable disk-backed pair spools and
    avoids making another full-size copy before exact W1 sorting.
    """
    a0, b0 = np.asarray(x), np.asarray(y)
    if a0.ndim != 1 or a0.shape != b0.shape:
        raise ValueError("pair arrays must be matching one-dimensional arrays")
    n = int(a0.size)
    if not n:
        return {
            "direct": asdict(PairDirectMetrics("NO_VALID_OVERLAP", 0, None, None, None, None)),
            "distribution": asdict(PairDistributionMetrics("NO_VALID_OVERLAP", 0, None, None, None, None, None, None, None)),
            "scratch_peak_bytes": 0,
        }
    if sort_in_place and (not a0.flags.writeable or not b0.flags.writeable):
        raise ValueError("in-place exact W1 requires writable arrays")
    a = a0 if sort_in_place else np.array(a0, dtype=np.float64, copy=True)
    b = b0 if sort_in_place else np.array(b0, dtype=np.float64, copy=True)
    sum_abs = sum_sq = sum_signed = 0.0
    mean_a_sum = mean_b_sum = 0.0
    with ExternalOrderStatistics(scratch_dir, prefix="abs-error") as abs_stats:
        for start in range(0, n, 1_000_000):
            if check:
                check()
            stop = min(n, start + 1_000_000)
            aa = np.asarray(a[start:stop], dtype=np.float64)
            bb = np.asarray(b[start:stop], dtype=np.float64)
            if not np.isfinite(aa).all() or not np.isfinite(bb).all():
                raise ValueError("nonfinite pair sample")
            delta = aa - bb
            absolute = np.abs(delta)
            sum_abs += float(absolute.sum(dtype=np.float64))
            sum_sq += float(np.square(delta).sum(dtype=np.float64))
            sum_signed += float(delta.sum(dtype=np.float64))
            mean_a_sum += float(aa.sum(dtype=np.float64))
            mean_b_sum += float(bb.sum(dtype=np.float64))
            abs_stats.add(absolute)
        p95ae = abs_stats.quantile(0.95)
        scratch_peak = abs_stats.peak_file_bytes
    mean_a, mean_b = mean_a_sum / n, mean_b_sum / n
    variance_a = variance_b = 0.0
    for start in range(0, n, 1_000_000):
        if check:
            check()
        stop = min(n, start + 1_000_000)
        aa = np.asarray(a[start:stop], dtype=np.float64)
        bb = np.asarray(b[start:stop], dtype=np.float64)
        variance_a += float(np.square(aa - mean_a).sum(dtype=np.float64))
        variance_b += float(np.square(bb - mean_b).sum(dtype=np.float64))
    std_a, std_b = math.sqrt(variance_a / n), math.sqrt(variance_b / n)
    if not sort_in_place:
        a.sort(kind="quicksort")
        b.sort(kind="quicksort")
    else:
        a.sort(kind="quicksort")
        b.sort(kind="quicksort")
    w1_sum = 0.0
    for start in range(0, n, 1_000_000):
        if check:
            check()
        stop = min(n, start + 1_000_000)
        w1_sum += float(np.abs(np.asarray(a[start:stop], dtype=np.float64)
                               - np.asarray(b[start:stop], dtype=np.float64)).sum(dtype=np.float64))
    return {
        "direct": asdict(PairDirectMetrics("EVALUABLE", n, sum_abs / n,
                                           math.sqrt(sum_sq / n), sum_signed / n, p95ae)),
        "distribution": asdict(PairDistributionMetrics(
            "EVALUABLE", n, mean_a, mean_b, std_a, std_b,
            abs(mean_a - mean_b), abs(std_a - std_b), w1_sum / n)),
        "scratch_peak_bytes": scratch_peak,
    }


def _iter_windows(shape: tuple[int, int], block_shape: tuple[int, int]):
    height, width = shape
    block_h, block_w = block_shape
    if block_h < 1 or block_w < 1:
        raise ValueError("block dimensions must be positive")
    for row in range(0, height, block_h):
        for col in range(0, width, block_w):
            yield Window(col, row, min(block_w, width - col), min(block_h, height - row))


def _read_halo(reader, window: Window, shape: tuple[int, int], halo: int = 1, fill=0):
    height, width = shape
    row0, col0 = int(window.row_off), int(window.col_off)
    rows, cols = int(window.height), int(window.width)
    r0, c0 = max(0, row0 - halo), max(0, col0 - halo)
    r1, c1 = min(height, row0 + rows + halo), min(width, col0 + cols + halo)
    source = np.asarray(reader(Window(c0, r0, c1 - c0, r1 - r0)))
    padded = np.full((rows + 2 * halo, cols + 2 * halo), fill,
                     dtype=source.dtype if np.issubdtype(source.dtype, np.number) else np.float64)
    dr, dc = r0 - (row0 - halo), c0 - (col0 - halo)
    padded[dr:dr + source.shape[0], dc:dc + source.shape[1]] = source
    return padded


def _gradient_core(values_halo: np.ndarray):
    values = np.asarray(values_halo, dtype=np.float64)
    gx = convolve(values, KX, mode="constant", cval=0.0)[1:-1, 1:-1]
    gy = convolve(values, KY, mode="constant", cval=0.0)[1:-1, 1:-1]
    return gx, gy


def _sobel_block(window, shape, input_reader, output_reader, support_reader):
    mask_halo = _read_halo(support_reader, window, shape, fill=False).astype(bool)
    x_halo = _read_halo(input_reader, window, shape, fill=0)
    y_halo = _read_halo(output_reader, window, shape, fill=0)
    core = np.s_[1:-1, 1:-1]
    if np.any(mask_halo & (~np.isfinite(x_halo) | ~np.isfinite(y_halo))):
        raise ValueError("nonfinite structure value on frozen support")
    x_halo = np.where(mask_halo, x_halo, 0.0)
    y_halo = np.where(mask_halo, y_halo, 0.0)
    gx, gy = _gradient_core(x_halo)
    hx, hy = _gradient_core(y_halo)
    eligible = convolve(mask_halo.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
                        mode="constant", cval=0)[core] == 9
    mag_x = np.hypot(gx, gy)
    mag_y = np.hypot(hx, hy)
    return gx, gy, hx, hy, mag_x, mag_y, eligible


def compute_sobel_structure_from_readers(shape, input_reader, output_reader, support_reader,
                                         *, block_shape=(512, 512), scratch_dir: str | Path, check=None):
    """Frozen Sobel metric streamed with one-pixel halo on every tile."""
    shape = tuple(map(int, shape))
    n = 0
    with ExternalOrderStatistics(scratch_dir, prefix="sobel-input-gradient") as stats:
        for window in _iter_windows(shape, block_shape):
            if check:
                check()
            values = _sobel_block(window, shape, input_reader, output_reader, support_reader)
            mag_x, eligible = values[4], values[6]
            n += int(eligible.sum())
            stats.add(mag_x[eligible])
        if not n:
            return asdict(StructureStats("NO_VALID_GRADIENT_CENTERS", 0, 0, None, None, None))
        tau = max(1e-12, stats.quantile(0.5))
        scratch_peak = stats.peak_file_bytes

    edge_count = 0
    direction_sum = 0.0
    gradient_sum = 0.0
    gradient_error_sum = 0.0
    for window in _iter_windows(shape, block_shape):
        if check:
            check()
        gx, gy, hx, hy, mag_x, mag_y, eligible = _sobel_block(
            window, shape, input_reader, output_reader, support_reader)
        edges = eligible & (mag_x > tau)
        ne = int(edges.sum())
        if ne:
            denominator = mag_x[edges] * mag_y[edges]
            cosine = np.zeros(ne, dtype=np.float64)
            nonzero = denominator > 1e-12
            cosine[nonzero] = ((gx[edges][nonzero] * hx[edges][nonzero]
                                + gy[edges][nonzero] * hy[edges][nonzero])
                               / denominator[nonzero])
            direction_sum += float((1.0 - np.clip(cosine, -1.0, 1.0)).sum(dtype=np.float64))
            edge_count += ne
        gradient_sum += float(mag_x[eligible].sum(dtype=np.float64))
        gradient_error_sum += float(np.abs(mag_y[eligible] - mag_x[eligible]).sum(dtype=np.float64))
    if not edge_count:
        return asdict(StructureStats("NOT_EVALUABLE_NO_EDGES", n, 0, tau, None, None))
    if gradient_sum == 0:
        return asdict(StructureStats("NOT_EVALUABLE_FLAT_INPUT", n, edge_count, tau, None, None))
    return {**asdict(StructureStats("EVALUABLE", n, edge_count, tau,
                                    direction_sum / edge_count, gradient_error_sum / gradient_sum)),
            "scratch_peak_bytes": scratch_peak}


def compute_range_retention_from_readers(shape, input_reader, output_reader, support_reader,
                                         *, block_shape=(512, 512), scratch_dir: str | Path, check=None):
    """Frozen P05/P95 range retention with exact external order statistics."""
    shape = tuple(map(int, shape))
    support_count = 0
    for window in _iter_windows(shape, block_shape):
        if check:
            check()
        mask = np.asarray(support_reader(window), dtype=bool)
        x = np.asarray(input_reader(window))
        y = np.asarray(output_reader(window))
        if x.shape != y.shape or x.shape != mask.shape:
            raise ValueError("range metric block shapes differ")
        if np.any(mask & (~np.isfinite(x) | ~np.isfinite(y))):
            raise ValueError("nonfinite range value on frozen support")
        support_count += int(mask.sum())
    if not support_count:
        return asdict(MetricValue(None, "ratio", "NO_VALID_SUPPORT", 0, 0, {}))

    quantiles = []
    scratch_peak = 0
    for reader, prefix in ((input_reader, "range-input"), (output_reader, "range-output")):
        with ExternalOrderStatistics(scratch_dir, prefix=prefix) as stats:
            for window in _iter_windows(shape, block_shape):
                if check:
                    check()
                mask = np.asarray(support_reader(window), dtype=bool)
                stats.add(np.asarray(reader(window), dtype=np.float64)[mask])
            q05, q95 = stats.quantile(0.05), stats.quantile(0.95)
            if stats.count != support_count:
                raise ValueError(f"RANGE_SUPPORT_COUNT_MISMATCH:{stats.count}!={support_count}")
            quantiles.append((q05, q95))
            scratch_peak = max(scratch_peak, stats.peak_file_bytes)

    input_range = float(quantiles[0][1] - quantiles[0][0])
    output_range = float(quantiles[1][1] - quantiles[1][0])
    if input_range == 0:
        return asdict(MetricValue(None, "ratio", "NOT_EVALUABLE_FLAT_INPUT", support_count,
                                  support_count, {"input_range": input_range}))
    return {"value": output_range / input_range, "unit": "ratio", "status": "EVALUABLE",
            "n_input": support_count, "n_output": support_count,
            "diagnostics": {"input_range": input_range, "output_range": output_range},
            "scratch_peak_bytes": scratch_peak}


def compute_seam_excess_from_readers(shape, mosaic_reader, labels_reader, valid_reader,
                                     *, block_rows: int = 64, scratch_dir: str | Path, check=None):
    """Stream the frozen hard-label seam formula by complete raster rows."""
    height, width = map(int, shape)
    if block_rows < 1:
        raise ValueError("block_rows must be positive")

    def accumulate(baseline: float | None, order_stats=None):
        cross_sum = within_sum = 0.0
        cross_count = within_count = 0
        for row0 in range(0, height, block_rows):
            if check:
                check()
            row1 = min(height, row0 + block_rows)
            read0, read1 = max(0, row0 - 1), min(height, row1 + 2)
            window = Window(0, read0, width, read1 - read0)
            x = np.asarray(mosaic_reader(window), dtype=np.float64)
            labels = np.asarray(labels_reader(window))
            valid = np.asarray(valid_reader(window), dtype=bool)
            if x.shape != labels.shape or x.shape != valid.shape:
                raise ValueError("seam metric block shapes differ")
            for row in range(row0, row1):
                local = row - read0
                values = x[local]
                source = labels[local]
                mask = valid[local]
                if width > 1:
                    a, b = values[:-1], values[1:]
                    la, lb = source[:-1], source[1:]
                    ok = mask[:-1] & mask[1:] & np.isfinite(a) & np.isfinite(b) & (la >= 0) & (lb >= 0)
                    edge = ok & (la != lb)
                    jumps = np.abs(a - b)[edge]
                    cross_count += int(jumps.size)
                    cross_sum += float(jumps.sum(dtype=np.float64))
                    if order_stats is not None and jumps.size:
                        order_stats.add(jumps - float(baseline))
                    if width > 2:
                        left = (edge[1:] & mask[:-2]
                                & (source[:-2] == source[1:-1]))
                        right = (edge[:-1] & mask[2:]
                                 & (source[2:] == source[1:-1]))
                        within_left = np.abs(values[1:-1] - values[:-2])[left]
                        within_right = np.abs(values[2:] - values[1:-1])[right]
                        within_count += int(within_left.size + within_right.size)
                        within_sum += float(within_left.sum(dtype=np.float64) + within_right.sum(dtype=np.float64))

                if row + 1 < height:
                    next_values = x[local + 1]
                    next_source = labels[local + 1]
                    next_mask = valid[local + 1]
                    a, b = values, next_values
                    la, lb = source, next_source
                    ok = mask & next_mask & np.isfinite(a) & np.isfinite(b) & (la >= 0) & (lb >= 0)
                    edge = ok & (la != lb)
                    jumps = np.abs(a - b)[edge]
                    cross_count += int(jumps.size)
                    cross_sum += float(jumps.sum(dtype=np.float64))
                    if order_stats is not None and jumps.size:
                        order_stats.add(jumps - float(baseline))
                    if row > 0:
                        prev_values, prev_source, prev_mask = x[local - 1], labels[local - 1], valid[local - 1]
                        within = edge & prev_mask & (prev_source == source)
                        differences = np.abs(values - prev_values)[within]
                        within_count += int(differences.size)
                        within_sum += float(differences.sum(dtype=np.float64))
                    if row + 2 < height:
                        after_values, after_source, after_mask = x[local + 2], labels[local + 2], valid[local + 2]
                        within = edge & after_mask & (after_source == next_source)
                        differences = np.abs(after_values - next_values)[within]
                        within_count += int(differences.size)
                        within_sum += float(differences.sum(dtype=np.float64))
        return cross_sum, cross_count, within_sum, within_count

    cross_sum, cross_count, within_sum, within_count = accumulate(None)
    if not cross_count:
        return {"status": "NOT_APPLICABLE_NO_SEAM", "edge_count": 0, "mean_excess": None}
    baseline = within_sum / within_count if within_count else 0.0
    with ExternalOrderStatistics(scratch_dir, prefix="seam-excess") as stats:
        _, second_count, _, _ = accumulate(baseline, stats)
        p95 = stats.quantile(0.95)
        if second_count != cross_count or stats.count != cross_count:
            raise ValueError("SEAM_EDGE_COUNT_CHANGED_DURING_PASSES")
        scratch_peak = stats.peak_file_bytes
    mean_cross = cross_sum / cross_count
    return {"status": "EVALUABLE", "edge_count": cross_count,
            "mean_cross_jump": mean_cross, "mean_within_jump": baseline,
            "mean_excess": mean_cross - baseline, "p95_excess": p95,
            "scratch_peak_bytes": scratch_peak}
