from __future__ import annotations

import numpy as np
import pytest
from src.benchmark_v1_unified.radiometric import (
    compute_pair_direct_metrics,
    compute_pair_distribution_metrics,
)
from src.benchmark_v1_unified.structure import (
    compute_range_retention,
    compute_sobel_structure,
)


def _reader(array):
    def read(window):
        r, c = int(window.row_off), int(window.col_off)
        h, w = int(window.height), int(window.width)
        return array[r:r+h, c:c+w]
    return read


def test_streaming_pair_metrics_match_frozen_array_metrics_with_ties(tmp_path):
    from src.benchmark_v1_unified.streaming_metrics import compute_pair_metrics_streaming

    x = np.array([[1, 1, 2, 4], [5, 8, 8, 9]], dtype=np.float32)
    y = np.array([[2, 1, 2, 3], [7, 8, 6, 9]], dtype=np.float32)
    support = np.array([[1, 1, 0, 1], [1, 1, 1, 0]], dtype=bool)
    blocks = [
        (x[:1, :2], y[:1, :2], support[:1, :2]),
        (x[:1, 2:], y[:1, 2:], support[:1, 2:]),
        (x[1:, :], y[1:, :], support[1:, :]),
    ]

    actual = compute_pair_metrics_streaming(blocks, int(support.sum()), tmp_path)
    direct = compute_pair_direct_metrics(x, y, support)
    distribution = compute_pair_distribution_metrics(x, y, support)

    for name, expected in direct.__dict__.items():
        assert actual["direct"][name] == pytest.approx(expected, abs=1e-12) if isinstance(expected, float) else actual["direct"][name] == expected
    for name, expected in distribution.__dict__.items():
        assert actual["distribution"][name] == pytest.approx(expected, abs=1e-12) if isinstance(expected, float) else actual["distribution"][name] == expected
    assert not list(tmp_path.iterdir())


def test_external_order_statistics_match_numpy_linear_quantiles(tmp_path):
    from src.benchmark_v1_unified.streaming_metrics import ExternalOrderStatistics

    values = np.array([9, -2, 4, 4, 4, 0, 13, -2, 6, 1, 7], dtype=np.float64)
    with ExternalOrderStatistics(tmp_path, chunk_size=3) as stats:
        for part in (values[:2], values[2:7], values[7:]):
            stats.add(part)
        for q in (0.05, 0.5, 0.95, 1.0):
            assert stats.quantile(q) == pytest.approx(np.quantile(values, q, method="linear"), abs=1e-12)
    assert not list(tmp_path.iterdir())


def test_pair_metrics_can_consume_disk_backed_arrays_without_metric_drift(tmp_path):
    from src.benchmark_v1_unified.streaming_metrics import compute_pair_metrics_from_arrays

    x = np.memmap(tmp_path / "x.bin", mode="w+", dtype=np.float64, shape=(8,))
    y = np.memmap(tmp_path / "y.bin", mode="w+", dtype=np.float64, shape=(8,))
    x[:] = [0, 1, 2, 2, 5, 7, 7, 9]
    y[:] = [1, 1, 3, 4, 4, 7, 8, 8]
    expected_direct = compute_pair_direct_metrics(x.copy(), y.copy(), np.ones(8, bool))
    expected_distribution = compute_pair_distribution_metrics(x.copy(), y.copy(), np.ones(8, bool))
    actual = compute_pair_metrics_from_arrays(x, y, tmp_path / "scratch")
    for section, expected in (("direct", expected_direct), ("distribution", expected_distribution)):
        for key, value in expected.__dict__.items():
            observed = actual[section][key]
            assert observed == pytest.approx(value, abs=1e-12) if isinstance(value, float) else observed == value
    assert not list((tmp_path / "scratch").iterdir())


def test_tiled_sobel_structure_matches_frozen_including_raster_and_tile_edges(tmp_path):
    from src.benchmark_v1_unified.streaming_metrics import compute_sobel_structure_from_readers

    rng = np.random.default_rng(1610)
    x = rng.normal(size=(19, 23)).astype(np.float32)
    y = (x * 1.1 + 3).astype(np.float32)
    valid = np.ones(x.shape, dtype=bool)
    valid[7:10, 11] = False
    valid[:1, :3] = False

    actual = compute_sobel_structure_from_readers(
        x.shape, _reader(x), _reader(y), _reader(valid),
        block_shape=(5, 7), scratch_dir=tmp_path,
    )
    expected = compute_sobel_structure(x, y, valid).__dict__

    assert actual["status"] == expected["status"]
    assert actual["gradient_centers"] == expected["gradient_centers"]
    assert actual["edge_centers"] == expected["edge_centers"]
    for key in ("input_edge_threshold", "direction_loss", "relative_gradient_mae"):
        assert actual[key] == pytest.approx(expected[key], abs=1e-12)
    assert not list(tmp_path.iterdir())


def test_tiled_range_retention_matches_frozen_exact_quantiles(tmp_path):
    from src.benchmark_v1_unified.streaming_metrics import compute_range_retention_from_readers

    x = np.arange(143, dtype=np.float32).reshape(11, 13)
    y = x * 2.0 - 7
    mask = np.ones(x.shape, dtype=bool)
    mask[2:4, 8] = False
    actual = compute_range_retention_from_readers(
        x.shape, _reader(x), _reader(y), _reader(mask),
        block_shape=(4, 5), scratch_dir=tmp_path,
    )
    expected = compute_range_retention(x, y, mask).__dict__
    assert actual["status"] == expected["status"]
    assert actual["n_input"] == expected["n_input"]
    assert actual["n_output"] == expected["n_output"]
    assert actual["value"] == pytest.approx(expected["value"], abs=1e-12)
    assert actual["diagnostics"] == pytest.approx(expected["diagnostics"], abs=1e-12)
    assert not list(tmp_path.iterdir())


def test_streaming_seam_metrics_match_frozen_formula_across_row_blocks(tmp_path):
    from src.benchmark_v1_unified.seams import compute_seam_excess
    from src.benchmark_v1_unified.streaming_metrics import compute_seam_excess_from_readers

    x = np.array([[1, 2, 5, 5, 6], [1, 3, 7, 8, 9], [0, 2, 4, 7, 9.]], dtype=np.float32)
    labels = np.array([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1]], dtype=np.int16)
    valid = np.ones(x.shape, dtype=bool)
    valid[1, 4] = False
    actual = compute_seam_excess_from_readers(
        x.shape, _reader(x), _reader(labels), _reader(valid),
        block_rows=1, scratch_dir=tmp_path,
    )
    expected = compute_seam_excess(x, labels, valid)
    assert actual["status"] == expected["status"]
    assert actual["edge_count"] == expected["edge_count"]
    for key in ("mean_cross_jump", "mean_within_jump", "mean_excess", "p95_excess"):
        assert actual[key] == pytest.approx(expected[key], abs=1e-12)
    assert not list(tmp_path.iterdir())
