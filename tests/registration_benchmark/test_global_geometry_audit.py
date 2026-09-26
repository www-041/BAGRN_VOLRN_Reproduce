from __future__ import annotations

import numpy as np

from src.multiscene_sift.global_geometry_audit import (
    build_translation_deltas,
    compute_edge_residual_metrics,
    compute_cycle_diagnostics,
    cycle_basis,
    summarize_residuals,
    tree_non_tree_summary,
    occupancy_count,
)


def _translation(dx: float, dy: float = 0.0) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def test_compute_edge_residual_metrics_exact_and_known_offset() -> None:
    points = np.array([[0.0, 0.0], [10.0, 4.0], [3.0, 7.0]])
    edges = [
        {"pair_i": 0, "pair_j": 1, "tree_edge": True},
        {"pair_i": 0, "pair_j": 2, "tree_edge": False},
    ]
    bundles = {
        (0, 1): {"ref_xy": points, "tgt_xy": points, "coverage": 0.8},
        (0, 2): {"ref_xy": points, "tgt_xy": points, "coverage": 0.7},
    }
    rows = compute_edge_residual_metrics(
        edges,
        bundles,
        {0: _translation(0), 1: _translation(0), 2: _translation(28)},
        pixel_size_m=14.0,
        matcher="synthetic",
        global_method="MST",
    )

    assert len(rows) == 2
    exact, offset = rows
    assert exact["inlier_count"] == 3
    assert exact["rmse_pixel"] == 0.0
    assert exact["p95_pixel"] == 0.0
    assert offset["rmse_pixel"] == 2.0
    assert offset["p95_pixel"] == 2.0
    assert offset["rmse_world_m"] == 28.0
    assert offset["coverage"] == 0.7


def test_edge_metrics_are_independent_of_other_edge_sample_count() -> None:
    easy = np.column_stack([np.arange(10_000, dtype=float), np.zeros(10_000)])
    hard = np.column_stack([np.arange(10, dtype=float), np.zeros(10)])
    edges = [
        {"pair_i": 0, "pair_j": 1, "tree_edge": True},
        {"pair_i": 0, "pair_j": 2, "tree_edge": False},
    ]
    bundles = {
        (0, 1): {"ref_xy": easy, "tgt_xy": easy},
        (0, 2): {"ref_xy": hard, "tgt_xy": hard},
    }
    rows = compute_edge_residual_metrics(
        edges,
        bundles,
        {0: _translation(0), 1: _translation(0), 2: _translation(70)},
        pixel_size_m=14.0,
    )
    assert rows[0]["rmse_pixel"] == 0.0
    assert rows[1]["rmse_pixel"] == 5.0
    assert rows[1]["p95_pixel"] == 5.0


def test_point_weighted_and_edge_balanced_statistics_are_both_reported() -> None:
    rows = [
        {"rmse_pixel": 0.0, "p95_pixel": 0.0},
        {"rmse_pixel": 10.0, "p95_pixel": 10.0},
    ]
    summary = summarize_residuals(rows, {
        (0, 1): np.zeros(100),
        (0, 2): np.full(1, 10.0),
    })
    assert summary["point_weighted"]["rmse_pixel"] < 2.0
    assert summary["edge_balanced"]["mean_edge_rmse_pixel"] == 5.0
    assert summary["point_weighted"]["p95_pixel"] != summary["edge_balanced"]["mean_edge_p95_pixel"]


def test_tree_non_tree_summary_uses_explicit_tree_flags() -> None:
    rows = [
        {"pair_i": 0, "pair_j": 1, "tree_edge": True, "p95_pixel": 1.0},
        {"pair_i": 0, "pair_j": 2, "tree_edge": True, "p95_pixel": 3.0},
        {"pair_i": 1, "pair_j": 2, "tree_edge": False, "p95_pixel": 5.0},
    ]
    out = tree_non_tree_summary(rows, "synthetic", "MST")
    assert out["tree_edge_count"] == 2
    assert out["tree_mean_p95_pixel"] == 2.0
    assert out["non_tree_edge_count"] == 1
    assert out["non_tree_max_p95_pixel"] == 5.0


def test_translation_deltas_join_by_unordered_pair() -> None:
    mst = [
        {"matcher": "x", "pair_i": 1, "pair_j": 0, "tree_edge": True, "rmse_pixel": 2.0, "p95_pixel": 3.0, "max_pixel": 4.0},
        {"matcher": "x", "pair_i": 0, "pair_j": 2, "tree_edge": False, "rmse_pixel": 5.0, "p95_pixel": 6.0, "max_pixel": 7.0},
    ]
    translation = [
        {"matcher": "x", "pair_i": 0, "pair_j": 2, "tree_edge": False, "rmse_pixel": 4.0, "p95_pixel": 6.0, "max_pixel": 8.0},
        {"matcher": "x", "pair_i": 0, "pair_j": 1, "tree_edge": True, "rmse_pixel": 1.0, "p95_pixel": 2.0, "max_pixel": 3.0},
    ]
    rows, summary = build_translation_deltas(mst, translation)
    assert [row["pair"] for row in rows] == ["0-1", "0-2"]
    assert rows[0]["delta_rmse_pixel"] == -1.0
    assert rows[1]["delta_rmse_pixel"] == -1.0
    assert summary["improved_count"] == 2


def test_cycle_diagnostics_reports_perfect_and_inconsistent_loops() -> None:
    identity = np.eye(3)
    bad = _translation(3.0)
    bundles = {
        (0, 1): {"pair_pixel_matrix": identity},
        (1, 2): {"pair_pixel_matrix": identity},
        (0, 2): {"pair_pixel_matrix": bad},
    }
    rows = compute_cycle_diagnostics(bundles, [(0, 1), (1, 2), (0, 2)], 3)
    assert len(rows) == 1
    assert rows[0]["cycle_length"] == 3
    assert rows[0]["translation_closure_px"] == 3.0
    assert rows[0]["linear_part_error"] == 0.0


def test_cycle_basis_is_deterministic_and_has_expected_rank() -> None:
    edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    cycles = cycle_basis(edges, 5)
    assert len(cycles) == 6
    assert cycles == cycle_basis(list(reversed(edges)), 5)


def test_coverage_occupancy_counts_known_cells_exactly() -> None:
    points = np.array([[1, 1], [1.5, 1.5], [15, 8], [19, 19]], dtype=float)
    assert occupancy_count(points, (0, 0, 20, 20), (2, 2)) == 3
