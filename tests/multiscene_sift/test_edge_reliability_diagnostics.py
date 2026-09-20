"""Synthetic tests for the edge-reliability diagnostics.

No real imagery or matching required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from rasterio.transform import Affine

from src.multiscene_sift import edge_reliability_diagnostics as erd


def _shape():
    return (640, 640)


def _uniform(shape, n=120, seed=1):
    rng = np.random.default_rng(seed)
    h, w = shape
    return np.column_stack([rng.uniform(20, w - 20, n),
                            rng.uniform(20, h - 20, n)])


def _clustered(shape, n=120, seed=2):
    rng = np.random.default_rng(seed)
    return rng.normal([64, 64], 12, (n, 2))


def test_diagnostic_edge_set_exact():
    assert erd.DIAGNOSTIC_EDGES == [
        (0, 6), (4, 6), (1, 8), (2, 5), (0, 5), (5, 6), (0, 1)]
    assert len(erd.DIAGNOSTIC_EDGES) == 7
    assert (0, 6) in erd.DIAGNOSTIC_EDGES
    assert (2, 5) in erd.DIAGNOSTIC_EDGES


def test_reproduce_gate_status():
    saved = {"inliers": 10, "raw_matches": 100, "residual_rmse": 0.5,
             "residual_p95": 0.9}

    class R:
        inliers = 10
        raw_matches = 100
        residual_rmse = 0.5
        residual_p95 = 0.9

    assert erd.reproduce_gate_status(saved, R()) == "EXACT"

    class R2(R):
        residual_rmse = 0.51

    assert erd.reproduce_gate_status(saved, R2()) == "CLOSE"

    class R3(R):
        inliers = 8

    assert erd.reproduce_gate_status(saved, R3()) == "MISMATCH"
    assert erd.reproduce_gate_status(None, R()) == "NO_SAVED_BASELINE"


def test_occupancy_uniform():
    m = erd.compute_inlier_spatial_metrics(_uniform(_shape()), None, _shape())
    assert m["occupancy_4x4"]["occupancy_ratio"] > 0.8
    assert m["occupancy_8x8"]["occupancy_ratio"] > 0.8


def test_occupancy_clustered_low_and_bias():
    m = erd.compute_inlier_spatial_metrics(_clustered(_shape()), None, _shape())
    assert m["occupancy_8x8"]["occupancy_ratio"] < 0.3
    assert m["centroid"]["centroid_offset_normalized"] > 0.2


def test_convex_hull_and_quadrants():
    shape = _shape()
    m = erd.compute_inlier_spatial_metrics(_uniform(shape), None, shape)
    assert m["hull_coverage"] > 0.5
    assert m["quadrants"]["entropy"] > 0.8
    # full square occupancy area for hull coverage denominator
    area = float(shape[0] * shape[1])
    assert 0.0 <= erd.convex_hull_coverage(_uniform(shape), area) <= 1.0


def test_nearest_control_uniform_vs_clustered():
    shape = _shape()
    samples = erd.sample_grid_points(shape, 32)
    d_u = erd.nearest_control_statistics(_uniform(shape), samples)
    d_c = erd.nearest_control_statistics(_clustered(shape), samples)
    assert d_u["p95_nearest_control_px"] < d_c["p95_nearest_control_px"]
    assert d_u["fraction_farther_than_128px"] < d_c["fraction_farther_than_128px"]


def test_residual_spatial_constant():
    shape = _shape()
    xy = _uniform(shape, n=60)
    r = np.full(len(xy), 0.5)
    s = erd.summarize_residual_spatial_pattern(xy, r, shape)
    assert s["quadrant_medians"] == pytest.approx([0.5] * 4)
    # correlation under zero variance is undefined (nan) or ~0
    assert np.isnan(s["corr_residual_x"]) or abs(s["corr_residual_x"]) < 1e-6


def test_residual_spatial_gradient():
    shape = _shape()
    xy = _uniform(shape, n=60)
    r = 0.2 + 0.01 * xy[:, 0]
    s = erd.summarize_residual_spatial_pattern(xy, r, shape)
    assert s["corr_residual_x"] > 0.9
    assert s["linear_fit"]["r2"] > 0.9


def test_direct_residual_zero_case():
    tiles = [{"accepted": True, "phase_mag_px": 0.4 - 0.1 * k} for k in range(12)]
    res = erd.summarize_direct_residual_field({"tiles": tiles})
    assert res["median_px"] <= 0.5
    assert res["fraction_tiles_gt5px"] == 0.0


def test_direct_residual_localized_shift():
    tiles = ([{"accepted": True, "phase_mag_px": 0.3} for _ in range(8)] +
             [{"accepted": True, "phase_mag_px": 8.1} for _ in range(4)])
    res = erd.summarize_direct_residual_field({"tiles": tiles})
    assert res["median_px"] < 2.0
    assert res["fraction_tiles_gt5px"] > 0.3


def test_residual_vs_control_support_positive():
    rng = np.random.default_rng(4)
    # control points sit at grid-pixel (14, 14); tiles run to the right,
    # so both nearest-control distance and phase residual increase together.
    pts = rng.normal([14, 14], 2, (40, 2))
    grid_t = Affine(14.0, 0.0, 0.0, 0.0, -14.0, 0.0)
    tiles = []
    for d in np.linspace(30, 400, 12):
        tiles.append({"accepted": True,
                      "center_world_x": float(d * 14.0),
                      "center_world_y": 200.0,
                      "phase_mag_px": float(d / 20.0)})
    r = erd.relate_residual_to_control_support(tiles, pts, grid_t)
    assert r["corr_spearman_phase_vs_distance"] > 0.9


def test_classifier_reliable():
    ev = {"inliers": 3000, "occupancy_8x8": 0.9,
          "direct_phase_median_px": 0.5, "fraction_tiles_gt5px": 0.0,
          "corr_spearman_phase_vs_distance": 0.2,
          "fraction_farther_than_128px": 0.1}
    assert erd.classify_edge_reliability_pattern(ev) == "DIRECT_GEOMETRY_RELIABLE"


def test_classifier_low_support():
    ev = {"inliers": 30, "occupancy_8x8": 0.05,
          "direct_phase_median_px": 40.0, "fraction_tiles_gt5px": 0.9,
          "corr_spearman_phase_vs_distance": None,
          "fraction_farther_than_128px": None}
    assert erd.classify_edge_reliability_pattern(ev) == "LOW_SUPPORT_EDGE"


def test_classifier_coverage_limited():
    ev = {"inliers": 3000, "occupancy_8x8": 0.4,
          "direct_phase_median_px": 7.0, "fraction_tiles_gt5px": 0.6,
          "corr_spearman_phase_vs_distance": 0.7,
          "fraction_farther_than_128px": 0.4}
    assert erd.classify_edge_reliability_pattern(
        ev) == "COVERAGE_LIMITED_EXTRAPOLATION_SUSPECT"


def test_classifier_local_inconsistent():
    ev = {"inliers": 3000, "occupancy_8x8": 0.85,
          "direct_phase_median_px": 9.0, "fraction_tiles_gt5px": 0.8,
          "corr_spearman_phase_vs_distance": 0.1,
          "fraction_farther_than_128px": 0.1}
    assert erd.classify_edge_reliability_pattern(
        ev) == "LOCAL_RESIDUAL_FIELD_INCONSISTENT"


def test_feature_table_schema():
    columns = ["edge", "group", "raw_matches", "inliers", "inlier_ratio",
               "rmse_px", "p95_px", "old_coverage", "occupancy_4x4",
               "occupancy_8x8", "hull_coverage", "quadrant_entropy",
               "centroid_bias", "p95_nearest_control_distance_px",
               "fraction_farther_than_128px", "local_phase_median_px",
               "fraction_tiles_gt5px", "corr_phase_vs_control_distance",
               "diagnosis"]
    assert set(columns) >= {"edge", "inliers", "rmse_px", "occupancy_8x8",
                            "hull_coverage", "local_phase_median_px",
                            "fraction_tiles_gt5px", "diagnosis"}