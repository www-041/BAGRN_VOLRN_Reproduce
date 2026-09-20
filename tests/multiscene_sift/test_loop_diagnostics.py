"""Synthetic tests for loop-closure diagnostics.

These tests never touch real DZ01V imagery — they construct artificial
consistency rows, spanning trees, pairwise rows and point sets and verify the
diagnostic functions on them.  Running the CLI on real data is the user's job.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from rasterio.transform import from_origin

from src.multiscene_sift import loop_diagnostics as diag
from src.multiscene_sift.global_registration import global_consistency_diagnostics
from src.multiscene_sift.models import PairwiseRegistration


def _tree_edges(entries):
    """Build the spanning-tree edge dicts (parent/child as in the runner)."""
    return [
        {"parent": p, "child": c, "depth": d, "weight": w}
        for (p, c, d, w) in entries
    ]


def _consistency_row(i, j, in_tree, p95, n=100, median=1.0, rmse=1.1, mx=2.0):
    return {
        "idx_i": i,
        "idx_j": j,
        "in_tree": in_tree,
        "n_points": n,
        "global_median_px": median,
        "global_rmse_px": rmse,
        "global_p90_px": p95 * 0.9,
        "global_p95_px": p95,
        "global_max_px": mx,
    }


def _make_pair(i, j, matrix=None, n=100, seed=7):
    """Synthetic OK pairwise registration in a 30 m/pix common grid."""
    if matrix is None:
        matrix = np.array(
            [[1.0, 0.01, 16.0], [-0.008, 1.0, -53.0], [0.0, 0.0, 1.0]]
        )
    rng = np.random.default_rng(seed)
    tgt = rng.uniform(50.0, 950.0, size=(n, 2))
    ref = (matrix[:2, :2] @ tgt.T).T + matrix[:2, 2]
    common = from_origin(500000.0, 4000000.0, 30.0, 30.0)
    return PairwiseRegistration(
        idx_i=i,
        idx_j=j,
        status="OK",
        raw_matches=n,
        inliers=n,
        inlier_ratio=1.0,
        coverage=0.5,
        residual_median=0.55,
        residual_rmse=0.62,
        residual_p95=1.1,
        pair_pixel_matrix=matrix.tolist(),
        pair_common_transform=common,
        runtime_sec=0.0,
        matcher="sift",
        matcher_runtime_sec=0.0,
        geometry_runtime_sec=0.0,
        inlier_ref_xy=ref,
        inlier_tgt_xy=tgt,
    )


# ---------------------------------------------------------------------------
# Task 1 — edge / path identification
# ---------------------------------------------------------------------------


def test_find_worst_non_tree_edge_uses_p95():
    rows = [
        _consistency_row(0, 1, in_tree=False, p95=10.0),
        _consistency_row(0, 2, in_tree=True, p95=60.0),   # tree edge, ignored
        _consistency_row(1, 3, in_tree=False, p95=3.0),
        _consistency_row(3, 4, in_tree=True, p95=1.0),
    ]
    assert diag.find_worst_non_tree_edge(rows) == (0, 1)


def test_find_worst_non_tree_edge_demo_five_scene_rows():
    # The SIFT five-scene run: 0-1 is the only non-tree edge.
    rows = [
        _consistency_row(0, 1, False, 53.6481, n=69, median=52.0864,
                         rmse=51.9735, mx=54.9142),
        _consistency_row(0, 2, True, 1.7439, n=56),
        _consistency_row(0, 4, True, 1.5636, n=128),
        _consistency_row(1, 4, True, 1.1427, n=3562),
        _consistency_row(3, 4, True, 1.7303, n=423),
    ]
    assert diag.find_worst_non_tree_edge(rows) == (0, 1)


def test_find_worst_non_tree_edge_none_when_all_tree():
    rows = [
        _consistency_row(0, 1, True, 1.0),
        _consistency_row(1, 2, True, 2.0),
    ]
    assert diag.find_worst_non_tree_edge(rows) is None


def test_load_consistency_rows_parses_types(tmp_path):
    csv_path = tmp_path / "global_edge_consistency.csv"
    csv_path.write_text(
        "idx_i,idx_j,in_tree,n_points,global_median_px,global_rmse_px,"
        "global_p90_px,global_p95_px,global_max_px\n"
        "0,1,False,69,52.0864,51.9735,53.4322,53.6481,54.9142\n"
        "3,4,true,423,0.967,1.061,1.5953,1.7303,2.2558\n",
        encoding="utf-8",
    )
    rows = diag.load_consistency_rows(csv_path)
    assert rows[0]["idx_i"] == 0 and rows[0]["in_tree"] is False
    assert abs(rows[0]["global_p95_px"] - 53.6481) < 1e-9
    assert rows[1]["in_tree"] is True
    assert rows[1]["n_points"] == 423


def test_find_tree_path_direct_and_multihop():
    edges = _tree_edges([
        (4, 1, 1, 11.0),
        (4, 3, 1, 9.0),
        (4, 0, 1, 8.0),
        (0, 2, 2, 4.0),
    ])
    assert diag.find_tree_path(edges, 0, 1) == [0, 4, 1]
    assert diag.find_tree_path(edges, 2, 1) == [2, 0, 4, 1]
    assert diag.find_tree_path(edges, 4, 4) == [4]


def test_find_tree_path_disconnected_returns_none():
    edges = _tree_edges([(3, 4, 1, 5.0)])
    assert diag.find_tree_path(edges, 0, 1) is None


# ---------------------------------------------------------------------------
# Task 2 — direct vs MST-implied transforms
# ---------------------------------------------------------------------------


def test_pair_direct_world_transform_direction():
    pair = _make_pair(0, 1)
    row = diag.registration_as_pair_row(pair)
    A = diag.pair_direct_world_transform(
        [row], 0, 1, pair.pair_common_transform
    )
    assert A is not None
    # A must map the target (scene 1) world point onto the ref (scene 0) point.
    k = 17
    tgt_world = np.array([*pair.pair_common_transform * pair.inlier_tgt_xy[k], 1.0])
    ref_world = np.array([*pair.pair_common_transform * pair.inlier_ref_xy[k], 1.0])
    np.testing.assert_allclose(A @ tgt_world, ref_world, atol=1e-6)


def test_pair_direct_world_transform_reversed_row_inverts():
    # A stored (j, i) row must produce the same j->i mapping as (i, j).
    # In _make_pair(1, 0) the ref points belong to scene 1 and the tgt points
    # belong to scene 0, so the returned matrix must map scene 1 -> scene 0.
    pair = _make_pair(1, 0)
    row = diag.registration_as_pair_row(pair)
    A = diag.pair_direct_world_transform(
        [row], 0, 1, pair.pair_common_transform
    )
    assert A is not None
    k = 5
    scene1_world = np.array([*pair.pair_common_transform * pair.inlier_ref_xy[k], 1.0])
    scene0_world = np.array([*pair.pair_common_transform * pair.inlier_tgt_xy[k], 1.0])
    np.testing.assert_allclose(A @ scene1_world, scene0_world, atol=1e-5)


def test_mst_implied_transform_direction():
    pair = _make_pair(0, 1)
    A_world = diag.pixel_affine_to_world(
        np.asarray(pair.pair_pixel_matrix), pair.pair_common_transform
    )
    G = [np.eye(3), A_world]
    T = diag.mst_implied_transform(G, 0, 1)
    np.testing.assert_allclose(T, A_world, atol=1e-12)


def test_compare_pure_translation_decomposition():
    direct = np.eye(3)
    mst = np.array([[1.0, 0.0, 40.0], [0.0, 1.0, 30.0], [0.0, 0.0, 1.0]])
    diff = diag.compare_direct_and_mst_transforms(direct, mst, 1.0, 1.0)
    assert diff["translation_x_px"] == pytest.approx(40.0, abs=1e-6)
    assert diff["translation_y_px"] == pytest.approx(30.0, abs=1e-6)
    assert diff["translation_magnitude_px"] == pytest.approx(50.0, abs=1e-6)
    assert diff["rotation_deg"] == pytest.approx(0.0, abs=1e-9)
    assert diff["scale_x"] == pytest.approx(1.0, abs=1e-9)
    assert diff["scale_y"] == pytest.approx(1.0, abs=1e-9)


def test_compare_rotation_decomposition():
    theta = math.radians(1.0)
    R = np.array(
        [[math.cos(theta), -math.sin(theta), 0.0],
         [math.sin(theta), math.cos(theta), 0.0],
         [0.0, 0.0, 1.0]]
    )
    diff = diag.compare_direct_and_mst_transforms(R, np.eye(3), 1.0, 1.0)
    assert abs(diff["rotation_deg"]) == pytest.approx(1.0, abs=1e-6)
    assert diff["scale_x"] == pytest.approx(1.0, abs=1e-6)
    assert diff["scale_y"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Task 4 — point residuals reproduce the runner's formula
# ---------------------------------------------------------------------------


def test_pair_point_residuals_match_runner_aggregates():
    pair = _make_pair(0, 1, n=120, seed=3)
    # Mismatched global transforms produce nonzero residuals.
    G = [np.eye(3), np.eye(3)]
    dx, dy, err = diag.pair_point_residuals_px(pair, G, 30.0, 30.0)

    rows = global_consistency_diagnostics([pair], G, [], pixel_size=30.0)
    row = rows[0]
    assert row["n_points"] == len(err)
    # The runner rounds its aggregates to 4 decimals.
    assert float(np.median(err)) == pytest.approx(row["global_median_px"], abs=5e-4)
    assert float(np.sqrt(np.mean(err**2))) == pytest.approx(row["global_rmse_px"], abs=5e-4)
    assert float(np.percentile(err, 95)) == pytest.approx(row["global_p95_px"], abs=5e-4)
    # A consistent transformed world should collapse all residuals.
    A_world = diag.pixel_affine_to_world(
        np.asarray(pair.pair_pixel_matrix), pair.pair_common_transform
    )
    dx2, dy2, err2 = diag.pair_point_residuals_px(pair, [np.eye(3), A_world], 30.0)
    assert float(np.max(err2)) < 1e-6


def test_idempotent_saved_comparison():
    pair = _make_pair(0, 1, n=69, seed=11)
    saved = {
        "idx_i": 0,
        "idx_j": 1,
        "status": "OK",
        "raw_matches": pair.raw_matches,
        "inliers": pair.inliers,
        "inlier_ratio": pair.inlier_ratio,
        "coverage": pair.coverage,
        "residual_rmse": pair.residual_rmse,
        "residual_p95": pair.residual_p95,
    }
    reproduction = diag.compare_pair_to_saved(saved, pair)
    assert reproduction["status"] == "exact"

    altered = dict(saved)
    altered["inliers"] = 30
    reproduction = diag.compare_pair_to_saved(altered, pair)
    assert reproduction["status"] == "mismatch"
    assert reproduction["warning"] is not None


# ---------------------------------------------------------------------------
# Task 5 — pattern classification
# ---------------------------------------------------------------------------


def test_classify_systematic_shift():
    rng = np.random.default_rng(0)
    n = 50
    ref = np.column_stack([np.linspace(0, 100, n), np.linspace(0, 60, n)])
    dx = np.full(n, 40.0) + rng.uniform(-0.5, 0.5, n)
    dy = np.full(n, 30.0) + rng.uniform(-0.5, 0.5, n)
    pattern = diag.classify_error_pattern(ref, dx, dy)
    assert pattern["classification"] == "SYSTEMATIC_SHIFT_LIKELY"
    assert pattern["direction_coherence"] > 0.99
    assert pattern["error_cv"] < 0.03


def test_classify_spatial_variation_not_systematic():
    n = 50
    ref = np.column_stack([np.linspace(0, 100, n), np.linspace(0, 60, n)])
    # dx grows with x: a rotation-like mismatch, not one global shift.
    dx = np.linspace(0.0, 50.0, n)
    dy = np.zeros(n)
    pattern = diag.classify_error_pattern(ref, dx, dy)
    assert pattern["classification"] != "SYSTEMATIC_SHIFT_LIKELY"
    assert pattern["spatial_r2"]["dx"] > 0.95
    assert pattern["classification"] == "SPATIAL_VARIATION_LIKELY"


def test_classify_outlier_driven_when_direction_mixed():
    n = 60
    rng = np.random.default_rng(11)
    dx = rng.uniform(-0.3, 0.3, n)
    dy = rng.uniform(-0.3, 0.3, n)
    # Enough outliers, balanced in direction, so the mean vector stays ~0
    # (low coherence) while p95/median is huge.
    for k, (ux, uy) in enumerate(
        [(40, 0), (-40, 0), (0, 40), (0, -40), (40, 40), (-40, -40)]
    ):
        dx[k], dy[k] = ux, uy
    ref = rng.uniform(0, 100, (n, 2))
    pattern = diag.classify_error_pattern(ref, dx, dy)
    assert pattern["max_error_px"] > 40.0
    assert pattern["p95_over_median"] > 6.0
    assert pattern["direction_coherence"] < 0.6
    assert pattern["classification"] == "OUTLIER_DRIVEN"


# ---------------------------------------------------------------------------
# Task 9 — final state is evidence-driven
# ---------------------------------------------------------------------------


def test_final_state_closure_consistent_when_transforms_match():
    diff = diag.compare_direct_and_mst_transforms(np.eye(3), np.eye(3), 1.0, 1.0)
    pattern = {"classification": "SYSTEMATIC_SHIFT_LIKELY"}
    state = diag.decide_final_state(diff, pattern, global_p95_px=0.3)
    assert state["final_state"] == "CLOSURE_CONSISTENT"


def test_final_state_metric_suspect_when_consistent_but_error_large():
    diff = diag.compare_direct_and_mst_transforms(np.eye(3), np.eye(3), 1.0, 1.0)
    pattern = {"classification": "SYSTEMATIC_SHIFT_LIKELY"}
    state = diag.decide_final_state(diff, pattern, global_p95_px=52.0)
    assert state["final_state"] == "METRIC_IMPLEMENTATION_SUSPECT"


def test_final_state_affine_suspect_on_rotation_difference():
    theta = math.radians(2.0)
    R = np.array(
        [[math.cos(theta), -math.sin(theta), 0.0],
         [math.sin(theta), math.cos(theta), 0.0],
         [0.0, 0.0, 1.0]]
    )
    diff = diag.compare_direct_and_mst_transforms(np.eye(3), R, 1.0, 1.0)
    state = diag.decide_final_state(diff, {"classification": "SPATIAL_VARIATION_LIKELY"}, 4.0)
    assert state["final_state"] == "AFFINE_MODEL_SUSPECT"
    # No configuration may ever auto-flag DIRECT_EDGE_SUSPECT.
    assert state["final_state"] != "DIRECT_EDGE_SUSPECT"


def test_final_state_undecided_when_comparison_unavailable():
    state = diag.decide_final_state(None, {}, global_p95_px=52.0)
    assert state["final_state"] == "NOT_YET_DETERMINED"


def test_final_state_systematic_shift_with_large_matrix_translation():
    # Mirrors the SIFT B14 run: clean point-level systematic shift while the
    # raw matrix difference carries a large coordinate-frame anchor offset.
    diff = diag.compare_direct_and_mst_transforms(
        np.eye(3),
        np.array(
            [[1.0012, 0.0023, -10473.8],
             [-0.0024, 0.9981, 8667.7],
             [0.0, 0.0, 1.0]]
        ),
        pixel_size_x=14.0, pixel_size_y=14.0,
    )
    pattern = {
        "classification": "SYSTEMATIC_SHIFT_LIKELY",
        "direction_coherence": 0.9996,
    }
    state = diag.decide_final_state(diff, pattern, global_p95_px=53.6)
    assert state["final_state"] == "NOT_YET_DETERMINED"
    assert state["evidence"]["coordinate_frame_warning"] is True


# ---------------------------------------------------------------------------
# Overlay rendering (synthetic images, no real DZ01V data)
# ---------------------------------------------------------------------------


@pytest.fixture
def synth_scenes(tmp_path):
    """Two synthetic overlapping scenes as Scene objects (B14 band)."""
    from tests.multiscene_sift.conftest import make_five_scene_path
    from src.multiscene_sift.dataset import discover_five_scenes

    root = make_five_scene_path(tmp_path)
    names = [
        "DZ01V_L2_E113.0_N36.4_20260222031837_01_T1",
        "DZ01V_L2_E113.4_N36.4_20260222031838_01_T2",
        "DZ01V_L2_E113.8_N36.4_20260222031839_01_T3",
        "DZ01V_L2_E113.4_N36.2_20260222031840_01_T4",
        "DZ01V_L2_E113.8_N36.2_20260222031841_01_T5",
    ]
    scenes, _ = discover_five_scenes(str(root), names, bands=("B14",))
    return scenes


def test_overlay_renders_shared_grid_with_large_world_offsets(synth_scenes, tmp_path):
    """The overlay frame must track the corrected scene-0 footprint.

    Large G translations (as in the real run) used to push scene 1 outside the
    raw-footprint grid, yielding 'No valid overlap pixels'.  The shared frame
    based on G0 provides a stable region for both overlays.
    """
    scene0, scene1 = synth_scenes[0], synth_scenes[1]
    band = "B14"
    eye5 = [np.eye(3) for _ in range(5)]

    def _translation(mx, my):
        return np.array([[1.0, 0.0, mx], [0.0, 1.0, my], [0.0, 0.0, 1.0]])

    # scene 1 tracks scene 0 but with a ~4 px separation at 30 m/pix.
    t_direct = _translation(2000.0, 1500.0)
    G = eye5
    G[0] = _translation(2000.0, 1500.0)
    G[1] = _translation(2120.0, 1500.0)  # 120 m ≈ 4 px from scene 0

    out_dir = tmp_path / "overlay"
    overlay = diag.plot_overlay_comparison(
        scene0, scene1, band, t_direct, G, out_dir, max_side=256
    )
    assert (out_dir / "09_direct_0_1_overlay.png").is_file()
    assert (out_dir / "10_mst_0_4_1_overlay.png").is_file()
    assert overlay["direct"].size > 0 and (overlay["direct"] > 0).any()
    assert overlay["mst"].size > 0 and (overlay["mst"] > 0).any()
    # Both overlays must share exactly the same pixel lattice.
    assert overlay["direct"].shape == overlay["mst"].shape
    assert "ncc_direct" in overlay and "ncc_mst" in overlay


def test_ncc_between_overlays_scoring():
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 100, (40, 40))
    valid = np.ones((40, 40), dtype=bool)
    assert diag._ncc_between_overlays(
        base, base, valid, valid
    ) == pytest.approx(1.0, abs=1e-9)
    noise = base + rng.normal(0, 30.0, base.shape)
    assert diag._ncc_between_overlays(base, noise, valid, valid) > 0.0
    assert diag._ncc_between_overlays(base, -base, valid, valid) < 0.0
    empty = np.zeros((40, 40), dtype=bool)
    assert math.isnan(diag._ncc_between_overlays(base, base, empty, empty))


# ---------------------------------------------------------------------------
# Sanity: the demo five-scene data shape, end to end (no real imagery)
# ---------------------------------------------------------------------------


def test_demo_run_artifacts_end_to_end(tmp_path):
    """Feed the SIFT B14 numbers through the diagnostic data flow.

    This validates Task 1 and 2 end-to-end on the shape of the real artifacts,
    without touching any real imagery.
    """
    from src.multiscene_sift.pairwise import save_pairwise_summary
    from src.multiscene_sift.global_registration import (
        save_global_registration_info,
        save_consistency_diagnostics,
    )

    run = tmp_path / "run"
    run.mkdir()
    consistency = [
        _consistency_row(0, 1, False, 53.6481, n=69, median=52.0864,
                         rmse=51.9735, mx=54.9142),
        _consistency_row(0, 2, True, 1.7439, n=56),
        _consistency_row(0, 4, True, 1.5636, n=128),
        _consistency_row(1, 4, True, 1.1427, n=3562),
        _consistency_row(3, 4, True, 1.7303, n=423),
    ]
    save_consistency_diagnostics(consistency, run)

    tree_edges = _tree_edges([
        (4, 1, 1, 1957.46),
        (4, 3, 1, 156.09),
        (4, 0, 1, 39.64),
        (0, 2, 2, 9.30),
    ])
    save_global_registration_info(
        {"reference_index": 4}, tree_edges, [np.eye(3)] * 5, run
    )

    pair0 = _make_pair(0, 1, n=69, seed=5)
    pair1 = _make_pair(1, 4, n=3562, seed=6)
    pair2 = _make_pair(3, 4, n=423, seed=8)
    save_pairwise_summary([pair0, pair1, pair2], run)

    rows = diag.load_consistency_rows(run / "global_edge_consistency.csv")
    edge = diag.find_worst_non_tree_edge(rows)
    assert edge == (0, 1)

    spanning = diag.load_spanning_tree(run / "spanning_tree.json")
    path = diag.find_tree_path(spanning["edges"], *edge)
    assert path == [0, 4, 1]

    G = diag.load_global_transforms(run / "global_transforms.json")
    assert len(G) == 5

    common = pair0.pair_common_transform
    T_direct = diag.pair_direct_world_transform(
        diag.load_pairwise_results(run / "pairwise_summary.json"),
        edge[0], edge[1], common,
    )
    T_mst = diag.mst_implied_transform(G, edge[0], edge[1])
    diff = diag.compare_direct_and_mst_transforms(
        T_direct, T_mst, pixel_size_x=14.0, pixel_size_y=14.0
    )
    assert "difference_matrix" in diff
    assert "translation_magnitude_px" in diff