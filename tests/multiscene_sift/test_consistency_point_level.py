"""Tests for point-level global consistency diagnostics (Task 3)."""

import numpy as np
import pytest
from rasterio.transform import from_origin

from src.multiscene_sift.models import PairwiseRegistration
from src.multiscene_sift.global_registration import (
    global_consistency_diagnostics,
)


def _make_pair_with_points(
    idx_i, idx_j, ref_pts, tgt_pts, pixel_mat=None, resolution=30
):
    """Create a PairwiseRegistration with inlier points."""
    if pixel_mat is None:
        pixel_mat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    T = from_origin(500000, 4000000, resolution, resolution)
    return PairwiseRegistration(
        idx_i=idx_i, idx_j=idx_j, status="OK",
        raw_matches=len(ref_pts),
        inliers=len(ref_pts),
        inlier_ratio=1.0, coverage=0.5,
        residual_median=0.0, residual_rmse=0.0, residual_p95=0.0,
        pair_pixel_matrix=pixel_mat,
        pair_common_transform=T,
        runtime_sec=1.0,
        inlier_ref_xy=np.asarray(ref_pts, dtype=np.float64),
        inlier_tgt_xy=np.asarray(tgt_pts, dtype=np.float64),
    )


class TestPointLevelConsistency:
    """Tests for point-level global consistency diagnostics."""

    def test_perfect_alignment_zero_residual(self):
        """3-scene chain with perfect alignment: all residuals ≈ 0."""
        # Three scenes in a chain, all with identity transforms
        T = from_origin(500000, 4000000, 30, 30)
        # Points in common-grid pixel coords
        pts = [[10, 10], [20, 30], [50, 60]]

        r01 = _make_pair_with_points(0, 1, pts, pts)
        r12 = _make_pair_with_points(1, 2, pts, pts)

        # Global transforms: all identity
        G = [np.eye(3) for _ in range(3)]
        tree = [{"parent": 0, "child": 1, "depth": 1, "weight": 10},
                {"parent": 1, "child": 2, "depth": 2, "weight": 10}]

        results = global_consistency_diagnostics(
            [r01, r12], G, tree, pixel_size=30,
        )

        for r in results:
            assert r["global_median_px"] == pytest.approx(0.0, abs=1e-6)
            assert r["global_rmse_px"] == pytest.approx(0.0, abs=1e-6)
            assert r["global_max_px"] == pytest.approx(0.0, abs=1e-6)
            assert r["n_points"] == 3

    def test_non_tree_has_visible_error(self):
        """Triangle 0-1-2: non-tree 0-2 with 3px point inconsistency must show error.

        Tree edges are 0-1 and 1-2 (both perfect). The non-tree edge 0-2
        carries a deterministic ~3px shift in its stored inlier point
        coordinates (tgt points shifted +3 in col), so — because
        consistency uses the inlier point coordinates directly — it must
        appear as a non-tree residual well above both tree edges.
        """
        pts = [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0],
               [40.0, 40.0], [50.0, 50.0], [60.0, 60.0],
               [70.0, 70.0], [80.0, 80.0], [90.0, 90.0], [100.0, 100.0]]

        # --- Tree edges: perfect -------------------------------------------------
        r01 = _make_pair_with_points(0, 1, pts, pts)
        r12 = _make_pair_with_points(1, 2, pts, pts)

        # --- Non-tree edge 0-2: scene-2 points shifted +3 in col (+90 m world,
        #     = 3 px at 30 m/px), applied to the stored inlier coordinates. -----
        tgt_pts_offset = np.asarray(pts, dtype=np.float64).copy()
        tgt_pts_offset[:, 0] += 3.0  # col shift
        r02 = _make_pair_with_points(0, 2, np.asarray(pts), tgt_pts_offset)

        # All global transforms identity: residuals keep pair-internal offsets.
        G = [np.eye(3) for _ in range(3)]
        tree = [{"parent": 0, "child": 1, "depth": 1, "weight": 10},
                {"parent": 1, "child": 2, "depth": 2, "weight": 10}]

        results = global_consistency_diagnostics(
            [r01, r12, r02], G, tree, pixel_size=30,
        )

        # --- Tree edges must be ~0 ----------------------------------------------
        tree_res = [r for r in results if r["in_tree"]]
        assert len(tree_res) == 2, f"expected 2 tree edges, got {len(tree_res)}"
        for r in tree_res:
            assert r["global_p95_px"] < 1e-6, (
                f"tree edge {r['idx_i']}-{r['idx_j']} should be ~0, "
                f"got P95={r['global_p95_px']}"
            )

        # --- Non-tree edge must be clearly above 2 px ---------------------------
        non_tree = [r for r in results if not r["in_tree"]]
        assert len(non_tree) == 1, f"expected 1 non-tree edge, got {len(non_tree)}"
        r_nt = non_tree[0]
        assert r_nt["idx_i"] == 0 and r_nt["idx_j"] == 2
        # All 10 points shifted by exactly 3 px → P95 = max = 3
        assert r_nt["global_p95_px"] == pytest.approx(3.0, abs=1e-6)
        assert r_nt["global_p95_px"] > 2.0

        # --- Non-tree error must exceed tree error ------------------------------
        max_tree_p95 = max(r["global_p95_px"] for r in tree_res)
        assert r_nt["global_p95_px"] > max_tree_p95

    def test_median_not_equal_max(self):
        """Verify median≠max and p95≠median for varied residuals."""
        T = from_origin(500000, 4000000, 30, 30)
        # ref pts matching tgt pts with different offsets per point
        ref_pts = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50],
                            [60, 60], [70, 70], [80, 80], [90, 90], [100, 100]],
                           dtype=np.float64)

        # tgt points with known varied offsets
        # Scene j G=identity + tx offset
        offsets = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
        tgt_pts = ref_pts.copy()
        tgt_pts[:, 0] += offsets

        # G[0]=I, G[1] has some residual
        # Actually G_j is I here, so the pair transform needs to map tgt→ref
        # If we set pair as identity and let the global transforms differ:
        G = [np.eye(3) for _ in range(2)]
        G[1] = np.array([[1, 0, -offsets[0] * 30], [0, 1, 0], [0, 0, 1]])  # mean offset

        r01 = _make_pair_with_points(0, 1, ref_pts, tgt_pts)
        tree = [{"parent": 0, "child": 1, "depth": 1, "weight": 10}]

        results = global_consistency_diagnostics(
            [r01], G, tree, pixel_size=30,
        )

        assert len(results) == 1
        r = results[0]
        assert r["n_points"] == 10
        # median and max must differ
        assert r["global_median_px"] != pytest.approx(r["global_max_px"], abs=1e-4)
        # p95 and median must differ
        assert r["global_p95_px"] != pytest.approx(r["global_median_px"], abs=1e-4)
        # rmse should reflect actual point errors
        assert r["global_rmse_px"] > 0

    def test_rmse_matches_manual_calculation(self):
        """RMSE must match manual sqrt(mean(errors^2))."""
        T = from_origin(500000, 4000000, 30, 30)
        ref_pts = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float64)
        tgt_pts = np.array([[13, 10], [20, 22], [30, 30]], dtype=np.float64)

        # Global all identity
        G = [np.eye(3) for _ in range(2)]
        r01 = _make_pair_with_points(0, 1, ref_pts, tgt_pts)
        tree = [{"parent": 0, "child": 1, "depth": 1, "weight": 10}]

        results = global_consistency_diagnostics(
            [r01], G, tree, pixel_size=30,
        )

        r = results[0]
        # Manual: points in world coords via transform
        # T * (10,10) = (500300, 3999700), T * (13,10) = (500390, 3999700)
        # error = 90m world / 30 = 3px → errors = [3, 2, 0]
        # rmse = sqrt((9+4+0)/3) = sqrt(13/3) = sqrt(4.333) ≈ 2.081
        expected_rmse = np.sqrt((9 + 4 + 0) / 3)
        assert r["global_rmse_px"] == pytest.approx(expected_rmse, abs=0.01)
        assert r["global_median_px"] == pytest.approx(2.0, abs=0.01)
        assert r["global_max_px"] == pytest.approx(3.0, abs=0.01)

    def test_reports_world_and_pixel_units_separately(self):
        """A 140 m residual at 14 m/pixel must report as 10 pixels."""
        ref_pts = np.array([[10.0, 10.0], [20.0, 20.0]])
        tgt_pts = ref_pts.copy()
        tgt_pts[:, 0] += 10.0
        pair = _make_pair_with_points(
            0, 1, ref_pts, tgt_pts, resolution=14
        )

        results = global_consistency_diagnostics(
            [pair], [np.eye(3), np.eye(3)], [], pixel_size_m=14.0
        )

        row = results[0]
        assert row["global_rmse_world_m"] == pytest.approx(140.0)
        assert row["global_rmse_pixel"] == pytest.approx(10.0)
        assert row["pixel_size_m"] == pytest.approx(14.0)

    def test_requires_explicit_pixel_size(self):
        """Pixel metrics must not silently default to one metre per pixel."""
        pair = _make_pair_with_points(
            0, 1, [[0.0, 0.0]], [[1.0, 0.0]], resolution=14
        )

        with pytest.raises(ValueError, match="pixel_size_m"):
            global_consistency_diagnostics([pair], [np.eye(3), np.eye(3)], [])

    def test_reports_fractional_pixel_residual_for_explicit_resolution(self):
        """A 7 m residual at 14 m/pixel must report as 0.5 pixel."""
        pair = _make_pair_with_points(
            0, 1, [[0.0, 0.0]], [[0.5, 0.0]], resolution=14
        )

        row = global_consistency_diagnostics(
            [pair], [np.eye(3), np.eye(3)], [], pixel_size_m=14.0
        )[0]

        assert row["global_rmse_world_m"] == pytest.approx(7.0)
        assert row["global_rmse_pixel"] == pytest.approx(0.5)
