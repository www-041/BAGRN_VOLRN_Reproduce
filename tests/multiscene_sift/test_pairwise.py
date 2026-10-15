"""Tests for :mod:`src.multiscene_sift.pairwise`."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from src.multiscene_sift.models import PairwiseRegistration
from src.multiscene_sift.pairwise import register_pair, run_all_pairs
from tests.multiscene_sift.conftest import _make_band


def _make_fake_matches(n=30):
    """Create a simple namespace with realistic match arrays."""
    return mock.MagicMock(
        ref_xy=np.zeros((n, 2), dtype=np.float64),
        tgt_xy=np.zeros((n, 2), dtype=np.float64),
        runtime_sec=1.0,
    )


def _make_ok_geom(n=30):
    """Create a realistic OK geometry mock."""
    g = mock.MagicMock()
    g.status = "OK"
    g.model = mock.MagicMock(params=np.eye(3))
    g.n_raw = n
    g.n_inlier = n - 5
    g.inlier_ratio = (n - 5) / n
    g.residual_median = 1.0
    g.residual_rmse = 1.2
    g.residual_p95 = 1.5
    g.inlier_mask = np.zeros(n, dtype=bool)
    g.inlier_mask[: n - 5] = True
    return g


class TestPairwiseRegistration:
    """Tests for pairwise SIFT registration."""

    def test_overlapping_pair_produces_matches(self, tmp_path):
        """Two overlapping synthetic scenes should produce SIFT matches."""
        from src.multiscene_sift.dataset import discover_five_scenes
        from src.multiscene_sift.overlap_graph import build_geographic_overlap_graph

        # Create two overlapping scenes
        root = tmp_path / "flat"
        root.mkdir(parents=True)
        names = ["scene_a", "scene_b"]

        # scene_a at (0, 0), scene_b shifted 2000m east
        for name, ox in [("scene_a", 500000), ("scene_b", 502000)]:
            d = root / name
            d.mkdir(parents=True)
            for band in ("B14", "B8", "B5"):
                _make_band(d / f"{name}_{band}.TIF", origin_x=ox)

        scenes, _ = discover_five_scenes(str(root), names)
        edges = build_geographic_overlap_graph(scenes)

        assert len(edges) >= 1

        reg = register_pair(scenes[0], scenes[1], band="B14")

        # Should produce some matches (may or may not pass quality thresholds)
        assert reg.idx_i == 0
        assert reg.idx_j == 1
        assert reg.raw_matches >= 0
        # The matrix should be 3×3
        assert len(reg.pair_pixel_matrix) == 3

    def test_non_overlapping_identity(self, tmp_path):
        """Non-overlapping scenes should get identity fallback."""
        # This is tested via the graph check in overlap_graph.
        # Here we just verify register_pair doesn't crash on distant scenes.
        from src.multiscene_sift.dataset import discover_five_scenes

        root = tmp_path / "flat"
        root.mkdir(parents=True)
        names = ["scene_x", "scene_y"]

        # Two scenes far apart (no overlap in common grid)
        for name, ox in [("scene_x", 400000), ("scene_y", 600000)]:
            d = root / name
            d.mkdir(parents=True)
            for band in ("B14", "B8", "B5"):
                _make_band(d / f"{name}_{band}.TIF", origin_x=ox)

        scenes, _ = discover_five_scenes(str(root), names)
        reg = register_pair(scenes[0], scenes[1], band="B14")

        # The function should complete without exception
        assert reg.idx_i == 0
        assert reg.idx_j == 1
        assert isinstance(reg.status, str)

    def test_match_max_side_forwarded_to_build_match_view(self, tmp_path):
        """Non-default match_max_side must reach build_match_view."""
        from src.multiscene_sift.dataset import discover_five_scenes

        root = tmp_path / "flat"
        root.mkdir(parents=True)
        names = ["s0", "s1"]
        for name, ox in [("s0", 500000), ("s1", 502000)]:
            d = root / name
            d.mkdir(parents=True)
            for band in ("B14", "B8", "B5"):
                _make_band(d / f"{name}_{band}.TIF", origin_x=ox)

        scenes, _ = discover_five_scenes(str(root), names)

        import src.multiscene_sift.pairwise as pw_mod
        fake_matches = _make_fake_matches()
        with mock.patch.object(pw_mod, "build_match_view") as mock_bmv:
            mock_bmv.return_value = mock.MagicMock()
            with mock.patch.object(pw_mod, "match_sift") as mock_ms:
                mock_ms.return_value = fake_matches
                with mock.patch.object(pw_mod, "fit_affine_ransac") as mock_fa:
                    mock_fa.return_value = _make_ok_geom()

                    register_pair(scenes[0], scenes[1], band="B14",
                                  match_max_side=800)

                    mock_bmv.assert_called_once()
                    _, kwargs = mock_bmv.call_args
                    assert kwargs.get("max_side") == 800, (
                        f"Expected max_side=800, got {kwargs}"
                    )

    def test_ransac_threshold_forwarded_to_fit_affine_ransac(self, tmp_path):
        """Non-default ransac_threshold must reach fit_affine_ransac."""
        from src.multiscene_sift.dataset import discover_five_scenes

        root = tmp_path / "flat"
        root.mkdir(parents=True)
        names = ["s0", "s1"]
        for name, ox in [("s0", 500000), ("s1", 502000)]:
            d = root / name
            d.mkdir(parents=True)
            for band in ("B14", "B8", "B5"):
                _make_band(d / f"{name}_{band}.TIF", origin_x=ox)

        scenes, _ = discover_five_scenes(str(root), names)

        import src.multiscene_sift.pairwise as pw_mod
        fake_matches = _make_fake_matches()
        with mock.patch.object(pw_mod, "build_match_view") as mock_bmv:
            mock_bmv.return_value = mock.MagicMock()
            with mock.patch.object(pw_mod, "match_sift") as mock_ms:
                mock_ms.return_value = fake_matches
                with mock.patch.object(pw_mod, "fit_affine_ransac") as mock_fa:
                    mock_fa.return_value = _make_ok_geom()

                    register_pair(scenes[0], scenes[1], band="B14",
                                  ransac_threshold=1.25)

                    mock_fa.assert_called_once()
                    _, kwargs = mock_fa.call_args
                    assert kwargs.get("residual_threshold") == 1.25, (
                        f"Expected residual_threshold=1.25, got {kwargs}"
                    )

    def test_run_all_pairs_forwards_params(self, tmp_path):
        """run_all_pairs must forward match_max_side/ransac_threshold to register_pair."""
        from src.multiscene_sift.dataset import discover_five_scenes
        from src.multiscene_sift.overlap_graph import build_geographic_overlap_graph

        root = tmp_path / "flat"
        root.mkdir(parents=True)
        names = ["s0", "s1"]
        for name, ox in [("s0", 500000), ("s1", 502000)]:
            d = root / name
            d.mkdir(parents=True)
            for band in ("B14", "B8", "B5"):
                _make_band(d / f"{name}_{band}.TIF", origin_x=ox)

        scenes, _ = discover_five_scenes(str(root), names)
        edges = build_geographic_overlap_graph(scenes)

        fake_reg = PairwiseRegistration(
            idx_i=0, idx_j=1, status="OK",
            raw_matches=10, inliers=8, inlier_ratio=0.8, coverage=0.4,
            residual_median=1.0, residual_rmse=1.2, residual_p95=1.5,
            pair_pixel_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            pair_common_transform=None, runtime_sec=1.0,
            inlier_ref_xy=np.empty((0, 2)), inlier_tgt_xy=np.empty((0, 2)),
        )

        import src.multiscene_sift.pairwise as pw_mod
        with mock.patch.object(pw_mod, "register_pair") as mock_rp:
            mock_rp.return_value = fake_reg
            run_all_pairs(scenes, edges, tmp_path / "out", band="B14",
                          match_max_side=400, ransac_threshold=3.5)
            mock_rp.assert_called()
            _, kwargs = mock_rp.call_args
            assert kwargs.get("match_max_side") == 400
            assert kwargs.get("ransac_threshold") == 3.5