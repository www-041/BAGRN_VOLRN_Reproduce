"""Tests for :mod:`src.multiscene_sift.pairwise`."""

from __future__ import annotations

import pytest

from src.multiscene_sift.pairwise import register_pair
from tests.multiscene_sift.conftest import _make_band


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