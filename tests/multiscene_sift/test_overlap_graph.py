"""Tests for :mod:`src.multiscene_sift.overlap_graph`."""

from __future__ import annotations

import pytest

from src.multiscene_sift.overlap_graph import (
    build_geographic_overlap_graph,
    _connected_components,
)
from tests.multiscene_sift.conftest import (
    make_five_scene_dataset,
    RESOLUTION,
    IMAGE_SIZE,
)


class TestOverlapGraph:
    """Tests for :func:`build_geographic_overlap_graph`."""

    def test_five_scenes_connected(self, tmp_path):
        """Synthetic 5-scene dataset should produce a connected graph."""
        from src.multiscene_sift.dataset import discover_five_scenes

        root = tmp_path / "flat"
        names = make_five_scene_dataset(root)

        scenes, _ = discover_five_scenes(str(root), names)
        edges = build_geographic_overlap_graph(scenes)

        assert len(edges) >= 4  # at least n-1
        # Check expected overlaps based on layout:
        # [0]-[1]-[2], [3]-[4] with [1]-[3] diagonal
        pairs = {(e.idx_i, e.idx_j) for e in edges}
        # 0-1, 1-2, 0-2?, 0-3, 1-3, 1-4, 2-4, 3-4
        assert (0, 1) in pairs  # neighbours
        assert (1, 2) in pairs  # neighbours
        assert (3, 4) in pairs  # neighbours

    def test_disconnected_raises(self, tmp_path):
        """Two scenes with no geographic overlap should raise."""
        from src.multiscene_sift.dataset import discover_five_scenes
        import rasterio
        from rasterio.transform import from_origin
        import numpy as np

        root = tmp_path / "flat"
        # Scene 0 at lon=500k, scene 1 at lon=600k (far apart)
        d0 = root / "s0"; d0.mkdir(parents=True)
        d1 = root / "s1"; d1.mkdir(parents=True)

        for d, ox in [(d0, 500000), (d1, 600000)]:
            for band in ("B14", "B8", "B5"):
                data = np.ones((16, 16), dtype="uint16")
                tx = from_origin(ox, 4000000, 30, 30)
                with rasterio.open(
                    d / f"{d.name}_{band}.TIF", "w",
                    driver="GTiff", height=16, width=16,
                    count=1, dtype="uint16", crs="EPSG:32650",
                    transform=tx,
                ) as dst:
                    dst.write(data, 1)

        scenes, _ = discover_five_scenes(str(root), ["s0", "s1"])
        with pytest.raises(ValueError, match="DISCONNECTED"):
            build_geographic_overlap_graph(scenes)

    def test_components_detection(self):
        """Connected components detection on a simple graph."""
        from src.multiscene_sift.models import OverlapEdge

        # Graph: 0-1  2-3 (two components)
        edges = [
            OverlapEdge(0, 1, 100, 0.5, 0.5),
            OverlapEdge(2, 3, 100, 0.5, 0.5),
        ]
        comps = _connected_components(4, edges)
        assert len(comps) == 2
        assert sorted(comps[0]) == [0, 1]
        assert sorted(comps[1]) == [2, 3]