"""Tests for :mod:`src.multiscene_sift.dataset`."""

from __future__ import annotations

from pathlib import Path

import pytest
import rasterio
from rasterio.transform import from_origin

from src.multiscene_sift.dataset import discover_five_scenes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESOLUTION = 30.0


def _write_tif(path: Path, crs: str = "EPSG:32650", **kwargs):
    """Write a minimal 16×16 GeoTIFF."""
    import numpy as np
    data = np.ones((16, 16), dtype="uint16")
    tx = kwargs.pop("transform", from_origin(500000, 4000000, RESOLUTION, RESOLUTION))
    with rasterio.open(
        path, "w", driver="GTiff", height=16, width=16,
        count=1, dtype="uint16", crs=crs, transform=tx,
        **kwargs,
    ) as dst:
        dst.write(data, 1)


def _make_scene(root: Path, name: str, bands=("B14", "B8", "B5"), **kwargs):
    d = root / name
    d.mkdir(parents=True)
    for b in bands:
        _write_tif(d / f"{name}_{b}.TIF", **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiscoverFiveScenes:
    """Tests for :func:`discover_five_scenes`."""

    def test_required_bands_discovered(self, tmp_path):
        """All three bands (B14, B8, B5) must be found for each scene."""
        root = tmp_path / "flat"
        names = [f"scene_{i}" for i in range(5)]
        for n in names:
            _make_scene(root, n)

        scenes, manifest = discover_five_scenes(str(root), names)

        assert len(scenes) == 5
        assert manifest["n_scenes"] == 5
        for s in scenes:
            for band in ("B14", "B8", "B5"):
                assert band in s.band_paths
                assert Path(s.band_paths[band]).exists()

    def test_missing_b8_raises(self, tmp_path):
        """If B8 is missing for any scene, FileNotFoundError is raised."""
        root = tmp_path / "flat"
        names = [f"scene_{i}" for i in range(5)]
        for n in names:
            _make_scene(root, n)
        # Remove B8 from scene 2
        b8_path = root / "scene_2" / "scene_2_B8.TIF"
        b8_path.unlink()

        with pytest.raises(FileNotFoundError, match="Band B8 not found"):
            discover_five_scenes(str(root), names)

    def test_missing_scene_directory_raises(self, tmp_path):
        """If a scene directory is missing, FileNotFoundError is raised."""
        root = tmp_path / "flat"
        names = ["exists", "missing"]
        _make_scene(root, "exists")

        with pytest.raises(FileNotFoundError, match="Scene directory not found"):
            discover_five_scenes(str(root), names)

    def test_crs_mismatch_raises(self, tmp_path):
        """If within-scene bands have different CRS, ValueError is raised."""
        root = tmp_path / "flat"
        d = root / "scene_0"
        d.mkdir(parents=True)
        _write_tif(d / "scene_0_B14.TIF", crs="EPSG:32650")
        _write_tif(d / "scene_0_B8.TIF", crs="EPSG:32651")  # different zone
        _write_tif(d / "scene_0_B5.TIF", crs="EPSG:32650")

        names = ["scene_0"]
        with pytest.raises(ValueError, match="CRS mismatch"):
            discover_five_scenes(str(root), names)

    def test_pixel_size_mismatch_raises(self, tmp_path):
        """If within-scene bands have different pixel sizes, ValueError is raised."""
        root = tmp_path / "flat"
        d = root / "scene_0"
        d.mkdir(parents=True)
        _write_tif(d / "scene_0_B14.TIF", transform=from_origin(500000, 4000000, 30, 30))
        _write_tif(d / "scene_0_B8.TIF", transform=from_origin(500000, 4000000, 60, 60))  # different res
        _write_tif(d / "scene_0_B5.TIF", transform=from_origin(500000, 4000000, 30, 30))

        names = ["scene_0"]
        with pytest.raises(ValueError, match="pixel size mismatch"):
            discover_five_scenes(str(root), names)

    def test_orientation_raises(self, tmp_path):
        """If a band has non-north-up orientation, ValueError is raised."""
        root = tmp_path / "flat"
        d = root / "scene_0"
        d.mkdir(parents=True)
        # B14 with flipped Y (e>0 → south-going)
        tx_bad = rasterio.Affine(30, 0, 500000, 0, 30, 4000000)
        _write_tif(d / "scene_0_B14.TIF", transform=tx_bad)
        _write_tif(d / "scene_0_B8.TIF")
        _write_tif(d / "scene_0_B5.TIF")

        names = ["scene_0"]
        with pytest.raises(ValueError, match="north-going"):
            discover_five_scenes(str(root), names)

    def test_manifest_structure(self, tmp_path):
        """Manifest dict should contain expected top-level keys."""
        root = tmp_path / "flat"
        names = [f"scene_{i}" for i in range(3)]
        for n in names:
            _make_scene(root, n)

        _, manifest = discover_five_scenes(str(root), names)

        assert "scenes" in manifest
        assert "bands" in manifest
        assert "n_scenes" in manifest
        s0 = manifest["scenes"][0]
        assert s0["index"] == 0
        assert "crs" in s0
        for band in ("B14", "B8", "B5"):
            assert band in s0["bands"]
            assert "path" in s0["bands"][band]
            assert "shape" in s0["bands"][band]
            assert "resolution" in s0["bands"][band]
            assert "bounds" in s0["bands"][band]