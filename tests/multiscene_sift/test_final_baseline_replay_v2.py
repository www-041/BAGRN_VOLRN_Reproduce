"""Regression tests for persisted BAGRN scene outputs used by replay v2."""

from __future__ import annotations

import json

import numpy as np
import rasterio

from src.multiscene_sift.radiometric_runner import run_fixed_geometry_radiometric
from tests.multiscene_sift.test_fixed_radiometric_runner import _inputs


def test_opt_in_bagrn_scene_cache_is_canonical_and_provenanced(tmp_path):
    """Removing per-scene persistence must make this replay input unavailable."""
    source, global_dir, grid = _inputs(tmp_path)
    run_dir = tmp_path / "bagrn"

    result = run_fixed_geometry_radiometric(
        source,
        global_dir,
        grid,
        run_dir,
        method="BAGRN",
        geometry_run="sift_mst",
        persist_normalized_scenes=True,
    )

    cache_dir = run_dir / "normalized_scenes"
    manifest_path = run_dir / "normalized_scenes_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["method"] == "BAGRN"
    assert manifest["geometry_run"] == "sift_mst"
    assert manifest["radiometric_control_idx"] == 0
    assert manifest["grid"]["width"] == 24
    assert manifest["grid"]["height"] == 16
    assert len(manifest["scenes"]) == 3
    assert result["normalized_scenes_manifest"] == "normalized_scenes_manifest.json"

    for scene in manifest["scenes"]:
        path = cache_dir / scene["path"]
        assert path.is_file()
        with rasterio.open(path) as dataset:
            assert (dataset.width, dataset.height) == (24, 16)
            assert dataset.crs.to_string() == "EPSG:32650"
            assert np.allclose(tuple(dataset.transform)[:6], (1, 0, 0, 0, -1, 32))
            data = dataset.read(1)
            finite = np.isfinite(data)
            assert finite.any()
            assert int(finite.sum()) == scene["valid_pixels"]


def test_default_bagrn_run_does_not_write_scene_cache(tmp_path):
    """The cache must remain opt-in so historical runs retain their layout."""
    source, global_dir, grid = _inputs(tmp_path)

    run_fixed_geometry_radiometric(
        source,
        global_dir,
        grid,
        tmp_path / "bagrn",
        method="BAGRN",
        geometry_run="sift_mst",
    )

    assert not (tmp_path / "bagrn" / "normalized_scenes").exists()
    assert not (tmp_path / "bagrn" / "normalized_scenes_manifest.json").exists()
