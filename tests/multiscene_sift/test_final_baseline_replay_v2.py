"""Regression tests for persisted BAGRN scene outputs used by replay v2."""

from __future__ import annotations

import json

import numpy as np
import pytest
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


def test_replay_regenerates_missing_cache_then_reuses_verified_cache(tmp_path):
    """A replay must produce BAGRN artifacts once, then consume the cache."""
    from src.multiscene_sift.final_baseline_replay import (
        BaselineReplaySpec,
        replay_baseline,
    )

    source, global_dir, grid = _inputs(tmp_path)
    spec = BaselineReplaySpec(
        name="synthetic_main",
        geometry_run="sift_mst",
        source_config=source,
        global_run_dir=global_dir,
        output_grid=grid,
        replay_root=tmp_path / "replay",
    )

    first = replay_baseline(spec)
    assert first.cache_status == "REGENERATED"
    assert first.normalized_scenes_manifest.is_file()
    assert first.mosaic_path.is_file()
    assert first.summary["radiometric_method"] == "BAGRN"
    assert set(first.summary["task10d_primary_metrics"]) == {
        "mamd", "msdd", "rdd", "local_mamd", "local_rdd",
        "seam_mae", "seam_rmse", "seam_rdd", "cgl_rad",
    }

    second = replay_baseline(spec)
    assert second.cache_status == "REUSED"
    assert second.mosaic_path.parent.name == "weighted_feather"
    assert second.mosaic_sha256 == first.mosaic_sha256


def test_replay_rejects_stale_cache_and_preserves_it_before_bagrn_fallback(tmp_path):
    """An invalid cache is retained while BAGRN writes a fresh replay cache."""
    from src.multiscene_sift.final_baseline_replay import BaselineReplaySpec, replay_baseline

    source, global_dir, grid = _inputs(tmp_path)
    spec = BaselineReplaySpec(
        name="stale", geometry_run="sift_mst", source_config=source,
        global_run_dir=global_dir, output_grid=grid, replay_root=tmp_path / "replay",
    )
    first = replay_baseline(spec)
    manifest = json.loads(first.normalized_scenes_manifest.read_text(encoding="utf-8"))
    manifest["geometry_run"] = "wrong_geometry"
    first.normalized_scenes_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    regenerated = replay_baseline(spec)

    assert regenerated.cache_status == "REGENERATED"
    assert regenerated.run_dir != first.run_dir
    assert first.run_dir.is_dir()
    assert regenerated.run_dir.name.startswith("stale__regen_")
    reused = replay_baseline(spec)
    assert reused.cache_status == "REUSED"
    assert reused.run_dir == regenerated.run_dir


def test_replay_rejects_relative_grid_shift_and_mosaic_holes(tmp_path):
    """Grid and support validation must not accept plausible-but-wrong rasters."""
    from src.multiscene_sift.final_baseline_replay import _validate_mosaic

    source, global_dir, grid = _inputs(tmp_path)
    run_dir = tmp_path / "mosaic"
    run_dir.mkdir()
    path = run_dir / "mosaic.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=24, height=16, count=1, dtype="float32",
        crs="EPSG:32650", transform=rasterio.Affine(1, 0, 0, 0, -1, 46), nodata=np.nan,
    ) as dst:
        values = np.full((16, 24), np.nan, dtype=np.float32)
        values[0, 0] = np.inf
        dst.write(values, 1)
    spec = type("Spec", (), {"output_grid": grid})()

    with pytest.raises(ValueError):
        _validate_mosaic(spec, path, np.ones((16, 24), dtype=bool))


def test_final_package_keeps_main_and_traditional_artifacts_distinct(tmp_path):
    """Swapping a baseline's geometry or mosaic must be visible in the package."""
    from src.multiscene_sift.final_baseline_replay import (
        BaselineReplaySpec,
        materialize_final_results,
        replay_baseline,
    )

    source, global_dir, grid = _inputs(tmp_path)
    main = replay_baseline(BaselineReplaySpec(
        name="main", geometry_run="sift_mst", source_config=source,
        global_run_dir=global_dir, output_grid=grid, replay_root=tmp_path / "replay",
    ))
    traditional = replay_baseline(BaselineReplaySpec(
        name="traditional", geometry_run="sift_mst", source_config=source,
        global_run_dir=global_dir, output_grid=grid, replay_root=tmp_path / "replay",
    ))

    manifest = materialize_final_results(main, traditional, tmp_path / "final_results")

    assert manifest["baselines"]["main"]["mosaic_sha256"] == main.mosaic_sha256
    assert manifest["baselines"]["traditional"]["mosaic_sha256"] == traditional.mosaic_sha256
    for name in ("EfficientLoFTR_Translation_BAGRN", "SIFT_MST_BAGRN"):
        root = tmp_path / "final_results" / name
        assert (root / "01_registration" / "pairwise_metrics.csv").is_file()
        assert (root / "01_registration" / "global_metrics.json").is_file()
        assert (root / "02_radiometric" / "radiometric_metrics.json").is_file()
        assert (root / "03_mosaic" / "final_mosaic.tif").is_file()
        assert (root / "03_mosaic" / "preview.png").is_file()
        assert (root / "03_mosaic" / "mosaic_summary.json").is_file()
        assert (root / "experiment_summary.md").is_file()
    assert (tmp_path / "final_results" / "paper_tables.md").is_file()


def test_materialize_revalidates_replay_mosaic_before_publishing(tmp_path):
    """A replay artifact cannot be published after its source mosaic is mutated."""
    from src.multiscene_sift.final_baseline_replay import BaselineReplaySpec, materialize_final_results, replay_baseline

    source, global_dir, grid = _inputs(tmp_path)
    spec = BaselineReplaySpec("main", "sift_mst", source, global_dir, grid, tmp_path / "replay")
    main = replay_baseline(spec)
    traditional = replay_baseline(BaselineReplaySpec("traditional", "sift_mst", source, global_dir, grid, tmp_path / "replay"))
    with main.mosaic_path.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError):
        materialize_final_results(main, traditional, tmp_path / "final_results")
