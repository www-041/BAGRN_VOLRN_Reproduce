"""Tests for :mod:`src.multiscene_sift.white_artifact_diagnostics`."""

from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin

from src.multiscene_sift.white_artifact_diagnostics import (
    stage_statistics,
    count_new_invalid,
    extreme_high_threshold,
    make_invalid_mask,
    make_extreme_high_mask,
    save_stage_tiff,
    block_valid_fraction_stats,
    coefficient_outlier_rows,
    affine_coordinate_diagnostic,
    write_json,
)


class TestStageStatistics:
    def test_plain_image(self):
        """Basic valid-image stats."""
        data = np.arange(0, 1000, 10, dtype=np.float64).reshape(10, 10)
        s = stage_statistics(data, None)
        assert s["total_pixels"] == 100
        assert s["valid_pixels"] == 100
        assert s["nan_pixels"] == 0
        assert s["inf_pixels"] == 0
        assert s["valid_min"] == 0.0
        assert s["valid_max"] == 990.0
        assert s["mean"] == pytest.approx(495.0)

    def test_nodata_and_nan_counts(self):
        """Nodata/nan/inf must be counted distinctly."""
        data = np.ones((10, 10)) * 100.0
        data[0, 0] = -1  # nodata
        data[0, 1] = -1  # nodata
        data[0, 2] = np.nan
        data[0, 3] = np.inf
        s = stage_statistics(data, nodata=-1)
        assert s["nodata_pixels"] == 2
        assert s["nan_pixels"] == 1
        assert s["inf_pixels"] == 1
        assert s["valid_pixels"] == 96

    def test_p99_9_above_p50(self):
        """Percentiles ordered."""
        rng = np.random.default_rng(7)
        data = rng.uniform(0, 1000, size=(64, 64))
        s = stage_statistics(data, None)
        assert s["p50"] <= s["p99"] <= s["p99_9"]
        assert s["p0_1"] <= s["p1"] <= s["p50"]


class TestCountNewInvalid:
    def test_no_new_invalid(self):
        before = np.arange(100, dtype=np.float64).reshape(10, 10)
        after = before.copy()
        r = count_new_invalid(before, None, after, None)
        assert r == {"new_nodata": 0, "new_nan": 0, "new_inf": 0}

    def test_new_nan_detected(self):
        before = np.ones((10, 10)) * 5.0
        after = np.ones((10, 10)) * 5.0
        after[3, 3] = np.nan
        after[4, 4] = np.inf
        r = count_new_invalid(before, None, after, None)
        assert r["new_nan"] == 1
        assert r["new_inf"] == 1

    def test_nodata_change_detected(self):
        before = np.ones((10, 10)) * 5.0
        after = before.copy()
        after[2, 2] = -9999.0
        r = count_new_invalid(before, 0, after, -9999.0)
        # before had one nodata pixel (0 stays valid? before has no 0s),
        # after has one nodata at (2,2)
        assert r["new_nodata"] == 1


class TestMasks:
    def test_invalid_mask(self):
        data = np.ones((4, 4))
        data[0, 0] = np.nan
        data[1, 1] = np.inf
        data[2, 2] = -1
        m = make_invalid_mask(data, nodata=-1)
        assert m[0, 0] and m[1, 1] and m[2, 2]
        assert m[3, 3] == False

    def test_extreme_high_mask(self):
        rng = np.random.default_rng(3)
        data = rng.uniform(0, 1000, size=(32, 32))
        thr = extreme_high_threshold(data, None, percentile=99.0)
        m = make_extreme_high_mask(data, None, thr)
        n_high = int(m.sum())
        # ~1% of 1024 pixels above p99
        assert 1 <= n_high <= 30


class TestSaveStageTiff:
    def test_roundtrip(self, tmp_path):
        tf = from_origin(500000, 4000000, 30, 30)
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        path = tmp_path / "stage.tif"
        save_stage_tiff(data, tf, "EPSG:32650", -1.0, path)
        with rasterio.open(path) as src:
            assert src.count == 1
            assert src.transform == tf
            assert src.crs.to_string() == "EPSG:32650"
            assert src.nodata == -1.0
            assert src.width == 10 and src.height == 10
            np.testing.assert_allclose(src.read(1), data)


class TestBlockDiagnostics:
    def _make_blocks(self):
        from src.volrn import BlockInfo

        rng = np.random.default_rng(1)
        blk = []
        # block 0: 100% valid
        m0 = np.ones((1, 50, 50), dtype=bool)
        blk.append(BlockInfo(0, 0, 0, 0, (0, 50, 0, 50), 500100.0, 3999900.0,
                             np.array([50.0]), np.array([10.0]), m0))
        # block 1: 2% valid
        m1 = np.zeros((1, 50, 50), dtype=bool)
        m1[:, :1, :1] = True
        blk.append(BlockInfo(1, 0, 0, 1, (0, 50, 50, 100), 500200.0, 3999900.0,
                             np.array([50.0]), np.array([10.0]), m1))
        # block 2: 40% valid
        m2 = rng.random((1, 50, 50)) > 0.6
        blk.append(BlockInfo(2, 1, 1, 0, (50, 100, 0, 50), 500100.0, 3999800.0,
                             np.array([60.0]), np.array([12.0]), m2))
        return blk

    def test_valid_fraction_stats(self):
        stats = block_valid_fraction_stats(self._make_blocks())
        assert stats["n_blocks"] == 3
        # block1 frac=4/2500=0.16% <1%, block2 ~40%
        assert stats["valid_fraction_lt_1pct"] == 1
        assert stats["valid_fraction_lt_30pct"] == 1  # block1 only

    def test_coefficient_outliers(self):
        details = []
        for k in range(10):
            details.append({
                "block_id": k, "image_idx": 0,
                "grid_m": 0, "grid_n": k,
                "center_x": 0.0, "center_y": 0.0,
                "mu": 50.0, "sigma": 10.0,
                "a": 1.0, "b": 0.0,
                "converged": True, "iterations": 10,
            })
        # inject an extreme outlier
        details[5]["a"] = 100.0
        details[5]["b"] = -500.0
        info = coefficient_outlier_rows(details)
        assert info["n_flagged"] >= 1
        ids = [r["block_id"] for r in info["flagged"]]
        assert 5 in ids
        assert "per_scene" in info and 0 in info["per_scene"]


class TestAffineCoordinateDiagnostic:
    def test_north_up_zero_cross_term(self):
        tf = from_origin(500000, 4000000, 30, 30)
        diag = affine_coordinate_diagnostic([tf], [128], [128], 30, 30, 200)
        assert diag["max_cross_term_shift_pixels"] == pytest.approx(0.0)

    def test_rotated_positive_cross_term(self):
        # 10-degree rotation: d and b become non-zero
        rad = math.radians(10)
        c, s = math.cos(rad), math.sin(rad)
        tf = Affine(30 * c, -30 * s, 500000, 30 * s, 30 * c, 4000000)
        diag = affine_coordinate_diagnostic([tf], [1024], [1024], 30, 30, 200)
        assert diag["max_cross_term_shift_pixels"] > 1.0
        assert diag["scenes"][0]["rotation_deg"] == pytest.approx(10.0, abs=0.5)
        # cross-block estimate must be positive
        assert diag["max_cross_term_shift_blocks"] > 0.0


class TestWriteJson:
    def test_nan_converted(self, tmp_path):
        obj = {"a": float("nan"), "b": [1, float("inf")], "c": 5}
        path = tmp_path / "d.json"
        write_json(path, obj)
        import json
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["a"] is None
        assert loaded["b"][1] is None
        assert loaded["c"] == 5


class TestUnionVsFinal:
    """Lightweight check of the mosaic union-vs-final comparison."""

    def _make_mosaic(self, tmp_path):
        from src.mosaic import create_mosaic
        from src.multiscene_sift.models import MosaicGrid

        tf = from_origin(500000, 4000000, 30, 30)
        # One scene, full valid coverage -> union == final
        data = (np.arange(64 * 64, dtype=np.float64).reshape(64, 64) % 1000)
        arr = data[np.newaxis, :, :]

        grid = MosaicGrid(
            crs=rasterio.crs.CRS.from_epsg(32650),
            transform=tf, width=64, height=64, resolution=30.0,
        )
        mosaic_path = tmp_path / "mosaic.tif"
        create_mosaic(
            arrays=[arr], transforms=[tf], crs="EPSG:32650",
            nodata_values=[None], output_path=str(mosaic_path),
            resolution=30.0, mode="weighted",
            output_transform=tf, output_width=64, output_height=64,
        )
        return arr, tf, grid, mosaic_path

    def test_full_coverage_zero_hole(self, tmp_path):
        from scripts.diagnose_b14_white_artifacts import _union_vs_final

        arr, tf, grid, mosaic_path = self._make_mosaic(tmp_path)
        stats = _union_vs_final(
            [arr], [tf], [None], grid,
            rasterio.crs.CRS.from_epsg(32650),
            mosaic_path, tmp_path,
        )
        assert stats["union_valid_pixels"] == 64 * 64
        assert stats["final_valid_pixels"] == 64 * 64
        assert stats["union_but_final_invalid_pixels"] == 0
        assert (tmp_path / "mosaic_union_but_final_invalid_mask.png").exists()