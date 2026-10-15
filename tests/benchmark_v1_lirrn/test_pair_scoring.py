import numpy as np
import rasterio
from affine import Affine
import subprocess
import sys
from pathlib import Path
import warnings
import json

from src.benchmark_v1_lirrn.frozen_crop import RasterCrop
from src.benchmark_v1_lirrn.pair_runner import score_pair_crop, write_corrected_geotiff


def test_geotiff_roundtrip_preserves_grid_mask_and_score_support(tmp_path):
    transform = Affine(14, 0, 500000, 0, -14, 3300000)
    ref_values = np.arange(64, dtype=np.float64).reshape(8, 8) + 10
    sub_values = ref_values * 0.8 + 5
    valid_ref = np.ones((8, 8), dtype=bool)
    valid_sub = np.ones((8, 8), dtype=bool)
    valid_ref[0, 0] = False
    valid_sub[1, 1] = False
    ref = RasterCrop(ref_values, valid_ref, transform, "EPSG:32650", 0)
    sub = RasterCrop(sub_values, valid_sub, transform, "EPSG:32650", 1)
    corrected = (1.1 * sub_values - 3).astype(np.float32)
    path = tmp_path / "corrected.tif"
    write_corrected_geotiff(path, sub, corrected)
    with rasterio.open(path) as ds:
        saved = ds.read(1)
        saved_valid = ds.dataset_mask() > 0
        assert ds.crs.to_string() == "EPSG:32650"
        assert ds.transform == transform
        assert (ds.height, ds.width) == (8, 8)
        assert ds.dtypes[0] == "float32"
    assert np.array_equal(saved_valid, valid_sub)
    assert np.array_equal(saved[valid_sub], corrected[valid_sub])
    result = score_pair_crop(ref, sub, saved, saved_valid)
    assert result["support_pixels"] == int(np.count_nonzero(valid_ref & valid_sub))
    assert result["full_support"] is True
    assert result["before"]["direct"]["n"] == result["after"]["direct"]["n"]


def test_writer_converts_nonfinite_masked_pixels_to_nodata(tmp_path):
    transform = Affine(14, 0, 500000, 0, -14, 3300000)
    values = np.array([[5.0, np.nan], [7.0, np.inf]])
    valid = np.array([[True, False], [True, False]])
    crop = RasterCrop(values, valid, transform, "EPSG:32650", 1)
    path = tmp_path / "masked_nan.tif"
    write_corrected_geotiff(path, crop, values.copy())
    with rasterio.open(path) as ds:
        saved = ds.read(1)
        saved_valid = ds.dataset_mask() > 0
        assert np.array_equal(saved_valid, valid)
        assert np.isfinite(saved[valid]).all()
        assert np.all(saved[~valid] == np.float32(ds.nodata))


def test_fixed_preview_masks_nodata_before_uint8_conversion(tmp_path):
    from src.benchmark_v1_lirrn.pair_runner import _fixed_preview
    values = np.array([[1.0, np.nan], [3.0, 4.0]])
    valid = np.array([[True, False], [True, True]])
    ref = RasterCrop(values, valid, Affine.identity(), "EPSG:32650", 0)
    sub = RasterCrop(values + 1, valid, Affine.identity(), "EPSG:32650", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        info = _fixed_preview(tmp_path / "preview.png", ref, sub, sub.values)
    assert Path(info["path"]).is_file()


def test_scoring_refuses_partial_output_support():
    values = np.arange(16, dtype=np.float64).reshape(4, 4) + 1
    valid = np.ones((4, 4), dtype=bool)
    ref = RasterCrop(values, valid, Affine.identity(), "EPSG:32650", 0)
    sub = RasterCrop(values * 2, valid, Affine.identity(), "EPSG:32650", 1)
    output_valid = valid.copy()
    output_valid[0, 0] = False
    result = score_pair_crop(ref, sub, sub.values, output_valid)
    assert result["full_support"] is False
    assert result["status"] == "PARTIAL_OUTPUT"
    assert result["after"] is None


def test_cli_imports_from_outside_repository(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_lirrn_phase3.py"
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--freeze" in result.stdout and "--pair" in result.stdout


def test_summary_requires_both_complete_diagnostic_pairs_and_keeps_scope(tmp_path):
    from src.benchmark_v1_lirrn.summary import build_summary
    for pair in ("00_01", "00_10"):
        pair_dir = tmp_path / pair
        pair_dir.mkdir()
        metrics = {"status": "FULL_SUPPORT", "full_support": True, "coverage": 1.0,
                   "support_pixels": 12, "output_support_pixels": 12, "missing_pixels": 0,
                   "before": {"direct": {"mae": 1.0, "rmse": 2.0},
                              "distribution": {"mamd": 0.0, "msdd": 0.0, "rdd": 0.0}},
                   "after": {"direct": {"mae": 0.5, "rmse": 1.0},
                             "distribution": {"mamd": 0.0, "msdd": 0.0, "rdd": 0.0}},
                   "timing": {"algorithm_s": 1.0, "total_wall_s": 2.0, "peak_rss_bytes": 10,
                              "rss_sample_count": 1, "rss_status": "SAMPLED"},
                   "output": {"path": "out.tif", "sha256": "abc"},
                   "preview": {"path": "preview.png", "sha256": "def"}}
        meta = {"scope": "1024_CROP_DIAGNOSTIC_ONLY"}
        regression = {"scale": 1.0, "offset": 0.0, "actual_p_n": 1000, "rng_seed": 1}
        for name, value in (("pair_metrics.json", metrics), ("method_metadata.json", meta),
                            ("regression_audit.json", regression)):
            (pair_dir / name).write_text(json.dumps(value), encoding="utf-8")
    result = build_summary(tmp_path)
    assert result["formal_ranking_eligible"] is False
    assert result["pair_count"] == 2
    assert (tmp_path / "comparison.csv").is_file()
    assert (tmp_path / "comparison.md").is_file()
