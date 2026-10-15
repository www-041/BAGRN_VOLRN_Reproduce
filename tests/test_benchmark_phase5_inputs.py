"""Synthetic checks for the Phase5 truth-free, bounded common input."""

from dataclasses import FrozenInstanceError, fields, replace
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_phase4_freeze import FrozenInputError, FrozenTrial
from benchmark_v1_phase5 import build_pair_manifest, load_config, prepare_pair
from benchmark_v1_phase5.protocol import PairManifest


def _raster(path, data, transform, mask=None):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype="float32", crs="EPSG:32650", transform=transform,
                       nodata=0) as dst:
        dst.write(data.astype("float32"), 1)
        if mask is not None:
            dst.write_mask(np.asarray(mask, dtype="uint8") * 255)
    return path


def _snapshot(tmp_path, left, right):
    records = []
    for index, path in ((0, left), (1, right)):
        records.append(dict(scene_index=index, scene_id=f"scene_{index}", path=str(path),
                            size_bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return FrozenTrial(tmp_path, {}, {("00_01", "secret"): {"x_i": "TEST_TRUTH"}},
                       {"b9_verified_inputs": records}, {}, "unused")


def _pair(tmp_path, left_data, right_data, left_mask=None, right_mask=None,
          left_transform=None, right_transform=None):
    transform = from_origin(100, 200, 14, 14)
    left = _raster(tmp_path / "left.tif", left_data, left_transform or transform, left_mask)
    right = _raster(tmp_path / "right.tif", right_data, right_transform or transform, right_mask)
    return _snapshot(tmp_path, left, right)


def _config():
    return load_config(ROOT / "configs/benchmark_v1_phase5.yaml")


def test_config_freezes_four_pairs_and_protocol_values():
    cfg = _config()
    assert cfg.pair_ids == ("00_01", "02_03", "05_07", "08_10")
    assert cfg.method_ids == ("SIFT", "KARIOS", "AROSICS_GLOBAL", "LoFTR",
                              "EfficientLoFTR", "LightGlue+DISK")
    assert (cfg.seed, cfg.ransac_hypotheses, cfg.ransac_radius_native_px) == (20261011, 1000, 2.0)
    assert (cfg.min_matches, cfg.min_inliers, cfg.min_inlier_fraction) == (12, 8, .10)
    assert (cfg.max_view_side, cfg.smoke_seconds, cfg.smoke_rss_gib) == (1600, 900, 12)


def test_config_stretch_is_frozen_to_2nd_and_98th_percentiles():
    cfg = _config()
    assert cfg.stretch_percentiles == (2, 98)


def test_lightglue_1024_is_extractor_resize_not_match_cap(tmp_path):
    cfg = _config()
    settings = dict(cfg.method_settings)["LightGlue+DISK"]
    assert dict(settings)["extractor_resize"] == 1024
    assert "matches" not in dict(settings)
    raw = (ROOT / "configs/benchmark_v1_phase5.yaml").read_text(encoding="utf-8")
    altered = tmp_path / "wrong.yaml"
    altered.write_text(raw.replace("extractor_resize: 1024", "matches: 1024"), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(altered)


def test_config_carries_immutable_exact_file_hash():
    path = ROOT / "configs/benchmark_v1_phase5.yaml"
    cfg = load_config(path)
    assert cfg.config_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FrozenInstanceError):
        cfg.config_sha256 = "changed"


@pytest.mark.parametrize("extra", ["pair_ids: [00_01, 02_03, 05_07, 08_10, 01_02]",
                                  "test_coordinates: [1, 2]", "candidate_id: hidden",
                                  "second_review: approved", "split: TEST"])
def test_config_rejects_fifth_pair_and_truth_keys(tmp_path, extra):
    raw = (ROOT / "configs/benchmark_v1_phase5.yaml").read_text(encoding="utf-8")
    if extra.startswith("pair_ids:"):
        raw = "\n".join(extra if line.startswith("pair_ids:") else line
                        for line in raw.splitlines())
    else:
        raw += "\n" + extra + "\n"
    path = tmp_path / "config.yaml"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_manifest_rejects_fifth_pair_and_changed_b9_bytes(tmp_path):
    data = np.ones((3, 3), dtype="float32")
    snapshot = _pair(tmp_path, data, data)
    with pytest.raises(ValueError):
        build_pair_manifest(snapshot, "01_02")
    manifest = build_pair_manifest(snapshot, "00_01")
    assert set(manifest.__dict__) == {f.name for f in fields(PairManifest)}
    assert "secret" not in repr(manifest) and "TEST_TRUTH" not in repr(manifest)
    with (tmp_path / "left.tif").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(FrozenInputError):
        build_pair_manifest(snapshot, "00_01")


def test_disjoint_valid_masks_return_explicit_no_valid_overlap(tmp_path):
    data = np.ones((4, 4), dtype="float32")
    left_mask = np.zeros((4, 4), dtype=bool)
    right_mask = np.zeros((4, 4), dtype=bool)
    left_mask[:, :2] = True
    right_mask[:, 2:] = True
    snapshot = _pair(tmp_path, data, data, left_mask, right_mask)
    result = prepare_pair(build_pair_manifest(snapshot, "00_01"), _config())
    assert result.status == "NO_VALID_OVERLAP"
    assert result.view is None
    assert result.native_joint_valid_count == result.view_joint_valid_count == 0


def test_per_scene_overlap_masks_are_retained_and_joint_is_intersection(tmp_path):
    data = np.ones((3, 3), dtype="float32")
    left_mask = np.ones((3, 3), dtype=bool)
    right_mask = np.ones((3, 3), dtype=bool)
    left_mask[0, 0] = False
    right_mask[0, 1] = False
    snapshot = _pair(tmp_path, data, data, left_mask, right_mask)
    result = prepare_pair(build_pair_manifest(snapshot, "00_01"), _config())
    view = result.view
    assert view.mask_i.shape == view.mask_j.shape == (3, 3)
    assert not view.mask_i[0, 0] and view.mask_j[0, 0]
    assert view.mask_i[0, 1] and not view.mask_j[0, 1]
    np.testing.assert_array_equal(view.joint_mask, view.mask_i & view.mask_j)
    assert result.native_joint_valid_count == result.view_joint_valid_count == 7


def test_native_support_lost_after_resize_is_preprocessing_failure(tmp_path):
    data = np.ones((2, 1601), dtype="float32")
    only_middle = np.zeros_like(data, dtype=bool)
    only_middle[0, 800] = True
    snapshot = _pair(tmp_path, data, data, only_middle, only_middle)
    result = prepare_pair(build_pair_manifest(snapshot, "00_01"), _config())
    assert result.status == "PREPROCESSING_FAIL"
    assert result.reason == "NO_USABLE_VIEW_AFTER_RESIZE"
    assert result.native_joint_valid_count == 1
    assert result.view_joint_valid_count == 0
    assert result.view is None


def test_stretch_uses_2nd_and_98th_percentiles_and_records_bounds(tmp_path):
    left = np.arange(1, 101, dtype="float32").reshape(10, 10)
    right = left + 100
    snapshot = _pair(tmp_path, left, right)
    view = prepare_pair(build_pair_manifest(snapshot, "00_01"), _config()).view
    assert view.provenance.stretch_percentiles == (2, 98)
    assert view.provenance.stretch_bounds_i == pytest.approx((2.98, 98.02))
    assert view.provenance.stretch_bounds_j == pytest.approx((102.98, 198.02))
    assert view.provenance.data_resampling == "bilinear"
    assert view.provenance.mask_resampling == "nearest"
    assert view.left[0, 0] == 0
    assert view.left[9, 9] == 1
    assert view.left[4, 9] == pytest.approx((50 - 2.98) / (98.02 - 2.98))
    assert view.right[4, 9] == pytest.approx(view.left[4, 9])


def test_common_crop_mask_stretch_and_reversible_provenance(tmp_path):
    left = np.arange(1, 17, dtype="float32").reshape(4, 4)
    right = left + 100
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    snapshot = _pair(tmp_path, left, right, mask, mask,
                     right_transform=from_origin(114, 200, 14, 14))
    result = prepare_pair(build_pair_manifest(snapshot, "00_01"), _config())
    assert result.status == "READY"
    view = result.view
    assert view.left.shape == view.right.shape == view.joint_mask.shape == (4, 3)
    assert not view.joint_mask[0, 0]
    assert np.all(view.left[~view.joint_mask] == 0)
    assert np.all(view.right[~view.joint_mask] == 0)
    assert np.all((view.left[view.joint_mask] >= 0) & (view.left[view.joint_mask] <= 1))
    assert view.provenance.resize_x == view.provenance.resize_y == 1
    assert view.provenance.view_to_native_i(0, 0) == pytest.approx((0, 1))
    assert view.provenance.view_to_native_j(0, 0) == pytest.approx((0, 0))
    assert not any(hasattr(result, name) for name in ("candidates", "assignments", "truth"))


def test_nonfinite_affine_rejected_before_resampling(tmp_path):
    data = np.ones((2, 2), dtype="float32")
    snapshot = _pair(tmp_path, data, data)
    manifest = build_pair_manifest(snapshot, "00_01")
    with rasterio.open(manifest.path_i, "r+") as src:
        src.transform = rasterio.Affine(14, 0, float("nan"), 0, -14, 200)
    # A separately frozen (synthetic) identity with that affine must still fail.
    manifest = replace(manifest, sha256_i=hashlib.sha256(manifest.path_i.read_bytes()).hexdigest(),
                       size_i=manifest.path_i.stat().st_size)
    with pytest.raises(ValueError, match="affine"):
        prepare_pair(manifest, _config())


def test_reads_are_windowed_and_view_side_is_bounded(tmp_path, monkeypatch):
    import benchmark_v1_phase5.inputs as inputs

    data = np.ones((300, 300), dtype="float32")
    snapshot = _pair(tmp_path, data, data)
    windows = []
    original = inputs._read_source_tile

    def observed(src, window):
        windows.append(window)
        return original(src, window)

    monkeypatch.setattr(inputs, "_read_source_tile", observed)
    result = prepare_pair(build_pair_manifest(snapshot, "00_01"), _config())
    assert result.status == "READY"
    assert windows and all(0 < w.width <= 258 and 0 < w.height <= 258 for w in windows)
    assert all(w.width < 300 and w.height < 300 for w in windows)
    assert max(result.view.left.shape) <= 1600


def test_realized_resize_factors_map_back_to_native_centres(tmp_path):
    data = np.ones((2, 1601), dtype="float32")
    snapshot = _pair(tmp_path, data, data)
    result = prepare_pair(build_pair_manifest(snapshot, "00_01"), _config())
    assert result.status == "READY"
    assert result.view.left.shape == (2, 1600)
    provenance = result.view.provenance
    assert provenance.resize_x == pytest.approx(1600 / 1601)
    assert provenance.resize_y == 1
    assert provenance.view_to_native_i(1599, 0) == pytest.approx((0, 1599.9996875))
