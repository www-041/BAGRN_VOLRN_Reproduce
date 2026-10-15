"""Synthetic pixel-centre and explicit matcher-frame contracts for Phase5."""

from dataclasses import replace
import hashlib
from pathlib import Path
import sys

from affine import Affine
import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_v1_phase5.inputs import PairInput, prepare_pair
from benchmark_v1_phase5.protocol import PairManifest, load_config
from registration_benchmark.models import MatchSet
from benchmark_v1_phase5.coordinates import (
    UnsupportedGridError, georef_only_translation, to_native_correspondences,
    view_xy_to_native_rc,
)


def _raster(path, transform, mask):
    with rasterio.open(path, "w", driver="GTiff", height=mask.shape[0],
                       width=mask.shape[1], count=1, dtype="float32",
                       crs="EPSG:32650", transform=transform, nodata=0) as dst:
        dst.write(np.ones(mask.shape, dtype="float32"), 1)
        dst.write_mask(mask.astype("uint8") * 255)
    return path


def _pair(tmp_path, *, mask_i=None, mask_j=None, resize_side=4):
    shape = (10, 12)
    mask_i = np.ones(shape, dtype=bool) if mask_i is None else mask_i
    mask_j = np.ones(shape, dtype=bool) if mask_j is None else mask_j
    i = _raster(tmp_path / "i.tif", Affine(14, 0, 100, 0, -14, 300), mask_i)
    j = _raster(tmp_path / "j.tif", Affine(14, 0, 128, 0, -14, 286), mask_j)
    manifest = PairManifest("00_01", "i", "j", i, j, i.stat().st_size,
                            j.stat().st_size, hashlib.sha256(i.read_bytes()).hexdigest(),
                            hashlib.sha256(j.read_bytes()).hexdigest())
    cfg = replace(load_config(ROOT / "configs/benchmark_v1_phase5.yaml"),
                  max_view_side=resize_side)
    pair = prepare_pair(manifest, cfg)
    assert pair.status == "READY"
    return pair


def _match(ref, tgt, *, frame="phase5_view", **metadata):
    return MatchSet("SIFT", np.asarray(ref, dtype=float), np.asarray(tgt, dtype=float),
                    np.ones(len(ref)), 0.0,
                    {"coordinate_frame": frame,
                     "confidence_semantics": "method_internal_only", **metadata})


def test_georef_only_includes_native_origin_offset_and_row_sign():
    i = Affine(14, 0, 100, 0, -14, 300)
    j = Affine(14, 0, 128, 0, -14, 286)
    assert georef_only_translation(i, j) == (-1.0, -2.0)
    assert georef_only_translation(j, i) == (1.0, 2.0)


def test_georef_only_rejects_relative_scale_and_rotation():
    i = Affine(14, 0, 100, 0, -14, 300)
    for j in (Affine(15, 0, 128, 0, -14, 286),
              Affine(14, 0.1, 128, 0, -14, 286)):
        with pytest.raises(UnsupportedGridError, match="UNSUPPORTED_GRID"):
            georef_only_translation(i, j)


def test_view_native_round_trip_after_crop_resize_and_subpixel_shifts(tmp_path):
    pair = _pair(tmp_path)
    view = pair.view
    assert view.provenance.resize_x < 1 and view.provenance.resize_y < 1
    assert view.provenance.overlap_affine.c != pair.native_affine_i.c
    xy = np.array([[0.2, 0.3], [1.4, 1.3], [2.2, 1.1]])
    for side, affine in (("i", pair.native_affine_i), ("j", pair.native_affine_j)):
        native = view_xy_to_native_rc(xy, pair, side)
        world_x, world_y = affine * (native[:, 1] + .5, native[:, 0] + .5)
        overlap_col, overlap_row = (~view.provenance.overlap_affine) * (world_x, world_y)
        back = np.column_stack((overlap_col * view.provenance.resize_x - .5,
                                overlap_row * view.provenance.resize_y - .5))
        np.testing.assert_allclose(back, xy, atol=1e-6)
    matches = to_native_correspondences(_match(xy[:2], xy[:2] + [[.25, -.2], [-.3, .2]]), pair)
    np.testing.assert_allclose(matches.j_rc - matches.i_rc,
                               np.array([[-1.0 - .2 / view.provenance.resize_y,
                                          -2.0 + .25 / view.provenance.resize_x],
                                         [-1.0 + .2 / view.provenance.resize_y,
                                          -2.0 - .3 / view.provenance.resize_x]]), atol=1e-6)
    np.testing.assert_array_equal(matches.i_view_xy, xy[:2])


def test_legacy_common_grid_is_inverted_once_and_keeps_view_positions(tmp_path):
    pair = _pair(tmp_path)
    view_xy = np.array([[1.1, 1.2], [2.1, 1.1]])
    # MatchView.to_common_grid implements origin + view / scale (not a
    # half-pixel-centred resize map). Reproduce that existing contract.
    legacy_origin = np.array([7.0, -3.0])
    legacy_scale = np.array([.25, .5])
    common_xy = legacy_origin + view_xy / legacy_scale
    matches = _match(common_xy, common_xy, frame="pair_common_grid",
                     phase5_common_grid="overlap_grid",
                     legacy_view_origin_xy=legacy_origin.tolist(),
                     legacy_view_scale_xy=legacy_scale.tolist())
    result = to_native_correspondences(matches, pair)
    np.testing.assert_allclose(result.i_view_xy, view_xy)
    np.testing.assert_allclose(result.j_view_xy, view_xy)
    np.testing.assert_allclose(result.i_rc, view_xy_to_native_rc(view_xy, pair, "i"))
    np.testing.assert_allclose(result.j_rc, view_xy_to_native_rc(view_xy, pair, "j"))
    with pytest.raises(ValueError, match="phase5_common_grid"):
        to_native_correspondences(_match(common_xy, common_xy, frame="pair_common_grid"), pair)


def test_coordinate_validity_uses_each_images_own_mask(tmp_path):
    left = np.ones((10, 12), bool)
    right = np.ones((10, 12), bool)
    right[2:5, 4:7] = False
    pair = _pair(tmp_path, mask_i=left, mask_j=right, resize_side=12)
    bad_y, bad_x = np.argwhere(~pair.view.joint_mask)[0]
    with pytest.raises(ValueError, match="scene-j"):
        to_native_correspondences(_match([[bad_x, bad_y]], [[bad_x, bad_y]]), pair)


def test_correspondence_survives_other_scene_mask_at_opposite_point(tmp_path):
    pair = _pair(tmp_path, resize_side=12)
    # The i feature is valid only in i at A; the j feature is valid only in j
    # at B. Corresponding points need not share the same raster coordinate.
    pair.view.mask_i[5, 5] = False
    pair.view.mask_j[4, 4] = False
    pair.view.joint_mask[4, 4] = False
    pair.view.joint_mask[5, 5] = False
    result = to_native_correspondences(_match([[4, 4]], [[5, 5]]), pair)
    assert result.i_rc.shape == (1, 2)


def test_scene_i_point_on_invalid_mask_is_rejected(tmp_path):
    pair = _pair(tmp_path, resize_side=12)
    pair.view.mask_i[4, 4] = False
    with pytest.raises(ValueError, match="scene-i"):
        to_native_correspondences(_match([[4, 4]], [[5, 5]]), pair)


def test_unknown_frame_and_invalid_native_coordinate_rejected(tmp_path):
    pair = _pair(tmp_path)
    with pytest.raises(ValueError, match="coordinate_frame"):
        to_native_correspondences(_match([[1, 1]], [[1, 1]], frame="mystery"), pair)
    with pytest.raises(ValueError, match="view"):
        view_xy_to_native_rc(np.array([[-1.0, 1.0]]), pair, "i")
