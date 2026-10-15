from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from src.benchmark_v1_radiometric_r1.roi_gui import canvas_to_world, candidate_list_entries, preview_stretch, RoiPairReader


def test_canvas_coordinate_maps_to_shared_map_grid():
    world = canvas_to_world(256, 128, Window(100, 200, 512, 512), from_origin(500000, 3300000, 14, 14), canvas_size=512)
    assert world == (500000 + (100 + 256) * 14, 3300000 - (200 + 128) * 14)


def test_display_stretch_does_not_mutate_raw_samples():
    raw = np.arange(100, dtype=np.uint16).reshape(10, 10)
    original = raw.copy()
    shown = preview_stretch(raw, np.ones(raw.shape, dtype=bool))
    assert shown.dtype == np.uint8
    assert np.array_equal(raw, original)


def test_saved_review_replaces_candidate_list_entry_without_duplicate():
    candidates = [{"properties": {"roi_id": "R1_00_01", "pair_id": "00_01"}}]
    saved = {"R1_00_01": {"pair_id": "00_01", "review_status": "DOUBTFUL"}}
    assert candidate_list_entries(candidates, saved, "00_01") == [("saved", "R1_00_01")]


def test_pair_reader_uses_bounded_window_and_aligns_second_affine(tmp_path):
    path_i, path_j = tmp_path / "i.tif", tmp_path / "j.tif"
    transform_i = from_origin(0, 64, 1, 1)
    transform_j = from_origin(1, 64, 1, 1)
    cols_i = np.tile(np.arange(64, dtype=np.uint16), (64, 1))
    cols_j = np.tile((np.arange(64, dtype=np.uint16) + 1), (64, 1))
    for path, transform, data in ((path_i, transform_i, cols_i), (path_j, transform_j, cols_j)):
        with rasterio.open(path, "w", driver="GTiff", width=64, height=64, count=1, dtype="uint16", crs="EPSG:32650", transform=transform, nodata=0) as dst:
            dst.write(data, 1)
    reader = RoiPairReader({9: path_i}, {9: path_j})
    a, b, mask_a, mask_b = reader.read_window(9, Window(8, 10, 16, 12))
    assert a.shape == b.shape == (12, 16)
    assert np.allclose(a[:, 1:-1], b[:, 1:-1])
    assert mask_a.shape == mask_b.shape == a.shape
    reader.close()


def test_pair_reader_pads_window_outside_reference_grid(tmp_path):
    path_i, path_j = tmp_path / "i.tif", tmp_path / "j.tif"
    data = np.ones((32, 32), dtype=np.uint16)
    for path in (path_i, path_j):
        with rasterio.open(path, "w", driver="GTiff", width=32, height=32, count=1, dtype="uint16", crs="EPSG:32650", transform=from_origin(0, 32, 1, 1), nodata=0) as dst:
            dst.write(data, 1)
    reader = RoiPairReader({9: path_i}, {9: path_j})
    a, b, mask_a, mask_b = reader.read_window(9, Window(-4, -3, 16, 12))
    assert a.shape == b.shape == mask_a.shape == mask_b.shape == (12, 16)
    assert not mask_a[:, :4].any()
    assert not mask_b[:, :4].any()
    assert mask_a[3:, 4:].all() and mask_b[3:, 4:].all()
    reader.close()
