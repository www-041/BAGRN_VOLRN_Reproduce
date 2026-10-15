from __future__ import annotations

import json
import hashlib

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box, mapping

from benchmark_v1_radiometric_r15.multiband_view import (
    COMPOSITES,
    difference_preview,
    read_window_padded,
    render_composite,
    read_raster_roi_stats,
    roi_band_stats,
    signed_difference_rgb,
    stretch_pair,
)


def test_composite_band_maps_follow_inventory_wavelength_order():
    assert COMPOSITES["TRUE_COLOR"] == (7, 4, 3)
    assert COMPOSITES["NIR_FALSE_COLOR"] == (14, 7, 4)


def test_independent_and_shared_stretch_are_distinct_and_nodata_safe():
    left = np.array([[0, 10], [20, 30]], dtype=np.float32)
    right = np.array([[100, 110], [120, 130]], dtype=np.float32)
    valid = np.array([[True, True], [True, False]])
    left_range, right_range = stretch_pair(left, right, valid, valid, mode="independent", low_pct=0, high_pct=100)
    common_left, common_right = stretch_pair(left, right, valid, valid, mode="shared", low_pct=0, high_pct=100)
    assert left_range == (0.0, 20.0)
    assert right_range == (100.0, 120.0)
    assert common_left == common_right == (0.0, 120.0)


def test_rgb_preview_refuses_missing_band_and_ignores_invalid_pixels():
    arrays = {3: np.full((2, 2), 30.0), 4: np.full((2, 2), 20.0), 7: np.full((2, 2), 10.0)}
    masks = {band: np.array([[True, True], [True, False]]) for band in arrays}
    rgb = render_composite(arrays, masks, (7, 4, 3), ranges=((0, 40),) * 3)
    assert rgb.shape == (2, 2, 3)
    assert tuple(rgb[0, 0]) == (64, 128, 191)
    assert tuple(rgb[1, 1]) == (0, 0, 0)
    with pytest.raises(ValueError, match="missing band"):
        render_composite({3: arrays[3]}, {3: masks[3]}, (7, 4, 3), ranges=((0, 40),) * 3)


def test_difference_preview_preserves_signed_raw_difference_and_common_mask():
    left = np.array([[10, 20], [30, 40]], dtype=np.float32)
    right = np.array([[7, 25], [29, 50]], dtype=np.float32)
    valid_left = np.array([[True, True], [False, True]])
    valid_right = np.array([[True, True], [True, False]])
    difference, valid = difference_preview(left, right, valid_left, valid_right)
    assert np.array_equal(valid, [[True, True], [False, False]])
    assert np.array_equal(difference[valid], [3.0, -5.0])


def test_signed_difference_preview_displays_sign_and_masks_invalid():
    difference = np.array([[-4.0, 0.0, 4.0], [1.0, -1.0, 99.0]], dtype=np.float32)
    valid = np.array([[True, True, True], [True, True, False]])
    rgb = signed_difference_rgb(difference, valid, limit=4.0)
    assert tuple(rgb[0, 0]) == (0, 0, 255)
    assert tuple(rgb[0, 1]) == (255, 255, 255)
    assert tuple(rgb[0, 2]) == (255, 0, 0)
    assert tuple(rgb[1, 2]) == (0, 0, 0)
    difference[1, 2] = np.nan
    with np.errstate(invalid="raise"):
        masked_rgb = signed_difference_rgb(difference, valid, limit=4.0)
    assert tuple(masked_rgb[1, 2]) == (0, 0, 0)


def test_window_reader_pads_edges_and_keeps_native_pixel_mapping():
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.windows import Window

    profile = {"driver": "GTiff", "height": 3, "width": 3, "count": 1, "dtype": "uint16",
               "crs": "EPSG:32650", "transform": Affine(14, 0, 100, 0, -14, 200), "nodata": 0}
    with MemoryFile() as mem:
        with mem.open(**profile) as ds:
            ds.write(np.arange(1, 10, dtype=np.uint16).reshape(3, 3), 1)
            data, valid = read_window_padded(ds, Window(-1, -1, 3, 3))
            assert data.shape == valid.shape == (3, 3)
            assert np.array_equal(data, [[0, 0, 0], [0, 1, 2], [0, 4, 5]])
            assert not valid[0, :].any() and valid[1, 1]
            x, y = ds.transform @ (1.5, 1.5)
            assert (~ds.transform) @ (x, y) == pytest.approx((1.5, 1.5))


def test_roi_stats_use_polygon_and_raster_mask_for_each_scene():
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    stats = roi_band_stats(data, valid, Affine(1, 0, 0, 0, -1, 4), box(1, 1, 3, 3))
    assert stats["valid_fraction"] == 1.0
    assert stats["valid_count"] == 4
    assert stats["median"] == 7.5
    assert stats["mad"] == 2.0
    assert stats["std"] == pytest.approx(np.std([5, 6, 9, 10]))


def test_raster_roi_stats_count_outside_scene_as_invalid_support():
    import rasterio
    from rasterio.io import MemoryFile

    profile = {"driver": "GTiff", "height": 2, "width": 2, "count": 1, "dtype": "uint16",
               "crs": "EPSG:32650", "transform": Affine(1, 0, 0, 0, -1, 2), "nodata": 0}
    with MemoryFile() as mem:
        with mem.open(**profile) as ds:
            ds.write(np.array([[1, 2], [3, 4]], dtype=np.uint16), 1)
            stats = read_raster_roi_stats(ds, box(0, 0, 4, 2))
    assert stats["roi_pixel_count"] == 8
    assert stats["valid_count"] == 4
    assert stats["valid_fraction"] == 0.5


def test_multiband_labels_are_separate_and_change_requires_evidence(tmp_path):
    from benchmark_v1_radiometric_r15.multiband_review import MultiBandReviewStore

    candidate = tmp_path / "candidates.geojson"
    candidate.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "geometry": mapping(box(10_500, 10_500, 11_000, 11_000)),
        "properties": {"pair_id": "05_07", "candidate_id": "05_07_c1", "role": "DEVELOPMENT",
                       "human_label": "UNKNOWN", "suggested_label": "UNKNOWN", "review_status": "UNVERIFIED"}
    }]}), encoding="utf-8")
    store = MultiBandReviewStore(candidate, tmp_path / "new" / "snapshot.json", tmp_path / "new" / "events.jsonl")
    key = ("05_07", "05_07_c1")
    assert store.get(*key)["landcover_type"] == "UNKNOWN"
    assert store.get(*key)["temporal_status"] == "UNKNOWN"
    assert store.get(*key)["formal_stable_truth_eligible"] is False
    with pytest.raises(ValueError, match="source:.*reference:"):
        store.record_labels(*key, reviewer="reviewer-a", decision="APPROVED",
                            landcover_type="VEGETATION", terrain_type="SLOPE",
                            temporal_status="CONFIRMED_CHANGE", interference_type="NONE",
                            evidence="It looks brighter here.")
    saved = store.record_labels(*key, reviewer="reviewer-a", decision="APPROVED",
                                landcover_type="VEGETATION", terrain_type="SLOPE",
                                temporal_status="NO_CHANGE", interference_type="NONE",
                                evidence="No visible land-cover change at native 14 m review scale.")
    assert saved["legacy_human_label"] == "UNKNOWN"
    assert saved["temporal_status"] == "NO_CHANGE"
    assert saved["formal_stable_truth_eligible"] is False
    assert (tmp_path / "new" / "events.jsonl").is_file()
    assert not (tmp_path / "legacy_review.json").exists()


def test_new_multiband_review_reads_legacy_state_but_never_modifies_legacy_files(tmp_path):
    from benchmark_v1_radiometric_r15.multiband_review import MultiBandReviewStore

    candidate = tmp_path / "candidates.geojson"
    candidate.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "geometry": mapping(box(2_500, 2_500, 3_000, 3_000)),
        "properties": {"pair_id": "05_07", "candidate_id": "05_07_c1", "role": "DEVELOPMENT",
                       "human_label": "UNKNOWN", "suggested_label": "UNKNOWN", "review_status": "UNVERIFIED"}
    }]}), encoding="utf-8")
    old_snapshot = tmp_path / "old" / "snapshot.json"
    old_events = tmp_path / "old" / "events.jsonl"
    old_snapshot.parent.mkdir()
    old_snapshot.write_text(json.dumps({"records": [{"pair_id": "05_07", "candidate_id": "05_07_c1",
        "human_label": "LEGACY_ONLY", "review_status": "APPROVED", "reviewer": "prior-human",
        "review_time_utc": "2026-01-01T00:00:00Z"}]}), encoding="utf-8")
    old_events.write_text('{"pair_id":"05_07","candidate_id":"05_07_c1","after":{"human_label":"OTHER_OLD"}}\n', encoding="utf-8")
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_snapshot, old_events)}
    store = MultiBandReviewStore(candidate, tmp_path / "new" / "snapshot.json", tmp_path / "new" / "events.jsonl",
                                 legacy_snapshot_path=old_snapshot, legacy_events_path=old_events)
    row = store.get("05_07", "05_07_c1")
    assert row["legacy_human_label"] == "OTHER_OLD"
    assert row["legacy_review_status"] == "APPROVED"
    assert row["landcover_type"] == "UNKNOWN"
    assert row["multiband_review_decision"] == "UNVERIFIED"
    store.record_labels("05_07", "05_07_c1", reviewer="reviewer-a", decision="DOUBTFUL",
                        landcover_type="UNKNOWN", terrain_type="UNKNOWN", temporal_status="UNCERTAIN",
                        interference_type="UNKNOWN", evidence="Need a second look.")
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_snapshot, old_events)}
    assert json.loads((tmp_path / "new" / "snapshot.json").read_text(encoding="utf-8"))["schema_version"] == 2
    event = json.loads((tmp_path / "new" / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert event["schema_version"] == 2


def test_multiband_test_review_requires_distinct_reviewer_and_matching_labels(tmp_path):
    from benchmark_v1_radiometric_r15.multiband_review import MultiBandReviewStore

    candidate = tmp_path / "candidates.geojson"
    candidate.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "geometry": mapping(box(10_500, 10_500, 11_000, 11_000)),
        "properties": {"pair_id": "05_07", "candidate_id": "05_07_c1", "role": "TEST",
                       "human_label": "UNKNOWN", "suggested_label": "UNKNOWN", "review_status": "UNVERIFIED"}
    }]}), encoding="utf-8")
    store = MultiBandReviewStore(candidate, tmp_path / "snapshot.json", tmp_path / "events.jsonl")
    key = ("05_07", "05_07_c1")
    fields = dict(landcover_type="BARE_GROUND", terrain_type="RIDGE",
                  temporal_status="POSSIBLE_CHANGE", interference_type="UNKNOWN")
    store.record_labels(*key, reviewer="www", decision="APPROVED", evidence="Reviewed both dates.", **fields)
    with pytest.raises(ValueError, match="different real reviewer"):
        store.record_labels(*key, reviewer="www", decision="APPROVED", evidence="Checked.",
                            second_review=True, **fields)
    approved = store.record_labels(*key, reviewer="reviewer-b", decision="APPROVED",
                                   evidence="Independently checked both dates.", second_review=True, **fields)
    assert approved["multiband_second_review_status"] == "APPROVED"
    assert approved["multiband_dispute_status"] == "NONE"
    assert approved["eligible_for_test"] is False


def test_multiband_geometry_edit_retains_frozen_role_and_rechecks_safety(tmp_path):
    from benchmark_v1_radiometric_r15.multiband_review import MultiBandReviewStore

    candidate = tmp_path / "candidates.geojson"
    from benchmark_v1_radiometric_r15.spatial import assign_frozen_role
    original_geometry = box(2_500, 2_500, 3_000, 3_000)
    frozen_role, frozen_cell, _ = assign_frozen_role(original_geometry)
    alternative = box(12_500, 2_500, 13_000, 3_000)
    assert assign_frozen_role(alternative)[0] != frozen_role
    feature = {"type": "Feature", "geometry": mapping(original_geometry),
               "properties": {"pair_id": "05_07", "candidate_id": "05_07_c1", "role": frozen_role,
                              "grid_cell_id": frozen_cell, "valid_support_fraction": 1.0,
                              "spatial_status": "SAFE_SINGLE_CELL", "human_label": "UNKNOWN",
                              "suggested_label": "UNKNOWN", "review_status": "UNVERIFIED"}}
    candidate.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}), encoding="utf-8")
    store = MultiBandReviewStore(candidate, tmp_path / "snapshot.json", tmp_path / "events.jsonl")
    # A move into a different role cell may be recorded but cannot reassign the frozen role.
    edited = store.edit_geometry("05_07", "05_07_c1", alternative,
                                 support_validator=lambda _g: {"B9": {"joint_valid_fraction": 1.0}},
                                 all_candidate_features=[feature], all_pairs_available=True, editor="reviewer-a")
    assert edited["frozen_role"] == edited["role"] == frozen_role
    assert edited["grid_cell_id"] == frozen_cell
    assert edited["spatial_edit_eligibility"] == "INELIGIBLE_FROZEN_ROLE_MISMATCH"
    assert edited["eligible_for_test"] is False
    assert edited["full_pairwise_leakage_recomputed"] is True
    assert edited["valid_support_recomputed"] is True


def test_multiband_gui_exposes_color_views_and_separate_labels(tmp_path):
    import tkinter as tk
    import rasterio
    from benchmark_v1_radiometric_r15.gui import AutoRoiReviewer
    from benchmark_v1_radiometric_r15.multiband_review import MultiBandReviewStore

    affine = Affine(14, 0, 0, 0, -14, 224)
    band_paths = ({}, {})
    for scene_index in range(2):
        for band in (3, 4, 7, 9, 14):
            path = tmp_path / f"s{scene_index}_B{band}.tif"
            data = (np.arange(256, dtype=np.uint16).reshape(16, 16) + 10 + band + scene_index)
            data[0, 0] = 0
            with rasterio.open(path, "w", driver="GTiff", width=16, height=16, count=1,
                               dtype="uint16", crs="EPSG:32650", transform=affine, nodata=0) as dst:
                dst.write(data, 1)
            band_paths[scene_index][band] = path
    candidate = tmp_path / "candidates.geojson"
    candidate.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "geometry": mapping(box(70, 70, 154, 154)),
            "properties": {"pair_id": "05_07", "candidate_id": "05_07_c1", "role": "TEST",
                       "human_label": "UNKNOWN", "suggested_label": "UNKNOWN", "review_status": "UNVERIFIED"}
    }]}), encoding="utf-8")
    store = MultiBandReviewStore(candidate, tmp_path / "review" / "snapshot.json", tmp_path / "review" / "events.jsonl")
    store.record_labels("05_07", "05_07_c1", reviewer="first-person", decision="APPROVED",
                        landcover_type="VEGETATION", terrain_type="SLOPE", temporal_status="NO_CHANGE",
                        interference_type="NONE", evidence="Initial visual pass only.")
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    try:
        gui = AutoRoiReviewer(root, "05_07", store, band_paths[0], band_paths[1], "second-person", second_review=True,
                              multiband_mode=True, band_centers_nm={3: 490, 4: 530.5, 7: 633.5, 9: 701.5, 14: 850.5})
        root.update_idletasks(); root.update()
        assert gui.composite_var.get() == "GRAY"
        assert gui.multiband_label_vars["terrain_type"].get() == "UNKNOWN"
        assert gui.multiband_label_vars["landcover_type"].get() == "UNKNOWN"
        gui.composite_var.set("TRUE_COLOR"); gui._render()
        gui.composite_var.set("NIR_FALSE_COLOR"); gui.stretch_var.set("SHARED"); gui._render()
        assert gui.canvas_i.find_all() and gui.canvas_j.find_all()
        assert "geometry" in gui.warning_var.get().lower()
        gui.display_var.set("DIFFERENCE"); gui._render()
        assert "scene_i-scene_j" in gui.render_mode_description
        assert "valid=" in gui.summary_var.get()
        assert store.get("05_07", "05_07_c1")["multiband_review_decision"] == "APPROVED"
    finally:
        root.destroy()
