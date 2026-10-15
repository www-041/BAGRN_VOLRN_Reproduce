from __future__ import annotations

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box

from src.benchmark_v1_radiometric_r1.roi import build_roi_candidates, classify_roi_role, replace_roi_geometry, valid_fraction_for_geometry


def test_phase3_grid_formula_assigns_only_safe_single_cell():
    safe_test = box(641000, 3241000, 642000, 3242000)
    role, cell, status = classify_roi_role(safe_test, "EPSG:32650")
    assert (role, cell, status) == ("TEST", "g64_324", "SAFE_SINGLE_CELL")
    assert classify_roi_role(box(649400, 3241000, 650000, 3242000), "EPSG:32650")[0] == "UNASSIGNED"
    assert classify_roi_role(box(641000, 3241000, 651000, 3242000), "EPSG:32650")[0] == "UNASSIGNED"


def test_candidate_proposals_never_claim_human_truth_or_approval():
    geojson = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[119.0, 29.0], [119.01, 29.0], [119.01, 29.01], [119.0, 29.0]]]},
        "properties": {"pair_id": "00_01", "region_id": "R1"},
    }]}
    result = build_roi_candidates([geojson])
    props = result["features"][0]["properties"]
    assert props["label"] == "UNKNOWN"
    assert props["review_status"] == "UNVERIFIED"
    assert props["reviewer"] == ""


def test_utm_candidate_geometry_is_transformed_to_declared_geojson_crs():
    from shapely.geometry import mapping
    source = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "geometry": mapping(box(650000, 3200000, 651000, 3201000)),
        "properties": {"pair_id": "00_01", "region_id": "UTM1", "source_crs": "EPSG:32650"},
    }]}
    candidate = build_roi_candidates([source])["features"][0]
    x, y = candidate["geometry"]["coordinates"][0][0]
    assert 115 < x < 125 and 25 < y < 35
    assert candidate["properties"]["geometry_crs"] == "EPSG:4326"
    assert candidate["properties"]["review_status"] == "UNVERIFIED"


def test_valid_fraction_uses_window_mask_not_full_dataset():
    transform = from_origin(0, 64, 1, 1)
    data = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    mask = np.full((64, 64), 255, dtype=np.uint8)
    mask[:32] = 0
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=64, width=64, count=1, dtype="uint16", crs="EPSG:32650", transform=transform) as dst:
            dst.write(data, 1)
            dst.write_mask(mask)
        with mem.open() as src:
            result = valid_fraction_for_geometry(src, box(0, 32, 32, 64))
    assert result == pytest.approx(0.0)


def test_valid_fraction_counts_roi_portion_outside_scene_as_invalid():
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=64, width=64, count=1, dtype="uint16", crs="EPSG:32650", transform=from_origin(0, 64, 1, 1)) as dst:
            dst.write(np.ones((64, 64), dtype=np.uint16), 1)
        with mem.open() as src:
            result = valid_fraction_for_geometry(src, box(-32, 0, 32, 64))
    assert result == pytest.approx(0.5)


def test_geometry_edit_invalidates_prior_human_reviews_without_erasing_history_fields():
    from shapely.geometry import box

    reviewed = {
        "roi_id": "R_TEST", "geometry": "old", "label": "STABLE", "review_status": "APPROVED",
        "reviewer": "first", "review_time_utc": "t1", "second_reviewer": "second",
        "second_review_status": "APPROVED", "second_review_time_utc": "t2", "second_review_label": "STABLE",
        "purpose": "EVALUATION",
    }
    updated = replace_roi_geometry(reviewed, box(641000, 3241000, 642000, 3242000))
    assert updated["geometry"] != "old"
    assert updated["role"] == "TEST"
    assert updated["label"] == "UNKNOWN" and updated["review_status"] == "UNVERIFIED"
    assert updated["reviewer"] == "" and updated["second_review_status"] == "UNVERIFIED"
    assert updated["second_reviewer"] == "" and updated["purpose"] == "UNASSIGNED"

