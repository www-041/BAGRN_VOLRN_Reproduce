from shapely.geometry import box, mapping

from src.benchmark_v1_radiometric_r1.roi_candidates import build_r1_roi_candidates


def test_candidates_cover_only_preselected_pairs_and_remain_unverified():
    pair_rows = [
        {"pair_id": "00_01", "scene_i": "s0", "scene_j": "s1", "days_apart": "1", "quality_i": "PASS", "quality_j": "PASS", "valid_overlap_pixels": "100"},
        {"pair_id": "02_03", "scene_i": "s2", "scene_j": "s3", "days_apart": "40", "quality_i": "RISK", "quality_j": "PASS", "valid_overlap_pixels": "100"},
    ]
    regions = [{"type": "FeatureCollection", "features": [{
        "type": "Feature", "geometry": mapping(box(650000, 3200000, 651000, 3201000)),
        "properties": {"pair_id": "00_01", "region_id": "R_00_01", "source_crs": "EPSG:32650"},
    }]}]
    scenes = [
        {"scene_id": "s0", "b9_bounds_left": "640000", "b9_bounds_bottom": "3200000", "b9_bounds_right": "660000", "b9_bounds_top": "3220000"},
        {"scene_id": "s1", "b9_bounds_left": "640000", "b9_bounds_bottom": "3200000", "b9_bounds_right": "660000", "b9_bounds_top": "3220000"},
        {"scene_id": "s2", "b9_bounds_left": "650000", "b9_bounds_bottom": "3200000", "b9_bounds_right": "670000", "b9_bounds_top": "3220000"},
        {"scene_id": "s3", "b9_bounds_left": "650000", "b9_bounds_bottom": "3200000", "b9_bounds_right": "670000", "b9_bounds_top": "3220000"},
    ]
    hashes = {"s0": "a" * 64, "s1": "b" * 64, "s2": "c" * 64, "s3": "d" * 64}
    result = build_r1_roi_candidates(pair_rows, regions, [], scenes, hashes)
    assert {f["properties"]["pair_id"] for f in result["features"]} == {"00_01", "02_03"}
    for feature in result["features"]:
        assert feature["properties"]["review_status"] == "UNVERIFIED"
        assert feature["properties"]["label"] == "UNKNOWN"
        assert feature["properties"]["source_sha256_i"] in hashes.values()
    fallback = next(f for f in result["features"] if f["properties"]["pair_id"] == "02_03")
    assert fallback["properties"]["candidate_geometry_source"] == "BBOX_INTERSECTION_UNVERIFIED"


def test_candidate_builder_rejects_selected_pair_without_real_overlap():
    import pytest

    with pytest.raises(ValueError, match="overlap"):
        build_r1_roi_candidates(
            [{"pair_id": "00_01", "valid_overlap_pixels": "0"}], [], [], [], {}
        )


def test_declared_utm_but_geographic_region_is_corrected_only_with_scene_extent_evidence():
    regions = [{"type": "FeatureCollection", "features": [{
        "type": "Feature", "geometry": mapping(box(119.0, 29.0, 119.1, 29.1)),
        "properties": {"pair_id": "00_01", "region_id": "CRS_MISMATCH", "source_crs": "EPSG:32650"},
    }]}]
    scenes = [
        {"scene_id": "s0", "product_id": "p0", "b9_bounds_left": "650000", "b9_bounds_bottom": "3200000", "b9_bounds_right": "670000", "b9_bounds_top": "3220000", "b9_lon_min": "118.9", "b9_lat_min": "28.9", "b9_lon_max": "119.2", "b9_lat_max": "29.2"},
        {"scene_id": "s1", "product_id": "p1", "b9_bounds_left": "650000", "b9_bounds_bottom": "3200000", "b9_bounds_right": "670000", "b9_bounds_top": "3220000", "b9_lon_min": "118.9", "b9_lat_min": "28.9", "b9_lon_max": "119.2", "b9_lat_max": "29.2"},
    ]
    result = build_r1_roi_candidates(
        [{"pair_id": "00_01", "scene_i": "p0", "scene_j": "p1", "valid_overlap_pixels": "10"}],
        regions, [], scenes, {},
    )["features"][0]
    x, y = result["geometry"]["coordinates"][0][0]
    assert 118 < x < 120 and 28 < y < 30
    assert result["properties"]["source_crs_resolution"] == "EPSG:4326_CONFIRMED_BY_SCENE_EXTENTS"
