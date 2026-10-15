from __future__ import annotations

import json

import pytest
from shapely.geometry import box, mapping, shape

from benchmark_v1_radiometric_r15.spatial import assign_frozen_role, audit_candidate_sets, revalidate_geometry_edit
from benchmark_v1_radiometric_r15.storage import load_candidate_collection, write_candidate_artifacts


def _candidate(cid, pair, geom, role=None):
    role = role or assign_frozen_role(geom)[0]
    return {"type": "Feature", "geometry": mapping(geom), "properties": {
        "candidate_id": cid, "pair_id": pair, "role": role,
        "suggested_label": "UNKNOWN", "review_status": "UNVERIFIED",
    }}


def test_frozen_cell_origin_and_role_formula_match_phase3():
    assert assign_frozen_role(box(1050, 1050, 2000, 2000)) == ("DEVELOPMENT", "g0_0", "SAFE_SINGLE_CELL")
    assert assign_frozen_role(box(11050, 1050, 12000, 2000)) == ("TEST", "g1_0", "SAFE_SINGLE_CELL")


@pytest.mark.parametrize("geom", [box(500, 1000, 2000, 2000), box(9000, 1000, 10050, 2000), box(9000, 9000, 11000, 11000)])
def test_500m_edge_and_crossing_cell_are_unassigned(geom):
    assert assign_frozen_role(geom)[0] == "UNASSIGNED"


def test_same_ground_overlap_across_pairs_and_duplicate_ids_are_reported():
    geom = box(1050, 1050, 2000, 2000)
    audit = audit_candidate_sets([
        [_candidate("same", "00_01", geom)],
        [_candidate("same", "00_01", geom), _candidate("other", "02_03", geom)],
    ])
    assert len(audit["duplicate_candidate_ids"]) == 1
    assert len(audit["overlaps"]) == 1


def test_opposite_role_nearby_areas_flagged_without_repair():
    development = box(8050, 1500, 9000, 2450)
    test = box(11050, 1500, 12000, 2450)
    audit = audit_candidate_sets([[_candidate("d", "00_01", development)],
                                  [_candidate("t", "02_03", test)]])
    assert audit["opposite_role_proximity"] == []
    conflicting = _candidate("t2", "02_03", box(11050, 1500, 12000, 2450), role="DEVELOPMENT")
    audit2 = audit_candidate_sets([[_candidate("d", "00_01", development)], [conflicting]])
    assert audit2["role_conflicts"]


def test_geometry_edit_recomputes_support_role_and_marks_leakage_pending():
    old = _candidate("c1", "00_01", box(1050, 1050, 2000, 2000))
    old["properties"].update({"eligible_for_test": True, "support_by_scene_band": {"old": 1.0}})
    updated, event = revalidate_geometry_edit(
        old, box(9500, 1500, 10500, 2450),
        support_validator=lambda geometry: {"scene_i_B9": 0.98, "scene_j_B9": 0.96, "auxiliary": "UNKNOWN"},
        all_candidates=[old],
    )
    assert updated["properties"]["role"] == "UNASSIGNED"
    assert updated["properties"]["support_by_scene_band"]["scene_i_B9"] == 0.98
    assert updated["properties"]["pairwise_leakage_audit"] == "PENDING_ALL_PAIR_REAUDIT"
    assert updated["properties"]["eligible_for_test"] is False
    assert event["before_geometry"] == old["geometry"]
    assert shape(event["after_geometry"]).equals(box(9500, 1500, 10500, 2450))


def test_geometry_edit_recomputes_full_four_pair_leakage_when_all_pairs_loaded():
    geometry = box(1050, 1050, 2000, 2000)
    current = _candidate("c1", "00_01", geometry)
    other = [_candidate(f"{pair}_c", pair, geometry) for pair in ("02_03", "05_07", "08_10")]
    updated, event = revalidate_geometry_edit(
        current, geometry, support_validator=lambda _g: {"B9": 1.0},
        all_candidates=[[current], *[[feature] for feature in other]], all_pairs_available=True)
    assert updated["properties"]["pairwise_leakage_audit"] == "CONFLICTS_REPORTED"
    assert updated["properties"]["full_pairwise_leakage_recomputed"] is True
    assert updated["properties"]["pairwise_leakage_audit_scope"] == "ALL_FOUR_PAIRS"
    assert len(event["provisional_pairwise_audit"]["overlaps"]) == 6


def test_candidate_artifact_round_trip_is_exact_and_never_overwrites(tmp_path):
    feature = _candidate("c1", "00_01", box(1050, 1050, 2000, 2000))
    path = tmp_path / "00_01.geojson"
    write_candidate_artifacts(path, [feature])
    loaded = load_candidate_collection(path, "00_01")
    assert loaded["features"][0]["properties"]["review_status"] == "UNVERIFIED"
    assert shape(loaded["features"][0]["geometry"]).equals(box(1050, 1050, 2000, 2000))
    with pytest.raises(FileExistsError):
        write_candidate_artifacts(path, [feature])
