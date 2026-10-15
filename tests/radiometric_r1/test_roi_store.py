from __future__ import annotations

import json

import pytest

from src.benchmark_v1_radiometric_r1.roi_store import ReviewStore, is_formal_test_eligible, validate_review_record


def base_record():
    return {
        "roi_id": "R001", "pair_id": "00_01", "scene_i": "i", "scene_j": "j",
        "geometry": "POLYGON ((0 0, 1 0, 1 1, 0 0))", "geometry_crs": "EPSG:32650",
        "label": "UNKNOWN", "review_status": "UNVERIFIED", "reviewer": "", "review_time_utc": "",
        "evidence_notes": "", "role": "TEST", "purpose": "EVALUATION",
        "second_reviewer": "", "second_review_status": "UNVERIFIED", "second_review_time_utc": "",
        "second_review_label": "UNKNOWN",
    }


def test_unknown_is_default_and_never_test_eligible():
    record = base_record()
    normalized = validate_review_record(record)
    assert normalized["label"] == "UNKNOWN"
    assert normalized["review_status"] == "UNVERIFIED"
    assert normalized["reviewer"] == ""
    assert not is_formal_test_eligible(normalized)


def test_test_approval_requires_distinct_human_second_review():
    record = {**base_record(), "label": "STABLE", "review_status": "APPROVED", "reviewer": "alice", "review_time_utc": "2026-10-11T00:00:00Z"}
    with pytest.raises(ValueError, match="different"):
        validate_review_record({**record, "second_reviewer": "alice", "second_review_status": "APPROVED", "second_review_time_utc": "2026-10-12T00:00:00Z", "second_review_label": "STABLE"})
    with pytest.raises(ValueError, match="agree"):
        validate_review_record({**record, "spatial_status": "SAFE_SINGLE_CELL", "spatial_overlap_audit": "PASS", "purpose": "EVALUATION", "second_reviewer": "bob", "second_review_status": "APPROVED", "second_review_time_utc": "2026-10-12T00:00:00Z", "second_review_label": "REAL_CHANGE"})
    eligible = {**record, "spatial_status": "SAFE_SINGLE_CELL", "spatial_overlap_audit": "PASS", "purpose": "EVALUATION", "second_reviewer": "bob", "second_review_status": "APPROVED", "second_review_time_utc": "2026-10-12T00:00:00Z", "second_review_label": "STABLE"}
    assert validate_review_record(eligible)["review_status"] == "APPROVED"
    assert is_formal_test_eligible(eligible)


def test_store_round_trip_preserves_snapshot_and_append_only_history(tmp_path):
    store = ReviewStore(tmp_path / "roi.csv", tmp_path / "events.jsonl")
    record = base_record()
    store.upsert(record, actor="", event_type="CREATE")
    store.save()
    reloaded = ReviewStore(store.csv_path, store.events_path)
    assert reloaded.records["R001"]["label"] == "UNKNOWN"
    assert reloaded.records["R001"]["review_status"] == "UNVERIFIED"
    assert len(store.events_path.read_text(encoding="utf-8").splitlines()) == 1
    reloaded.upsert({**record, "label": "CLOUD"}, actor="reviewer", event_type="EDIT")
    reloaded.save()
    events = [json.loads(line) for line in store.events_path.read_text(encoding="utf-8").splitlines()]
    assert [e["event_type"] for e in events] == ["CREATE", "EDIT"]
    assert len({e["event_id"] for e in events}) == len(events)
    assert events[-1]["before"]["label"] == "UNKNOWN"

