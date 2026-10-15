"""Synthetic second-review policy tests; never create real review events."""

from pathlib import Path
import sys

from affine import Affine
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_phase4_freeze import FrozenTrial
from benchmark_phase4_review import (ImageGeometry, ReviewError, append_review_event,
                                     load_review_events, propose_review, review_state,
                                     validate_event_destination, validate_review_geometry_history)


KEY = ("00_01", "00_01_C0001")
TRANSFORM = Affine.translation(650000, 3280000) * Affine.scale(1, 1)


def _snapshot():
    row = {"pair_id": KEY[0], "candidate_id": KEY[1], "scene_i": "L", "scene_j": "R",
           "split": "TEST", "grid_cell_id": "g65_328", "source_review_status": "APPROVED",
           "effective_x_i": "650501", "effective_y_i": "3280501",
           "effective_x_j": "650502", "effective_y_j": "3280502"}
    source = {"pair_id": KEY[0], "candidate_id": KEY[1], "scene_i": "L", "scene_j": "R",
              "primary_reviewer": "first", "reviewer": "first", "human_review_status": "APPROVED",
              "confidence": "HIGH", "row_i": "500.5", "col_i": "500.5",
              "row_j": "501.5", "col_j": "501.5", "human_adjusted": "False"}
    return FrozenTrial(Path("."), {KEY: row}, {KEY: source}, {}, {KEY[0]: "sourcehash"}, "phase3hash")


def _geometry(valid=lambda row, col: True):
    return {"L": ImageGeometry(TRANSFORM, 2000, 2000, valid),
            "R": ImageGeometry(TRANSFORM, 2000, 2000, valid)}


def _propose(*, left=(500.5, 500.5), right=(501.5, 501.5), reviewer="second",
             outcome="APPROVED", previous=(), disagreement="NO", reason=""):
    return propose_review(_snapshot(), KEY, reviewer, outcome, left, right,
                          {"left_original_resolution": True, "right_original_resolution": True},
                          "Human estimate: 0.7 px", reason, disagreement, "", _geometry(), list(previous))


def test_distinct_reviewer_and_explicit_both_image_inspection():
    with pytest.raises(ReviewError, match="distinct"):
        _propose(reviewer="first")
    with pytest.raises(ReviewError, match="right"):
        propose_review(_snapshot(), KEY, "second", "APPROVED", (500.5, 500.5), (501.5, 501.5),
                       {"left_original_resolution": True, "right_original_resolution": False},
                       "Human estimate: 0.7 px", "", "NO", "", _geometry(), [])
    event = _propose()
    assert event["human_outcome"] == "APPROVED"
    assert event["spatial_status"] == "VALID"
    assert event["reviewer"] == "second"
    assert event["before"]["left"]["x"] == pytest.approx(650501)


def test_adjustment_outside_original_safe_core_stays_ineligible_after_reversal():
    bad = _propose(left=(499.5, 500.5), reason="manual correction")
    assert bad["spatial_status"] == "NOT_ELIGIBLE_AFTER_COORDINATE_ADJUSTMENT"
    assert "BUFFER_ZONE" in bad["spatial_reasons"]
    later = _propose(previous=[bad])
    state = review_state(_snapshot(), [bad, later])[KEY]
    assert state["frozen_role"] == "TEST"
    assert state["eligible"] is False
    assert state["spatial_status"] == "NOT_ELIGIBLE_AFTER_COORDINATE_ADJUSTMENT"
    validate_review_geometry_history(_snapshot(), [bad, later], _geometry())
    concealed = {**later, "spatial_status": "VALID"}
    with pytest.raises(ReviewError, match="spatial status"):
        validate_review_geometry_history(_snapshot(), [bad, concealed], _geometry())


def test_disagreement_and_rejection_do_not_become_truth():
    doubtful = _propose(outcome="DOUBTFUL")
    rejected = _propose(outcome="REJECTED", previous=[doubtful])
    assert review_state(_snapshot(), [doubtful, rejected])[KEY]["eligible"] is False
    with pytest.raises(ReviewError, match="explanation"):
        _propose(disagreement="YES")


def test_append_replay_preserves_history_and_rejects_tampering(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    first = _propose(outcome="DOUBTFUL")
    append_review_event(path, first)
    second = _propose(previous=[first], outcome="APPROVED")
    append_review_event(path, second)
    loaded = load_review_events(path, _snapshot())
    assert [item["human_outcome"] for item in loaded] == ["DOUBTFUL", "APPROVED"]
    assert review_state(_snapshot(), loaded)[KEY]["eligible"] is True
    path.write_bytes(path.read_bytes() + b"{broken\n")
    with pytest.raises(ReviewError, match="JSONL"):
        load_review_events(path, _snapshot())


def test_event_destination_cannot_be_an_original_candidate_or_phase3_file(tmp_path):
    source = ROOT / "docs/benchmark/v1/phase2/auto_candidates/00_01.csv"
    phase3 = ROOT / "docs/benchmark/v1/phase3/checkpoint_split_assignments.csv"
    for path in (source, phase3):
        with pytest.raises(ReviewError, match="read-only"):
            validate_event_destination(path, _snapshot())
    assert validate_event_destination(tmp_path / "events.jsonl", _snapshot()) == tmp_path / "events.jsonl"


@pytest.mark.parametrize("change,expected", [
    ({"schema_version": "unknown_schema"}, "schema"),
    ({"frozen_grid_cell_id": "g99_99"}, "cell"),
    ({"reviewer": "first"}, "distinct"),
])
def test_replay_rejects_invalid_event_identity_even_with_valid_hash_chain(tmp_path, change, expected):
    event = _propose()
    event.update(change)
    path = tmp_path / "synthetic.jsonl"
    append_review_event(path, event)
    with pytest.raises(ReviewError, match=expected):
        load_review_events(path, _snapshot())
