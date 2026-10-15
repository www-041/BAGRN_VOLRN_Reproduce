"""Candidate metadata must remain separate from human truth and review."""

import csv
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_candidate_core import CandidateReviewStore
from benchmark_candidate_metadata import (
    apply_batch_labels, build_candidate_context, load_frozen_split,
    load_scene_metadata, save_candidate_session,
)
from benchmark_phase2_core import load_formal_test_points


def _csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidate(number):
    return {"candidate_id": f"00_01_C{number:04d}", "pair_id": "00_01", "scene_i": "A",
            "scene_j": "B", "row_i": "10.0", "col_i": "20.0", "x_i": "100.0", "y_i": "200.0",
            "row_j": "11.0", "col_j": "21.0", "x_j": "101.0", "y_j": "201.0",
            "recommendation_method": "ShiTomasi_gradientNCC_bidirectional", "matching_score": "0.99",
            "forward_backward_error": "0.0", "grid_cell_id": "r03_c02"}


def _store(path):
    store = CandidateReviewStore(path, "00_01", "A", "B")
    store.replace_suggestions([_candidate(1), _candidate(2)])
    store.save()
    return store


def test_objective_context_uses_manifest_dates_quality_and_keeps_unknown_labels(tmp_path):
    manifest = tmp_path / "manifest.csv"
    inventory = tmp_path / "inventory.csv"
    _csv(manifest, ["scene_id", "acquisition_time"], [
        {"scene_id": "A", "acquisition_time": "2025-11-14T02:38:47"},
        {"scene_id": "B", "acquisition_time": "2025-12-14T02:38:47"},
    ])
    _csv(inventory, ["product_id", "b9_quality_status_tag"], [
        {"product_id": "A", "b9_quality_status_tag": "PASS"},
        {"product_id": "B", "b9_quality_status_tag": "RISK"},
    ])
    scene_meta = load_scene_metadata(manifest, inventory)
    context = build_candidate_context(_candidate(1), scene_meta)
    assert context["pair_id"] == "00_01"
    assert context["scene_i"] == "A" and context["scene_j"] == "B"
    assert context["candidate_source"] == "AUTO_SUGGESTED"
    assert context["recommendation_method"] == "ShiTomasi_gradientNCC_bidirectional"
    assert context["original_row_i"] == "10.0" and context["original_x_j"] == "101.0"
    assert context["acquisition_time_i"] == "2025-11-14T02:38:47"
    assert context["elapsed_days"] == 30.0
    assert (context["quality_i"], context["quality_j"]) == ("PASS", "RISK")
    assert context["geographic_zone"] == "PAIR_GRID:00_01:r03_c02"
    assert context["terrain_suggestion"] == "TERRAIN_UNKNOWN"
    assert context["landcover_suggestion"] == "UNKNOWN"
    assert context["split_suggestion"] == "UNASSIGNED"
    assert "review_status" not in context and "confidence" not in context


def test_split_requires_explicit_frozen_nonambiguous_polygon(tmp_path):
    path = tmp_path / "split.geojson"
    split = {"type": "FeatureCollection", "metadata": {"status": "DRAFT", "version": "v1",
             "crs": "EPSG:32650"}, "features": [{"type": "Feature",
             "geometry": {"type": "Polygon", "coordinates": [[[90, 190], [110, 190],
             [110, 210], [90, 210], [90, 190]]]},
             "properties": {"split_id": "cell-1", "role": "TEST"}}]}
    path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match="FROZEN"):
        load_frozen_split(path)
    split["metadata"]["status"] = "FROZEN"
    path.write_text(json.dumps(split), encoding="utf-8")
    frozen = load_frozen_split(path)
    context = build_candidate_context(_candidate(1), {"A": {"acquisition_time": "2025-11-14T00:00:00",
                  "quality_status": "PASS"}, "B": {"acquisition_time": "2025-11-15T00:00:00",
                  "quality_status": "PASS"}}, frozen)
    assert context["split_suggestion"] == "TEST"
    assert context["split_source"].startswith("FROZEN:")
    assert context["split_id"] == "cell-1"
    shifted = _candidate(1) | {"human_adjusted": "True", "adjusted_x_i": "130.0",
                               "adjusted_y_i": "200.0", "adjusted_x_j": "131.0", "adjusted_y_j": "201.0"}
    shifted_context = build_candidate_context(shifted, {"A": {"acquisition_time": "2025-11-14T00:00:00",
                    "quality_status": "PASS"}, "B": {"acquisition_time": "2025-11-15T00:00:00",
                    "quality_status": "PASS"}}, frozen)
    assert shifted_context["split_suggestion"] == "UNASSIGNED"
    overlap = dict(split["features"][0])
    overlap["properties"] = {"split_id": "cell-2", "role": "DEVELOPMENT"}
    split["features"].append(overlap)
    path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_frozen_split(path)


def test_batch_labels_are_transactional_and_cannot_touch_reviewed_points(tmp_path):
    store = _store(tmp_path / "candidates.csv")
    store.review("00_01_C0002", "APPROVED", "reviewer-1", "DEVELOPMENT", "MEDIUM", "0.5 px")
    before = [dict(r) for r in store.rows]
    with pytest.raises(ValueError, match="UNVERIFIED"):
        apply_batch_labels(store, ["00_01_C0001", "00_01_C0002"], reviewer="reviewer-2",
                           terrain="MOUNTAIN")
    assert store.rows == before
    changes = apply_batch_labels(store, ["00_01_C0001"], reviewer="reviewer-2",
                                 terrain="MOUNTAIN", landcover="UNKNOWN")
    assert len(changes) == 1
    assert changes[0]["before"]["terrain_class"] == "TERRAIN_UNKNOWN"
    assert changes[0]["after"]["terrain_class"] == "MOUNTAIN"
    assert store.rows[0]["human_review_status"] == "UNVERIFIED"
    assert store.rows[0]["confidence"] == "UNKNOWN"
    assert store.rows[0]["split"] == "UNASSIGNED"
    assert store.rows[0]["row_i"] == "10.0"
    assert store.rows[1] == before[1]


def test_human_approval_without_frozen_split_is_not_formal_truth(tmp_path):
    store = _store(tmp_path / "candidates.csv")
    store.review("00_01_C0001", "APPROVED", "first-human", "UNASSIGNED", "MEDIUM", "0.5 px")
    assert store.rows[0]["human_review_status"] == "APPROVED"
    assert store.rows[0]["split"] == "UNASSIGNED"
    assert store.approved_truth() == []
    with pytest.raises(ValueError, match="distinct second reviewer"):
        store.review("00_01_C0001", "APPROVED", "first-human", "TEST", "MEDIUM", "0.5 px",
                     second_reviewer="first-human")
    store.review("00_01_C0001", "APPROVED", "second-human", "TEST", "MEDIUM", "0.5 px",
                 second_reviewer="second-human")
    assert len(store.approved_test_truth()) == 1


def test_save_reload_keeps_original_labels_suggestions_and_review_gates(tmp_path):
    path = tmp_path / "candidates.csv"
    store = _store(path)
    before = {r["candidate_id"]: dict(r) for r in store.rows}
    apply_batch_labels(store, ["00_01_C0001"], reviewer="human", terrain="MOUNTAIN")
    store.review("00_01_C0001", "APPROVED", "human", "TEST", "MEDIUM", "0.5 px")
    context = {"00_01_C0001": {"terrain_suggestion": "TERRAIN_UNKNOWN",
                              "split_suggestion": "UNASSIGNED", "geographic_zone": "PAIR_GRID:00_01:r03_c02"}}
    truth = tmp_path / "truth.csv"
    save_candidate_session(store, before, context, truth, reviewer="human")
    reloaded = CandidateReviewStore(path, "00_01", "A", "B")
    assert reloaded.rows[0]["terrain_class"] == "MOUNTAIN"
    assert reloaded.rows[0]["human_review_status"] == "PENDING_SECOND_REVIEW"
    assert reloaded.rows[0]["row_i"] == "10.0"
    assert json.loads(path.with_suffix(".context.json").read_text(encoding="utf-8"))["00_01_C0001"]["terrain_suggestion"] == "TERRAIN_UNKNOWN"
    audit = [json.loads(line) for line in path.with_suffix(".audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert audit[0]["before"]["terrain_class"] == "TERRAIN_UNKNOWN"
    assert audit[0]["after"]["terrain_class"] == "MOUNTAIN"
    assert path.with_suffix(".baseline.csv").is_file()
    assert load_formal_test_points(None, truth, pair_id="00_01") == []
    reloaded.review("00_01_C0001", "APPROVED", "second-human", "TEST", "MEDIUM", "0.5 px",
                    second_reviewer="second-human", terrain="MOUNTAIN")
    save_candidate_session(reloaded, {r["candidate_id"]: dict(r) for r in store.rows}, context, truth,
                           reviewer="second-human")
    assert len(load_formal_test_points(None, truth, pair_id="00_01")) == 1
