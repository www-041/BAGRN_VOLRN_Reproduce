"""Formal TEST must exclude unreviewed and invalid candidates explicitly."""

from pathlib import Path
import sys

from affine import Affine
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_phase4_freeze import FrozenTrial, load_frozen_trial
from benchmark_phase4_review import ImageGeometry, ReviewError, propose_review
from benchmark_phase4_export import ensure_new_export_dir, validate_test_candidates


def test_real_frozen_trial_without_reviews_or_attestation_has_no_truth():
    snapshot = load_frozen_trial(ROOT)
    truth, excluded = validate_test_candidates(snapshot, {}, None, {})
    assert truth == []
    assert len(excluded) == 84
    assert all("MISSING_SECOND_REVIEW" in row["reasons"] for row in excluded)
    assert all("MISSING_NONUSE_ATTESTATION" in row["reasons"] for row in excluded)


def test_synthetic_explicit_second_approval_and_nonuse_are_both_required():
    key = ("00_01", "00_01_C1")
    row = {"pair_id": key[0], "candidate_id": key[1], "split": "TEST", "grid_cell_id": "g65_328",
           "scene_i": "L", "scene_j": "R", "source_review_status": "APPROVED",
           "effective_x_i": "650501", "effective_y_i": "3280501",
           "effective_x_j": "650502", "effective_y_j": "3280502"}
    source = {"pair_id": key[0], "candidate_id": key[1], "primary_reviewer": "first", "reviewer": "first",
              "human_review_status": "APPROVED", "human_adjusted": "False", "confidence": "HIGH",
              "row_i": "500.5", "col_i": "500.5", "row_j": "501.5", "col_j": "501.5",
              "scene_i": "L", "scene_j": "R"}
    snapshot = FrozenTrial(Path("."), {key: row}, {key: source}, {}, {"00_01": "sourcehash"}, "phase3hash")
    transform = Affine.translation(650000, 3280000)
    geometry = {scene: ImageGeometry(transform, 2000, 2000, lambda r, c: True) for scene in ("L", "R")}
    approved = propose_review(snapshot, key, "second", "APPROVED", (500.5, 500.5), (501.5, 501.5),
                              {"left_original_resolution": True, "right_original_resolution": True},
                              "Human estimate 0.7 px", "", "NO", "", geometry, [])
    attestation = {"dataset_version": "phase3hash", "responsible_person": "lead",
                   "signature": "lead 2026-10-11", "run_ids": ["synthetic-run"],
                   "excluded_test_ids": [key[1]], "declared_no_test_use": True}
    truth, excluded = validate_test_candidates(snapshot, {key[0]: [approved]}, attestation, geometry)
    assert len(truth) == 1 and excluded == []
    assert truth[0]["point_id"] == key[1]
    assert truth[0]["row_i"] == 500.5
    assert truth[0]["second_reviewer"] == "second"
    truth, excluded = validate_test_candidates(snapshot, {key[0]: [approved]}, None, geometry)
    assert not truth and "MISSING_NONUSE_ATTESTATION" in excluded[0]["reasons"]
    forged = {**approved, "after": {**approved["after"], "left": {**approved["after"]["left"], "row": 700.0}}}
    with pytest.raises(ReviewError, match="projected coordinate"):
        validate_test_candidates(snapshot, {key[0]: [forged]}, attestation, geometry)
    concealed = {**approved, "after": {**approved["after"], "left": {"row": 499.5, "col": 500.5,
                  "x": 650501.0, "y": 3280500.0}}, "adjusted": True,
                  "spatial_status": "VALID", "spatial_reasons": [], "adjustment_reason": "manual"}
    with pytest.raises(ReviewError, match="spatial status"):
        validate_test_candidates(snapshot, {key[0]: [concealed]}, attestation, geometry)


def test_synthetic_human_approval_does_not_override_spatial_disqualification():
    key = ("00_01", "00_01_C1")
    row = {"pair_id": key[0], "candidate_id": key[1], "split": "TEST", "grid_cell_id": "g65_328",
           "scene_i": "L", "scene_j": "R", "source_review_status": "APPROVED",
           "effective_x_i": "650501", "effective_y_i": "3280501",
           "effective_x_j": "650502", "effective_y_j": "3280502"}
    source = {"pair_id": key[0], "candidate_id": key[1], "primary_reviewer": "first", "reviewer": "first",
              "human_review_status": "APPROVED", "human_adjusted": "False", "confidence": "HIGH",
              "row_i": "500.5", "col_i": "500.5", "row_j": "501.5", "col_j": "501.5",
              "scene_i": "L", "scene_j": "R"}
    snapshot = FrozenTrial(Path("."), {key: row}, {key: source}, {}, {"00_01": "sourcehash"}, "phase3hash")
    geometry = {scene: ImageGeometry(Affine.translation(650000, 3280000), 2000, 2000,
                                     lambda r, c: True) for scene in ("L", "R")}
    invalid = propose_review(snapshot, key, "second", "APPROVED", (499.5, 500.5), (501.5, 501.5),
                             {"left_original_resolution": True, "right_original_resolution": True},
                             "Human estimate 0.7 px", "adjusted landmark", "NO", "", geometry, [])
    attestation = {"dataset_version": "phase3hash", "responsible_person": "lead",
                   "signature": "lead 2026-10-11", "run_ids": ["synthetic-run"],
                   "excluded_test_ids": [key[1]], "declared_no_test_use": True}
    truth, excluded = validate_test_candidates(snapshot, {key[0]: [invalid]}, attestation, geometry)
    assert truth == []
    assert "NOT_ELIGIBLE_AFTER_COORDINATE_ADJUSTMENT" in excluded[0]["reasons"]


def test_formal_export_refuses_directory_with_stale_truth(tmp_path):
    output = tmp_path / "formal"
    output.mkdir()
    (output / "formal_test_truth.csv").write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="empty"):
        ensure_new_export_dir(output)


def test_formal_export_cannot_write_into_frozen_phase3(tmp_path):
    destination = tmp_path / "docs/benchmark/v1/phase3/new_export"
    with pytest.raises(ValueError, match="Phase4"):
        ensure_new_export_dir(destination, root=tmp_path)
    assert not destination.exists()
