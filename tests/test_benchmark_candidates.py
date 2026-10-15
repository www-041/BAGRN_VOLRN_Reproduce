"""Synthetic contracts for candidate-only matching and human review persistence."""

import csv
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
from affine import Affine


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_candidate_core import (CandidateReviewStore, deduplicate_candidates,
                                      generate_candidates_arrays, match_bidirectional,
                                      select_spatially_balanced)
from benchmark_phase2_core import load_formal_test_points


def textured_pair(shift_x=3, shift_y=-2):
    rng = np.random.default_rng(42)
    left = (rng.normal(100, 25, (160, 160))).astype(np.float32)
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    right = cv2.warpAffine(left, matrix, (160, 160), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    right = right * 1.3 + 17  # cross-date-style gain and offset
    return left, right


def test_bidirectional_ncc_recovers_shift_without_assuming_same_indices():
    left, right = textured_pair()
    mask = np.ones((160, 160), dtype=bool)
    result = match_bidirectional(left, right, mask, mask, (80, 80), search_radius=7)
    assert result is not None
    assert result["row_j"] == pytest.approx(78, abs=.5)
    assert result["col_j"] == pytest.approx(83, abs=.5)
    assert result["forward_backward_error"] <= 1.0


def test_generator_uses_shared_mask_and_keeps_suggestions_unapproved():
    left, right = textured_pair()
    mask_i = np.zeros((160, 160), dtype=bool)
    mask_j = np.zeros((160, 160), dtype=bool)
    mask_i[25:135, 25:135] = True
    mask_j[25:135, 25:135] = True
    rows = generate_candidates_arrays(left, right, mask_i, mask_j,
                                      Affine(14, 0, 0, 0, -14, 2240),
                                      Affine(14, 0, 0, 0, -14, 2240),
                                      pair_id="00_01", scene_i="A", scene_j="B",
                                      max_candidates=12, grid_side=4, search_radius=7)
    assert rows
    assert len(rows) <= 12
    assert len({r["grid_cell_id"] for r in rows}) >= 3
    for row in rows:
        assert row["recommendation_status"] == "SUGGESTED"
        assert row["human_review_status"] == "UNVERIFIED"
        assert row["reviewer"] == ""
        assert mask_i[round(float(row["row_i"])), round(float(row["col_i"]))]
        assert mask_j[round(float(row["row_j"])), round(float(row["col_j"]))]
        assert abs(float(row["col_j"]) - float(row["col_i"]) - 3) < 1


def test_low_texture_and_mask_boundary_can_return_no_candidates():
    blank = np.ones((96, 96), dtype=np.float32)
    mask = np.ones((96, 96), dtype=bool)
    kwargs = dict(pair_id="00_01", scene_i="A", scene_j="B", max_candidates=40, grid_side=4)
    assert generate_candidates_arrays(blank, blank, mask, mask, Affine.identity(), Affine.identity(), **kwargs) == []
    left, right = textured_pair()
    edge = np.zeros((160, 160), dtype=bool)
    edge[:10, :10] = True
    assert generate_candidates_arrays(left, right, edge, edge, Affine.identity(), Affine.identity(), **kwargs) == []


def test_deduplication_preserves_highest_score_and_spatial_spread():
    rows = [dict(row_i=10., col_i=10., matching_score=.9, grid_cell_id="0_0"),
            dict(row_i=12., col_i=11., matching_score=.8, grid_cell_id="0_0"),
            dict(row_i=100., col_i=100., matching_score=.7, grid_cell_id="1_1")]
    result = deduplicate_candidates(rows, min_distance_px=10)
    assert len(result) == 2
    assert result[0]["matching_score"] == .9


def test_spatial_selection_does_not_take_only_first_rows():
    cells = []
    for gy in range(8):
        for gx in range(8):
            cells.append([dict(row_i=float(gy*100), col_i=float(gx*100),
                               matching_score=.9, grid_cell_id=f"r{gy:02d}_c{gx:02d}")])
    chosen = select_spatially_balanced(cells, max_candidates=16, min_distance_px=20)
    quadrants = {(int(r["row_i"] >= 400), int(r["col_i"] >= 400)) for r in chosen}
    assert len(chosen) == 16
    assert quadrants == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_review_adjustment_persists_original_and_manual_file_is_untouched(tmp_path):
    manual = tmp_path / "annotations" / "00_01.csv"
    manual.parent.mkdir()
    manual.write_text("point_id\n00_01_P0001\n", encoding="utf-8")
    original_bytes = manual.read_bytes()
    candidate_path = tmp_path / "auto_candidates" / "00_01.csv"
    store = CandidateReviewStore(candidate_path, "00_01", "A", "B")
    store.replace_suggestions([{"candidate_id": "00_01_C0001", "pair_id": "00_01",
                                "scene_i": "A", "scene_j": "B", "row_i": 20., "col_i": 21.,
                                "row_j": 22., "col_j": 23., "x_i": 301., "y_i": 302.,
                                "x_j": 303., "y_j": 304., "recommendation_method": "ShiTomasi_gradNCC",
                                "matching_score": .91, "forward_backward_error": .2,
                                "grid_cell_id": "2_3", "recommendation_status": "SUGGESTED",
                                "human_review_status": "UNVERIFIED", "reviewer": "", "review_notes": ""}])
    store.adjust("00_01_C0001", (20.25, 21.5), (22.75, 23.125),
                 Affine.identity(), Affine.identity(), reviewer="alice")
    store.save()
    restored = CandidateReviewStore(candidate_path, "00_01", "A", "B")
    row = restored.rows[0]
    assert float(row["row_i"]) == 20
    assert float(row["adjusted_row_i"]) == 20.25
    assert float(row["adjusted_col_j"]) == 23.125
    assert row["human_adjusted"] == "True"
    assert row["human_review_status"] == "UNVERIFIED"
    assert manual.read_bytes() == original_bytes


def test_only_explicitly_approved_second_reviewed_test_enters_truth(tmp_path):
    path = tmp_path / "00_01.csv"
    store = CandidateReviewStore(path, "00_01", "A", "B")
    store.replace_suggestions([{"candidate_id": "00_01_C0001", "pair_id": "00_01",
                                "scene_i": "A", "scene_j": "B", "row_i": 10., "col_i": 11.,
                                "row_j": 12., "col_j": 13., "x_i": 11.5, "y_i": 10.5,
                                "x_j": 13.5, "y_j": 12.5, "recommendation_method": "ShiTomasi_gradNCC",
                                "matching_score": .95, "forward_backward_error": .1,
                                "grid_cell_id": "1_1", "recommendation_status": "SUGGESTED",
                                "human_review_status": "UNVERIFIED", "reviewer": "", "review_notes": ""}])
    assert store.approved_test_truth() == []
    store.review("00_01_C0001", "APPROVED", "alice", "TEST", "HIGH",
                 "±0.5 px visual localization", second_reviewer="")
    assert store.approved_test_truth() == []
    store.review("00_01_C0001", "APPROVED", "bob", "TEST", "HIGH",
                 "±0.5 px visual localization", second_reviewer="bob")
    truth = store.approved_test_truth()
    assert len(truth) == 1
    assert truth[0]["point_source"] == "AUTO_SUGGESTED"
    assert truth[0]["origin_candidate_id"] == "00_01_C0001"
    assert truth[0]["review_status"] == "APPROVED"


def test_formal_loader_rejects_candidate_file_even_if_passed_as_truth(tmp_path):
    candidate = tmp_path / "candidates.csv"
    store = CandidateReviewStore(candidate, "00_01", "A", "B")
    store.replace_suggestions([{"candidate_id": "00_01_C0001", "pair_id": "00_01",
                                "scene_i": "A", "scene_j": "B", "row_i": 10, "col_i": 10,
                                "row_j": 10, "col_j": 10, "x_i": 10.5, "y_i": 10.5,
                                "x_j": 10.5, "y_j": 10.5, "recommendation_method": "test",
                                "matching_score": 1, "forward_backward_error": 0,
                                "grid_cell_id": "0_0", "recommendation_status": "SUGGESTED",
                                "human_review_status": "UNVERIFIED", "reviewer": "", "review_notes": ""}])
    store.save()
    with pytest.raises(ValueError, match="candidate CSV is not formal truth"):
        load_formal_test_points(None, candidate, "00_01")
    auto = tmp_path / "auto_truth.csv"
    store.save_approved_truth(auto)
    assert load_formal_test_points(None, auto, "00_01") == []


def test_reviewed_candidate_export_survives_reload_and_replace_is_refused(tmp_path):
    path = tmp_path / "00_01.csv"
    store = CandidateReviewStore(path, "00_01", "A", "B")
    store.replace_suggestions([{"candidate_id": "00_01_C0001", "pair_id": "00_01",
                                "scene_i": "A", "scene_j": "B", "row_i": 10, "col_i": 11,
                                "row_j": 12, "col_j": 13, "x_i": 11.5, "y_i": 10.5,
                                "x_j": 13.5, "y_j": 12.5, "recommendation_method": "test",
                                "matching_score": .9, "forward_backward_error": 0,
                                "grid_cell_id": "0_0", "recommendation_status": "SUGGESTED",
                                "human_review_status": "UNVERIFIED", "reviewer": "", "review_notes": ""}])
    store.review("00_01_C0001", "APPROVED", "alice", "TEST", "HIGH", "±0.5px")
    store.save()
    restored = CandidateReviewStore(path, "00_01", "A", "B")
    assert restored.rows[0]["human_review_status"] == "PENDING_SECOND_REVIEW"
    restored.review("00_01_C0001", "APPROVED", "bob", "TEST", "HIGH", "±0.5px",
                    second_reviewer="bob")
    restored.save()
    truth_path = tmp_path / "auto_truth.csv"
    restored.save_approved_truth(truth_path)
    points = load_formal_test_points(None, truth_path, "00_01")
    assert len(points) == 1
    assert points[0]["point_source"] == "AUTO_SUGGESTED"
    with pytest.raises(FileExistsError):
        CandidateReviewStore(path, "00_01", "A", "B").replace_suggestions([])


def test_skipped_candidate_is_saved_but_never_exported_as_truth(tmp_path):
    path = tmp_path / "00_01.csv"
    store = CandidateReviewStore(path, "00_01", "A", "B")
    store.replace_suggestions([{"candidate_id": "00_01_C0001", "pair_id": "00_01",
                                "scene_i": "A", "scene_j": "B", "row_i": 1, "col_i": 1,
                                "row_j": 2, "col_j": 2, "x_i": 1.5, "y_i": 1.5,
                                "x_j": 2.5, "y_j": 2.5, "recommendation_method": "test",
                                "matching_score": .9, "forward_backward_error": 0,
                                "grid_cell_id": "0_0", "recommendation_status": "SUGGESTED",
                                "human_review_status": "UNVERIFIED", "reviewer": "", "review_notes": ""}])
    store.review("00_01_C0001", "SKIPPED", "alice", "UNASSIGNED", "UNKNOWN", "")
    store.save()
    restored = CandidateReviewStore(path, "00_01", "A", "B")
    assert restored.rows[0]["human_review_status"] == "SKIPPED"
    assert restored.approved_truth() == []
