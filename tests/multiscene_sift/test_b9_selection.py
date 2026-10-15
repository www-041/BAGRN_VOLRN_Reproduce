"""Synthetic tests for B9 five-scene topology recommendation."""

from __future__ import annotations

import json

from src.multiscene_sift.b9_selection import (
    rank_five_scene_candidates,
    write_candidate_outputs,
)


def _records(n=6):
    return [{"scene_id": f"scene_{i}"} for i in range(n)]


def _pairs(n=6):
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "idx_i": i,
                "idx_j": j,
                "scene_i": f"scene_{i}",
                "scene_j": f"scene_{j}",
                "intersection_area": 100.0,
                "overlap_area_i_ratio": 0.5,
                "overlap_area_j_ratio": 0.5,
                "symmetric_overlap_ratio": 0.5,
                "has_overlap": True,
            })
    return pairs


def test_ranker_returns_top_five_only_for_hard_valid_combinations():
    candidates = rank_five_scene_candidates(_records(), _pairs())

    assert len(candidates) == 5
    assert all(candidate["edge_count"] >= 6 for candidate in candidates)
    assert all(candidate["cycle_rank"] >= 1 for candidate in candidates)
    assert all(candidate["min_degree"] >= 1 for candidate in candidates)
    assert candidates[0]["score"] >= candidates[-1]["score"]


def test_candidate_outputs_record_formula_and_recommendation(tmp_path):
    records = _records()
    candidates = rank_five_scene_candidates(records, _pairs())

    payload = write_candidate_outputs(records, candidates, tmp_path)

    assert payload["recommended"]["scene_ids"] == candidates[0]["scene_ids"]
    assert "3 * cycle_rank" in payload["scoring_formula"]
    assert (tmp_path / "03_five_scene_candidates.csv").exists()
    assert (tmp_path / "03_recommended_five_scene.json").exists()
    assert (tmp_path / "03_recommended_subgraph.png").exists()
    saved = json.loads(
        (tmp_path / "03_recommended_five_scene.json").read_text(encoding="utf-8")
    )
    assert len(saved["top_candidates"]) == 5
