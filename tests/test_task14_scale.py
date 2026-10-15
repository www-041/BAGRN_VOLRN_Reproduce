import json
from pathlib import Path

import pytest

from scripts.run_task14_13scene_scale import (
    HARD_STOP_RESUME_PROVENANCE_MISMATCH,
    STAGES,
    _task14_downstream_stages,
    _task14_window_from_bounds,
    canonical_sha256,
    graph_statistics,
    stage_is_reusable,
    unavailable_value,
)
import scripts.run_task14_13scene_scale as task14


def test_graph_statistics_reports_complete_pair_space_and_topology():
    records = [
        {"scene_id": "a", "left": 0, "bottom": 0, "right": 10, "top": 10},
        {"scene_id": "b", "left": 5, "bottom": 0, "right": 15, "top": 10},
        {"scene_id": "c", "left": 30, "bottom": 0, "right": 40, "top": 10},
    ]
    pairs = [
        {"idx_i": 0, "idx_j": 1, "scene_i": "a", "scene_j": "b", "intersection_area": 50.0, "has_overlap": True},
        {"idx_i": 0, "idx_j": 2, "scene_i": "a", "scene_j": "c", "intersection_area": 0.0, "has_overlap": False},
        {"idx_i": 1, "idx_j": 2, "scene_i": "b", "scene_j": "c", "intersection_area": 0.0, "has_overlap": False},
    ]
    stats = graph_statistics(records, pairs, bbox_candidate_pairs={(0, 1)})
    assert stats["possible_pair_count"] == 3
    assert stats["exact_overlap_edge_count"] == 1
    assert stats["non_overlap_pair_count"] == 2
    assert stats["connected_component_count"] == 2
    assert stats["cycle_rank"] == 0
    assert stats["spatial_index_false_negative_count"] == 0


def test_stage_resume_requires_matching_input_and_output_hashes(tmp_path: Path):
    stage = tmp_path / "stage"
    stage.mkdir()
    inp = tmp_path / "input.txt"
    out = tmp_path / "output.txt"
    inp.write_text("input", encoding="utf-8")
    out.write_text("output", encoding="utf-8")
    payload = {
        "input_hashes": {str(inp): canonical_sha256(inp)},
        "output_hashes": {str(out): canonical_sha256(out)},
        "status": "SUCCESS",
    }
    (stage / "_SUCCESS.json").write_text(json.dumps(payload), encoding="utf-8")
    assert stage_is_reusable(stage, payload["input_hashes"], payload["output_hashes"])
    inp.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match=HARD_STOP_RESUME_PROVENANCE_MISMATCH):
        stage_is_reusable(stage, {str(inp): canonical_sha256(inp)}, payload["output_hashes"])


def test_unavailable_telemetry_is_explicit_and_not_zero():
    assert unavailable_value() == "UNAVAILABLE"
    assert unavailable_value() != 0


def test_streaming_hard_stop_prevents_thirteen_scene_preflight(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(task14, "_streaming_equivalence", lambda root: ("HARD_STOP", "HARD_STOP_STREAMING_REFERENCE_MISSING", [], []))
    def unexpected_discovery(_):
        raise AssertionError("13-scene discovery must not run after Stage -1 hard stop")
    monkeypatch.setattr(task14, "discover_b9_scenes", unexpected_discovery)
    result = task14.run_task14(stage="all", input_root=tmp_path / "missing", output_root=tmp_path / "out")
    assert result["decision"] == "NOT_READY_FOR_LARGE_SCALE"
    assert result["hard_stops"] == ["HARD_STOP_STREAMING_REFERENCE_MISSING"]


def test_registration_metrics_uses_accepted_for_global_field():
    rows = [{"accepted_for_global": True, "pairwise_rmse_px": 1.0, "pairwise_p95_px": 2.0, "coverage": 0.5}]
    assert task14._registration_metrics(rows)["accepted_edges"] == 1


def test_task14_window_from_bounds_clips_to_canonical_grid():
    grid = {
        "transform": [14.0, 0.0, 100.0, 0.0, -14.0, 200.0],
        "width": 10,
        "height": 8,
    }
    assert _task14_window_from_bounds((72.0, 100.0, 156.0, 212.0), grid) == (0, 8, 0, 4)


def test_task14_downstream_stages_start_after_bagrn():
    assert _task14_downstream_stages() == STAGES[8:]
