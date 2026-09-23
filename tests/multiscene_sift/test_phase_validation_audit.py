import csv
import json

import numpy as np
import pytest

from src.multiscene_sift.phase_validation_audit import (
    AUDIT_EDGES,
    load_phase_audit_baseline,
)


def _write_minimal_baseline(root):
    edge_dir = root / "edge"
    local_dir = root / "local"
    selected_dir = root / "nine_scene_selected_pair_consistency"
    edge_dir.mkdir()
    local_dir.mkdir()
    selected_dir.mkdir()
    (edge_dir / "00_edge_reliability_baseline.json").write_text(json.dumps({
        "edges": [
            {"edge": [0, 6], "full_overlap_phase_residual_px": 0.58},
            {"edge": [2, 5], "full_overlap_phase_residual_px": 6.92},
        ]
    }), encoding="utf-8")
    (edge_dir / "07_direct_residual_summary.json").write_text(json.dumps({
        "0-6": {"median_px": 0.6324555320336759},
        "2-5": {"median_px": 6.715078575495168},
    }), encoding="utf-8")
    with (edge_dir / "07_direct_residual_tiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "edge", "grid_n", "tile_row", "tile_col", "phase_mag_px", "accepted"
        ])
        writer.writeheader()
        writer.writerow({"edge": "0-6", "grid_n": 4, "tile_row": 1, "tile_col": 1,
                         "phase_mag_px": 0.6324555320336759, "accepted": "True"})
        writer.writerow({"edge": "2-5", "grid_n": 4, "tile_row": 1, "tile_col": 1,
                         "phase_mag_px": 6.715078575495168, "accepted": "True"})
    (local_dir / "00_local_affine_baseline.json").write_text(json.dumps({"edges": {}}), encoding="utf-8")
    (local_dir / "06_global_vs_local_phase_summary.json").write_text(json.dumps({"0-6": {}, "2-5": {}}), encoding="utf-8")
    (selected_dir / "06_canonical_world_edge_transforms.json").write_text(json.dumps({
        "edges": {"0-6": {"matrix": np.eye(3).tolist()}, "2-5": {"matrix": np.eye(3).tolist()}}
    }), encoding="utf-8")
    (selected_dir / "02_new_pairwise_registration_results.json").write_text(json.dumps({
        "results": [
            {"idx_i": 0, "idx_j": 6, "pixel_matrix": np.eye(3).tolist()},
            {"idx_i": 2, "idx_j": 5, "pixel_matrix": np.eye(3).tolist()},
        ]
    }), encoding="utf-8")
    return edge_dir, local_dir, selected_dir


def test_audit_edges_are_exactly_working_control_and_suspect():
    assert AUDIT_EDGES == {
        (0, 6): "WORKING_CONTROL",
        (2, 5): "SUSPECT_PHASE_FALSE_ALARM",
    }


def test_baseline_loader_reads_old_phase_values_without_hardcoding(tmp_path):
    edge_dir, local_dir, selected_dir = _write_minimal_baseline(tmp_path)
    result = load_phase_audit_baseline(edge_dir, local_dir, selected_pair_dir=selected_dir)
    assert result["edge_keys"] == ["0-6", "2-5"]
    assert result["edges"]["0-6"]["old_phase_median_px"] == pytest.approx(0.6324555320336759)
    assert result["edges"]["2-5"]["old_phase_median_px"] == pytest.approx(6.715078575495168)
    assert result["edges"]["2-5"]["pixel_matrix"] == np.eye(3).tolist()


def test_baseline_loader_fails_when_required_artifact_is_missing(tmp_path):
    edge_dir, local_dir, selected_dir = _write_minimal_baseline(tmp_path)
    (edge_dir / "07_direct_residual_summary.json").unlink()
    with pytest.raises(FileNotFoundError, match="07_direct_residual_summary"):
        load_phase_audit_baseline(edge_dir, local_dir, selected_pair_dir=selected_dir)
