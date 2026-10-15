"""Focused tests for the frozen global DZ01V checkpoint split."""

import csv
import hashlib
import json
from pathlib import Path
import sys

from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_benchmark_phase3_split import (assign_candidate, audit_assignments,
                                          build_regions, collect_assignments)
from benchmark_candidate_metadata import load_frozen_split


def _candidate(candidate_id, xi, yi, xj=None, yj=None, *, adjusted=False):
    return {
        "candidate_id": candidate_id, "pair_id": "00_01", "scene_i": "left", "scene_j": "right",
        "x_i": str(xi), "y_i": str(yi), "x_j": str(xi if xj is None else xj),
        "y_j": str(yi if yj is None else yj), "human_adjusted": str(adjusted),
        "adjusted_x_i": "670501" if adjusted else "", "adjusted_y_i": "3280501" if adjusted else "",
        "adjusted_x_j": "670502" if adjusted else "", "adjusted_y_j": "3280502" if adjusted else "",
        "human_review_status": "APPROVED", "reviewer": "www", "primary_reviewer": "www",
        "second_reviewer": "", "review_time_utc": "2026-10-10T10:00:00+00:00",
        "coordinate_precision_note": "Manual visual estimate: ±0.5 pixel per image",
    }


def test_global_cells_and_edge_margin_assign_both_scene_coordinates():
    # (ix+2*iy)%3: cell (67,328) is DEV; (65,328) is TEST.
    dev = assign_candidate(_candidate("a", 670501, 3280501, 670502, 3280502))
    assert dev["split"] == "DEVELOPMENT" and dev["grid_cell_id"] == "g67_328"
    assert dev["formal_test_eligible"] is False
    test = assign_candidate(_candidate("b", 650501, 3280501, 650502, 3280502))
    assert test["split"] == "TEST" and test["second_review_status"] == "PENDING_SECOND_REVIEW"
    assert test["formal_test_eligible"] is False
    edge = assign_candidate(_candidate("c", 650500, 3280501))
    assert edge["split"] == "UNASSIGNED" and edge["reason"] == "BUFFER_ZONE"
    across = assign_candidate(_candidate("d", 659000, 3281000, 660001, 3281000))
    assert across["split"] == "UNASSIGNED" and across["reason"] == "DIFFERENT_CELLS"
    adjusted = assign_candidate(_candidate("e", 650001, 3280001, adjusted=True))
    assert adjusted["split"] == "DEVELOPMENT" and adjusted["grid_cell_id"] == "g67_328"


def test_frozen_geojson_is_usable_by_existing_gui_without_opposite_role_contact(tmp_path):
    regions = build_regions((650000, 3280000, 680000, 3290000))
    assert regions["metadata"]["status"] == "FROZEN"
    assert regions["metadata"]["freeze_scope"] == "TRIAL4_ONLY"
    assert regions["metadata"]["origin_m"] == [0, 0]
    assert regions["metadata"]["cell_size_m"] == 10000
    assert regions["metadata"]["edge_margin_m"] == 500
    path = tmp_path / "regions.geojson"
    path.write_text(json.dumps(regions), encoding="utf-8")
    assert load_frozen_split(path) is not None
    shapes = [(shape(f["geometry"]), f["properties"]["role"]) for f in regions["features"]]
    gaps = [a.distance(b) for a, role_a in shapes for b, role_b in shapes if role_a != role_b]
    assert min(gaps) >= 1000


def test_audit_detects_cross_pair_nearby_opposite_roles():
    assignments = [
        {"candidate_id": "a", "pair_id": "00_01", "split": "DEVELOPMENT", "grid_cell_id": "g67_328",
         "effective_x_i": 670001, "effective_y_i": 3280001, "effective_x_j": 670002, "effective_y_j": 3280002,
         "reason": "SAFE_CORE", "source_review_status": "APPROVED", "source_second_reviewer": ""},
        {"candidate_id": "b", "pair_id": "02_03", "split": "TEST", "grid_cell_id": "g65_328",
         "effective_x_i": 670100, "effective_y_i": 3280100, "effective_x_j": 670101, "effective_y_j": 3280101,
         "reason": "SAFE_CORE", "source_review_status": "APPROVED", "source_second_reviewer": ""},
    ]
    audit = audit_assignments(assignments)
    assert audit["leakage_pass"] is False
    assert audit["opposite_role_point_conflicts"]


def test_real_trial_counts_and_original_candidate_files_remain_read_only():
    assignments, hashes = collect_assignments(ROOT)
    assert len(assignments) == 160 and len(hashes) == 4
    expected = {"00_01": (11, 19, 10), "02_03": (9, 22, 9),
                "05_07": (9, 26, 5), "08_10": (14, 17, 9)}
    for pair, counts in expected.items():
        rows = [row for row in assignments if row["pair_id"] == pair]
        assert tuple(sum(row["split"] == role for row in rows)
                     for role in ("DEVELOPMENT", "TEST", "UNASSIGNED")) == counts
        assert all(row["formal_test_eligible"] is False for row in rows)
    assert audit_assignments(assignments)["leakage_pass"] is True


def test_written_artifacts_reload_and_preserve_source_review_history():
    output = ROOT / "docs/benchmark/v1/phase3"
    audit = json.loads((output / "split_leakage_audit.json").read_text(encoding="utf-8"))
    with (output / "checkpoint_split_assignments.csv").open(newline="", encoding="utf-8") as stream:
        derived = list(csv.DictReader(stream))
    assert len(derived) == 160
    for pair in ("00_01", "02_03", "05_07", "08_10"):
        source_path = ROOT / "docs/benchmark/v1/phase2/auto_candidates" / f"{pair}.csv"
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == audit["source_candidate_sha256"][pair]
        with source_path.open(newline="", encoding="utf-8") as stream:
            source = {row["candidate_id"]: row for row in csv.DictReader(stream)}
        for row in (r for r in derived if r["pair_id"] == pair):
            original = source[row["candidate_id"]]
            for key in ("row_i", "col_i", "x_i", "y_i", "row_j", "col_j", "x_j", "y_j",
                        "primary_reviewer", "second_reviewer", "review_time_utc", "review_notes",
                        "human_review_status", "coordinate_precision_note"):
                assert row[key] == original[key]
            assert row["source_split"] == original["split"] == "UNASSIGNED"
            assert row["formal_test_eligible"] == "False"
    for name, expected in audit["output_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert len(audit["b9_verified_inputs"]) == 8
    assert audit["remaining_scene_indices_not_verified"] == [4, 6, 9, 11, 12]
    assert audit["formal_test_eligible_count"] == 0
    assert audit["overall"]["unique_assigned_cells"] == 38
    assert audit["unassigned_reason_counts"] == {"BUFFER_ZONE": 33}
    assert audit["minimum_opposite_role_point_distance_m"] >= 1000
    assert audit["per_pair"]["00_01"]["role_cells"]["DEVELOPMENT"]["concentration_flag"] is True
