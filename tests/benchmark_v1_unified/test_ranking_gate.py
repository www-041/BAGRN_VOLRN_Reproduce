import csv
import hashlib
import json
from itertools import combinations

from src.benchmark_v1_unified.ranking_gate import (
    apply_ranking_gate,
    evaluate_formal_13scene_ranking,
    evaluate_regime_eligibility,
)
from tests.benchmark_v1_unified.test_submission_scoring import make_thirteen_scene_fixture


def test_complete_small_synthetic_regime_can_be_individually_eligible():
    result = {"status": "EVALUATED", "B1": {"status": "EVALUABLE", "expected_pair_rows": 1,
                                                "pair_rows": [{"pair_id": "a_b", "status": "EVALUABLE"}]}}
    assert evaluate_regime_eligibility(result, "B1")["eligible"] is True


def test_b1_only_diagnostic_never_satisfies_formal_13scene_gate(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    evaluation = {"status": "EVALUATED", "B1": {"status": "EVALUABLE"}, "ranking_eligible": True}

    result = evaluate_formal_13scene_ranking(submission, evaluation, freeze)

    assert result["eligible"] is False
    assert "MISSING_REQUIRED_REGIMES" in result["reasons"]


def test_legacy_ranking_flag_is_overridden_by_formal_gate(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    evaluation = {"status": "EVALUATED", "B1": {"status": "EVALUABLE"}, "ranking_eligible": True}

    result = apply_ranking_gate(submission, evaluation, freeze)

    assert result["ranking_eligible"] is False
    assert result["formal_13scene_ranking_eligibility"]["eligible"] is False


def test_formal_gate_rejects_pair_only_unverified_science_and_bad_rss(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    submission.update({"method_kind": "PAIR_ONLY", "science_status": "LSSTAA_PY_ADAPTED_DN_PROXY_UNVERIFIED"})
    evaluation = {
        "status": "EVALUATED",
        "A": {"status": "EVALUABLE", "mosaic_coverage": {"holes": 0, "extras": 0}},
        "B1": {"status": "EVALUABLE"}, "B2": {"status": "EVALUABLE"},
        "runtime": {"measurement_provenance_status": "INCOMPLETE", "rss_status": "NOT_MEASURED",
                     "rss_sample_count": 0},
    }

    result = evaluate_formal_13scene_ranking(submission, evaluation, freeze)

    assert result["eligible"] is False
    assert "PAIR_ONLY_NOT_FORMAL_13SCENE" in result["reasons"]
    assert "SCIENCE_STATUS_NOT_FORMAL" in result["reasons"]
    assert "RESOURCE_MEASUREMENT_INVALID" in result["reasons"]


def test_formal_gate_rejects_missing_frozen_b2_windows_and_mosaic_holes(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    evaluation = {
        "status": "EVALUATED",
        "A": {"status": "INCOMPLETE_MOSAIC_SUPPORT", "mosaic_coverage": {"holes": 1, "extras": 0}},
        "B1": {"status": "EVALUABLE"}, "B2": {"status": "PARTIAL_OUTPUT", "window_count": 1},
        "runtime": {"measurement_provenance_status": "PASS", "rss_status": "SAMPLED", "rss_sample_count": 2},
    }

    result = evaluate_formal_13scene_ranking(submission, evaluation, freeze)

    assert result["eligible"] is False
    assert "MOSAIC_COVERAGE_INCOMPLETE" in result["reasons"]
    assert "B2_WINDOW_POPULATION_MISMATCH" in result["reasons"]


def test_formal_gate_requires_hard_source_value_provenance(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    submission.update({"method_kind": "FULL_MOSAIC", "status": "COMPLETED", "science_status": "VALID"})
    evaluation = {
        "status": "EVALUATED",
        "A": {"status": "EVALUABLE", "mosaic_coverage": {"holes": 0, "extras": 0},
              "seam": {"status": "EVALUABLE"}, "single_source_structure": {"status": "EVALUABLE"}},
        "B1": {"status": "EVALUABLE", "expected_pair_rows": 78, "positive_overlap_pairs": 48,
               "no_valid_overlap_pairs": 30},
        "B2": {"status": "EVALUABLE", "window_count": 1533, "expected_window_rows": 1533},
        "runtime": {"measurement_provenance_status": "PASS", "rss_status": "SAMPLED", "rss_sample_count": 2},
    }

    result = evaluate_formal_13scene_ranking(submission, evaluation, freeze, provenance_audit={"status": "NOT_RUN"})

    assert result["eligible"] is False
    assert "HARD_PROVENANCE_AUDIT_REQUIRED" in result["reasons"]
def test_exact_frozen_population_and_complete_a_b1_b2_can_pass_formal_gate(tmp_path):
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    scenes = [f"SCENE_{i:02d}" for i in range(13)]
    pair_rows = []
    for index, (i, j) in enumerate(combinations(range(13), 2)):
        pair_rows.append({"pair_id": f"{i:02d}_{j:02d}", "scene_i": scenes[i], "scene_j": scenes[j],
                          "status": "OVERLAP" if index < 48 else "NO_VALID_OVERLAP"})
    window_rows = [{"window_id": f"W{i:04d}", "pair_id": "00_01"} for i in range(1533)]
    for filename, rows in (("pair_overlap_inventory.csv", pair_rows), ("fixed_windows.csv", window_rows)):
        with (freeze / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    frozen = {"status": "FROZEN", "protocol_id": "DZ01V_UNIFIED_A_B_V1_20261013", "scene_count": 13,
              "input_freeze_sha256": "a" * 64, "geometry_freeze_sha256": "b" * 64}
    frozen_path = freeze / "frozen_manifest.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    locked = [freeze / "pair_overlap_inventory.csv", freeze / "fixed_windows.csv", frozen_path]
    entries = [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in locked]
    (freeze / "freeze_lock.json").write_text(json.dumps({"schema_version": 1, "files": entries}), encoding="utf-8")
    submission = {"method_kind": "FULL_MOSAIC", "method_id": "test-method", "run_id": "r1",
                  "status": "COMPLETED", "science_status": "VALID", "seam_type": "HARD",
                  "input_freeze_sha256": "a" * 64, "geometry_freeze_sha256": "b" * 64}
    evaluation = {
        "status": "EVALUATED", "submission_audit": {"status": "PASS", "artifact_status": "PASS"},
        "A": {"status": "EVALUABLE", "mosaic_coverage": {"coverage": 1.0, "holes": 0, "extras": 0},
              "single_source_structure": {"status": "EVALUABLE"}, "seam": {"status": "EVALUABLE"}},
        "B1": {"status": "EVALUABLE", "expected_pair_rows": 78, "positive_overlap_pairs": 48,
               "no_valid_overlap_pairs": 30,
               "pair_rows": [{"status": "EVALUABLE"}] * 48 + [{"status": "NO_VALID_OVERLAP"}] * 30},
        "B2": {"status": "EVALUABLE", "window_count": 1533, "expected_window_rows": 1533,
               "window_rows": [{"status": "EVALUABLE", "metrics": {"direct": {}, "distribution": {}}}] * 1533},
        "runtime": {"measurement_provenance_status": "PASS", "wall_seconds": 10.0,
                    "wall_measurement_source": "time.perf_counter", "peak_rss_bytes": 1000,
                    "rss_status": "MEASURED", "rss_sample_count": 10,
                    "rss_measurement_source": "psutil.ProcessTree", "resource_budget": {"max_rss_bytes": 2000}},
    }

    result = evaluate_formal_13scene_ranking(submission, evaluation, freeze, provenance_audit={"status": "PASS"})

    assert result["eligible"] is True, result["reasons"]
    assert result["frozen_population"] == {"scene_count": 13, "pair_count": 78,
                                            "overlap_pair_count": 48, "no_overlap_pair_count": 30,
                                            "window_count": 1533}
