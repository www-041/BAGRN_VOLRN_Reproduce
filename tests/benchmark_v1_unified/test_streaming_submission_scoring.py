from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import rasterio

from src.benchmark_v1_unified.submission_scoring import evaluate_submission
from tests.benchmark_v1_unified.test_submission_scoring import make_thirteen_scene_fixture, _write


def _assert_nested_close(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for observed, reference in zip(actual, expected):
            _assert_nested_close(observed, reference)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-10)
    else:
        assert actual == expected


def test_streaming_scorer_matches_array_scorer_for_all_fixture_regimes(tmp_path, monkeypatch):
    from src.benchmark_v1_unified.streaming_submission_scoring import evaluate_submission_streaming
    import src.benchmark_v1_unified.streaming_submission_scoring as streaming_module

    # This is a scorer-equivalence test; resource-policy behavior is tested
    # independently below. Keep host RAM fluctuations out of this fixture.
    monkeypatch.setattr(streaming_module, "_resource_preflight", lambda *args, **kwargs: {
        "status": "PASS", "reasons": [], "available_ram_bytes": 2**40,
        "available_ram_measurement_source": "TEST_FIXTURE", "evaluator_ram_budget_bytes": 128 * 1024**2,
        "ram_safety_reserve_bytes": 0, "declared_method_max_rss_bytes": None,
        "hard_os_memory_limit": False, "available_scratch_disk_bytes": 2**40,
        "scratch_measurement_source": "TEST_FIXTURE", "estimated_pair_sample_scratch_bytes": 0,
        "estimated_exact_order_scratch_bytes": 0, "estimated_peak_scratch_bytes": 0,
        "disk_safety_reserve_bytes": 0,
    })

    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    reference = evaluate_submission(freeze, submission, submission_dir=tmp_path / "method",
                                   regimes=("A", "B1", "B2"))
    actual = evaluate_submission_streaming(freeze, submission, submission_dir=tmp_path / "method",
                                           regimes=("A", "B1", "B2"),
                                           scratch_dir=tmp_path / "scratch", block_shape=(37, 41))
    assert actual["status"] == reference["status"] == "EVALUATED", actual
    assert actual["A"]["status"] == reference["A"]["status"]
    assert actual["A"]["mosaic_coverage"] == reference["A"]["mosaic_coverage"]
    assert actual["A"]["seam"]["status"] == reference["A"]["seam"]["status"]
    assert actual["A"]["seam"]["edge_count"] == reference["A"]["seam"]["edge_count"]
    for key in ("mean_cross_jump", "mean_within_jump", "mean_excess", "p95_excess"):
        assert actual["A"]["seam"][key] == pytest.approx(reference["A"]["seam"][key], abs=1e-12)
    assert actual["B1"]["expected_pair_rows"] == reference["B1"]["expected_pair_rows"]
    assert actual["B1"]["positive_overlap_pairs"] == reference["B1"]["positive_overlap_pairs"]
    assert actual["B1"]["no_valid_overlap_pairs"] == reference["B1"]["no_valid_overlap_pairs"]
    for expected, observed in zip(reference["B1"]["pair_rows"], actual["B1"]["pair_rows"]):
        assert observed["pair_id"] == expected["pair_id"]
        assert observed["status"] == expected["status"]
        assert observed["n"] == expected["n"]
        if expected["metrics"]["direct"]:
            for key, value in expected["metrics"]["direct"].items():
                assert observed["metrics"]["direct"][key] == pytest.approx(value, abs=1e-10) if isinstance(value, float) else observed["metrics"]["direct"][key] == value
            for key, value in expected["metrics"]["distribution"].items():
                assert observed["metrics"]["distribution"][key] == pytest.approx(value, abs=1e-10) if isinstance(value, float) else observed["metrics"]["distribution"][key] == value
    assert actual["B2"]["window_count"] == reference["B2"]["window_count"]
    assert actual["B2"]["expected_window_rows"] == reference["B2"]["expected_window_rows"]
    assert len(actual["B2"]["structure_by_scene_window"]) == len(reference["B2"]["structure_by_scene_window"])
    for expected, observed in zip(reference["B2"]["window_rows"], actual["B2"]["window_rows"]):
        assert observed["window_id"] == expected["window_id"]
        assert observed["status"] == expected["status"]
        assert observed["n"] == expected["n"]
        if expected["metrics"]:
            for section in ("direct", "distribution"):
                for key, value in expected["metrics"][section].items():
                    assert observed["metrics"][section][key] == pytest.approx(value, abs=1e-10) if isinstance(value, float) else observed["metrics"][section][key] == value
    _assert_nested_close(actual["scene_structure"], reference["scene_structure"])
    _assert_nested_close(actual["B2"]["structure_by_scene_window"], reference["B2"]["structure_by_scene_window"])
    assert actual["ranking_eligible"] == reference["ranking_eligible"]
    assert not list((tmp_path / "scratch").iterdir())


def test_streaming_scorer_checks_frozen_mask_support_hashes(tmp_path):
    from src.benchmark_v1_unified.streaming_submission_scoring import evaluate_submission_streaming

    freeze, submission, masks = make_thirteen_scene_fixture(tmp_path)
    # Mutate a frozen mask after its lock was written: the fail-closed lock check must fire first.
    mask_path = freeze / "aligned_inputs" / "scene_000_input_mask.tif"
    with rasterio.open(mask_path, "r+") as ds:
        values = ds.read(1)
        values[0, 0] = 0 if values[0, 0] else 1
        ds.write(values, 1)
    result = evaluate_submission_streaming(freeze, submission, submission_dir=tmp_path / "method",
                                           regimes=("B2",), scratch_dir=tmp_path / "scratch")
    assert result["status"] == "FREEZE_INVALID"


def test_streaming_resource_preflight_fails_before_opening_outputs(tmp_path):
    from src.benchmark_v1_unified.streaming_submission_scoring import evaluate_submission_streaming

    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    submission["resource_budget"]["max_rss_bytes"] = 1
    submission["corrected_scene_paths"] = {"SCENE_00": "missing.tif"}
    result = evaluate_submission_streaming(freeze, submission, submission_dir=tmp_path / "method",
                                           regimes=("B1",), scratch_dir=tmp_path / "scratch",
                                           ram_budget_bytes=1)
    assert result["status"] == "RESOURCE_PREFLIGHT_FAILED"
    assert "RAM_BUDGET_BELOW_STREAMING_ESTIMATE" in result["resource_preflight"]["reasons"]
