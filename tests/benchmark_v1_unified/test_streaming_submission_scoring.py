from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import rasterio
from affine import Affine

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


def test_streaming_b1_b2_does_not_open_missing_mosaic_artifacts(tmp_path, monkeypatch):
    from src.benchmark_v1_unified.streaming_submission_scoring import evaluate_submission_streaming
    import src.benchmark_v1_unified.streaming_submission_scoring as streaming_module

    monkeypatch.setattr(streaming_module, "_resource_preflight", lambda *args, **kwargs: {
        "status": "PASS", "estimated_peak_scratch_bytes": 0,
    })
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    submission["output_mosaic_path"] = "absent-mosaic.tif"
    submission["output_mosaic_mask_path"] = "absent-mask.tif"
    result = evaluate_submission_streaming(freeze, submission, submission_dir=tmp_path / "method",
                                           regimes=("B1", "B2"), scratch_dir=tmp_path / "scratch")
    assert result["status"] == "EVALUATED", result
    assert result["A"] is None
    assert result["B1"]["status"] == "EVALUABLE"
    assert result["B2"]["status"] == "EVALUABLE"


def test_resource_preflight_checks_actual_scratch_directory_volume(tmp_path, monkeypatch):
    import src.benchmark_v1_unified.streaming_submission_scoring as streaming_module

    scratch = tmp_path / "scratch-volume" / "nested"
    scratch.mkdir(parents=True)
    captured = []

    class Disk:
        free = 10**12

    monkeypatch.setattr(streaming_module.shutil, "disk_usage", lambda path: captured.append(Path(path)) or Disk())
    monkeypatch.setattr("psutil.virtual_memory", lambda: type("VM", (), {"available": 10**12})())
    result = streaming_module._resource_preflight({}, (8, 8), [], scratch, 128 * 1024**2, 0, 0)
    assert result["status"] == "PASS"
    assert captured == [scratch.resolve()]
    assert result["scratch_path"] == str(scratch.resolve())


def test_partial_b2_windows_remain_rows_with_null_metrics_and_valid_denominators(tmp_path, monkeypatch):
    from src.benchmark_v1_unified.streaming_submission_scoring import evaluate_submission_streaming
    import src.benchmark_v1_unified.streaming_submission_scoring as streaming_module

    monkeypatch.setattr(streaming_module, "_resource_preflight", lambda *args, **kwargs: {
        "status": "PASS", "estimated_peak_scratch_bytes": 0,
    })
    freeze, submission, masks = make_thirteen_scene_fixture(tmp_path)
    output_path = Path(submission["corrected_scene_paths"]["SCENE_00"])
    partial = masks[0].copy()
    partial[20:30, 20:30] = False
    _write(output_path, np.full((128, 128), 100), partial, Affine(14, 0, 633346, 0, -14, 3359874))
    result = evaluate_submission_streaming(freeze, submission, submission_dir=tmp_path / "method",
                                           regimes=("B2",), scratch_dir=tmp_path / "scratch")
    assert result["B2"]["status"] == "PARTIAL_OUTPUT"
    assert result["B2"]["window_count"] == result["B2"]["expected_window_rows"]
    partial_rows = [row for row in result["B2"]["window_rows"] if row["status"] == "PARTIAL_OUTPUT"]
    assert partial_rows
    assert all(row["metrics"] is None for row in partial_rows)
    assert all(row["n"] == row["output_pixels"] + 100 for row in partial_rows)
    assert all(sum(row["n"] for row in result["B2"]["metric_distributions"][key]["per_pair"].values())
               < result["B2"]["expected_window_rows"]
               for key in ("mae", "rmse", "bias", "p95ae", "mamd", "msdd", "rdd"))


def test_streaming_and_array_single_source_structure_use_raw_reference_for_affine_output(tmp_path, monkeypatch):
    from src.benchmark_v1_unified.streaming_submission_scoring import evaluate_submission_streaming
    import src.benchmark_v1_unified.streaming_submission_scoring as streaming_module

    monkeypatch.setattr(streaming_module, "_resource_preflight", lambda *args, **kwargs: {
        "status": "PASS", "estimated_peak_scratch_bytes": 0,
    })
    freeze, submission, masks = make_thirteen_scene_fixture(tmp_path)
    labels_path = Path(submission["source_label_path"])
    with rasterio.open(labels_path) as ds:
        labels = ds.read(1)
    rr, cc = np.mgrid[:128, :128]
    raw_base = 100 + rr * 0.5 + cc * 0.25 + np.sin(cc / 7) * 3
    corrected = []
    for index, (scene_id, path) in enumerate(submission["corrected_scene_paths"].items()):
        values = 1.2 * (raw_base + index) + 50
        corrected.append(values)
        _write(Path(path), values, masks[index], Affine(14, 0, 633346, 0, -14, 3359874))
    mosaic = np.zeros((128, 128), dtype=np.float32)
    for index in range(13):
        selected = labels == index
        mosaic[selected] = corrected[index][selected]
    _write(Path(submission["output_mosaic_path"]), mosaic, np.logical_or.reduce(masks),
           Affine(14, 0, 633346, 0, -14, 3359874))

    reference = evaluate_submission(freeze, submission, submission_dir=tmp_path / "method", regimes=("A",))
    actual = evaluate_submission_streaming(freeze, submission, submission_dir=tmp_path / "method",
                                           regimes=("A",), scratch_dir=tmp_path / "scratch",
                                           block_shape=(31, 37))
    ref_struct = reference["A"]["single_source_structure"]
    actual_struct = actual["A"]["single_source_structure"]
    assert ref_struct["structure"]["direction_loss"] == pytest.approx(0.0, abs=1e-7)
    assert actual_struct["structure"]["direction_loss"] == pytest.approx(0.0, abs=1e-7)
    assert ref_struct["structure"]["relative_gradient_mae"] == pytest.approx(0.2, rel=1e-5)
    assert actual_struct["structure"]["relative_gradient_mae"] == pytest.approx(0.2, rel=1e-5)
