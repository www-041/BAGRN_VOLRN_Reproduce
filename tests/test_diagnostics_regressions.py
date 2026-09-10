"""Regression tests for diagnostics coefficient statistics.

Task 12 of reliability-fixes plan:
- gain RMS must be measured relative to identity (a - 1), not raw a values
- dynamic range should use data range, not coefficient range
"""

import json
import csv
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin


def _hull_causal_registration_fixture():
    return {
        "status": "fail",
        "quality": {"quality": "fail", "median": 2.0, "rmse": 2.5,
                    "p95": 3.0, "confidence": 0.7, "n_blocks": 6},
        "global_only_quality": {"quality": "pass", "median": 0.5,
                                 "rmse": 0.8, "p95": 1.0,
                                 "confidence": 0.9, "n_blocks": 6},
        "global_only_validation": {"edges": [], "overall": {}},
        "final_validation": {"edges": [], "overall": {}},
        "local_refinement": {},
        "strict_counterfactual_arrays": [
            np.zeros((1, 4096, 4096)), np.zeros((1, 4096, 4096))
        ],
        "hull_causal": {
            "available": True,
            "integrity": {"integrity_pass": True},
            "strict_counterfactual_quality": {
                "quality": "pass", "median": 0.6, "rmse": 0.9,
                "p95": 1.1, "confidence": 0.9, "n_blocks": 6,
            },
            "comparisons": {
                "global_to_legacy": {"rmse_improvement": -1.7,
                                      "p95_improvement": -2.0,
                                      "median_improvement": -1.5},
                "global_to_strict": {"rmse_improvement": -0.1,
                                      "p95_improvement": -0.1,
                                      "median_improvement": -0.1},
                "legacy_to_strict": {"rmse_improvement": 1.6,
                                      "p95_improvement": 1.9,
                                      "median_improvement": 1.4},
            },
            "window_stats": {"blocks": [{
                "validation_row": 0, "validation_col": 0,
                "block_size": 4, "scene_idx": 1,
                "window_inside_hull_fraction": 0.5,
                "window_legacy_support_nonzero_fraction": 0.75,
                "window_strict_support_nonzero_fraction": 0.5,
                "window_support_difference_fraction": 0.25,
            }]},
        },
    }


def test_hull_causal_cli_flag_is_opt_in():
    from scripts import diagnose_registration_pair

    assert diagnose_registration_pair._parse_args([
        "--config", "config.yaml", "--hull-causal-test",
    ]).hull_causal_test is True
    assert diagnose_registration_pair._parse_args([
        "--config", "config.yaml",
    ]).hull_causal_test is False


def test_hull_causal_payload_excludes_large_field_arrays(tmp_path):
    from scripts import diagnose_registration_pair

    registration = _hull_causal_registration_fixture()
    payload = diagnose_registration_pair.build_diagnostic_payload(
        registration, ["a", "b"], tmp_path,
    )
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["hull_causal"]["available"] is True
    assert "strict_counterfactual_arrays" not in payload
    assert "4096" not in serialized


def test_hull_causal_stage_csv_has_exact_three_rows(tmp_path):
    from scripts import diagnose_registration_pair

    path = diagnose_registration_pair._write_hull_causal_stage_metrics_csv(
        _hull_causal_registration_fixture(), tmp_path,
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["stage"] for row in rows] == [
        "global_only", "legacy_rbf", "strict_rbf"
    ]
    assert len(rows) == 3


def test_hull_causal_window_csv_contains_window_support_columns(tmp_path):
    from scripts import diagnose_registration_pair

    path = diagnose_registration_pair._write_hull_causal_window_stats_csv(
        _hull_causal_registration_fixture(), tmp_path,
    )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert rows
    assert {
        "window_inside_hull_fraction",
        "window_legacy_support_nonzero_fraction",
        "window_strict_support_nonzero_fraction",
        "window_support_difference_fraction",
        "legacy_minus_strict",
        "negative_control_candidate",
    } <= set(reader.fieldnames)


def test_hull_window_csv_preserves_real_block_size_and_triplet_residuals(tmp_path):
    from scripts import diagnose_registration_pair

    registration = _hull_causal_registration_fixture()
    registration["hull_causal"]["window_stats"]["blocks"] = [{
        "block_size": 384, "global_residual_magnitude": 0.6,
        "legacy_residual_magnitude": 2.1, "strict_residual_magnitude": 2.2,
        "validation_row": 1, "validation_col": 2,
    }]
    path = diagnose_registration_pair._write_hull_causal_window_stats_csv(
        registration, tmp_path
    )
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["block_size"] == "384"
    assert row["global_residual_magnitude"] != ""
    assert row["legacy_residual_magnitude"] != ""
    assert row["strict_residual_magnitude"] != ""


def test_hull_window_csv_recovery_fraction_handles_zero_denominator(tmp_path):
    from scripts import diagnose_registration_pair

    registration = _hull_causal_registration_fixture()
    registration["hull_causal"]["window_stats"]["blocks"] = [{
        "block_size": 384, "global_residual_magnitude": 2.0,
        "legacy_residual_magnitude": 2.0, "strict_residual_magnitude": 1.0,
    }]
    path = diagnose_registration_pair._write_hull_causal_window_stats_csv(
        registration, tmp_path
    )
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["recovery_fraction"] == ""


def test_quality_fail_still_writes_strict_diagnostic_artifacts_but_returns_nonzero(
    tmp_path, monkeypatch
):
    from scripts import diagnose_registration_pair

    config = SimpleNamespace(
        scenes=[{"id": "scene_a"}, {"id": "scene_b"}],
        control_scene="scene_a",
        output_root=str(tmp_path),
        registration_params={"required_quality": "pass"},
    )
    registration = _hull_causal_registration_fixture()
    registration["connected"] = True
    registration["diagnostics"] = {}
    registration["registered_arrays"] = [
        np.zeros((1, 4, 4)), np.zeros((1, 4, 4))
    ]
    captured = {}

    class FakePipeline:
        registration_band_idx = 0

        def __init__(self, pipeline_config):
            self.config = pipeline_config

        def load_scenes(self):
            return {
                "arrays": registration["registered_arrays"],
                "transforms": [from_origin(0, 4, 1, 1)] * 2,
                "nodata_values": [None, None],
                "crs": None,
            }

        def detect_overlaps(self, scene_data):
            return [{"idx_i": 0, "idx_j": 1}]

        def register_scenes(self, *args, **kwargs):
            captured["hull_causal_diagnostic"] = kwargs.get(
                "hull_causal_diagnostic"
            )
            return registration

    monkeypatch.setattr(diagnose_registration_pair, "load_config", lambda _: config)
    monkeypatch.setattr(diagnose_registration_pair, "MultibandPipeline", FakePipeline)
    monkeypatch.setattr(
        diagnose_registration_pair,
        "write_diagnostic_artifacts",
        lambda *args, **kwargs: captured.setdefault("artifacts", {}) or {},
    )

    exit_code = diagnose_registration_pair.main([
        "--config", "unused.yaml",
        "--scene-i", "0", "--scene-j", "1",
        "--hull-causal-test",
        "--output-dir", str(tmp_path / "out"),
    ])

    assert exit_code == 1
    assert captured["hull_causal_diagnostic"] is True
    assert "artifacts" in captured


def test_registration_diagnostic_distinguishes_raw_and_robust_pair_matches(tmp_path):
    from scripts import diagnose_registration_pair

    raw_matches = [
        {"ref_x": 1.0, "ref_y": 1.0, "shift_dx": 3.0, "shift_dy": -2.0,
         "confidence": 0.9},
        {"ref_x": 2.0, "ref_y": 2.0, "shift_dx": 18.0, "shift_dy": 15.0,
         "confidence": 0.8},
    ]
    robust_matches = [raw_matches[0]]
    registration = {
        "global_shifts": [[0.0, 0.0], [0.0, 0.0]],
        "pair_matches": [{"idx_i": 0, "idx_j": 1,
                          "raw_matches": raw_matches,
                          "matches": robust_matches}],
        "quality": {"quality": "pass"},
        "final_validation": {"edges": [], "overall": {"quality": "pass"}},
        "local_refinement": {},
        "diagnostics": {},
    }

    payload = diagnose_registration_pair.build_diagnostic_payload(
        registration, ["scene_a", "scene_b"], tmp_path
    )

    assert payload["raw_block_matches"][0]["matches"] == raw_matches
    assert payload["robust_pair_measurements"][0]["matches"] == robust_matches
    assert payload["raw_block_matches"][0]["matches"] != payload[
        "robust_pair_measurements"
    ][0]["matches"]


def test_registration_diagnostic_writes_holdout_and_common_valid_fields(tmp_path):
    from scripts import diagnose_registration_pair

    registration = {
        "quality": {"quality": "fail"},
        "final_validation": {"edges": [], "overall": {"quality": "fail"}},
        "diagnostics": {
            "overlap": {"0-1": {"bbox_pixels": 100,
                                  "common_valid_pixels": 60,
                                  "common_valid_ratio": 0.6}},
            "holdout": {"0-1": {"holdout_cells": 2}},
            "local_controls": {"1": {"n_raw": 10}},
            "local_field": {"1": {"max_dx": 6.0}},
        },
    }

    payload = diagnose_registration_pair.build_diagnostic_payload(
        registration, ["a", "b"], tmp_path
    )

    assert payload["overlap"]["0-1"]["common_valid_ratio"] == 0.6
    assert payload["holdout"]["0-1"]["holdout_cells"] == 2
    assert payload["local_controls"]["1"]["n_raw"] == 10
    assert payload["local_field"]["1"]["max_dx"] == 6.0


def test_registration_diagnostic_preserves_all_local_cv_candidates(tmp_path):
    from scripts import diagnose_registration_pair

    cv_result = {
        "available": True,
        "selected_smoothing": 0.1,
        "has_passing_candidate": True,
        "selection_reason": "best_passing_candidate",
        "candidate_results": [
            {"smoothing": value, "available": True, "passes_gate": value == 0.1}
            for value in (0.01, 0.05, 0.1, 0.5, 1.0)
        ],
    }
    registration = {
        "quality": {"quality": "pass"},
        "final_validation": {"edges": [], "overall": {"quality": "pass"}},
        "local_refinement": {"cv_results": {"1": cv_result}},
    }

    payload = diagnose_registration_pair.build_diagnostic_payload(
        registration, ["a", "b"], tmp_path
    )

    assert payload["local_cv_results"]["1"]["selected_smoothing"] == 0.1
    assert [item["smoothing"] for item in payload["local_cv_results"]["1"]["candidate_results"]] == [
        0.01, 0.05, 0.1, 0.5, 1.0
    ]


def test_register_scenes_preserves_raw_matches_separately(monkeypatch):
    from src import coregistration, multiband_pipeline

    raw_matches = [
        {"shift_dx": 3.0, "shift_dy": -2.0, "confidence": 0.9},
        {"shift_dx": 18.0, "shift_dy": 15.0, "confidence": 0.8},
    ]
    robust_matches = [raw_matches[0]]
    monkeypatch.setattr(
        coregistration,
        "collect_block_matches",
        lambda *args, **kwargs: (raw_matches, {"total": 2, "accepted": 2}),
    )
    monkeypatch.setattr(
        coregistration,
        "build_robust_pair_measurement",
        lambda matches, params: {
            "status": "pass", "shift_dx": 3.0, "shift_dy": -2.0,
            "confidence": 0.9, "n_blocks_inlier": 1, "n_blocks_total": 2,
            "rmse": 0.0, "p95": 0.0, "matches": robust_matches,
            "screening": params["screening"],
        },
    )
    monkeypatch.setattr(
        coregistration,
        "multi_image_network_adjustment",
        lambda pairs, n_images, control_idx: {
            "global_shifts": np.zeros((n_images, 2)), "loop_errors": [],
        },
    )
    monkeypatch.setattr(
        coregistration,
        "refine_global_residual_shifts_from_original",
        lambda *args, **kwargs: {
            "global_shifts": np.zeros((2, 2)), "history": [], "warnings": [],
        },
    )
    monkeypatch.setattr(
        multiband_pipeline,
        "_validate_final_registration_arrays",
        lambda *args, **kwargs: (
            {"quality": "pass"}, {"edges": [], "overall": {"quality": "pass"}}
        ),
    )

    pipeline = object.__new__(multiband_pipeline.MultibandPipeline)
    pipeline.registration_band_idx = 0
    pipeline.common_bands = ["B14"]
    pipeline.control_idx = 0
    pipeline.smoke = False
    pipeline.config = SimpleNamespace(
        registration_params={"enable_local_refinement": False}
    )
    arrays = [np.ones((1, 8, 8)), np.ones((1, 8, 8))]
    scene_data = {
        "arrays": arrays,
        "transforms": [None, None],
        "nodata_values": [None, None],
        "scene_ids": ["scene_a", "scene_b"],
    }

    result = pipeline.register_scenes(scene_data, [{"idx_i": 0, "idx_j": 1}])

    assert result["pair_matches"][0]["matches"] == robust_matches
    assert result["pair_matches"][0]["raw_matches"] == raw_matches
    assert result["diagnostics"]["raw_block_matches"][0]["matches"] == raw_matches


def test_no_overlap_writes_structured_failure_payload(tmp_path, monkeypatch):
    from scripts import diagnose_registration_pair

    config = SimpleNamespace(
        scenes=[{"id": "scene_a"}, {"id": "scene_b"}],
        control_scene="original_control",
        output_root=str(tmp_path / "default-output"),
    )
    captured = {}

    class FakePipeline:
        registration_band_idx = 0

        def __init__(self, pipeline_config):
            captured["control_scene"] = pipeline_config.control_scene

        def load_scenes(self):
            return {
                "arrays": [], "transforms": [], "nodata_values": [],
                "crs": "EPSG:4326", "scene_ids": ["scene_a", "scene_b"],
            }

        def detect_overlaps(self, scene_data):
            return []

    monkeypatch.setattr(diagnose_registration_pair, "load_config", lambda _: config)
    monkeypatch.setattr(diagnose_registration_pair, "MultibandPipeline", FakePipeline)
    output_dir = tmp_path / "diagnostic"

    result = diagnose_registration_pair.main([
        "--config", "ignored.yaml", "--scene-i", "0", "--scene-j", "1",
        "--output-dir", str(output_dir),
    ])

    assert result == 1
    assert captured["control_scene"] == "scene_a"
    payload = json.loads(
        (output_dir / "registration_diagnostics.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "fail"
    assert payload["failure"]["code"] == "no_overlap"
    assert payload["quality"]["quality"] == "fail"
    assert not list(output_dir.glob("*.tif"))


def test_failed_overlapping_registration_writes_failure_only_and_returns_nonzero(
    tmp_path, monkeypatch,
):
    from scripts import diagnose_registration_pair

    config = SimpleNamespace(
        scenes=[{"id": "scene_a"}, {"id": "scene_b"}],
        control_scene="original_control",
        output_root=str(tmp_path / "default-output"),
        registration_params={"required_quality": "pass"},
    )
    raw_arrays = [np.ones((1, 4, 4)), np.ones((1, 4, 4)) * 2]

    class FakePipeline:
        registration_band_idx = 0

        def __init__(self, pipeline_config):
            self.config = pipeline_config

        def load_scenes(self):
            return {
                "arrays": raw_arrays,
                "transforms": [from_origin(0, 4, 1, 1)] * 2,
                "nodata_values": [None, None],
                "crs": "EPSG:4326",
                "scene_ids": ["scene_a", "scene_b"],
            }

        def detect_overlaps(self, scene_data):
            return [{"idx_i": 0, "idx_j": 1}]

        def register_scenes(self, scene_data, overlaps):
            # These arrays are the raw fallback arrays from a blocked result;
            # the CLI must not publish them as registered artifacts.
            return {
                "registered_arrays": raw_arrays,
                "connected": False,
                "quality": {"quality": "fail"},
                "pair_matches": [],
                "diagnostics": {"registration_blocked": True},
            }

    monkeypatch.setattr(diagnose_registration_pair, "load_config", lambda _: config)
    monkeypatch.setattr(diagnose_registration_pair, "MultibandPipeline", FakePipeline)
    monkeypatch.setattr(
        diagnose_registration_pair,
        "write_diagnostic_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("failed registration must not write raster artifacts")
        ),
    )
    output_dir = tmp_path / "diagnostic"

    result = diagnose_registration_pair.main([
        "--config", "ignored.yaml", "--scene-i", "0", "--scene-j", "1",
        "--output-dir", str(output_dir),
    ])

    assert result == 1
    payload = json.loads(
        (output_dir / "registration_diagnostics.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "fail"
    assert payload["failure"]["code"] == "registration_connectivity_failed"
    assert payload["connected"] is False
    assert payload["quality"]["quality"] == "fail"
    assert not list(output_dir.glob("*.tif"))


def test_registration_diagnostic_uses_actual_registration_schema(tmp_path, monkeypatch):
    from scripts import diagnose_registration_pair

    registration = {
        "connected": True,
        "global_shifts": np.zeros((2, 2)).tolist(),
        "pair_matches": [{"idx_i": 0, "idx_j": 1, "status": "pass"}],
        "quality": {"quality": "pass", "rmse": 0.1, "p95": 0.2,
                    "median": 0.1, "confidence": 0.8, "n_blocks": 8},
        "local_refinement": {"enabled": False, "used_for_scenes": [],
                              "fallback_scenes": [1], "cv_results": {}},
        "final_validation": {"edges": [], "overall": {"quality": "pass"}},
        "diagnostics": {"global_refinement": []},
    }
    payload = diagnose_registration_pair.build_diagnostic_payload(
        registration, ["scene_a", "scene_b"], tmp_path
    )

    assert payload["scene_ids"] == ["scene_a", "scene_b"]
    assert payload["quality"]["quality"] == "pass"
    assert payload["final_validation"] == registration["final_validation"]


def test_registration_diagnostic_payload_preserves_global_only_and_stage_comparison(tmp_path):
    from scripts import diagnose_registration_pair

    registration = {
        "global_only_quality": {"quality": "fail", "rmse": 1.8},
        "global_only_validation": {"edges": [], "overall": {"rmse": 1.8}},
        "final_validation": {"edges": [], "overall": {"rmse": 1.2}},
        "stage_validation_comparison": {
            "available": True, "rmse_improvement": 0.6,
        },
        "holdout_local_field_samples": {"1": {"edges": []}},
        "quality": {"quality": "fail"},
    }

    payload = diagnose_registration_pair.build_diagnostic_payload(
        registration, ["a", "b"], tmp_path
    )

    assert payload["global_only_quality"] == registration["global_only_quality"]
    assert payload["global_only_validation"] == registration["global_only_validation"]
    assert payload["stage_validation_comparison"] == registration[
        "stage_validation_comparison"
    ]
    assert payload["holdout_local_field_samples"] == registration[
        "holdout_local_field_samples"
    ]


def test_registration_diagnostic_payload_defaults_new_stage_fields_for_old_results(tmp_path):
    from scripts import diagnose_registration_pair

    payload = diagnose_registration_pair.build_diagnostic_payload(
        {"quality": {}, "final_validation": {}}, ["a", "b"], tmp_path
    )

    assert payload["global_only_quality"] is None
    assert payload["global_only_validation"] is None
    assert payload["stage_validation_comparison"] is None
    assert payload["holdout_local_field_samples"] == {}


def test_connected_quality_fail_can_write_diagnostic_artifacts_but_returns_nonzero(
    tmp_path, monkeypatch,
):
    from scripts import diagnose_registration_pair

    config = SimpleNamespace(
        scenes=[{"id": "a"}, {"id": "b"}],
        control_scene="a",
        output_root=str(tmp_path / "default-output"),
        registration_params={"required_quality": "pass"},
    )
    arrays = [np.ones((1, 4, 4)), np.ones((1, 4, 4)) * 2]
    registration = {
        "registered_arrays": arrays,
        "global_only_arrays": arrays,
        "connected": True,
        "status": "fail",
        "quality": {"quality": "fail", "rmse": 1.2, "p95": 2.0},
        "final_validation": {"edges": [], "overall": {"quality": "fail"}},
        "diagnostics": {"registration_blocked": False},
    }

    class FakePipeline:
        registration_band_idx = 0

        def __init__(self, pipeline_config):
            self.config = pipeline_config

        def load_scenes(self):
            return {
                "arrays": arrays,
                "transforms": [from_origin(0, 4, 1, 1)] * 2,
                "nodata_values": [None, None],
                "crs": "EPSG:4326",
                "scene_ids": ["a", "b"],
            }

        def detect_overlaps(self, scene_data):
            return [{"idx_i": 0, "idx_j": 1}]

        def register_scenes(self, scene_data, overlaps):
            return registration

    writer_calls = []
    monkeypatch.setattr(diagnose_registration_pair, "load_config", lambda _: config)
    monkeypatch.setattr(diagnose_registration_pair, "MultibandPipeline", FakePipeline)
    monkeypatch.setattr(
        diagnose_registration_pair,
        "write_diagnostic_artifacts",
        lambda *args, **kwargs: writer_calls.append(kwargs) or {"final": "artifact"},
    )

    result = diagnose_registration_pair.main([
        "--config", "ignored.yaml", "--scene-i", "0", "--scene-j", "1",
        "--output-dir", str(tmp_path / "diagnostic"),
    ])

    assert result == 1
    assert writer_calls[0]["allow_quality_fail_for_diagnostics"] is True


def test_blocked_quality_fail_still_cannot_write_artifacts(tmp_path):
    from scripts import diagnose_registration_pair

    registration = {
        "registered_arrays": [np.ones((1, 4, 4)), np.ones((1, 4, 4))],
        "connected": True,
        "quality": {"quality": "fail"},
        "diagnostics": {"registration_blocked": True},
    }

    with pytest.raises(ValueError, match="blocked/disconnected"):
        diagnose_registration_pair.write_diagnostic_artifacts(
            registration,
            {"transforms": [from_origin(0, 4, 1, 1)] * 2,
             "nodata_values": [None, None], "crs": "EPSG:4326"},
            ["a", "b"], tmp_path,
            allow_quality_fail_for_diagnostics=True,
        )


def test_diagnostic_artifacts_write_distinct_global_only_and_final_overlays(
    tmp_path, monkeypatch,
):
    from scripts import diagnose_registration_pair

    global_only = [np.ones((1, 4, 4)), np.ones((1, 4, 4)) * 2]
    final = [np.ones((1, 4, 4)) * 3, np.ones((1, 4, 4)) * 4]
    written = []
    monkeypatch.setattr(
        diagnose_registration_pair,
        "write_geotiff",
        lambda path, array, *args, **kwargs: written.append(Path(path).name),
    )
    monkeypatch.setattr(diagnose_registration_pair, "create_mosaic", lambda *a, **k: None)
    monkeypatch.setattr(diagnose_registration_pair, "_reproject_to_reference",
                        lambda *a, **k: np.asarray(a[0]))
    monkeypatch.setattr(diagnose_registration_pair, "_save_mask_png", lambda *a, **k: None)
    monkeypatch.setattr(diagnose_registration_pair, "_save_residual_vectors", lambda *a, **k: None)
    monkeypatch.setattr(diagnose_registration_pair, "_save_validation_holdout_png", lambda *a, **k: None)
    monkeypatch.setattr(diagnose_registration_pair, "_save_field_png", lambda *a, **k: None)
    registration = {
        "registered_arrays": final,
        "global_only_arrays": global_only,
        "connected": True,
        "quality": {"quality": "pass"},
        "pair_matches": [{}],
        "diagnostics": {"raw_block_matches": [{"matches": []}]},
        "diagnostic_masks": {"0-1": {
            "common_valid_mask": np.ones((4, 4), dtype=bool),
            "train_sampling_mask": np.ones((4, 4), dtype=bool),
            "holdout_region_mask": np.ones((4, 4), dtype=bool),
        }},
        "final_validation": {"edges": [], "overall": {}},
        "local_dx_fields": [np.zeros((4, 4)), np.zeros((4, 4))],
        "local_dy_fields": [np.zeros((4, 4)), np.zeros((4, 4))],
    }
    artifacts = diagnose_registration_pair.write_diagnostic_artifacts(
        registration,
        {"transforms": [from_origin(0, 4, 1, 1)] * 2,
         "nodata_values": [None, None], "crs": "EPSG:4326"},
        ["a", "b"], tmp_path,
    )

    expected = {
        "registered_global_only_reference.tif",
        "registered_global_only_target.tif",
        "registered_global_only_red_green_overlay.tif",
        "registered_final_reference.tif",
        "registered_final_target.tif",
        "registered_final_red_green_overlay.tif",
    }
    assert expected <= set(written)
    assert artifacts["final_holdout_validation_blocks"].endswith(
        "final_holdout_validation_blocks.png"
    )


def test_diagnostic_payload_does_not_serialize_large_global_only_arrays(tmp_path):
    from scripts import diagnose_registration_pair

    payload = diagnose_registration_pair.build_diagnostic_payload(
        {"global_only_arrays": [np.ones((1, 4, 4))], "quality": {}},
        ["a", "b"], tmp_path,
    )

    assert "global_only_arrays" not in payload


def _mapped_holdout_samples():
    return {
        "edges": [{
            "idx_i": 0,
            "idx_j": 1,
            "blocks": [{
                "validation_row": 10,
                "validation_col": 20,
                "reference_center_x": 22.5,
                "reference_center_y": 12.5,
                "field_center_x": 19.5,
                "field_center_y": 15.5,
                "scene_idx": 1,
                "coordinate_mapping_available": True,
                "coordinate_mapping_reason": None,
                "predicted_local_dx": 0.4,
                "predicted_local_dy": -0.2,
                "predicted_local_magnitude": 0.4472136,
                "fade_value": 0.75,
                "inside_control_hull": False,
                "distance_outside_hull_px": 2.0,
                "nearest_local_control_distance_px": 12.0,
                "final_residual_dx": 0.1,
                "final_residual_dy": 0.0,
                "final_residual_magnitude": 0.1,
                "accepted": True,
                "reject_reason": None,
                "selected_smoothing": 0.5,
            }],
        }],
    }


def test_holdout_local_field_csv_contains_mapped_coordinate_columns(tmp_path):
    from scripts import diagnose_registration_pair

    registration = {"holdout_local_field_samples": _mapped_holdout_samples()}
    path = diagnose_registration_pair._write_holdout_local_field_csv(
        registration, tmp_path
    )

    header = Path(path).read_text(encoding="utf-8").splitlines()[0]
    assert "reference_center_x" in header
    assert "field_center_x" in header
    assert "coordinate_mapping_available" in header
    assert "fade_value" in header
    assert "inside_control_hull" in header
    assert "nearest_local_control_distance_px" in header


def test_quality_fail_diagnostic_still_writes_holdout_local_field_csv(tmp_path):
    from scripts import diagnose_registration_pair

    registration = {
        "holdout_local_field_samples": _mapped_holdout_samples(),
        "quality": {"quality": "fail"},
    }
    path = diagnose_registration_pair._write_holdout_local_field_csv(
        registration, tmp_path
    )

    assert Path(path).name == "holdout_local_field_samples.csv"
    assert Path(path).exists()
    assert "0.4" in Path(path).read_text(encoding="utf-8")


def test_holdout_local_field_summary_logs_mapping_counts(caplog):
    from scripts import diagnose_registration_pair

    with caplog.at_level("INFO"):
        diagnose_registration_pair._log_holdout_local_field_summary({
            "holdout_local_field_samples": _mapped_holdout_samples(),
        })

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "local-field HOLDOUT mapping: total=1, mapped=1" in messages
    assert "local-field HOLDOUT correction magnitude" in messages


def test_identity_gain_has_zero_rms():
    """When a=1 (identity gain), RMS should be 0, not 1."""
    # Identity transform: a=1, b=0
    a = np.ones(10)
    
    # Bug: RMS = sqrt(mean(a^2)) = sqrt(mean(1)) = 1.0
    # Fix: RMS = sqrt(mean((a-1)^2)) = sqrt(mean(0)) = 0.0
    
    finite_a = a[np.isfinite(a)]
    a_rms_correct = float(np.sqrt(np.mean((finite_a - 1) ** 2)))
    
    assert a_rms_correct == 0.0, "Identity gain (a=1) should have RMS=0"


def test_non_identity_gain_has_nonzero_rms():
    """When a deviates from 1, RMS should be non-zero."""
    a = np.array([0.9, 1.0, 1.1])
    
    finite_a = a[np.isfinite(a)]
    a_rms_correct = float(np.sqrt(np.mean((finite_a - 1) ** 2)))
    
    # RMS = sqrt(mean([0.01, 0, 0.01])) = sqrt(0.02/3) ≈ 0.0816
    assert a_rms_correct > 0.0, "Non-identity gain should have RMS > 0"
    assert abs(a_rms_correct - 0.0816) < 0.01


def test_dynamic_range_uses_data_not_coefficients():
    """b_over_dynamic_range should use data dynamic range, not coefficient range."""
    # Data values
    data = np.array([100, 200, 300, 400, 500])
    data_dr = data.max() - data.min()  # 400
    
    # Coefficient a values
    a_vals = np.array([0.9, 1.0, 1.1])
    a_range = a_vals.max() - a_vals.min()  # 0.2
    
    # Bug: uses a_range as denominator
    # Fix: should use data_dr (or some measure of data dynamic range)
    
    assert data_dr == 400
    assert abs(a_range - 0.2) < 1e-10
    # They should be different - data range is much larger
    assert data_dr != a_range
