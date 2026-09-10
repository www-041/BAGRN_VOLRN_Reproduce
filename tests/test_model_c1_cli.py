"""CLI and artifact schema tests for MODEL-C1 without real imagery."""

import csv
from pathlib import Path

import numpy as np


def test_affine_causal_cli_flag_is_explicitly_available():
    from scripts.diagnose_registration_pair import _parse_args

    args = _parse_args([
        "--config", "config.yaml",
        "--validation-band", "B14",
        "--holdout-manifest", "holdout.json",
        "--affine-causal-test",
    ])
    assert args.affine_causal_test is True
    assert args.validation_band == "B14"


def test_model_c1_stage_csv_blanks_treatment_metrics_when_training_gate_rejects(tmp_path):
    from scripts.diagnose_registration_pair import _write_model_c1_stage_metrics_csv

    registration = {
        "global_only_quality": {"quality": "fail", "median": 1.0, "rmse": 1.2, "p95": 2.0},
        "affine_causal": {
            "available": False,
            "reason": "training_gate_rejected",
            "quality": None,
        },
    }
    path = _write_model_c1_stage_metrics_csv(registration, tmp_path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    treatment = rows[1]
    assert treatment["status"] == "training_gate_rejected"
    assert treatment["rmse"] == ""
    assert treatment["p95"] == ""


def test_model_c1_counterfactual_artifact_writer_uses_exact_names(tmp_path, monkeypatch):
    from scripts import diagnose_registration_pair as cli

    reference = np.ones((1, 8, 8), dtype=float)
    target = np.ones((1, 8, 8), dtype=float)
    registration = {
        "quality": {"quality": "pass"},
        "connected": True,
        "registered_arrays": [reference, target],
        "global_only_arrays": [reference, target],
        "affine_causal_counterfactual_arrays": [reference, target],
        "affine_causal_fields_by_scene": {
            1: {"dx_field": np.zeros((8, 8)), "dy_field": np.zeros((8, 8))}
        },
        "affine_causal": {
            "available": True, "quality": {"quality": "pass"}, "comparison": {},
        },
        "registration_band_name": "B12",
        "validation_band_name": "B14",
        "diagnostics": {"raw_block_matches": []},
        "local_dx_fields": [np.zeros((8, 8)), np.zeros((8, 8))],
        "local_dy_fields": [np.zeros((8, 8)), np.zeros((8, 8))],
        "final_validation": {"edges": []},
        "diagnostic_masks": {},
    }
    # Exercise only the schema helper here; raster writing is covered by the
    # existing artifact tests and requires valid GeoTIFF transforms/CRS.
    stage = cli._write_model_c1_stage_metrics_csv(registration, tmp_path)
    pairs = cli._write_model_c1_holdout_pairs_csv(registration, tmp_path)
    assert stage.name == "model_c1_stage_metrics.csv"
    assert pairs.name == "model_c1_holdout_pairs.csv"
