"""Protocol and paired-HOLDOUT comparison tests for MODEL-C1."""

from pathlib import Path
import json

from src.experiment_config import load_config
from src.registration_model_c1 import (
    compare_fixed_holdout_validations,
    validate_model_c1_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def test_model_c1_protocol_accepts_frozen_b12_b14_manifest():
    config = load_config(str(ROOT / "configs" / "dz01_model_c1_affine_b12.yaml"))
    manifest = json.loads(
        (ROOT / "configs" / "dz01_model_c1_holdout_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    result = validate_model_c1_protocol(config, manifest, validation_band="B14")
    assert result["valid"] is True
    assert result["reserved_count"] == 7


def test_fixed_holdout_comparison_pairs_all_measurable_blocks_and_tracks_transitions():
    global_validation = {"edges": [{"idx_i": 0, "idx_j": 1, "blocks": [
        {"validation_row": 10, "validation_col": 20, "residual_magnitude": 2.0, "accepted": True},
        {"validation_row": 30, "validation_col": 40, "residual_magnitude": 4.0, "accepted": False},
    ]}]}
    affine_validation = {"edges": [{"idx_i": 0, "idx_j": 1, "blocks": [
        {"validation_row": 10, "validation_col": 20, "residual_magnitude": 1.0, "accepted": True},
        {"validation_row": 30, "validation_col": 40, "residual_magnitude": 3.0, "accepted": True},
    ]}]}

    result = compare_fixed_holdout_validations(global_validation, affine_validation)

    assert result["n_paired"] == 2
    assert result["n_common_accepted"] == 1
    assert result["all_measurable"]["global"]["n"] == 2
    assert result["all_measurable"]["affine"]["n"] == 2
    assert result["status_transitions"][1]["affine_status"] == "accepted"
