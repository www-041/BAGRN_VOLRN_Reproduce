"""Six-method preparation exposes DEVELOPMENT truth only and retains FAILs."""

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_phase4_freeze import load_frozen_trial
from benchmark_phase4_development import (METHODS, environment_preflight,
                                          evaluate_development, load_development_truth,
                                          normalize_result, prepare_trial_inputs,
                                          validate_manifest_destination)


def test_only_43_frozen_development_points_leave_truth_loader():
    snapshot = load_frozen_trial(ROOT)
    truth = load_development_truth(snapshot)
    assert {pair: len(rows) for pair, rows in truth.items()} == {
        "00_01": 11, "02_03": 9, "05_07": 9, "08_10": 14}
    test_ids = {key[1] for key, row in snapshot.assignments.items() if row["split"] == "TEST"}
    assert not test_ids.intersection({row["point_id"] for rows in truth.values() for row in rows})
    assert all(row["split"] == "DEVELOPMENT" for rows in truth.values() for row in rows)
    manifest = prepare_trial_inputs(snapshot, ROOT)
    assert len(manifest) == 4
    assert all(row["input_band"] == "B9" and row["crs"] == "EPSG:32650" for row in manifest)
    assert all("test_points" not in row for row in manifest)


def test_six_method_schema_preserves_failures_missing_pairs_and_model_direction():
    assert METHODS == ("SIFT", "KARIOS", "AROSICS_GLOBAL", "LoFTR",
                       "EfficientLoFTR", "LightGlue+DISK")
    success = normalize_result("SIFT", "00_01", "SUCCESS", (0.0, 0.0), 1.5, "",
                               {"frame": "native", "translation_semantics": "full_native_i_to_j"})
    assert success["direction"] == "scene_i_to_scene_j"
    failed = normalize_result("KARIOS", "00_01", "FAIL", None, 0.2, "NOT_INSTALLED", {})
    dev_truth = {"00_01": [{"point_id": "D1", "pair_id": "00_01", "split": "DEVELOPMENT",
                             "row_i": 0.0, "col_i": 0.0, "row_j": 1.0, "col_j": 0.0}],
                 "02_03": [], "05_07": [], "08_10": []}
    scored = evaluate_development([success, failed], dev_truth)
    assert len(scored["rows"]) == 24
    s = next(row for row in scored["rows"] if row["method"] == "SIFT" and row["pair_id"] == "00_01")
    assert s["metrics"]["rmse_px"] == pytest.approx(1.0)
    assert s["metrics"]["p95_px"] == pytest.approx(1.0)
    assert s["metrics"]["pck_1px"] == pytest.approx(1.0)
    f = next(row for row in scored["rows"] if row["method"] == "KARIOS" and row["pair_id"] == "00_01")
    assert f["status"] == "FAIL" and f["metrics"] is None
    assert sum(row["status"] == "NOT_ATTEMPTED" for row in scored["rows"]) == 22
    with pytest.raises(ValueError, match="local"):
        normalize_result("AROSICS_LOCAL", "00_01", "SUCCESS", (0, 0), 1.0, "", {})


def test_environment_preflight_does_not_claim_external_methods_passed():
    status = environment_preflight()
    assert status["KARIOS"]["end_to_end"] == "NOT_VERIFIED_END_TO_END"
    assert status["AROSICS_GLOBAL"]["end_to_end"] == "NOT_VERIFIED_END_TO_END"


def test_residual_shift_cannot_be_mislabeled_as_full_native_translation():
    with pytest.raises(ValueError, match="full native"):
        normalize_result("SIFT", "00_01", "SUCCESS", (0.0, 0.0), 1.0, "",
                         {"frame": "native", "translation_semantics": "residual_to_georeferenced_base"})
    result = normalize_result("SIFT", "00_01", "SUCCESS", (100.0, 200.0), 1.0, "",
                              {"frame": "native", "translation_semantics": "full_native_i_to_j"})
    score = evaluate_development([result], {"00_01": [
        {"point_id": "D1", "pair_id": "00_01", "split": "DEVELOPMENT",
         "row_i": 0.0, "col_i": 0.0, "row_j": 100.0, "col_j": 200.0}],
        "02_03": [], "05_07": [], "08_10": []})
    assert score["rows"][0]["metrics"]["rmse_px"] == 0.0
    forged = {**result, "direction": "scene_j_to_scene_i"}
    with pytest.raises(ValueError, match="Unnormalized"):
        evaluate_development([forged], {"00_01": [], "02_03": [], "05_07": [], "08_10": []})


def test_development_manifest_cannot_overwrite_frozen_files(tmp_path):
    with pytest.raises(ValueError, match="Phase4"):
        validate_manifest_destination(tmp_path / "docs/benchmark/v1/phase3/assignments.json", tmp_path)
    allowed = tmp_path / "docs/benchmark/v1/phase4/dev_manifest.json"
    assert validate_manifest_destination(allowed, tmp_path) == allowed.resolve()
    allowed.parent.mkdir(parents=True)
    allowed.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        validate_manifest_destination(allowed, tmp_path)
