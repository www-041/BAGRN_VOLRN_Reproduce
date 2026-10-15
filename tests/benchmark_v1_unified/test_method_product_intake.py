from src.benchmark_v1_unified.method_products import prepare_method_submission
from tests.benchmark_v1_unified.test_submission_scoring import make_thirteen_scene_fixture


def test_raw_intake_uses_frozen_scene_paths_and_shared_hard_mosaic_products(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)

    result = prepare_method_submission({
        "method_id": "RAW", "run_id": "fixture", "method_kind": "FULL_MOSAIC",
        "status": "COMPLETED", "science_status": "NOT_APPLICABLE",
        "output_mosaic_path": submission["output_mosaic_path"],
        "output_mosaic_mask_path": submission["output_mosaic_mask_path"],
        "source_label_path": submission["source_label_path"], "seam_type": "HARD",
    }, freeze)

    assert result["status"] == "READY_FOR_SCORING"
    assert result["submission"]["corrected_scene_paths"] == {
        scene: str(freeze / "aligned_inputs" / f"scene_{i:03d}_B9_input.tif")
        for i, scene in enumerate(submission["corrected_scene_paths"])
    }


def test_bagrn_full_scene_intake_requires_all_outputs_and_never_substitutes_identity(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    config = {**submission, "method_id": "BAGRN", "run_id": "fixture", "method_kind": "FULL_MOSAIC",
              "science_status": "VALID"}
    config["corrected_scene_paths"] = dict(list(config["corrected_scene_paths"].items())[:-1])

    result = prepare_method_submission(config, freeze)

    assert result["status"] == "NOT_READY"
    assert any(reason.startswith("MISSING_SCENE_OUTPUTS") for reason in result["reasons"])


def test_lirrn_pair_only_is_kept_diagnostic_and_not_expanded_to_13_scene(tmp_path):
    freeze, base, _ = make_thirteen_scene_fixture(tmp_path)
    result = prepare_method_submission({"method_id": "LIRRN", "run_id": "pair", "method_kind": "PAIR_ONLY",
                                       "status": "COMPLETED", "pair_specific_outputs": {"00_01": base["corrected_scene_paths"]["SCENE_01"]}},
                                      freeze)
    assert result["status"] == "PAIR_ONLY_DIAGNOSTIC"
    assert result["submission"]["corrected_scene_paths"] == {}
    assert result["submission"]["method_kind"] == "PAIR_ONLY"


def test_unverified_or_disconnected_lsstaa_stays_not_ready(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    config = {**submission, "method_id": "LSSTAA", "run_id": "proxy", "method_kind": "SCENE_OUTPUTS",
              "science_status": "LSSTAA_PY_ADAPTED_DN_PROXY_UNVERIFIED", "status": "COMPLETED"}

    result = prepare_method_submission(config, freeze)

    assert result["status"] == "NOT_READY"
    assert "SCIENCE_STATUS_UNVERIFIED" in result["reasons"]
