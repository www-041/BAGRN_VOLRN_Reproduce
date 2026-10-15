import json
import hashlib

from src.benchmark_v1_spectralmatch_phase2 import freeze_audit
from src.benchmark_v1_spectralmatch_phase2.freeze_audit import audit_frozen_inputs


def test_freeze_audit_classifies_positive_and_zero_overlap_pairs(tmp_path):
    root = tmp_path / "freeze"
    root.mkdir()
    (root / "freeze_lock.json").write_text(json.dumps({"files": []}), encoding="utf-8")
    (root / "frozen_manifest.json").write_text(json.dumps({
        "protocol_id": "DZ01V_UNIFIED_A_B_V1_20261013",
        "git_head": "locked",
        "status": "FROZEN",
        "scene_count": 13,
        "pair_counts": {"total_pairs": 3, "positive_overlap_pairs": 2, "no_valid_overlap_pairs": 1},
        "grid": {"crs": "EPSG:32650", "width": 4, "height": 4,
                 "transform": [14, 0, 0, 0, -14, 56]},
        "aligned_input_artifacts": [],
    }), encoding="utf-8")
    (root / "code_manifest.json").write_text(json.dumps({"git_head": "locked", "files": []}), encoding="utf-8")
    (root / "pair_overlap_inventory.csv").write_text(
        "pair_id,scene_i,scene_j,status,aligned_valid_overlap_pixels,days_apart,time_stratum\n"
        "00_01,a,b,OVERLAP,4,0,SAME_DATE\n00_10,a,k,OVERLAP,5,61,CROSS_DATE_GE_30D\n"
        "00_02,a,c,NO_VALID_OVERLAP,0,,UNKNOWN\n",
        encoding="utf-8",
    )

    report = audit_frozen_inputs(root, expected_head="locked", check_rasters=False)

    assert report["status"] == "PASS", report["errors"]
    assert report["pair_counts"] == {"total": 3, "positive_overlap": 2, "no_valid_overlap": 1}
    assert report["positive_pair_ids"] == ["00_01", "00_10"]


def test_freeze_audit_detects_pair_inventory_count_drift(tmp_path):
    root = tmp_path / "freeze"
    root.mkdir()
    (root / "freeze_lock.json").write_text(json.dumps({"files": []}), encoding="utf-8")
    (root / "frozen_manifest.json").write_text(json.dumps({
        "protocol_id": "DZ01V_UNIFIED_A_B_V1_20261013", "status": "FROZEN",
        "git_head": "locked",
        "scene_count": 13,
        "pair_counts": {"total_pairs": 78, "positive_overlap_pairs": 48, "no_valid_overlap_pairs": 30},
        "grid": {"crs": "EPSG:32650", "width": 4, "height": 4,
                 "transform": [14, 0, 0, 0, -14, 56]},
        "aligned_input_artifacts": [],
    }), encoding="utf-8")
    (root / "code_manifest.json").write_text(json.dumps({"git_head": "locked", "files": []}), encoding="utf-8")
    (root / "pair_overlap_inventory.csv").write_text(
        "pair_id,scene_i,scene_j,status,aligned_valid_overlap_pixels\n00_01,a,b,OVERLAP,4\n", encoding="utf-8")

    report = audit_frozen_inputs(root, expected_head="locked", check_rasters=False)

    assert report["status"] == "FREEZE_INVALID"
    assert "PAIR_INVENTORY_COUNT_MISMATCH" in report["errors"]


def test_method_git_head_change_is_informational_when_locked_content_matches(tmp_path, monkeypatch):
    root = tmp_path / "freeze"
    root.mkdir()
    (root / "freeze_lock.json").write_text(json.dumps({"files": []}), encoding="utf-8")
    frozen = {
        "protocol_id": "DZ01V_UNIFIED_A_B_V1_20261013", "status": "FROZEN", "git_head": "locked",
        "scene_count": 13, "pair_counts": {"total_pairs": 2, "positive_overlap_pairs": 2, "no_valid_overlap_pairs": 0},
        "grid": {"crs": "EPSG:32650", "width": 4, "height": 4, "transform": [14, 0, 0, 0, -14, 56]},
        "aligned_input_artifacts": [],
    }
    code = {"git_head": "locked", "files": []}
    (root / "frozen_manifest.json").write_text(json.dumps(frozen), encoding="utf-8")
    (root / "code_manifest.json").write_text(json.dumps(code), encoding="utf-8")
    (root / "pair_overlap_inventory.csv").write_text(
        "pair_id,scene_i,scene_j,status,aligned_valid_overlap_pixels\n00_01,a,b,OVERLAP,4\n00_10,a,c,OVERLAP,4\n", encoding="utf-8"
    )
    monkeypatch.setattr(freeze_audit, "_git_head", lambda: "method-head")

    report = audit_frozen_inputs(root, check_rasters=False)

    assert report["status"] == "PASS", report["errors"]
    assert report["locked_git_head"] == "locked"
    assert report["method_git_head"] == "method-head"
    assert "GIT_HEAD_MISMATCH" not in report["errors"]


def test_locked_source_hash_drift_still_fails(tmp_path, monkeypatch):
    root = tmp_path / "freeze"
    root.mkdir()
    (root / "freeze_lock.json").write_text(json.dumps({"files": []}), encoding="utf-8")
    locked_file = tmp_path / "locked.py"
    locked_file.write_text("frozen", encoding="utf-8")
    code = {"git_head": "locked", "files": [{"path": str(locked_file), "sha256": "0" * 64}]}
    code_path = root / "code_manifest.json"
    code_path.write_text(json.dumps(code), encoding="utf-8")
    frozen = {
        "protocol_id": "DZ01V_UNIFIED_A_B_V1_20261013", "status": "FROZEN", "git_head": "locked",
        "code_manifest_sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
        "scene_count": 13, "pair_counts": {"total_pairs": 1, "positive_overlap_pairs": 1, "no_valid_overlap_pairs": 0},
        "grid": {"crs": "EPSG:32650", "width": 4, "height": 4, "transform": [14, 0, 0, 0, -14, 56]},
        "aligned_input_artifacts": [],
    }
    (root / "frozen_manifest.json").write_text(json.dumps(frozen), encoding="utf-8")
    (root / "pair_overlap_inventory.csv").write_text(
        "pair_id,scene_i,scene_j,status,aligned_valid_overlap_pixels\n00_01,a,b,OVERLAP,4\n", encoding="utf-8"
    )
    monkeypatch.setattr(freeze_audit, "_git_head", lambda: "method-head")

    report = audit_frozen_inputs(root, check_rasters=False)

    assert report["status"] == "FREEZE_INVALID"
    assert any(error.startswith("CODE_FILE_HASH_MISMATCH") for error in report["errors"])
