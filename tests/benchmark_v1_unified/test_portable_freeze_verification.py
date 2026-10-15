import hashlib
import json
from pathlib import Path

from src.benchmark_v1_unified.portable_freeze_verification import verify_portable_lock
from src.benchmark_v1_unified.portable_freeze_verification import as_freeze_verification
from src.benchmark_v1_unified.streaming_submission_scoring import evaluate_submission_streaming
from tests.benchmark_v1_unified.test_submission_scoring import make_thirteen_scene_fixture


def _lock(path, entries):
    path.write_text(json.dumps({"schema_version": 1, "files": entries}, indent=2), encoding="utf-8")


def test_two_windows_roots_map_and_verify_exact_hashes_without_lock_mutation(tmp_path):
    repo = tmp_path / "repo-root"
    data = tmp_path / "data-root"
    (repo / "docs").mkdir(parents=True)
    (data / "scene").mkdir(parents=True)
    file_a, file_b = repo / "docs" / "frozen.txt", data / "scene" / "input.tif"
    file_a.write_bytes(b"frozen-code")
    file_b.write_bytes(b"frozen-raster-bytes")
    lock = tmp_path / "freeze_lock.json"
    entries = [
        {"path": r"D:\old\repo\docs\frozen.txt", "sha256": hashlib.sha256(file_a.read_bytes()).hexdigest()},
        {"path": r"E:\old\dataset\scene\input.tif", "sha256": hashlib.sha256(file_b.read_bytes()).hexdigest()},
    ]
    _lock(lock, entries)
    before = hashlib.sha256(lock.read_bytes()).hexdigest()
    mappings = [
        {"source_root": r"D:\old\repo", "target_root": str(repo)},
        {"source_root": r"E:\old\dataset", "target_root": str(data)},
    ]

    result = verify_portable_lock(lock, mappings)

    assert result["status"] == "PASS"
    assert result["locked_file_count"] == result["matched_file_count"] == 2
    assert hashlib.sha256(lock.read_bytes()).hexdigest() == before


def test_missing_mapping_fails_closed(tmp_path):
    lock = tmp_path / "freeze_lock.json"
    _lock(lock, [{"path": r"D:\input\x.tif", "sha256": "0" * 64}])
    result = verify_portable_lock(lock, [])
    assert result["status"] == "FREEZE_PATH_MAPPING_INCOMPLETE"
    assert result["matched_file_count"] == 0


def test_ambiguous_mapping_fails_closed(tmp_path):
    lock = tmp_path / "freeze_lock.json"
    target = tmp_path / "target"
    target.mkdir()
    _lock(lock, [{"path": r"D:\source\x", "sha256": "0" * 64}])
    mappings = [{"source_root": r"D:\source", "target_root": str(target)},
                {"source_root": r"D:\source", "target_root": str(target)}]
    result = verify_portable_lock(lock, mappings)
    assert result["status"] == "FREEZE_PATH_MAPPING_AMBIGUOUS"


def test_hash_mismatch_and_traversal_are_rejected(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "safe.txt").write_bytes(b"changed")
    lock = tmp_path / "freeze_lock.json"
    _lock(lock, [
        {"path": r"D:\source\safe.txt", "sha256": "0" * 64},
        {"path": r"D:\source\..\escape.txt", "sha256": "0" * 64},
    ])
    result = verify_portable_lock(lock, [{"source_root": r"D:\source", "target_root": str(target)}])
    assert result["status"] == "FREEZE_PATH_MAPPING_INVALID"
    assert any(item["status"] == "HASH_MISMATCH" for item in result["files"])
    assert any(item["status"] == "UNSAFE_PATH" for item in result["files"])


def test_portable_verification_adapter_can_authorize_evaluator_without_rewriting_hashes(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    lock_path = freeze / "freeze_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for entry in lock["files"]:
        relative = Path(entry["path"]).relative_to(tmp_path)
        entry["path"] = str(Path("Z:/frozen-root").joinpath(*relative.parts))
    original_hashes = [entry["sha256"] for entry in lock["files"]]
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    portable = verify_portable_lock(lock_path, [{"source_root": "Z:/frozen-root", "target_root": str(tmp_path)}])

    result = evaluate_submission_streaming(
        freeze, submission, submission_dir=tmp_path / "method", regimes=("B1",),
        scratch_dir=tmp_path / "scratch", freeze_verifier=lambda _root: as_freeze_verification(portable),
    )

    assert portable["status"] == "PASS"
    assert [entry["sha256"] for entry in lock["files"]] == original_hashes
    assert result["status"] == "EVALUATED"
    assert result["B1"]["status"] == "EVALUABLE"
