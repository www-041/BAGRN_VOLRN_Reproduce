"""Single-pair Phase5 CLI, persistence order, and resource-limit tests."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from affine import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_benchmark_v1_phase5_worker as worker
import benchmark_v1_phase5.runner as runner
from benchmark_v1_phase5.inputs import MatchView, PairInput, ViewProvenance
from benchmark_v1_phase5.protocol import PairManifest, load_config


def _config_path():
    return ROOT / "configs/benchmark_v1_phase5.yaml"


def _pair():
    i = Affine(14, 0, 100, 0, -14, 300)
    j = Affine(14, 0, 128, 0, -14, 286)
    view = MatchView(np.ones((8, 8), np.float32), np.ones((8, 8), np.float32),
                     np.ones((8, 8), bool), np.ones((8, 8), bool), np.ones((8, 8), bool),
                     ViewProvenance(i, j, i, 1, 1, (2, 98), (0, 1), (0, 1),
                                    "bilinear", "nearest"))
    return PairInput("00_01", "READY", view, i, j, 64, 64)


def _manifest(tmp_path):
    i, j = tmp_path / "i.tif", tmp_path / "j.tif"
    return PairManifest("00_01", "scene_i", "scene_j", i, j, 1, 1, "a" * 64, "b" * 64)


def test_cli_rejects_implicit_mode_missing_pair_method_and_unknown_method():
    from run_benchmark_v1_phase5 import build_parser, validate_args
    parser = build_parser()
    with pytest.raises(SystemExit):
        validate_args(parser.parse_args(["--config", str(_config_path())]))
    with pytest.raises(SystemExit):
        validate_args(parser.parse_args(["--config", str(_config_path()), "--synthetic"]))
    with pytest.raises(SystemExit):
        validate_args(parser.parse_args(["--config", str(_config_path()), "--synthetic",
                                         "--pair", "00_01"]))
    with pytest.raises(SystemExit):
        validate_args(parser.parse_args(["--config", str(_config_path()), "--synthetic",
                                         "--pair", "00_01", "--method", "NOT_A_METHOD"]))


def test_smoke_non00_pair_is_rejected_before_frozen_snapshot_read(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("B9 freeze must not be opened for an out-of-scope smoke")
    monkeypatch.setattr(runner, "_load_snapshot_manifest", forbidden)
    monkeypatch.setattr(runner, "OUTPUT_BASE", tmp_path)
    with pytest.raises(ValueError, match="00_01"):
        runner.run_one("02_03", "SIFT", _config_path(), tmp_path / "bad_smoke", "smoke")


def test_synthetic_run_persists_lock_and_translation_without_truth_or_overwrite(tmp_path,
                                                                                monkeypatch):
    monkeypatch.setattr(runner, "load_approved_development_truth",
                        lambda *_: (_ for _ in ()).throw(AssertionError("truth accessed")))
    runner.OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="phase5_test_", dir=runner.OUTPUT_BASE) as temp_root:
        out = Path(temp_root) / "synthetic_run"
        result = runner.run_one("00_01", "SIFT", _config_path(), out, "synthetic")
        assert result["execution_status"] == "SUCCESS"
        assert result["mode"] == "synthetic"
        for name in ("protocol_lock.json", "input_provenance.json", "method_result.json",
                     "resource_usage.json", "run_status.json"):
            assert (out / name).is_file()
        lock = json.loads((out / "protocol_lock.json").read_text(encoding="utf-8"))
        method_result = json.loads((out / "method_result.json").read_text(encoding="utf-8"))
        assert lock["config_sha256"] == load_config(_config_path()).config_sha256
        assert method_result["delta_row_col"] == pytest.approx([2.25, -3.5], abs=0.25)
        request = json.loads((out / "worker_request.json").read_text(encoding="utf-8"))
        serialized = json.dumps(request).lower()
        assert "test_truth" not in serialized and "checkpoint" not in serialized
        with pytest.raises(FileExistsError):
            runner.run_one("00_01", "SIFT", _config_path(), out, "synthetic")


def test_mutated_protocol_config_is_rejected_before_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUTPUT_BASE", tmp_path)
    changed = tmp_path / "changed.yaml"
    changed.write_text(_config_path().read_text(encoding="utf-8").replace(
        "max_view_side: 1600", "max_view_side: 1599"), encoding="utf-8")
    output = tmp_path / "mutated_run"
    with pytest.raises(ValueError, match="frozen truth-free protocol"):
        runner.run_one("00_01", "SIFT", changed, output, "synthetic")
    assert not output.exists()


def test_development_truth_is_loaded_only_after_transform_is_persisted(tmp_path, monkeypatch):
    out = tmp_path / "development_run"
    snapshot = object()
    manifest = _manifest(tmp_path)
    snapshot = SimpleNamespace(assignment_sha256="frozen-assignments",
                               audit={"output_sha256": {"split_leakage_audit.json": "frozen-audit"}})
    monkeypatch.setattr(runner, "OUTPUT_BASE", tmp_path)
    monkeypatch.setattr(runner, "_load_snapshot_manifest", lambda pair_id: (snapshot, manifest))
    monkeypatch.setattr(runner, "verify_manifest_bytes", lambda _manifest: None)

    def fake_worker(command, output_dir, timeout_s, rss_gib):
        result = {"method": "SIFT", "pair_id": "00_01", "status": "SUCCESS",
                  "delta_row_col": [0.0, 0.0], "direction": "scene_i_to_scene_j",
                  "frame": "native", "translation_semantics": "full_native_i_to_j",
                  "elapsed_s": 0.1, "reason": None, "provenance": {}}
        runner._write_json_atomic(output_dir / "method_result.json", result)
        return {"status": "SUCCESS", "peak_rss_bytes": 123, "wall_seconds": 0.1}

    monkeypatch.setattr(runner, "_run_worker", fake_worker)
    monkeypatch.setattr(runner, "prepare_pair", lambda *_: _pair())

    def fake_truth(loaded_snapshot):
        assert loaded_snapshot is snapshot
        assert (out / "method_result.json").is_file()
        return {"00_01": [{"point_id": "p1", "pair_id": "00_01", "split": "DEVELOPMENT",
                            "human_review_status": "APPROVED", "row_i": 1, "col_i": 2,
                            "row_j": 1, "col_j": 2}]}

    monkeypatch.setattr(runner, "load_approved_development_truth", fake_truth)
    result = runner.run_one("00_01", "SIFT", _config_path(), out, "development")
    assert result["execution_status"] == "SUCCESS"
    assert (out / "development_metrics.json").is_file()
    assert (out / "georef_only_metrics.json").is_file()


@pytest.mark.parametrize("mode,expected", [("timeout", "ALGORITHM_FAIL"),
                                             ("rss", "ALGORITHM_FAIL"),
                                             ("rss_unavailable", "ENV_UNAVAILABLE")])
def test_process_monitor_classifies_timeout_and_rss_budget(mode, expected, tmp_path, monkeypatch):
    class FakeProcess:
        pid = 99999
        def __init__(self): self.killed = False
        def poll(self): return None if not self.killed else -9
        def terminate(self): self.killed = True
        def kill(self): self.killed = True
        def wait(self, timeout=None): return -9
    fake = FakeProcess()
    class FakePopen:
        def __new__(cls, *args, **kwargs): return fake
    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    times = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(times, 2.0))
    if mode == "timeout":
        monkeypatch.setattr(runner, "_process_rss_bytes", lambda _pid: 0)
        timeout = 0.5
        ceiling = 10**12
    elif mode == "rss":
        monkeypatch.setattr(runner, "_process_rss_bytes", lambda _pid: 10**12)
        timeout = 50
        ceiling = 1
    else:
        monkeypatch.setattr(runner, "_process_rss_bytes", lambda _pid: None)
        timeout = 50
        ceiling = 10**12
    result = runner._run_process_with_limits(["fake"], tmp_path, timeout, ceiling)
    assert result["status"] == expected
    assert result["reason"].startswith("RESOURCE_")
    if mode == "rss_unavailable":
        assert result["peak_rss_bytes"] is None


def test_atomic_json_removes_temporary_file_if_replace_fails(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    monkeypatch.setattr(runner.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("blocked")))
    with pytest.raises(OSError):
        runner._write_json_atomic(path, {"frozen": True})
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_parent_rejects_worker_result_identity_or_schema_mismatch():
    valid = {"method": "SIFT", "pair_id": "00_01", "status": "SUCCESS",
             "delta_row_col": [1.0, 2.0], "direction": "scene_i_to_scene_j",
             "frame": "native", "translation_semantics": "full_native_i_to_j",
             "elapsed_s": 1.0, "reason": None, "provenance": {}}
    runner._validate_worker_result(valid, "SIFT", "00_01")
    with pytest.raises(ValueError, match="identity mismatch"):
        runner._validate_worker_result({**valid, "method": "LoFTR"}, "SIFT", "00_01")
    with pytest.raises(ValueError, match="schema mismatch"):
        runner._validate_worker_result({**valid, "test_points": []}, "SIFT", "00_01")


def test_standalone_worker_rejects_out_of_scope_and_existing_result_paths(tmp_path, monkeypatch):
    allowed = tmp_path / "phase5_runs"
    allowed.mkdir()
    monkeypatch.setattr(worker, "OUTPUT_BASE", allowed)
    outside = tmp_path / "outside"
    outside.mkdir()
    request = {"output_dir": str(outside)}
    with pytest.raises(ValueError, match="Phase5 output directory"):
        worker._validate_output_target(request)
    output = allowed / "one_run"
    output.mkdir()
    (output / "protocol_lock.json").write_text("{}", encoding="utf-8")
    (output / "worker_request.json").write_text("{}", encoding="utf-8")
    request["output_dir"] = str(output)
    assert worker._validate_output_target(request) == output.resolve()
    (output / "method_result.json").write_text("locked", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        worker._validate_output_target(request)


def test_standalone_worker_rejects_manifest_pair_id_mismatch_before_file_access(monkeypatch):
    request = {"pair_id": "00_01", "pair_manifest": {"pair_id": "02_03"}}
    with pytest.raises(ValueError, match="does not match frozen request"):
        worker._verify_frozen_b9_manifest(request)
