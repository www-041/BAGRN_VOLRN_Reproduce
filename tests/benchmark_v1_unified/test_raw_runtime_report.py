from scripts.benchmark.run_raw_baseline_v1 import build_runtime_report
from tests.benchmark_v1_unified.test_submission_scoring import make_thirteen_scene_fixture
import scripts.benchmark.run_raw_baseline_v1 as raw_baseline
import json
import sys


def test_runtime_report_retains_measurements_and_scratch_on_failure(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "partial-sort.f64").write_bytes(b"01234567")

    class ActiveMonitor:
        def report(self):
            return {"status": "MEASURED", "peak_rss_bytes": 1234, "rss_sample_count": 4,
                    "hard_os_memory_limit": False}

    result = build_runtime_report(
        stage_reports={"mosaic": {"status": "MEASURED", "peak_rss_bytes": 2048}},
        active_stage="evaluator", active_monitor=ActiveMonitor(), scratch_dir=scratch,
        failure={"status": "STOPPED", "error": "RuntimeError:forced"},
    )

    assert result["stage_measurements"]["mosaic"]["peak_rss_bytes"] == 2048
    assert result["stage_measurements"]["evaluator"]["rss_sample_count"] == 4
    assert result["scratch"]["file_count"] == 1
    assert result["scratch"]["bytes"] == 8
    assert result["scratch"]["cleanup_status"] == "RETAINED_ON_FAILURE"
    assert result["hard_os_memory_limit"] is False
    assert result["failure"]["status"] == "STOPPED"


def test_raw_runner_persists_monitor_report_and_partial_scratch_after_mosaic_error(tmp_path, monkeypatch):
    freeze, _, _ = make_thirteen_scene_fixture(tmp_path)
    output = tmp_path / "raw-output"
    monkeypatch.setattr(raw_baseline, "_preflight", lambda *args, **kwargs: {
        "status": "PASS", "scene_ids": [f"SCENE_{i:02d}" for i in range(13)],
        "grid": {"shape": [128, 128], "transform": [14, 0, 633346, 0, -14, 3359874], "crs": "EPSG:32650"},
        "fixed_priority": list(range(13)),
    })

    def fail_during_mosaic(*args, **kwargs):
        scratch = output / "scratch"
        scratch.mkdir(exist_ok=True)
        (scratch / "partial-run.tmp").write_bytes(b"partial")
        raise RuntimeError("synthetic mosaic stop")

    monkeypatch.setattr(raw_baseline, "build_fixed_source_mosaic", fail_during_mosaic)
    monkeypatch.setattr(sys, "argv", ["run_raw_baseline_v1", "--freeze", str(freeze),
                                       "--output-root", str(output)])

    assert raw_baseline.main() == 2
    runtime = json.loads((output / "runtime.json").read_text(encoding="utf-8"))
    assert runtime["stage_measurements"]["mosaic"]["rss_sample_count"] > 0
    assert runtime["scratch"]["cleanup_status"] == "RETAINED_ON_FAILURE"
    assert runtime["scratch"]["bytes"] == len(b"partial")
    assert runtime["hard_os_memory_limit"] is False
