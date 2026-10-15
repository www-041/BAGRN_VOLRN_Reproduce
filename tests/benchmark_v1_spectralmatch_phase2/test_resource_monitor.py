import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import ctypes


def _pid_running(pid):
    kernel = ctypes.windll.kernel32
    handle = kernel.OpenProcess(0x00100000 | 0x001000, False, int(pid))
    if not handle:
        return False
    try:
        return kernel.WaitForSingleObject(handle, 0) == 0x00000102
    finally:
        kernel.CloseHandle(handle)

from src.benchmark_v1_spectralmatch_phase2 import resource_monitor


class _FakeProcess:
    def __init__(self, pid, child_pid_file, trigger_file):
        self.pid = int(pid)
        self.child_pid_file = Path(child_pid_file)
        self.trigger_file = Path(trigger_file) if trigger_file else None

    def children(self, recursive=True):
        if not self.child_pid_file.exists():
            return []
        try:
            pid = int(self.child_pid_file.read_text().strip())
            if not _pid_running(pid):
                return []
            return [_FakeProcess(pid, self.child_pid_file, self.trigger_file)]
        except (ValueError, OSError, SystemError):
            return []

    def memory_info(self):
        value = 100 if self.trigger_file is None or self.trigger_file.exists() else 10
        return type("Mem", (), {"rss": value})()

    def terminate(self):
        try:
            os.kill(self.pid, signal.SIGTERM)
        except (OSError, SystemError):
            pass

    kill = terminate

    def is_running(self):
        return _pid_running(self.pid)


def _fake_psutil(child_pid_file, trigger_file=None):
    return type(
        "FakePsutil",
        (),
        {
            "Process": staticmethod(lambda pid: _FakeProcess(pid, child_pid_file, trigger_file)),
            "NoSuchProcess": ProcessLookupError,
            "AccessDenied": PermissionError,
        },
    )


def _tree_worker_command(child_pid_file, heartbeat):
    child_code = (
        "import pathlib,time; f=pathlib.Path(r'" + str(heartbeat) + "').open('a'); "
        "exec('while True: f.write(\"x\"); f.flush(); time.sleep(.04)')"
    )
    parent_code = (
        "import pathlib,subprocess,sys,time; c=subprocess.Popen([sys.executable,'-c',sys.argv[1]]); "
        "pathlib.Path(sys.argv[2]).write_text(str(c.pid)); time.sleep(30)"
    )
    return [sys.executable, "-c", parent_code, child_code, str(child_pid_file)]


def test_missing_psutil_fails_closed_before_worker_launch(tmp_path, monkeypatch):
    sentinel = tmp_path / "started.txt"
    monkeypatch.setattr(resource_monitor, "_load_psutil", lambda: None)
    result = resource_monitor.run_monitored(
        [sys.executable, "-c", f"open({str(sentinel)!r}, 'w').write('started')"],
        cwd=tmp_path, env=os.environ.copy(), timeout_s=1, rss_limit_bytes=1000,
        sample_interval_s=0.02, output_dir=tmp_path / "out",
    )
    assert result["status"] == "MONITOR_UNAVAILABLE"
    assert not sentinel.exists()
    assert result["rss_sample_count"] == 0


def test_normal_exit_reports_measured_rss(tmp_path, monkeypatch):
    monkeypatch.setattr(resource_monitor, "_load_psutil", lambda: _fake_psutil(tmp_path / "child.pid"))
    result = resource_monitor.run_monitored(
        [sys.executable, "-c", "print('ok')"], cwd=tmp_path, env=os.environ.copy(),
        timeout_s=2, rss_limit_bytes=1000, sample_interval_s=0.02, output_dir=tmp_path / "out",
    )
    assert result["status"] == "COMPLETED"
    assert result["peak_rss_bytes"] >= 100
    assert result["rss_sample_count"] > 0
    assert result["os_hard_memory_limit_applied"] is False


def test_runtime_zero_process_samples_fails_closed_and_terminates(tmp_path, monkeypatch):
    child_pid_file = tmp_path / "child.pid"
    monkeypatch.setattr(resource_monitor, "_load_psutil", lambda: _fake_psutil(child_pid_file))
    sentinel = tmp_path / "started.txt"

    def no_samples_after_start(root, psutil):
        deadline = time.monotonic() + 2
        while not sentinel.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        return 0, 0, [root]

    monkeypatch.setattr(resource_monitor, "_sample_tree", no_samples_after_start)
    result = resource_monitor.run_monitored(
        [sys.executable, "-c", f"import time; open({str(sentinel)!r}, 'w').write('started'); time.sleep(5)"],
        cwd=tmp_path, env={**os.environ, "PYTHONSTARTUP": ""}, timeout_s=2,
        rss_limit_bytes=1000, sample_interval_s=0.02, output_dir=tmp_path / "out",
    )
    assert sentinel.exists()
    assert result["status"] == "MONITOR_ERROR"
    assert result["termination_reason"] == "rss_sampling_unavailable"
    assert result["rss_sample_count"] == 0
    assert result["process_tree_terminated"] is True


def test_process_handle_initialization_failure_terminates_descendant(tmp_path, monkeypatch):
    child_pid_file = tmp_path / "child.pid"
    heartbeat = tmp_path / "heartbeat.txt"

    def broken_process(pid):
        deadline = time.monotonic() + 3
        while not child_pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        raise RuntimeError("psutil process handle unavailable")

    broken_psutil = type("BrokenPsutil", (), {
        "Process": staticmethod(broken_process),
        "NoSuchProcess": ProcessLookupError,
        "AccessDenied": PermissionError,
    })
    monkeypatch.setattr(resource_monitor, "_load_psutil", lambda: broken_psutil)
    result = resource_monitor.run_monitored(
        _tree_worker_command(child_pid_file, heartbeat), cwd=tmp_path, env=os.environ.copy(),
        timeout_s=3, rss_limit_bytes=10000, sample_interval_s=0.02, output_dir=tmp_path / "out",
    )
    assert child_pid_file.exists()
    child_pid = int(child_pid_file.read_text())
    assert result["status"] == "MONITOR_ERROR"
    assert result["process_tree_terminated"] is True
    assert not _pid_running(child_pid)


def test_timeout_terminates_descendant_process(tmp_path, monkeypatch):
    child_pid_file = tmp_path / "child.pid"
    heartbeat = tmp_path / "heartbeat.txt"
    monkeypatch.setattr(resource_monitor, "_load_psutil", lambda: _fake_psutil(child_pid_file, heartbeat))
    result = resource_monitor.run_monitored(
        _tree_worker_command(child_pid_file, heartbeat), cwd=tmp_path, env=os.environ.copy(),
        timeout_s=0.7, rss_limit_bytes=10000, sample_interval_s=0.03, output_dir=tmp_path / "out",
    )
    assert result["status"] == "TIMEOUT"
    assert result["process_tree_terminated"] is True
    before = heartbeat.stat().st_size
    time.sleep(0.15)
    assert heartbeat.stat().st_size == before


def test_rss_threshold_terminates_process_tree(tmp_path, monkeypatch):
    child_pid_file = tmp_path / "child.pid"
    heartbeat = tmp_path / "heartbeat.txt"
    monkeypatch.setattr(resource_monitor, "_load_psutil", lambda: _fake_psutil(child_pid_file, heartbeat))
    result = resource_monitor.run_monitored(
        _tree_worker_command(child_pid_file, heartbeat), cwd=tmp_path, env=os.environ.copy(),
        timeout_s=3, rss_limit_bytes=150, sample_interval_s=0.03, output_dir=tmp_path / "out",
    )
    assert result["status"] == "RESOURCE_LIMIT"
    assert result["process_tree_terminated"] is True
    before = heartbeat.stat().st_size
    time.sleep(0.15)
    assert heartbeat.stat().st_size == before
