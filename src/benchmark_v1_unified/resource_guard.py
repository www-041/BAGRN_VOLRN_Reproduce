"""Fail-closed in-process RSS polling and scratch high-water telemetry."""
from __future__ import annotations

import threading
import time
from pathlib import Path


class ResourceLimitExceeded(RuntimeError):
    pass


class ProcessResourceMonitor:
    """Measure this process tree and stop cooperatively at a polling ceiling.

    This is a sampled software guard, not an operating-system memory limit.
    Long-running loops must call ``check`` between tiles/chunks.
    """

    def __init__(self, max_rss_bytes: int, scratch_paths=(), *, sample_interval: float = 0.2):
        if int(max_rss_bytes) < 1:
            raise ValueError("max_rss_bytes must be positive")
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        self.max_rss_bytes = int(max_rss_bytes)
        self.scratch_paths = [Path(p) for p in scratch_paths]
        self.sample_interval = float(sample_interval)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._process = None
        self._started = None
        self._peak_rss = 0
        self._samples = 0
        self._scratch_peak = 0
        self._triggered = False
        self._error = None
        self._finished = False

    def __enter__(self):
        try:
            import psutil
        except Exception as exc:
            raise RuntimeError(f"RSS_MONITOR_UNAVAILABLE:{exc}") from exc
        self._process = psutil.Process()
        self._started = time.perf_counter()
        self._sample_once()
        self._thread = threading.Thread(target=self._poll, name="unified-v1-rss-monitor", daemon=True)
        self._thread.start()
        return self

    def _scratch_size(self):
        total = 0
        for root in self.scratch_paths:
            if root.is_file():
                try:
                    total += root.stat().st_size
                except OSError:
                    pass
            elif root.is_dir():
                for path in root.rglob("*"):
                    if path.is_file():
                        try:
                            total += path.stat().st_size
                        except OSError:
                            pass
        return total

    def _sample_once(self):
        try:
            rss = int(self._process.memory_info().rss)
            for child in self._process.children(recursive=True):
                try:
                    rss += int(child.memory_info().rss)
                except Exception:
                    continue
            scratch = self._scratch_size()
            with self._lock:
                self._samples += 1
                self._peak_rss = max(self._peak_rss, rss)
                self._scratch_peak = max(self._scratch_peak, scratch)
                if rss > self.max_rss_bytes:
                    self._triggered = True
        except Exception as exc:
            with self._lock:
                self._error = f"RSS_MONITOR_ERROR:{type(exc).__name__}:{exc}"

    def _poll(self):
        while not self._stop.wait(self.sample_interval):
            self._sample_once()

    def check(self):
        with self._lock:
            error = self._error
            triggered = self._triggered
        if error:
            raise RuntimeError(error)
        if triggered:
            raise ResourceLimitExceeded("RSS_POLLING_LIMIT_TRIGGERED")

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval * 4))
        self._sample_once()
        self._finished = True

    def report(self):
        with self._lock:
            error = self._error
            triggered = self._triggered
            samples = self._samples
            peak = self._peak_rss
            scratch_peak = self._scratch_peak
        status = "MONITOR_ERROR" if error or samples == 0 else "RSS_LIMIT_TRIGGERED" if triggered else "MEASURED"
        return {
            "status": status,
            "peak_rss_bytes": peak if samples else None,
            "rss_sample_count": samples,
            "rss_measurement_source": "psutil.ProcessTree.memory_info.rss",
            "max_rss_bytes": self.max_rss_bytes,
            "threshold_triggered": triggered,
            "scratch_peak_bytes": scratch_peak,
            "scratch_measurement_source": "filesystem_size_polling",
            "hard_os_memory_limit": False,
            "wall_seconds": None if self._started is None else time.perf_counter() - self._started,
            "wall_measurement_source": "time.perf_counter",
            "error": error,
        }
