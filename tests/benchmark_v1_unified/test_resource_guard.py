from __future__ import annotations

import time
import pytest


def test_resource_monitor_samples_rss_and_scratch_usage(tmp_path):
    from src.benchmark_v1_unified.resource_guard import ProcessResourceMonitor

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "probe.bin").write_bytes(b"1234567890")
    with ProcessResourceMonitor(512 * 1024**2, [scratch], sample_interval=0.05) as monitor:
        monitor.check()
        time.sleep(0.12)
    report = monitor.report()
    assert report["status"] == "MEASURED"
    assert report["rss_sample_count"] >= 2
    assert report["peak_rss_bytes"] > 0
    assert report["scratch_peak_bytes"] >= 10
    assert report["hard_os_memory_limit"] is False


def test_resource_monitor_stops_at_polling_threshold_without_claiming_hard_limit(tmp_path):
    from src.benchmark_v1_unified.resource_guard import ProcessResourceMonitor, ResourceLimitExceeded

    with ProcessResourceMonitor(1, [tmp_path], sample_interval=0.01) as monitor:
        time.sleep(0.03)
        with pytest.raises(ResourceLimitExceeded, match="RSS_POLLING_LIMIT_TRIGGERED"):
            monitor.check()
    report = monitor.report()
    assert report["status"] == "RSS_LIMIT_TRIGGERED"
    assert report["threshold_triggered"] is True
    assert report["hard_os_memory_limit"] is False
