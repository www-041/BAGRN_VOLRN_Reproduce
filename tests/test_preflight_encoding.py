import io
import sys


def test_preflight_summary_is_safe_on_gbk_console(monkeypatch):
    from scripts import preflight_mosaic_series

    monkeypatch.setattr(
        preflight_mosaic_series, "preflight_mosaic_series",
        lambda *args: {
            "config": "config.yaml", "band": "B14", "total_scenes": 2,
            "scene_counts": [2], "checks": {"connected": True},
        },
    )
    stream = io.TextIOWrapper(io.BytesIO(), encoding="gbk", write_through=True)
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "argv", ["preflight", "--config", "config.yaml",
                                        "--scene-count", "2"])

    assert preflight_mosaic_series.main() == 0
    stream.flush()
