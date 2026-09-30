from pathlib import Path

from src.pipeline.final_metrics import five_scene_regression_gate


def test_completed_five_scene_fresh_replay_passes_the_frozen_regression_gate():
    root = Path("data/output/final_pipeline/b9_5scene")
    if not root.is_dir():
        import pytest

        pytest.skip("five-scene fresh replay has not been executed")
    assert five_scene_regression_gate(root)["status"] == "PASS"
