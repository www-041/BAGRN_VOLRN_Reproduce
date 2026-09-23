from __future__ import annotations

from pathlib import Path

import pytest


def test_loader_rejects_missing_equal_l2_baseline(tmp_path):
    from src.multiscene_sift.robust_translation_adjustment import (
        load_robust_translation_inputs,
    )

    with pytest.raises(FileNotFoundError):
        load_robust_translation_inputs(tmp_path, tmp_path, tmp_path)


def test_equal_l2_baseline_is_loaded_not_recomputed_with_new_weights():
    from src.multiscene_sift.robust_translation_adjustment import (
        load_robust_translation_inputs,
    )

    root = Path("data/output")
    inputs = load_robust_translation_inputs(
        root / "five_scene_sift_B14",
        root / "five_scene_inlier_recovery",
        root / "five_scene_global_adjustment",
    )
    assert inputs["equal_l2_summary"]["zero_one"]["p95_px"] > 0
    assert inputs["candidate_weighting_used_for_baseline"] is False
    assert inputs["equal_l2_solution"]["status"] == "OK"
