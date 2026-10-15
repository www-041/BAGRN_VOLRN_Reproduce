from __future__ import annotations

import importlib


def test_force_mode_lowers_only_runtime_registration_gates(monkeypatch):
    geometry = importlib.import_module("src.registration_benchmark.geometry")
    pairwise = importlib.import_module("src.multiscene_sift.pairwise")
    original = (
        geometry.MIN_INLIERS,
        geometry.MIN_INLIER_RATIO,
        pairwise.MIN_INLIERS,
        pairwise.MIN_INLIER_RATIO,
        geometry._is_geometry_valid,
    )

    launcher = importlib.import_module("scripts.run_mountain_b9_6scene_force_all_edges")
    manifest = launcher.install_force_registration_gates()

    assert manifest == {
        "minimum_inliers": 3,
        "minimum_inlier_ratio": 0.0,
        "residual_threshold_px": 2.0,
        "allow_invalid_geometry": True,
        "scope": "process-local experimental launcher only",
    }
    assert geometry.MIN_INLIERS == 3
    assert geometry.MIN_INLIER_RATIO == 0.0
    assert pairwise.MIN_INLIERS == 3
    assert pairwise.MIN_INLIER_RATIO == 0.0
    assert geometry._is_geometry_valid(0.37, 0.66, -28.0, 43.0)

    monkeypatch.setattr(geometry, "MIN_INLIERS", original[0])
    monkeypatch.setattr(geometry, "MIN_INLIER_RATIO", original[1])
    monkeypatch.setattr(pairwise, "MIN_INLIERS", original[2])
    monkeypatch.setattr(pairwise, "MIN_INLIER_RATIO", original[3])
    monkeypatch.setattr(geometry, "_is_geometry_valid", original[4])
