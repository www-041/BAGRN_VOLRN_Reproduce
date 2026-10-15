"""Synthetic tests for the selected-pair / cycle-consistency diagnostics.

No real registration or imagery is required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.multiscene_sift import network_consistency_diagnostics as ncd
from src.multiscene_sift.models import Scene
from src.multiscene_sift.nine_scene_overlap_diagnostic import SCENE_NAMES_9


def _scene(idx, cx, cy, size=0.8):
    from rasterio.coords import BoundingBox

    return Scene(
        index=idx, name=SCENE_NAMES_9[idx], directory="", band_paths={"B14": ""},
        crs="EPSG:32650", transforms={"B14": None},
        shapes={"B14": (100, 100)}, nodata={"B14": 0.0},
        bounds={"B14": BoundingBox(left=cx - size / 2, bottom=cy - size / 2,
                                   right=cx + size / 2, top=cy + size / 2)},
    )


def _translation(tx, ty):
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _three_scenes_small():
    return [
        _scene(0, 113.4, 36.6), _scene(4, 113.7, 36.7), _scene(6, 113.5, 36.9),
        _scene(1, 113.6, 36.5), _scene(2, 113.0, 36.4), _scene(5, 113.1, 36.8),
        _scene(7, 113.9, 36.0), _scene(8, 113.75, 36.35),
    ]


def test_selected_new_pairs_are_exact():
    assert ncd.SELECTED_NEW_PAIRS == [
        (0, 6), (4, 6), (0, 5), (2, 5),
        (5, 6), (1, 8), (1, 7), (7, 8),
    ]
    assert len(ncd.SELECTED_NEW_PAIRS) == 8


def test_selected_pairs_no_duplicates():
    assert len(set(ncd.SELECTED_NEW_PAIRS)) == 8


def test_selected_pairs_exist_in_bounds_graph(tmp_path):
    from src.multiscene_sift.nine_scene_overlap_diagnostic import all_pair_overlaps
    scenes = [
        _scene(i, 113.4 + 0.28 * (i % 3), 36.4 + 0.28 * (i // 3))
        for i in range(9)
    ]
    overlaps = all_pair_overlaps(scenes)
    by = {(r["idx_i"], r["idx_j"]): r for r in overlaps}
    for (i, j) in ncd.SELECTED_NEW_PAIRS:
        assert by[(i, j)]["intersection_exists"] is True


def test_baseline_config_required_fields(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_config.json").write_text(
        '{"matcher": "sift"}', encoding="utf-8")
    with pytest.raises(ValueError):
        ncd.load_baseline_registration_config(run)


def test_wrapper_runs_exactly_selected_pairs(monkeypatch):
    calls = []

    def _fake(scene_i, scene_j, band, match_max_side, ransac_threshold,
              random_seed, matcher):
        calls.append((scene_i.index, scene_j.index))
        return type("R", (), {
            "idx_i": scene_i.index, "idx_j": scene_j.index, "status": "OK",
            "raw_matches": 10, "inliers": 10, "inlier_ratio": 1.0,
            "coverage": 0.5, "residual_rmse": 1.0, "residual_p95": 1.5,
            "residual_median": 0.8, "pair_pixel_matrix": [],
            "matcher": "sift", "runtime_sec": 0.1,
            "pair_common_transform": None,
        })()

    monkeypatch.setattr("src.multiscene_sift.pairwise.register_pair", _fake)
    config = {
        "matcher": "sift", "registration_band": "B14", "match_max_side": 1600,
        "ransac_threshold": 2.0, "random_seed": 0,
    }
    scenes = [_scene(i, 113.4, 36.6) for i in range(9)]
    for i, j in ncd.SELECTED_NEW_PAIRS:
        ncd.run_selected_pair_registration(scenes[i], scenes[j], config)
    assert sorted(map(tuple, calls)) == sorted(ncd.SELECTED_NEW_PAIRS)
    assert len(calls) == 8


def test_config_snapshot_marks_no_relaxation(tmp_path):
    import json

    run = tmp_path / "run"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({
        "matcher": "sift", "registration_band": "B14",
        "match_max_side": 1600, "ransac_threshold": 2.0, "random_seed": 0,
    }), encoding="utf-8")
    snap = ncd.load_baseline_registration_config(run)
    assert snap["matcher"] == "sift"
    assert snap["no_threshold_relaxation"] is True
    assert snap["no_loftr_fallback"] is True
    assert snap["fixed_sift_constants"]["ransac_model"] == "Affine"


def test_compose_cycle_reversed_edge_uses_inverse():
    # consistent triangle stored only in the reverse directions
    A = _translation(10.0, -20.0)   # scene1 from scene0
    B = _translation(-5.0, 12.0)    # scene4 from scene1
    C = _translation(-5.0, 8.0)     # scene0 from scene4
    edges = {
        (0, 1): np.linalg.inv(A),
        (1, 4): np.linalg.inv(B),
        (4, 0): np.linalg.inv(C),
    }
    closure = ncd.compose_cycle_world((0, 1, 4), edges)
    np.testing.assert_allclose(closure, np.eye(3), atol=1e-9)


def test_cycle_closure_identity_for_consistent_edges():
    cycle = (0, 1, 4)
    X = _translation(20.0, 30.0)     # 1 from 0
    Y = _translation(-40.0, 10.0)    # 4 from 1
    Z = np.linalg.inv(Y @ X)         # 0 from 4 (consistent)
    edges = {(1, 0): X, (4, 1): Y, (0, 4): Z}
    closure = ncd.compose_cycle_world(cycle, edges)
    np.testing.assert_allclose(closure, np.eye(3), atol=1e-9)


def test_cycle_injected_translation_detected():
    cycle = (0, 1, 4)
    X = _translation(20.0, 30.0)
    Y = _translation(-40.0, 10.0)
    Z = np.linalg.inv(Y @ X)
    Z_inj = Z @ _translation(50.0 * 14.0, 0.0)  # +50 px east in 0-from-4
    edges = {(1, 0): X, (4, 1): Y, (0, 4): Z_inj}
    closure = ncd.compose_cycle_world(cycle, edges)
    pts = np.array([[0.0, 0.0]])
    stats = ncd.evaluate_cycle_on_points(closure, pts, pixel_size_m=14.0)
    assert stats["median_px"] == pytest.approx(50.0, abs=0.5)


def test_cycle_edge_eligibility_gate():
    eligible = {(0, 1), (1, 4), (4, 0)}
    ok, missing = ncd.cycle_edge_eligibility((0, 1, 4), eligible)
    assert ok is True and missing == []
    ok2, m2 = ncd.cycle_edge_eligibility((0, 2, 5), {(0, 2), (2, 5)})
    assert ok2 is False and (0, 5) in m2


def test_cycle_unavailable_on_missing_edge():
    eligible = {(0, 1), (1, 4)}  # missing 0-4
    ok, missing = ncd.cycle_edge_eligibility((0, 1, 4), eligible)
    assert not ok and (0, 4) in missing


def test_point_residual_pure_translation():
    # translation stored in metres: 20/40 px * 14 m
    closure = _translation(20.0 * 14.0, 40.0 * 14.0)
    pts = np.array([[100.0, 100.0], [2000.0, -500.0], [50.0, 30.0]])
    stats = ncd.evaluate_cycle_on_points(closure, pts, pixel_size_m=14.0)
    expected = math.hypot(20.0, 40.0)
    assert stats["median_px"] == pytest.approx(expected, abs=1e-6)
    assert stats["std_px"] < 1e-6


def test_classifier_closed():
    closure = _translation(0.5, -0.3)
    stats = {"median_px": 0.7, "std_px": 0.1}
    assert ncd.classify_cycle(closure, stats) == "CLOSED"


def test_classifier_large_systematic_translation():
    closure = _translation(40.0, 90.0)
    stats = {"median_px": 98.0, "std_px": 0.4}
    assert ncd.classify_cycle(closure, stats) == "LARGE_SYSTEMATIC_TRANSLATION"


def test_classifier_non_translational():
    th = math.radians(2.0)
    rot = np.array([
        [math.cos(th), -math.sin(th), 0.0],
        [math.sin(th), math.cos(th), 0.0],
        [0.0, 0.0, 1.0],
    ])
    closure = _translation(30.0, 10.0) @ rot
    stats = {"median_px": 40.0, "std_px": 3.0}
    assert ncd.classify_cycle(closure, stats) == "NON_TRANSLATIONAL_INCONSISTENCY"


def _pair_row(i, j, status="OK"):
    return {"idx_i": i, "idx_j": j, "status": status}


def test_scene0_evidence_aggregation():
    cycle_states = {
        "C1_scene0_ref4": ("CLOSED", None),
        "C2_scene0_ref2": ("SMALL_RESIDUAL", None),
        "C3_scene0_new": ("CLOSED", None),
        "C4_scene1_new": ("CLOSED", None),
        "C0_baseline": ("LARGE_SYSTEMATIC_TRANSLATION", None),
    }
    pairs = [
        _pair_row(0, 6), _pair_row(4, 6), _pair_row(0, 5), _pair_row(2, 5),
        _pair_row(5, 6), _pair_row(1, 8), _pair_row(1, 7), _pair_row(7, 8),
    ]
    ev = ncd.build_scene_evidence(cycle_states, pairs)
    assert ev["scene0_cycles"]["C1_scene0_ref4"] == "CLOSED"
    assert ev["baseline_contrast"]["C0_baseline"] == "LARGE_SYSTEMATIC_TRANSLATION"
    assert "0-6" in ev["scene0_new_edges"]
    assert "1-8" in ev["scene1_new_edges"]


def test_pair_0_1_specific_classifier():
    evidence = {
        "c0_status": "LARGE_SYSTEMATIC_TRANSLATION",
        "c1_status": "CLOSED", "c2_status": "SMALL_RESIDUAL",
        "c3_status": "CLOSED", "c4_status": "CLOSED",
        "new_pair_ok": 8, "new_pair_tried": 8,
    }
    assert ncd.classify_network_inconsistency(evidence) == \
        "PAIR_0_1_OVERLAP_SPECIFIC_SUPPORTED"


def test_insufficient_edge_classifier():
    evidence = {
        "c0_status": "LARGE_SYSTEMATIC_TRANSLATION",
        "c1_status": "UNAVAILABLE", "c2_status": "UNAVAILABLE",
        "c3_status": "UNAVAILABLE", "c4_status": "UNAVAILABLE",
        "new_pair_ok": 2, "new_pair_tried": 8,
    }
    assert ncd.classify_network_inconsistency(evidence) == "NEW_EDGES_INSUFFICIENT"


def test_scene0_level_classifier():
    evidence = {
        "c0_status": "LARGE_SYSTEMATIC_TRANSLATION",
        "c1_status": "LARGE_SYSTEMATIC_TRANSLATION",
        "c2_status": "NON_TRANSLATIONAL_INCONSISTENCY",
        "c3_status": "SMALL_RESIDUAL",
        "c4_status": "CLOSED",
        "new_pair_ok": 8, "new_pair_tried": 8,
    }
    assert ncd.classify_network_inconsistency(evidence) == \
        "SCENE0_LEVEL_INCONSISTENCY_SUSPECT"