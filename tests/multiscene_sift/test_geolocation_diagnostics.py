"""Synthetic tests for the geolocation consistency diagnostics.

Everything here runs on constructed matrices / tiny synthetic bands — no real
DZ01V imagery and no matcher is involved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.multiscene_sift import geolocation_diagnostics as gd
from src.multiscene_sift.loop_diagnostics import _phase_cross_correlation_shift

RES = 14.0


def _A(origin_x, origin_y, res=RES, rot_deg=0.0):
    """Native pixel -> world matrix (north-up, optional small rotation)."""
    th = math.radians(rot_deg)
    return np.array([
        [res * math.cos(th), -res * math.sin(th), origin_x],
        [res * math.sin(th), res * math.cos(th), origin_y],
        [0.0, 0.0, 1.0],
    ])


def _world(tx, ty, rot_deg=0.15, scale=1.001):
    th = math.radians(rot_deg)
    return np.array([
        [scale * math.cos(th), -scale * math.sin(th), tx],
        [scale * math.sin(th), scale * math.cos(th), ty],
        [0.0, 0.0, 1.0],
    ])


def _field_samples(dx, dy):
    """Build sample dicts with per-point correction components."""
    dx = np.atleast_1d(np.asarray(dx, dtype=float))
    dy = np.atleast_1d(np.asarray(dy, dtype=float))
    rows = []
    for k in range(len(dx)):
        rows.append({
            "valid": True,
            "sample_fraction_x": (0.1 + 0.4 * (k % 3)),
            "sample_fraction_y": (0.1 + 0.4 * (k // 3)),
            "correction_dx_px": float(dx[k]),
            "correction_dy_px": float(dy[k]),
        })
    return rows


# ---------------------------------------------------------------------------
# Task 2 — metadata relation
# ---------------------------------------------------------------------------


def test_metadata_native_relation_math():
    A0 = _A(700000.0, 4070000.0)
    A1 = _A(700280.0, 4069860.0)
    T_meta = gd.metadata_native_relation(A0, A1)
    p1 = np.array([[100.0, 50.0]])
    # Synthetic convention here is world y-up (like rasterio): a row increase
    # of 50 moves world y to 4069860 + 50*14 = 4070560, which maps back to
    # native-0 row (4070560 - 4070000) / 14 = 40.  All signs come from the
    # matrices; this test only locks the algebra.
    expected_native1 = (np.linalg.inv(A0) @ (A1 @ np.array([100.0, 50.0, 1.0])))[:2]
    got = (T_meta @ np.array([100.0, 50.0, 1.0]))[:2]
    np.testing.assert_allclose(got, expected_native1, atol=1e-9)
    assert got[0] == pytest.approx(120.0, abs=1e-6)
    assert got[1] == pytest.approx(40.0, abs=1e-6)


def test_metadata_native_relation_roundtrip():
    rng = np.random.default_rng(0)
    A0 = _A(700000.0, 4070000.0)
    A1 = _A(700294.0, 4069342.0, rot_deg=0.03)
    T_meta = gd.metadata_native_relation(A0, A1)
    pts_j = rng.uniform(0, 4000, (50, 2))
    for pj in pts_j:
        world = (A1 @ np.append(pj, 1.0))[:2]
        pi_manual = (np.linalg.inv(A0) @ np.append(world, 1.0))[:2]
        pi_meta = (T_meta @ np.append(pj, 1.0))[:2]
        np.testing.assert_allclose(pi_meta, pi_manual, atol=1e-9)


def test_world_correction_identity_when_metadata_matches_direct():
    A0 = _A(700000.0, 4070000.0)
    A1 = _A(700280.0, 4069860.0)
    T_meta = gd.metadata_native_relation(A0, A1)
    C = gd.compute_world_correction(A0, T_meta, A1)
    np.testing.assert_allclose(C, np.eye(3), atol=1e-9)


def test_world_correction_recovers_known_translation():
    A0 = _A(700000.0, 4070000.0)
    A1 = _A(700280.0, 4069860.0)
    C_true = _world(140.0, -70.0, rot_deg=0.0, scale=1.0)
    T_direct = np.linalg.inv(A0) @ C_true @ A1
    C = gd.compute_world_correction(A0, T_direct, A1)
    np.testing.assert_allclose(C, C_true, atol=1e-6)


def test_direct_native_relation_frame():
    A0 = _A(700000.0, 4070000.0)
    A1 = _A(700280.0, 4069860.0)
    T_meta = gd.metadata_native_relation(A0, A1)
    # a world translation T should appear as a native translation via
    # inv(A0) @ T @ A1
    T_world = _world(350.0, 140.0, rot_deg=0.0, scale=1.0)
    T_native = gd.direct_native_relation(T_world, A0, A1)
    C = gd.compute_world_correction(A0, T_native, A1)
    np.testing.assert_allclose(C, T_world, atol=1e-9)


# ---------------------------------------------------------------------------
# Task 4 — correction field classification
# ---------------------------------------------------------------------------


def test_constant_correction_field_classification():
    summary = gd.correction_field_summary(_field_samples(
        [30.0] * 9, [40.0] * 9))
    assert summary["total_std_px"] < 1e-9
    assert summary["classification"] == "NEAR_CONSTANT_WITHIN_OVERLAP"


def test_spatial_correction_field_classification():
    dx = np.linspace(0.0, 50.0, 9)
    dy = np.linspace(20.0, -10.0, 9)
    summary = gd.correction_field_summary(_field_samples(dx, dy))
    assert summary["classification"] == "SPATIALLY_VARYING_WITHIN_OVERLAP"


def test_correction_field_lacks_valid_samples():
    summary = gd.correction_field_summary(
        [{"valid": False} for _ in range(9)])
    assert summary["classification"] == "INSUFFICIENT_VALID_SAMPLES"


# ---------------------------------------------------------------------------
# Task 5 — phase correlation synthetic (reuses the existing implementation)
# ---------------------------------------------------------------------------


def test_phase_shift_recovers_known_translation():
    rng = np.random.default_rng(3)
    ref = rng.uniform(0, 100, (256, 256))
    moving = np.roll(ref, (17, -11), axis=(0, 1))  # +17 rows, -11 cols
    dx, dy, score = _phase_cross_correlation_shift(ref, moving)
    aligned = np.roll(moving, (int(round(dy)), int(round(dx))), axis=(0, 1))
    assert np.allclose(aligned, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# Task 6/7 — triangle closure and scene-level model
# ---------------------------------------------------------------------------


def _world_points(n_merge=9):
    fx = np.array([0.1, 0.5, 0.9] * 3)
    fy = np.repeat([0.1, 0.5, 0.9], 3)
    return np.column_stack([700000.0 + fx * 20000.0,
                            4060000.0 + fy * 20000.0])


def test_triangle_closure_exact():
    H0 = _world(3000.0, -2000.0, rot_deg=0.1)
    H1 = _world(9000.0, 1500.0, rot_deg=-0.05)
    C01 = np.linalg.inv(H0) @ H1
    C04 = np.linalg.inv(H0)            # H4 = I
    C41 = H1
    closure = gd.world_correction_triangle_closure(
        C01, C04, C41, _world_points(), pixel_size=RES
    )
    assert closure["sample_mean_px"] < 1e-6
    assert closure["classification"] == "NEAR_CLOSED"


def test_triangle_closure_injected_translation_conflict():
    H0 = _world(3000.0, -2000.0, rot_deg=0.1)
    H1 = _world(9000.0, 1500.0, rot_deg=-0.05)
    C01 = np.linalg.inv(H0) @ H1.copy()
    C01[0, 2] += 50.0 * RES              # +50 px east content conflict
    C04 = np.linalg.inv(H0)
    C41 = H1
    closure = gd.world_correction_triangle_closure(
        C01, C04, C41, _world_points(), pixel_size=RES
    )
    assert closure["sample_mean_px"] > 40.0
    assert closure["sample_std_px"] < 2.0
    assert closure["classification"] == "NEAR_CONSTANT_TRANSLATION_CONFLICT"


def test_scene_level_constant_model_consistent():
    H0 = _world(3000.0, -2000.0, rot_deg=0.1)
    H1 = _world(9000.0, 1500.0, rot_deg=-0.05)
    result = gd.scene_level_constant_correction_test(
        np.linalg.inv(H0) @ H1, np.linalg.inv(H0), H1,
        _world_points(), pixel_size=RES,
    )
    assert result["status"] == "CONSISTENT_WITH_SCENE_LEVEL_CONSTANT_CORRECTIONS"


def test_scene_level_constant_model_inconsistent():
    H0 = _world(3000.0, -2000.0, rot_deg=0.1)
    H1 = _world(9000.0, 1500.0, rot_deg=-0.05)
    C01 = np.linalg.inv(H0) @ H1
    C01[1, 2] += 60.0 * RES
    result = gd.scene_level_constant_correction_test(
        C01, np.linalg.inv(H0), H1, _world_points(), pixel_size=RES,
    )
    assert result["status"] == "INCONSISTENT_WITH_SCENE_LEVEL_CONSTANT_CORRECTIONS"
    assert result["prediction_error_mean_px"] > 50.0


# ---------------------------------------------------------------------------
# Task 8 — within-overlap spatial-variation classification
# ---------------------------------------------------------------------------


def _tiles(dx_plane, dy_plane, n=6):
    rng = np.random.default_rng(1)
    tiles = []
    for k in range(n):
        row, col = k % 3, k // 3
        base = np.array([col * 0.5, row * 0.5])
        tiles.append({
            "tile_row": float(row),
            "tile_col": float(col),
            "phase_dx_px": float(dx_plane[0] + dx_plane[1] * base[0]
                                 + rng.normal(0, 0.1)),
            "phase_dy_px": float(dy_plane[0] + dy_plane[1] * base[1]
                                 + rng.normal(0, 0.1)),
            "accepted": True,
        })
    return tiles


def test_spatial_variation_picks_planar_field():
    tiles = _tiles([0.0, 30.0], [0.0, 30.0], n=9)
    result = gd.within_overlap_spatial_variation(tiles)
    assert result["classification"] == "SPATIALLY_VARYING"
    assert result["improvement_ratio"] > 0.3
    assert result["field_range_px"] > 5.0


def test_spatial_variation_constant():
    tiles = _tiles([10.0, 0.0], [20.0, 0.0], n=6)
    result = gd.within_overlap_spatial_variation(tiles)
    assert result["classification"] == "CONSTANT_LIKE"
    assert result["improvement_ratio"] < 0.2


# ---------------------------------------------------------------------------
# Task 10 — classifier
# ---------------------------------------------------------------------------


def test_classifier_globally_consistent():
    state = gd.classify_geolocation_inconsistency(
        {"triangle_closure_mean_px": 0.4})
    assert state == "PAIRWISE_RELATIONS_GLOBALLY_CONSISTENT"


def test_classifier_scene_level_insufficient():
    state = gd.classify_geolocation_inconsistency({
        "triangle_closure_mean_px": 71.6,
        "scene_level_constant_model": "INCONSISTENT_WITH_SCENE_LEVEL_CONSTANT_CORRECTIONS",
        "spatial_variation_supported": False,
    })
    assert state == "SCENE_LEVEL_CONSTANT_CORRECTION_INSUFFICIENT"


def test_classifier_spatial_variation_supported():
    state = gd.classify_geolocation_inconsistency({
        "triangle_closure_mean_px": 20.0,
        "spatial_variation_supported": True,
        "scene_level_constant_model": "INCONSISTENT_WITH_SCENE_LEVEL_CONSTANT_CORRECTIONS",
    })
    assert state == "WITHIN_OVERLAP_SPATIAL_VARIATION_SUPPORTED"


def test_classifier_underdetermined_conflict():
    state = gd.classify_geolocation_inconsistency({
        "triangle_closure_mean_px": 71.6,
        "scene_level_constant_model": "INCONSISTENT_WITH_SCENE_LEVEL_CONSTANT_CORRECTIONS",
        "spatial_variation_supported": False,
        "within_overlap_displacement_constant": True,
    })
    assert state == "SCENE_LEVEL_CONSTANT_CORRECTION_INSUFFICIENT" or \
        state == "LOCAL_RELATIONS_CONFLICT_BUT_SOURCE_UNDERDETERMINED"


def test_classifier_default_insufficient():
    state = gd.classify_geolocation_inconsistency({})
    assert state == "INSUFFICIENT_EVIDENCE"


# ---------------------------------------------------------------------------
# Task 1 — baseline loader
# ---------------------------------------------------------------------------


def test_load_geolocation_baseline(tmp_path):
    from src.multiscene_sift.frame_diagnostics import write_json as fw

    frame = tmp_path / "frame"
    frame.mkdir()
    fw(frame / "12_frame_conclusion.json",
       {"state": "PAIR_LOCAL_GRID_NOT_ROOT_CAUSE"})
    fw(frame / "09_tree_edge_independent_checks.json", {"pair_0_4": {}})
    baseline = gd.load_geolocation_baseline(str(tmp_path), str(frame))
    assert baseline["frame_conclusion"]["state"] == "PAIR_LOCAL_GRID_NOT_ROOT_CAUSE"
    assert baseline["tree_checks"]["pair_0_4"] == {}
    assert baseline["frame_semantics"] is None