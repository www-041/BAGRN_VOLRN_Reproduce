"""Synthetic tests for the frame root-cause diagnostics.

These tests build entirely synthetic common grids, match-view frames and
world transforms — no real DZ01V imagery and no matcher is ever run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from rasterio.transform import Affine

from src.multiscene_sift import frame_diagnostics as fd
from src.multiscene_sift.models import Scene

RES = 14.0


def _grid(origin_x, origin_y, window=None):
    """A pair-common-grid record in the shape produced by the rebuilders."""
    if window is None:
        window = (60, 460, 40, 440)  # (row_start, row_end, col_start, col_end)
    return {
        "idx_i": 0,
        "idx_j": 0,
        "crs": "EPSG:32650",
        "width": 1000,
        "height": 1000,
        "transform": [RES, 0.0, origin_x, 0.0, -RES, origin_y],
        "origin_world_x": origin_x,
        "origin_world_y": origin_y,
        "pixel_size_x": RES,
        "pixel_size_y": -RES,
        "overlap_window": list(window),
    }


def _mv(origin_x, origin_y, scale):
    return {
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
        "scale_x": float(scale),
        "scale_y": float(scale),
        "scale": float(scale),
    }


def _world_affine(tx, ty, rot_deg=0.15, scale=1.001):
    theta = math.radians(rot_deg)
    return np.array([
        [scale * math.cos(theta), -scale * math.sin(theta), tx],
        [scale * math.sin(theta), scale * math.cos(theta), ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Frame math helpers
# ---------------------------------------------------------------------------


def test_frame_semantics_summary_is_complete():
    s = fd.frame_semantics_summary()
    for key in ("pixel_matrix_direction", "pixel_matrix_frame",
                "common_transform_semantics", "global_transform_direction",
                "global_transform_frame", "compose_rule_in_production",
                "pixel_affine_to_world_rule",
                "match_view_to_common_conversion"):
        assert key in s
        assert isinstance(s[key], str) and s[key]
    assert "match-view" in s["pixel_matrix_frame"]


def test_transform_points_homogeneous():
    M = np.array([[2.0, 0.0, 5.0], [0.0, 3.0, 7.0], [0.0, 0.0, 1.0]])
    pts = np.array([[1.0, 1.0], [0.0, 0.0]])
    out = fd.transform_points(M, pts)
    assert out[0].tolist() == pytest.approx([7.0, 10.0])
    assert out[1].tolist() == pytest.approx([5.0, 7.0])


def test_match_view_to_common_affine_roundtrip():
    T = fd.match_view_to_common_affine(1454.0, 2336.0, 0.7383, 0.7383)
    mv = np.array([[100.0, 90.0]])
    common = fd.transform_points(T, mv)
    assert common[0].tolist() == pytest.approx(
        [1454.0 + 100.0 / 0.7383, 2336.0 + 90.0 / 0.7383]
    )
    back = fd.transform_points(np.linalg.inv(T), common)
    np.testing.assert_allclose(back, mv, atol=1e-9)


def test_match_view_frame_params_matches_build_contract():
    # 1600×1339 match view comes from a 2168-row, 1995-col overlap at 1600 max.
    params = fd.match_view_frame_params((0, 2168, 0, 1995), max_side=1600)
    assert params["scale"] == pytest.approx(1600 / 2168, abs=1e-9)
    assert params["match_view_height"] == 1600  # round(2168 * scale)
    assert params["match_view_width"] == 1472
    assert params["origin_x"] == 0.0 and params["origin_y"] == 0.0


# ---------------------------------------------------------------------------
# Task 3 — origin invariance
# ---------------------------------------------------------------------------


def test_origin_invariance_probe_passes():
    probe = fd.origin_invariance_probe(t_true=None, res=RES, origins=[
        (700000.0, 4070000.0), (700294.0, 4069342.0)])
    assert probe["passes"] is True
    for grid in probe["grids"]:
        assert grid["max_abs_difference"] < 1e-6


def test_origin_invariance_pure_translation_negative_y():
    t_true = np.array([[1.0, 0.0, 42.0], [0.0, 1.0, -70.0], [0.0, 0.0, 1.0]])
    grids = [(700000.0, 4070000.0), (700294.0, 4069342.0)]
    probe = fd.origin_invariance_probe(t_true=t_true, res=RES, origins=grids)
    assert probe["passes"] is True
    for grid in probe["grids"]:
        # The recovered world translation is exactly +42 / -70 m, sign comes
        # from the matrices (never hand-flipped for image-y vs world-y).
        rec = grid["recovered_world_transform"]
        assert rec[0][2] == pytest.approx(42.0, abs=1e-6)
        assert rec[1][2] == pytest.approx(-70.0, abs=1e-6)
        assert grid["max_abs_difference"] < 1e-6


# ---------------------------------------------------------------------------
# Task 5 — scene-4 frame trace
# ---------------------------------------------------------------------------


def _minimal_scene4():
    t4 = Affine(RES, 0.0, 699200.0, 0.0, -RES, 4072000.0)
    return Scene(
        index=4, name="s4", directory="", band_paths={"B14": ""},
        crs="EPSG:32650", transforms={"B14": t4},
        shapes={"B14": (2000, 2000)}, nodata={"B14": 0.0},
        bounds={"B14": (t4.c, t4.f - 14.0 * 2000, t4.c + 14.0 * 2000, t4.f)},
    )


def test_trace_scene4_roundtrip_consistent_across_grids():
    scene4 = _minimal_scene4()
    g04 = _grid(700000.0, 4070000.0)
    g14 = _grid(700294.0, 4069342.0)
    rows, summary = fd.trace_scene4_through_grids(
        scene4, "B14", g04, g14
    )
    assert len(rows) == 5
    assert summary["max_roundtrip_error_m"] < 1e-6
    assert summary["max_c04_vs_c14_world_delta_m"] < 1e-6
    # The two grids must place the same physical point at different pixels.
    assert rows[0]["c04_col"] != pytest.approx(rows[0]["c14_col"], abs=1e-3)


# ---------------------------------------------------------------------------
# Task 6/7/8 — consistent three-node world and the frame-correction switch
# ---------------------------------------------------------------------------


def _consistent_world_system():
    """A synthetic 4—0, 4—1, 0—1 system with one true world geometry."""
    g04 = _grid(700000.0, 4070000.0)
    g14 = _grid(700294.0, 4069342.0)
    g01 = _grid(700150.0, 4069700.0, window=(100, 500, 120, 480))
    mv04 = _mv(60.0, 40.0, 0.7383)
    mv14 = _mv(70.0, 50.0, 0.7383)
    mv01 = _mv(120.0, 100.0, 0.7383)

    C04 = Affine(*g04["transform"])
    C14 = Affine(*g14["transform"])  # noqa: F841  (kept for parity/readability)
    C01 = Affine(*g01["transform"])

    # True world (scene -> reference 4).
    A04 = _world_affine(5000.0, -3000.0)     # scene0 -> scene4
    A14 = _world_affine(12000.0, 4000.0)     # scene1 -> scene4

    def _to_pixel_matrix(common, mv, world):
        A = fd.affine_to_matrix(common)
        T = fd.match_view_to_common_affine(
            mv["origin_x"], mv["origin_y"], mv["scale_x"], mv["scale_y"])
        M_common = np.linalg.inv(A) @ world @ A
        # represent the common-frame matrix in the match-view pixel frame
        return (np.linalg.inv(T) @ M_common @ T).tolist()

    pair04 = {
        "idx_i": 4, "idx_j": 0, "status": "OK",
        "pixel_matrix": _to_pixel_matrix(C04, mv04, A04),
    }
    pair14 = {
        "idx_i": 4, "idx_j": 1, "status": "OK",
        "pixel_matrix": _to_pixel_matrix(C14, mv14, A14),
    }

    # 0-1 same-name points in the reference frame; map to both scenes, then to
    # the 0-1 match-view frame.
    rng = np.random.default_rng(0)
    ref_pts = np.column_stack([
        rng.uniform(700400, 700850, 120),
        rng.uniform(4069600, 4070100, 120),
    ])
    p0_w = fd.transform_points(np.linalg.inv(A04), ref_pts)
    p1_w = fd.transform_points(np.linalg.inv(A14), ref_pts)

    def _to_mv(world):
        inv_C01 = ~C01
        c = np.array([inv_C01 * (x, y) for x, y in world])
        T01 = fd.match_view_to_common_affine(
            mv01["origin_x"], mv01["origin_y"],
            mv01["scale_x"], mv01["scale_y"])
        return fd.transform_points(np.linalg.inv(T01), c)

    ref_mv = _to_mv(p0_w)
    tgt_mv = _to_mv(p1_w)
    return dict(g04=g04, g14=g14, g01=g01, mv04=mv04, mv14=mv14, mv01=mv01,
                pair04=pair04, pair14=pair14, ref_mv=ref_mv, tgt_mv=tgt_mv,
                A04=A04, A14=A14, C01=C01)


def test_world_transform_frame_adjustment_recovers_true_world():
    system = _consistent_world_system()
    prod = fd.world_transform_from_saved_pair(
        4, 0, [system["pair04"]], Affine(*system["g04"]["transform"]),
        system["mv04"], adjust_frame=False,
    )
    ref = fd.world_transform_from_saved_pair(
        4, 0, [system["pair04"]], Affine(*system["g04"]["transform"]),
        system["mv04"], adjust_frame=True,
    )
    np.testing.assert_allclose(ref, system["A04"], atol=1e-6)
    # production (match-view matrix conjugated directly) differs from truth.
    assert np.max(np.abs(prod - system["A04"])) > 1.0


def test_closure_zero_with_frame_corrected_but_not_production():
    system = _consistent_world_system()
    g01, mv01 = system["g01"], system["mv01"]
    prod0 = fd.world_transform_from_saved_pair(
        4, 0, [system["pair04"]], Affine(*system["g04"]["transform"]),
        system["mv04"], adjust_frame=False)
    prod1 = fd.world_transform_from_saved_pair(
        4, 1, [system["pair14"]], Affine(*system["g14"]["transform"]),
        system["mv14"], adjust_frame=False)
    ref0 = fd.world_transform_from_saved_pair(
        4, 0, [system["pair04"]], Affine(*system["g04"]["transform"]),
        system["mv04"], adjust_frame=True)
    ref1 = fd.world_transform_from_saved_pair(
        4, 1, [system["pair14"]], Affine(*system["g14"]["transform"]),
        system["mv14"], adjust_frame=True)

    explicit = fd.closure_stats_on_points(
        system["ref_mv"], system["tgt_mv"], g01, mv01, ref0, ref1, RES, RES)
    production = fd.closure_stats_on_points(
        system["ref_mv"], system["tgt_mv"], g01, mv01, prod0, prod1, RES, RES)
    assert explicit["median_px"] < 1e-6
    # The production (wrong-frame) composition must introduce a material,
    # systematic residual that disappears under frame correction.
    assert production["median_px"] > 5.0
    assert explicit["median_px"] < production["median_px"] * 0.01


# ---------------------------------------------------------------------------
# Task 11 — classifier gates
# ---------------------------------------------------------------------------


def test_classifier_pixel_to_world_when_correction_collapses_closure():
    evidence = {
        "origin_invariance_pass": True,
        "production_closure_p95_px": 52.0,
        "explicit_closure_p95_px": 0.8,
        "direct_pair_p95_px": 1.95,
        "production_equals_explicit": False,
    }
    verdict = fd.classify_frame_root_cause(evidence)
    assert verdict["state"] == "PIXEL_TO_WORLD_FRAME_BUG_CONFIRMED"


def test_classifier_tree_edge_inconsistent():
    evidence = {
        "origin_invariance_pass": True,
        "tree_edge_04_phase_mag_px": 48.0,
        "tree_edge_14_phase_mag_px": 1.0,
    }
    verdict = fd.classify_frame_root_cause(evidence)
    assert verdict["state"] == "TREE_EDGE_GEOMETRY_INCONSISTENT"


def test_classifier_origin_invariance_failure():
    evidence = {"origin_invariance_pass": False}
    verdict = fd.classify_frame_root_cause(evidence)
    assert verdict["state"] == "PIXEL_TO_WORLD_FRAME_BUG_CONFIRMED"


def test_classifier_pair_local_grid_excluded():
    evidence = {
        "origin_invariance_pass": True,
        "production_closure_p95_px": 53.0,
        "explicit_closure_p95_px": 53.5,
        "production_equals_explicit": True,
    }
    verdict = fd.classify_frame_root_cause(evidence)
    assert verdict["state"] == "PAIR_LOCAL_GRID_NOT_ROOT_CAUSE"


def test_classifier_default_not_yet_determined():
    evidence = {"origin_invariance_pass": True}
    verdict = fd.classify_frame_root_cause(evidence)
    assert verdict["state"] == "NOT_YET_DETERMINED"
    assert verdict["state"] in fd.ALLOWED_STATES