"""Synthetic tests for the dense multiscale 0-1 phase-field diagnostics.

All tests use constructed arrays or the tiny synthetic raster fixture from the
test conftest — no real DZ01V imagery and no matcher is run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from rasterio.transform import Affine

from src.multiscene_sift import dense_phase_diagnostics as dpd


def _texture(shape=(512, 512), seed=3):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    base = (np.sin(x * 0.2) * 200 + np.cos(y * 0.15) * 180
            + rng.uniform(-40, 40, shape))
    return (base + 5000).astype(np.float64)


def _overlap(image0, image1, mask0=None, mask1=None):
    if mask0 is None:
        mask0 = np.ones(image0.shape, dtype=bool)
    if mask1 is None:
        mask1 = np.ones(image1.shape, dtype=bool)
    h, w = image0.shape
    return {
        "image0": image0, "image1": image1,
        "mask0": mask0, "mask1": mask1,
        "joint_mask": mask0 & mask1,
        "transform": Affine(14.0, 0.0, 700000.0, 0.0, -14.0, 4070000.0),
        "bounds": {"left": 700000.0, "right": 700000.0 + 14.0 * w,
                   "bottom": 4070000.0 - 14.0 * h, "top": 4070000.0},
        "width": w, "height": h,
        "resolution": 14.0,
        "joint_valid_pixels": int((mask0 & mask1).sum()),
        "joint_valid_fraction": float((mask0 & mask1).mean()),
    }


def _rows(dx, dy, n=12, accepted=True):
    if np.ndim(dx) == 0:
        dx = np.full(n, float(dx))
    if np.ndim(dy) == 0:
        dy = np.full(n, float(dy))
    n = min(len(dx), len(dy))
    rows = []
    for k in range(n):
        rows.append({
            "grid_n": 4,
            "center_pixel_x": 50.0 + k * 10,
            "center_pixel_y": 50.0 + k * 10,
            "phase_dx_px": float(dx[k]),
            "phase_dy_px": float(dy[k]),
            "accepted": accepted,
        })
    return rows


# ---------------------------------------------------------------------------
# Task 1 — baseline loader
# ---------------------------------------------------------------------------


def test_load_dense_phase_baseline(tmp_path):
    from src.multiscene_sift.frame_diagnostics import write_json as fw

    geo = tmp_path / "geo"
    geo.mkdir()
    fw(geo / "13_revised_state.json", {
        "state": "SCENE_LEVEL_CONSTANT_CORRECTION_INSUFFICIENT",
        "edge": [0, 1],
    })
    fw(geo / "05_metadata_only_shift_summary.json", {
        "edge_0_1": {"n_tiles_accepted": 6,
                     "mean_dx_px": 18.0, "mean_dy_px": -41.5},
    })
    result = dpd.load_dense_phase_baseline(geo)
    assert result["revised_state"]["edge"] == [0, 1]
    assert result["metadata_only_shift"]["edge_0_1"]["n_tiles_accepted"] == 6


# ---------------------------------------------------------------------------
# Task 2 — metadata-only overlap semantics (synthetic tiny scenes)
# ---------------------------------------------------------------------------


def test_metadata_only_overlap_semantics(tmp_path):
    from tests.multiscene_sift.conftest import make_five_scene_path
    from src.multiscene_sift.dataset import discover_five_scenes

    root = make_five_scene_path(tmp_path)
    names = [
        "DZ01V_L2_E113.0_N36.4_20260222031837_01_T1",
        "DZ01V_L2_E113.4_N36.4_20260222031838_01_T2",
        "DZ01V_L2_E113.8_N36.4_20260222031839_01_T3",
        "DZ01V_L2_E113.4_N36.2_20260222031840_01_T4",
        "DZ01V_L2_E113.8_N36.2_20260222031841_01_T5",
    ]
    scenes, _ = discover_five_scenes(str(root), names, bands=("B14",))
    overlap = dpd.build_metadata_only_overlap_01(scenes[0], scenes[1], "B14")
    h, w = overlap["image0"].shape
    assert overlap["image1"].shape == (h, w)
    assert overlap["joint_mask"].shape == (h, w)
    assert overlap["joint_valid_pixels"] > 1000
    meta = dpd.overlap_grid_metadata(overlap)
    assert meta["applied_corrections"] == "none"
    assert meta["transforms_used"] == "scene0/scene1 original geotransforms only"


# ---------------------------------------------------------------------------
# Task 3 — tile windows
# ---------------------------------------------------------------------------


def test_tile_windows_exact_coverage():
    windows = dpd.make_tile_windows(800, 960, 8)
    assert len(windows) == 64
    assert dpd.assert_full_coverage(windows, 800, 960)


def test_tile_windows_uneven_dimensions():
    windows = dpd.make_tile_windows(803, 967, 6)
    assert len(windows) == 36
    assert dpd.assert_full_coverage(windows, 803, 967)
    # final segments must reach the raster edge
    assert max(w["row1"] for w in windows) == 803
    assert max(w["col1"] for w in windows) == 967


def test_tile_windows_minimum_size():
    windows = dpd.make_tile_windows(256, 256, 8, min_tile_size_px=64)
    assert all(w["tile_height"] == 32 and w["tile_width"] == 32 for w in windows)
    assert all(w["scale_too_small"] for w in windows)


# ---------------------------------------------------------------------------
# Task 4 — quality gates
# ---------------------------------------------------------------------------


def test_tile_quality_low_valid_fraction():
    img = _texture((64, 64))
    mask = np.zeros((64, 64), dtype=bool)
    q = dpd.evaluate_tile_quality(img, img, mask, mask)
    assert q["reject_reason"] == "LOW_VALID_FRACTION"
    assert q["accepted_for_phase"] is False


def test_tile_quality_low_texture():
    flat = np.full((64, 64), 100.0)
    mask = np.ones((64, 64), dtype=bool)
    q = dpd.evaluate_tile_quality(flat, flat, mask, mask)
    assert q["reject_reason"] == "LOW_TEXTURE"
    assert q["accepted_for_phase"] is False


def test_tile_quality_textured_accepted():
    img = _texture((64, 64))
    mask = np.ones((64, 64), dtype=bool)
    q = dpd.evaluate_tile_quality(img, img, mask, mask)
    assert q["accepted_for_phase"] is True
    assert q["reject_reason"] is None
    assert q["joint_valid_fraction"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Task 5 — phase adapter
# ---------------------------------------------------------------------------


def test_tile_phase_known_shift():
    ref = _texture((256, 256))
    moving = np.roll(ref, (24, -16), axis=(0, 1))
    joint = np.ones(ref.shape, dtype=bool)
    r = dpd.measure_tile_phase_shift(ref, moving, joint)
    assert r["status"] == "OK"
    mag = math.hypot(r["dx_px"], r["dy_px"])
    assert mag == pytest.approx(math.hypot(24, -16), abs=0.6)
    aligned = np.roll(moving, (round(r["dy_px"]), round(r["dx_px"])),
                      axis=(0, 1))
    assert np.allclose(aligned, ref, atol=1e-6)


def test_tile_phase_flat_returns_not_ok():
    flat = np.full((128, 128), 50.0)
    joint = np.ones(flat.shape, dtype=bool)
    r = dpd.measure_tile_phase_shift(flat, flat, joint)
    assert r["status"] != "OK"


# ---------------------------------------------------------------------------
# Task 6 — field measurement + accepted counts
# ---------------------------------------------------------------------------


def test_dense_field_accepted_count():
    base = _texture((512, 512), seed=7)
    shifted = np.roll(base, (24, -16), axis=(0, 1))
    overlap = _overlap(base, shifted)
    rows = dpd.measure_dense_shift_field(overlap, 4, {})
    assert len(rows) == 16
    accepted = [r for r in rows if r["accepted"]]
    assert len(accepted) == 16
    # magnitude consistent with the known 24/-16 shift
    mags = [r["phase_mag_px"] for r in accepted]
    assert np.median(mags) == pytest.approx(math.hypot(24, 16), abs=0.5)


def test_dense_field_invalid_tiles_are_kept_with_reason():
    base = _texture((512, 512), seed=9)
    shifted = np.roll(base, (8, 8), axis=(0, 1))
    mask0 = np.ones((512, 512), dtype=bool)
    mask1 = np.ones((512, 512), dtype=bool)
    mask1[0:256, :] = False  # top half invalid in the target
    overlap = _overlap(base, shifted, mask0, mask1)
    rows = dpd.measure_dense_shift_field(overlap, 4, {})
    assert len(rows) == 16
    rejected = [r for r in rows if not r["accepted"]]
    assert rejected
    for r in rejected:
        assert r["reject_reason"] is not None
    assert any(r["reject_reason"] == "LOW_VALID_FRACTION" for r in rejected)


# ---------------------------------------------------------------------------
# Task 7 — robust summary
# ---------------------------------------------------------------------------


def test_summary_robust_to_outlier():
    dx = np.full(12, 20.0) + np.random.default_rng(1).uniform(-1, 1, 12)
    dy = np.full(12, 40.0) + np.random.default_rng(2).uniform(-1, 1, 12)
    dx[0] = 150.0
    dy[0] = 200.0
    rows = _rows(dx, dy)
    s = dpd.summarize_dense_shift_field(rows)
    assert s["median_dx_px"] == pytest.approx(20.0, abs=1.2)
    assert s["median_dy_px"] == pytest.approx(40.0, abs=1.2)
    assert s["summary_status"] == "OK"


def test_summary_insufficient_tiles():
    rows = _rows([1.0] * 3, [2.0] * 3)
    s = dpd.summarize_dense_shift_field(rows)
    assert s["summary_status"] == "INSUFFICIENT_TILES"


# ---------------------------------------------------------------------------
# Task 8 — spatial trend fit
# ---------------------------------------------------------------------------


def test_trend_constant_field_small_range():
    dx = np.full(12, 20.0)
    dy = np.full(12, 40.0)
    t = dpd.fit_shift_spatial_trend(_rows(dx, dy))
    assert t["trend_vector_change_px"] < 0.5
    assert t["predicted_dx_range"] < 0.5
    assert t["predicted_dy_range"] < 0.5


def test_trend_gradient_field_recovers_slope():
    x = np.linspace(-1, 1, 12)
    y = np.linspace(-1, 1, 12)
    dx = 10.0 + 15.0 * x
    dy = 40.0 + 8.0 * y
    rows = _rows(dx, dy)
    t = dpd.fit_shift_spatial_trend(rows)
    assert t["trend_status"] == "OK"
    assert t["dx_r2"] > 0.9 and t["dy_r2"] > 0.9
    assert abs(t["dx_coefficients"][1] - 15.0) < 2.0
    assert abs(t["dy_coefficients"][2] - 8.0) < 2.0
    assert t["predicted_dx_range"] > 15.0
    assert t["predicted_dy_range"] > 8.0


# ---------------------------------------------------------------------------
# Task 10 — multiscale classifier
# ---------------------------------------------------------------------------


def _summary(dx, dy, n=12, rng_px=1.0):
    return {
        "n_phase_ok": n, "n_total": n,
        "median_dx_px": float(dx), "median_dy_px": float(dy),
        "dx_range_px": rng_px, "dy_range_px": rng_px,
        "summary_status": "OK",
    }


def test_classifier_stable_constant():
    summaries = {
        "grid_4": _summary(20.0, 40.0, rng_px=1.0),
        "grid_6": _summary(21.0, 39.0, rng_px=1.5),
        "grid_8": _summary(20.5, 40.0, rng_px=2.0),
    }
    trends = {g: {"trend_status": "OK", "trend_vector_change_px": 0.5}
              for g in summaries}
    v = dpd.classify_multiscale_shift_behavior(summaries, trends)
    assert v["state"] == "STABLE_CONSTANT_SHIFT"


def test_classifier_stable_gradient():
    summaries = {
        "grid_4": _summary(20.0, 40.0, rng_px=6.0),
        "grid_6": _summary(20.0, 40.0, rng_px=8.0),
        "grid_8": _summary(20.0, 40.0, rng_px=9.0),
    }
    trends = {}
    for g in (6, 8):
        trends[f"grid_{g}"] = {
            "trend_status": "OK", "n_points": 12,
            "trend_vector_change_px": 10.0,
            "dx_coefficients": [0.0, 5.0, 0.0],
            "dy_coefficients": [0.0, 0.0, 6.0],
        }
    trends["grid_4"] = {"trend_status": "OK", "trend_vector_change_px": 3.0,
                        "dx_coefficients": [0, 1, 0],
                        "dy_coefficients": [0, 0, 1]}
    v = dpd.classify_multiscale_shift_behavior(summaries, trends)
    assert v["state"] == "STABLE_SPATIAL_GRADIENT"


def test_classifier_unstable_insufficient():
    summaries = {
        "grid_4": {"n_phase_ok": 8, "n_total": 16, "summary_status": "OK"},
        "grid_6": {"n_phase_ok": 3, "n_total": 36, "summary_status": "INSUFFICIENT_TILES"},
        "grid_8": {"n_phase_ok": 4, "n_total": 64, "summary_status": "INSUFFICIENT_TILES"},
    }
    trends = {}
    v = dpd.classify_multiscale_shift_behavior(summaries, trends)
    assert v["state"] == "UNSTABLE_OR_INSUFFICIENT"


def test_classifier_ambiguous():
    summaries = {
        "grid_4": _summary(20.0, 40.0, rng_px=7.0),
        "grid_6": _summary(21.0, 41.0, rng_px=8.0),
        "grid_8": _summary(20.5, 40.5, rng_px=8.5),
    }
    trends = {g: {"trend_status": "OK", "trend_vector_change_px": 2.0}
              for g in summaries}
    v = dpd.classify_multiscale_shift_behavior(summaries, trends)
    assert v["state"] == "MIXED_OR_AMBIGUOUS"