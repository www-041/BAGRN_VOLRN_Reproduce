"""Synthetic checks only; no matcher, radiometric solver, or full-scene reads."""

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.io import MemoryFile
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_phase2_core import (Viewport, evaluate_pair, load_truth, new_row,
                                   pair_metrics, pixel_to_projected, projected_to_pixel,
                                   save_truth, summarize_pairs)
from build_benchmark_phase2_regions import best_valid_tile
from check_benchmark_phase2_leakage import inspect
from src.multiscene_sift.radiometric_metrics import compute_pair_mamd, compute_pair_msdd, compute_pair_rdd


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 4.0, 8.0])
def test_window_screen_native_roundtrip(scale):
    view = Viewport(123, 456, scale, 512, 512)
    for row, col in [(456.0, 123.0), (480.375, 200.125), (500.5, 250.5)]:
        x, y = view.native_to_screen(row, col)
        actual = view.screen_to_native(x, y)
        assert actual == pytest.approx((row, col), abs=1e-10)


def test_one_to_one_display_pixel_zero_is_native_pixel_zero():
    view = Viewport(123, 456, 1.0, 512, 512)
    assert view.screen_to_native(0, 0) == (456, 123)
    assert view.native_to_screen(456, 123) == (0, 0)


def test_native_projected_roundtrip_no_half_pixel_shift():
    transform = Affine(14, 0, 689766, 0, -14, 3336382)
    for row, col in [(0, 0), (12.25, 7.75), (3000.5, 2000.125)]:
        x, y = pixel_to_projected(transform, row, col)
        assert projected_to_pixel(transform, x, y) == pytest.approx((row, col), abs=1e-9)
    assert pixel_to_projected(transform, 0, 0) == (689773, 3336375)


def test_truth_roundtrip_and_pair_guard(tmp_path):
    tr = from_origin(0, 42, 14, 14)
    row = new_row("00_01", "A", "B", "human", "p1", (1.25, 2.75), (2.5, 3.0), tr, tr)
    assert row["review_status"] == "UNVERIFIED"
    path = tmp_path / "truth.csv"
    save_truth(path, [row])
    assert float(load_truth(path, "00_01", "A", "B")[0]["col_i"]) == 2.75
    with pytest.raises(ValueError, match="another pair"):
        load_truth(path, "02_03", "C", "D")


def test_registration_metrics_known_errors_and_denominators():
    points = [dict(review_status="APPROVED", split="TEST", row_i=0, col_i=0,
                   row_j=0, col_j=-1),
              dict(review_status="APPROVED", split="TEST", row_i=0, col_i=0,
                   row_j=0, col_j=-2),
              dict(review_status="APPROVED", split="TEST", row_i=0, col_i=0,
                   row_j=0, col_j=-4)]
    result = evaluate_pair("00_01", points, lambda row, col: (row, col), "SUCCESS")
    m = result["metrics"]
    assert m["rmse_px"] == pytest.approx(np.sqrt(7))
    assert m["p95_px"] == pytest.approx(3.8)
    assert m["max_px"] == 4
    assert [m[f"pck_{n}px"] for n in (1, 2, 3)] == pytest.approx([1/3, 2/3, 2/3])
    second = evaluate_pair("02_03", [points[0]], lambda row, col: (row, col-1), "SUCCESS")
    failure = evaluate_pair("05_07", [], None, "FAIL")
    summary = summarize_pairs([result, second, failure], {"00_01", "02_03", "05_07"})
    assert summary["eligible_pairs"] == 3 and summary["failed_pairs"] == 1
    assert summary["macro_mean_success_only"]["rmse_px"] == pytest.approx(np.sqrt(7)/2)
    with pytest.raises(ValueError, match="APPROVED TEST"):
        evaluate_pair("00_01", [{**points[0], "split": "DEVELOPMENT"}], lambda r, c: (r, c), "SUCCESS")
    with pytest.raises(ValueError, match="ineligible"):
        summarize_pairs([result], {"02_03"})  # no-overlap pair must never be FAIL


def test_best_tile_is_native_mask_verified():
    profile = dict(driver="GTiff", width=8, height=8, count=1, dtype="uint8",
                   crs="EPSG:32650", transform=from_origin(0, 112, 14, 14), nodata=0)
    with MemoryFile() as left, MemoryFile() as right:
        with left.open(**profile) as a, right.open(**profile) as b:
            data = np.zeros((8, 8), dtype=np.uint8)
            data[4:8, 4:8] = 1
            a.write(data, 1)
            b.write(data, 1)
            fraction, count, wa, wb = best_valid_tile(a, b, tile_size=4)
            assert (fraction, count) == (1.0, 16)
            assert (wa.row_off, wa.col_off) == (4, 4)


def test_spatial_role_checker_flags_touching_dev_test(tmp_path):
    from rasterio.warp import transform_geom
    def feature(key, left):
        coords = [[left, 0], [left+100, 0], [left+100, 100], [left, 100], [left, 0]]
        return {"type": "Feature", "properties": {"region_id": key},
                "geometry": transform_geom("EPSG:32650", "EPSG:4326",
                                           {"type": "Polygon", "coordinates": [coords]})}
    old = tmp_path / "old.geojson"
    new = tmp_path / "new.geojson"
    old.write_text(json.dumps({"type": "FeatureCollection", "features": [feature("a", 500000)]}))
    new.write_text(json.dumps({"type": "FeatureCollection", "features": [feature("b", 500200)]}))
    review = tmp_path / "review.csv"
    with review.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["region_id", "role", "review_status"])
        writer.writeheader()
        writer.writerows([{"region_id": "a", "role": "DEVELOPMENT", "review_status": "APPROVED"},
                          {"region_id": "b", "role": "TEST", "review_status": "APPROVED"}])
    assert len(inspect(review, old, new, buffer_m=140)["conflicts"]) == 1
    assert inspect(review, old, new, buffer_m=50)["ready_for_spatial_split"] is True


def test_existing_radiometric_formula_mapping_on_shared_support():
    a = np.array([[1., 3., 100.]])
    b = np.array([[2., 4., -100.]])
    mask = np.array([[True, True, False]])
    assert compute_pair_mamd(a, b, mask) == pytest.approx(1.)
    assert compute_pair_msdd(a, b, mask) == pytest.approx(0.)
    assert compute_pair_rdd(a, b, mask) == pytest.approx(1.)
