from __future__ import annotations

import json
import numpy as np
import rasterio
import inspect
from rasterio.transform import from_origin

from src.multiscene_sift.structural_metrics import stream_structure_metrics, structure_metrics
from src.pipeline.final_metrics import _label_legality, classify_structural_gate, compute_final_metrics
from src.seam_local.config import SeamLocalRuntimeConfig
from src.seam_local.seam import SeamResult
from src.seam_local.source_side import SourceSideResult
from src.task16_volrn_comparison import _build_report, run_strict_local_ablation, run_volrn_end_to_end
from scripts.run_task14_13scene_scale import run_task14
from scripts.run_task14a_resume_13 import _pair_normalized_transition_zone


def _grid(width: int, height: int) -> dict:
    return {
        "crs": "EPSG:32650", "width": width, "height": height,
        "transform": list(from_origin(100.0, 200.0, 1.0, 1.0)),
    }


def _write(path, data, grid):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype="float32", crs=grid["crs"],
                       transform=from_origin(100.0, 200.0, 1.0, 1.0), nodata=np.nan) as dst:
        dst.write(data.astype(np.float32), 1)


def test_structural_metrics_are_identity_and_nodata_safe():
    image = np.arange(100, dtype=np.float64).reshape(10, 10)
    valid = np.ones_like(image, dtype=bool)
    valid[:2, :] = False
    image[~valid] = np.nan
    metrics = structure_metrics(image, image.copy(), valid)
    assert metrics["gradient_magnitude_ncc"] == 1.0
    assert metrics["cgl_rad"] == 0.0
    assert metrics["support_pixels"] > 0


def test_streaming_structure_metrics_matches_identity_contract(tmp_path):
    grid = _grid(12, 12)
    image = np.arange(144, dtype=np.float32).reshape(12, 12)
    image[:1, :] = np.nan
    path = tmp_path / "scene.tif"
    _write(path, image, grid)
    mask = np.isfinite(image)
    metrics = stream_structure_metrics(path, path, mask, tile_size=4, halo=1)
    assert metrics["finite"] is True
    assert metrics["gradient_magnitude_ncc"] == 1.0


def test_ncc_quality_review_does_not_fail_finite_gate():
    gate = classify_structural_gate([{"finite": True, "gradient_magnitude_ncc": 0.8}])
    assert gate == {"finite": True, "quality_review": "REVIEW"}


def test_task14_stage02_registration_parameters_are_defined():
    signature = inspect.signature(run_task14)
    assert signature.parameters["band"].default == "B9"
    assert signature.parameters["match_max_side"].default == 1024
    assert signature.parameters["ransac_threshold"].default == 2.0
    assert signature.parameters["random_seed"].default == 0


def test_strict_ablation_uses_one_v1_weight_layout(tmp_path):
    grid = _grid(8, 8)
    masks = [np.ones((8, 8), dtype=bool), np.ones((8, 8), dtype=bool)]
    paths = []
    ours = []
    volrn = []
    weights = []
    for index in range(2):
        base = np.full((8, 8), index + 1, dtype=np.float32)
        p = tmp_path / f"base{index}.tif"; _write(p, base, grid); paths.append(p)
        p = tmp_path / f"ours{index}.tif"; _write(p, base + 2, grid); ours.append(p)
        p = tmp_path / f"volrn{index}.tif"; _write(p, base + 3, grid); volrn.append(p)
        p = tmp_path / f"weight{index}.tif"; _write(p, np.ones((8, 8), dtype=np.float32), grid); weights.append(p)
    result = run_strict_local_ablation(paths, ours, volrn, weights, masks, ["a", "b"], grid, tmp_path / "strict")
    assert result["weight_manifest"]["same_weight_hashes"] is True
    assert set(result["routes"]) == {"A0_BAGRN_V1", "A1_OURS_V1", "A2_VOLRN_V1"}
    assert result["weight_manifest"]["transition_balance_threshold"] == 0.25
    assert set(result["fixed_support"]) == set(result["routes"])
    assert (tmp_path / "strict/fixed_support_metrics.json").is_file()


def test_volrn_end_to_end_calls_shared_refinement(tmp_path):
    grid = _grid(8, 8)
    paths = []
    masks = [np.ones((8, 8), dtype=bool), np.ones((8, 8), dtype=bool)]
    for index in range(2):
        path = tmp_path / f"scene{index}.tif"
        _write(path, np.arange(64, dtype=np.float32).reshape(8, 8) + index, grid)
        paths.append(path)
    seam = SeamResult("vertical", np.column_stack((np.arange(8), np.full(8, 4))), 0.0, 0.0, 0.0, "OK")
    calls = []

    def fake_refine(a, b, va, vb, initial, *, refine_half_width, cost_config):
        calls.append((a.shape, refine_half_width, cost_config))
        return initial

    result = run_volrn_end_to_end(
        paths, masks, [{"pair_id": "00_01", "scene_i": 0, "scene_j": 1, "initial_seam": seam}],
        grid, tmp_path / "e2e", refine_fn=fake_refine, runtime_config=SeamLocalRuntimeConfig(),
    )
    assert len(calls) == 1
    assert result["shared_refine_function"] == "fake_refine"
    assert (tmp_path / "e2e/mosaic.tif").is_file()


def test_e2e_unresolved_source_side_has_no_initial_or_scene_fallback(monkeypatch, tmp_path):
    grid = _grid(8, 8)
    paths = []
    masks = [np.ones((8, 8), dtype=bool), np.ones((8, 8), dtype=bool)]
    for index in range(2):
        path = tmp_path / f"scene{index}.tif"
        _write(path, np.arange(64, dtype=np.float32).reshape(8, 8) + index, grid)
        paths.append(path)
    seam = SeamResult("vertical", np.column_stack((np.arange(8), np.full(8, 4))), 0.0, 0.0, 0.0, "OK")

    def fake_refine(a, b, va, vb, initial, *, refine_half_width, cost_config):
        return initial

    def unresolved(*args, **kwargs):
        return SourceSideResult("REQUIRES_MULTISCENE_LABELING", None, None, "TEST", 0.0, {})

    def forbidden_pairwise(*args, **kwargs):
        raise AssertionError("unresolved source-side pair must not create a pairwise field")

    monkeypatch.setattr("src.seam_local.source_side.resolve_source_sides", unresolved)
    monkeypatch.setattr("src.seam_local.multiscene_label.build_pairwise_preference_field", forbidden_pairwise)
    result = run_volrn_end_to_end(
        paths, masks, [{"pair_id": "00_01", "scene_i": 0, "scene_j": 1,
                        "initial_side_1_source": "B", "initial_seam": seam}],
        grid, tmp_path / "e2e", refine_fn=fake_refine, runtime_config=SeamLocalRuntimeConfig(),
    )
    row = result["refinement_rows"][0]
    assert row["refinement_status"] == "OK"
    assert row["source_side_status"] == "REQUIRES_MULTISCENE_LABELING"
    assert row["label_status"] == "REQUIRES_MULTISCENE_LABELING"


def test_e2e_unresolved_labels_propagate_to_top_level_status(monkeypatch, tmp_path):
    grid = _grid(8, 8)
    paths = []
    masks = [np.ones((8, 8), dtype=bool), np.ones((8, 8), dtype=bool)]
    for index in range(2):
        path = tmp_path / f"scene{index}.tif"
        _write(path, np.arange(64, dtype=np.float32).reshape(8, 8) + index, grid)
        paths.append(path)
    seam = SeamResult("vertical", np.column_stack((np.arange(8), np.full(8, 4))), 0.0, 0.0, 0.0, "OK")

    def fake_refine(a, b, va, vb, initial, *, refine_half_width, cost_config):
        return initial

    def unresolved_labels(tile_masks, fields, raw, *, p95_edt, tie_tolerance):
        shape = tile_masks.shape[1:]
        return (
            np.zeros(shape, dtype=np.int16),
            np.zeros(shape, dtype=np.uint8),
            {"cycle_pixels": 0, "top_score_tie_pixels": 0, "interiority_fallback_pixels": 0,
             "resolved_by_unclipped_interiority": 0, "resolved_by_raw_edt": 0,
             "unresolved_pixels": 1, "two_scene_disagreement_pixels": 0,
             "score_margin": np.zeros(shape, dtype=np.float32)},
        )

    monkeypatch.setattr("src.seam_local.adapter.aggregate_labels_with_ties", unresolved_labels)
    result = run_volrn_end_to_end(
        paths, masks, [{"pair_id": "00_01", "scene_i": 0, "scene_j": 1, "initial_seam": seam}],
        grid, tmp_path / "e2e", refine_fn=fake_refine, runtime_config=SeamLocalRuntimeConfig(),
    )
    assert result["labels"]["unresolved_pixels"] > 0
    assert result["status"] == "HARD_STOP_UNRESOLVED_LABELS"


def test_final_label_legality_requires_both_v1_and_v2_clean():
    v1 = _label_legality({"unresolved_pixels": 1, "invalid_label_pixels": 0, "two_scene_disagreement_pixels": 0})
    v2 = _label_legality({"unresolved_pixels": 0, "invalid_label_pixels": 0, "two_scene_disagreement_pixels": 0})
    assert v1["clean"] is False
    assert v2["clean"] is True
    assert v1["measured"] is True and v2["measured"] is True
    assert not (v1["clean"] and v2["clean"])


def test_final_gate_rejects_dirty_v1_even_when_v2_is_clean(tmp_path):
    labels_dir = tmp_path / "stages/08_multiscene_labeling"
    metrics_dir = tmp_path / "stages/11_metrics"
    scale_dir = tmp_path / "stages/12_report"
    mosaic_dir = tmp_path / "stages/10_mosaics"
    labels_dir.joinpath("v1").mkdir(parents=True)
    labels_dir.joinpath("v2").mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    scale_dir.mkdir(parents=True)
    mosaic_dir.mkdir(parents=True)
    clean = {"unresolved_pixels": 0, "invalid_label_pixels": 0, "two_scene_disagreement_pixels": 0}
    (labels_dir / "labeling_summary.json").write_text(json.dumps(clean), encoding="utf-8")
    (labels_dir / "v1/labeling_summary.json").write_text(json.dumps({**clean, "unresolved_pixels": 1}), encoding="utf-8")
    (labels_dir / "v2/labeling_summary.json").write_text(json.dumps(clean), encoding="utf-8")
    boundary = {name: 1.0 for name in ("bagrn_weighted_mae", "v2_weighted_mae", "bagrn_weighted_rdd", "v2_weighted_rdd", "median_bagrn_mae", "median_v2_mae", "median_bagrn_rdd", "median_v2_rdd")}
    (metrics_dir / "metrics_summary.json").write_text(json.dumps({"status": "SUCCESS", "boundary_metrics": boundary, "structural": [{"finite": True, "gradient_magnitude_ncc": 0.8}]}), encoding="utf-8")
    (scale_dir / "scale_summary.json").write_text(json.dumps({"scene_count": 0, "union_valid_pixels": 1}), encoding="utf-8")
    (mosaic_dir / "mosaic_summary.json").write_text(json.dumps({"v0": "BAGRN + weighted feather semantics"}), encoding="utf-8")
    for name in ("v0_bagrn_weighted.tif", "v1_multiscene_label_blend.tif", "v2_local_corrected_multiscene.tif"):
        (mosaic_dir / name).write_bytes(b"test")
    result = compute_final_metrics(tmp_path)
    assert result["quality"]["label_validity"]["V1"]["clean"] is False
    assert result["quality"]["label_validity"]["V2"]["clean"] is True
    assert result["quality"]["labels_clean"] is False
    assert result["quality"]["scientific_gate_status"] == "MIXED_SCALE_REVIEW"


def test_transition_support_uses_pair_normalized_threshold_and_is_scale_invariant():
    wi = np.array([0.9, 0.8, 0.6, 0.5])
    wj = np.array([0.1, 0.2, 0.4, 0.5])
    expected = np.array([False, False, True, True])
    assert np.array_equal(_pair_normalized_transition_zone(wi, wj, 0.25), expected)
    assert np.array_equal(_pair_normalized_transition_zone(10 * wi, 10 * wj, 0.25), expected)


def test_v0_v1_v2_transition_routes_share_one_threshold_definition():
    routes = [
        (np.array([9.0, 8.0, 6.0, 5.0]), np.array([1.0, 2.0, 4.0, 5.0])),
        (np.array([0.9, 0.8, 0.6, 0.5]), np.array([0.1, 0.2, 0.4, 0.5])),
        (np.array([90.0, 80.0, 60.0, 50.0]), np.array([10.0, 20.0, 40.0, 50.0])),
    ]
    masks = [_pair_normalized_transition_zone(left, right, 0.25) for left, right in routes]
    assert all(int(mask.sum()) == 2 for mask in masks)


def test_task16_report_uses_v1_label_derived_wording():
    report_source = inspect.getsource(_build_report)
    assert "same frozen V1 label-derived cosine weight layout" in report_source
    assert "weighted-feather mode" not in report_source
