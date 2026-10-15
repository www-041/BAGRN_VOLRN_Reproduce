import numpy as np
import json
import csv
import hashlib
from pathlib import Path

import pytest

from src.benchmark_v1_spectralmatch_phase2.scoring import (
    audit_mosaic_arrays,
    score_pair_arrays,
    score_submission,
)
from src.benchmark_v1_spectralmatch_phase2 import scoring


def test_pair_score_refuses_to_shrink_support_when_an_output_pixel_is_missing():
    valid_i = np.array([[1, 1], [0, 1]], dtype=bool)
    valid_j = np.array([[1, 0], [0, 1]], dtype=bool)
    oi = np.array([[1.0, 2.0], [0.0, 3.0]])
    oj = np.array([[1.5, 0.0], [0.0, 3.5]])
    output_valid_i = np.ones((2, 2), dtype=bool)
    output_valid_j = np.array([[1, 0], [1, 0]], dtype=bool)

    result = score_pair_arrays(oi, oj, valid_i, valid_j, output_valid_i, output_valid_j)

    assert result["status"] == "PARTIAL_OUTPUT"
    assert result["input_support_pixels"] == 2
    assert result["evaluated_pixels"] == 1
    assert result["missing_pixels"] == 1
    assert result["metrics"] is None


def test_pair_score_reports_exact_metrics_when_support_is_complete():
    valid = np.array([[1, 1], [0, 1]], dtype=bool)
    result = score_pair_arrays(
        np.array([[1.0, 2.0], [0.0, 3.0]]), np.array([[2.0, 2.0], [0.0, 5.0]]),
        valid, valid, valid, valid,
    )
    assert result["status"] == "EVALUABLE"
    assert result["input_support_pixels"] == 3
    assert result["metrics"]["direct"]["mae"] == 1.0
    assert result["metrics"]["direct"]["rmse"] > 1.0


def test_mosaic_audit_counts_missing_and_extra_without_changing_denominator():
    expected = np.array([[1, 1, 0]], dtype=bool)
    result = audit_mosaic_arrays(expected, np.array([[4.0, np.nan, 2.0]]), np.array([[1, 0, 1]], dtype=bool))
    assert result == {
        "status": "COVERAGE_INVALID", "expected_pixels": 2, "output_valid_inside": 1,
        "missing_pixels": 1, "extra_pixels": 1, "coverage": 0.5,
    }


def test_crop_submission_never_eligible_for_formal_ranking():
    result = score_submission("unused", {}, "B3", expected_scope="FORMAL_FULL", scope="DIAGNOSTIC_CROP_ONLY")
    assert result["status"] == "NOT_RUN_B3"
    formal = score_submission("unused", {}, "B1", expected_scope="FORMAL_FULL", scope="DIAGNOSTIC_CROP_ONLY")
    assert formal["status"] == "SCOPE_REJECTED"
    assert formal["formal_ranking_eligible"] is False


def _write_raster(path, data, *, nodata=-9999.0, mask=None):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine

    if np.issubdtype(data.dtype, np.integer) and nodata == -9999.0:
        nodata = None
    profile = {"driver": "GTiff", "width": data.shape[1], "height": data.shape[0], "count": 1,
               "dtype": str(data.dtype), "crs": "EPSG:32650", "transform": Affine(14, 0, 100, 0, -14, 200),
               "nodata": nodata}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        if mask is not None:
            dst.write_mask(mask.astype("uint8") * 255)
    return path


def _tiny_freeze(tmp_path):
    np = pytest.importorskip("numpy")
    freeze = tmp_path / "freeze"
    aligned = freeze / "aligned_inputs"
    aligned.mkdir(parents=True)
    artifacts = []
    scene_names = []
    masks = np.ones((128, 128), dtype=np.uint8)
    data = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    for index in range(13):
        name = f"scene{index:02d}"
        scene_names.append(name)
        image_path = aligned / f"scene_{index:03d}_B9_input.tif"
        mask_path = aligned / f"scene_{index:03d}_input_mask.tif"
        _write_raster(image_path, data + index)
        _write_raster(mask_path, masks)
        artifacts.append({"scene_index": index, "aligned_path": str(image_path), "mask_path": str(mask_path),
                          "source_path": f"{name}_B9.TIF"})
    pairs = []
    for i in range(13):
        for j in range(i + 1, 13):
            pair_id = f"{i:02d}_{j:02d}"
            pairs.append({"pair_id": pair_id, "scene_i": scene_names[i], "scene_j": scene_names[j],
                          "status": "OVERLAP" if len(pairs) < 48 else "NO_VALID_OVERLAP",
                          "aligned_valid_overlap_pixels": str(128 * 128 if len(pairs) < 48 else 0)})
    (freeze / "pair_overlap_inventory.csv").write_text(
        "pair_id,scene_i,scene_j,status,aligned_valid_overlap_pixels\n" +
        "".join(",".join(row[key] for key in ("pair_id", "scene_i", "scene_j", "status", "aligned_valid_overlap_pixels")) + "\n" for row in pairs),
        encoding="utf-8",
    )
    manifest = {"status": "FROZEN", "protocol_id": "DZ01V_UNIFIED_A_B_V1_20261013",
                "input_freeze_sha256": "a" * 64, "geometry_freeze_sha256": "b" * 64,
                "grid": {"crs": "EPSG:32650", "transform": [14, 0, 100, 0, -14, 200], "width": 128, "height": 128},
                "aligned_input_artifacts": artifacts}
    (freeze / "frozen_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return freeze, manifest, scene_names, data


def test_formal_a_b1_b2_accept_exact_synthetic_support_and_report_status(tmp_path, monkeypatch):
    pytest.importorskip("rasterio")
    monkeypatch.setattr(scoring, "_audit_phase1_freeze", lambda root: {"status": "PASS", "errors": []})
    from src.benchmark_v1_unified import radiometric
    monkeypatch.setattr(
        radiometric, "compute_pair_distribution_metrics",
        lambda x, y, support: radiometric.PairDistributionMetrics(
            "EVALUABLE", int(np.asarray(support).sum()), float(np.mean(x)), float(np.mean(y)),
            float(np.std(x)), float(np.std(y)), abs(float(np.mean(x))-float(np.mean(y))),
            abs(float(np.std(x))-float(np.std(y))), abs(float(np.mean(x))-float(np.mean(y))),
        ),
    )
    freeze, manifest, scene_names, data = _tiny_freeze(tmp_path)
    output_paths = {}
    for index, name in enumerate(scene_names):
        path = tmp_path / f"corrected_{name}.tif"
        _write_raster(path, data + index, mask=np.ones((128, 128), bool))
        output_paths[name] = str(path)
    mosaic_path = tmp_path / "mosaic.tif"
    _write_raster(mosaic_path, data, mask=np.ones((128, 128), bool))
    submission = {"schema_version": 1, "method_id": "synthetic", "run_id": "synth-run", "method_kind": "FULL_MOSAIC",
                  "status": "COMPLETED", "input_freeze_sha256": "a" * 64,
                  "geometry_freeze_sha256": "b" * 64, "band": "B9", "pixel_value_unit": "UNKNOWN",
                  "corrected_scene_paths": output_paths, "output_mosaic_path": str(mosaic_path)}
    a = score_submission(freeze, submission, "A")
    b1 = score_submission(freeze, submission, "B1")
    assert a["status"] == "PASS"
    assert a["expected_union_pixels"] == 128 * 128
    assert b1["status"] == "FULL_SUPPORT"
    assert b1["expected_positive_pairs"] == 48
    assert b1["pair_macro"]["mae"]["all_pairs_eligible"] is True

    support = np.ones((128, 128), dtype=bool)
    import hashlib
    window_path = freeze / "fixed_windows.csv"
    support_hash = hashlib.sha256(np.packbits(support, bitorder="little").tobytes()).hexdigest()
    window_path.write_text(
        "window_id,pair_id,row_off,col_off,height,width,valid_count,support_sha256\n"
        f"00_01_r00000_c00000,00_01,0,0,128,128,16384,{support_hash}\n", encoding="utf-8"
    )
    b2 = score_submission(freeze, submission, "B2")
    assert b2["status"] == "FULL_SUPPORT"
    assert b2["evaluated_windows"] == 1


def test_experiment_a_preserves_regime_specific_ranking_eligibility(tmp_path, monkeypatch):
    freeze, manifest, _, _ = _tiny_freeze(tmp_path)
    (freeze / "frozen_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(scoring, "_audit_phase1_freeze", lambda root: {"status": "PASS", "errors": []})
    monkeypatch.setattr(scoring, "_score_a", lambda root, manifest, submission: {
        "status": "PASS", "formal_ranking_eligible": True, "scope": "FORMAL_FULL",
    })
    submission = {"schema_version": 1, "method_id": "synthetic", "run_id": "run",
                  "method_kind": "FULL_MOSAIC", "status": "COMPLETED",
                  "input_freeze_sha256": manifest["input_freeze_sha256"],
                  "geometry_freeze_sha256": manifest["geometry_freeze_sha256"],
                  "band": "B9", "pixel_value_unit": "UNKNOWN"}
    result = score_submission(freeze, submission, "A")
    assert result["status"] == "PASS"
    assert result["formal_ranking_eligible"] is True


def test_formal_a_rejects_ambiguous_nodata_and_wrong_grid(tmp_path, monkeypatch):
    pytest.importorskip("rasterio")
    monkeypatch.setattr(scoring, "_audit_phase1_freeze", lambda root: {"status": "PASS", "errors": []})
    freeze, _, _, data = _tiny_freeze(tmp_path)
    ambiguous = tmp_path / "ambiguous.tif"
    _write_raster(ambiguous, data, nodata=None)
    submission = {"schema_version": 1, "method_id": "synthetic", "run_id": "synth-run", "method_kind": "FULL_MOSAIC",
                  "status": "COMPLETED", "input_freeze_sha256": "a" * 64,
                  "geometry_freeze_sha256": "b" * 64, "band": "B9", "pixel_value_unit": "UNKNOWN",
                  "output_mosaic_path": str(ambiguous)}
    assert score_submission(freeze, submission, "A")["status"] == "AMBIGUOUS_OUTPUT_MASK"
    wrong_grid = tmp_path / "wrong_grid.tif"
    _write_raster(wrong_grid, data[:64], nodata=-9999.0)
    submission["output_mosaic_path"] = str(wrong_grid)
    assert score_submission(freeze, submission, "A")["status"] == "INVALID_GRID"
    submission["output_mosaic_path"] = str(tmp_path / "missing.tif")
    assert score_submission(freeze, submission, "A")["status"] == "MISSING_OUTPUT"


def test_b1_missing_scene_outputs_are_kept_in_denominator(tmp_path, monkeypatch):
    pytest.importorskip("rasterio")
    monkeypatch.setattr(scoring, "_audit_phase1_freeze", lambda root: {"status": "PASS", "errors": []})
    freeze, _, _, _ = _tiny_freeze(tmp_path)
    submission = {"schema_version": 1, "method_id": "synthetic", "run_id": "synth-run", "method_kind": "SCENE_OUTPUTS",
                  "status": "COMPLETED", "input_freeze_sha256": "a" * 64,
                  "geometry_freeze_sha256": "b" * 64, "band": "B9", "pixel_value_unit": "UNKNOWN"}
    result = score_submission(freeze, submission, "B1")
    assert result["status"] == "MISSING_OUTPUT"
    assert result["missing_scene_outputs"] == list(range(13))


def test_formal_scores_are_blocked_when_locked_code_hashes_drift(tmp_path, monkeypatch):
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    (freeze / "frozen_manifest.json").write_text(json.dumps({
        "status": "FROZEN", "protocol_id": "DZ01V_UNIFIED_A_B_V1_20261013",
        "input_freeze_sha256": "a" * 64, "geometry_freeze_sha256": "b" * 64,
    }), encoding="utf-8")
    monkeypatch.setattr(scoring, "_audit_phase1_freeze", lambda root: {
        "status": "FREEZE_INVALID", "errors": ["CODE_FILE_HASH_MISMATCH:6"]
    })
    submission = {"schema_version": 1, "method_id": "synthetic", "run_id": "run", "method_kind": "SCENE_OUTPUTS",
                  "status": "COMPLETED", "input_freeze_sha256": "a" * 64,
                  "geometry_freeze_sha256": "b" * 64, "band": "B9", "pixel_value_unit": "UNKNOWN"}
    result = score_submission(freeze, submission, "B1")
    assert result["status"] == "INVALID_FREEZE"
    assert result["freeze_errors"] == ["CODE_FILE_HASH_MISMATCH:6"]
    assert result["formal_ranking_eligible"] is False
