import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from src.benchmark_v1_unified.freeze import write_lock
from src.benchmark_v1_unified.submission_scoring import evaluate_submission


def _write(path, values, mask, transform, *, dtype="float32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
                       count=1, dtype=dtype, crs="EPSG:32650", transform=transform,
                       nodata=-9999 if dtype != "uint8" else 0) as ds:
        ds.write(values.astype(dtype), 1)
        ds.write_mask(mask.astype("uint8") * 255)


def make_thirteen_scene_fixture(tmp_path):
    freeze = tmp_path / "freeze"
    aligned = freeze / "aligned_inputs"
    aligned.mkdir(parents=True)
    shape = (128, 128)
    transform = Affine(14, 0, 633346, 0, -14, 3359874)
    rr, cc = np.mgrid[:128, :128]
    base = (100 + rr * 0.5 + cc * 0.25 + np.sin(cc / 7) * 3).astype(np.float32)
    ids = [f"SCENE_{i:02d}" for i in range(13)]
    masks = []
    input_paths = []
    mask_paths = []
    for i in range(13):
        mask = np.ones(shape, dtype=bool)
        if i < 12:
            mask[120:, :] = False
            for k in range(12):
                if k != i:
                    mask[:7, k * 7:(k + 1) * 7] = False
        else:
            mask[:] = False
            mask[120:, :] = True
        image_path = aligned / f"scene_{i:03d}_B9_input.tif"
        mask_path = aligned / f"scene_{i:03d}_input_mask.tif"
        _write(image_path, base + i, np.ones(shape, bool), transform)
        _write(mask_path, mask.astype(np.uint8), np.ones(shape, bool), transform, dtype="uint8")
        masks.append(mask)
        input_paths.append(image_path)
        mask_paths.append(mask_path)

    pairs_path = freeze / "pair_overlap_inventory.csv"
    windows_path = freeze / "fixed_windows.csv"
    pair_rows, window_rows = [], []
    for i in range(13):
        for j in range(i + 1, 13):
            pid = f"{i:02d}_{j:02d}"
            support = masks[i] & masks[j]
            pair_rows.append({"pair_id": pid, "scene_i": ids[i], "scene_j": ids[j],
                              "status": "OVERLAP" if support.any() else "NO_VALID_OVERLAP",
                              "aligned_valid_overlap_pixels": int(support.sum())})
            if support.any():
                window_rows.append({"window_id": f"{pid}_r00000_c00000", "pair_id": pid,
                                    "row_off": 0, "col_off": 0, "height": 128, "width": 128,
                                    "valid_count": int(support.sum()),
                                    "support_sha256": hashlib.sha256(np.packbits(support, bitorder="little").tobytes()).hexdigest(),
                                    "time_stratum": "SYNTHETIC", "quality_i": "PASS", "quality_j": "PASS"})
    for path, rows in ((pairs_path, pair_rows), (windows_path, window_rows)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    frozen = {"protocol_id": "DZ01V_UNIFIED_A_B_V1_20261013", "scene_count": 13,
              "grid": {"crs": "EPSG:32650", "transform": list(transform)[:6], "width": 128, "height": 128},
              "input_freeze_sha256": "a" * 64, "geometry_freeze_sha256": "b" * 64,
              "aligned_input_artifacts": [{"scene_index": i} for i in range(13)]}
    (freeze / "frozen_manifest.json").write_text(json.dumps(frozen), encoding="utf-8")
    write_lock(freeze, [*input_paths, *mask_paths, pairs_path, windows_path, freeze / "frozen_manifest.json"])

    output = tmp_path / "method"
    corrected_paths, corrected, labels = {}, [], np.zeros(shape, dtype=np.uint8)
    for i, scene_id in enumerate(ids):
        values = base + i + (0.25 * i)
        path = output / f"scene_{i:03d}.tif"
        _write(path, values, masks[i], transform)
        corrected_paths[scene_id] = str(path)
        corrected.append(values.astype(np.float32))
    for i in range(12):
        labels[:7, i * 7:(i + 1) * 7] = i
    labels[120:, :] = 12
    mosaic_mask = np.logical_or.reduce(masks)
    mosaic = np.zeros(shape, dtype=np.float32)
    for i in range(13):
        take = mosaic_mask & (labels == i)
        mosaic[take] = corrected[i][take]
    mosaic_path, mosaic_mask_path, labels_path = output / "mosaic.tif", output / "mosaic_mask.tif", output / "source_labels.tif"
    _write(mosaic_path, mosaic, mosaic_mask, transform)
    _write(mosaic_mask_path, mosaic_mask.astype(np.uint8), np.ones(shape, bool), transform, dtype="uint8")
    _write(labels_path, labels, mosaic_mask, transform, dtype="uint8")
    submission = {"schema_version": 1, "method_id": "synthetic-13", "run_id": "smoke-1",
                  "method_kind": "FULL_MOSAIC", "input_freeze_sha256": "a" * 64,
                  "geometry_freeze_sha256": "b" * 64, "band": "B9", "pixel_value_unit": "UNKNOWN",
                  "status": "COMPLETED", "corrected_scene_paths": corrected_paths,
                  "output_mosaic_path": str(mosaic_path), "output_mosaic_mask_path": str(mosaic_mask_path),
                  "source_label_path": str(labels_path), "seam_type": "HARD",
                  "resource_budget": {"max_rss_bytes": 2_000_000_000, "gpu_memory_bytes": None},
                  "runtime": {"status": "NOT_MEASURED", "wall_seconds": None,
                              "wall_measurement_source": "NOT_MEASURED",
                              "rss_status": "NOT_MEASURED", "peak_rss_bytes": None,
                              "rss_sample_count": 0, "rss_measurement_source": "NOT_MEASURED",
                              "gpu_status": "NOT_MEASURED", "peak_gpu_memory_bytes": None,
                              "gpu_sample_count": 0, "gpu_measurement_source": "NOT_MEASURED",
                              "hard_os_memory_limit": False}}
    return freeze, submission, masks


def test_geo_tiff_submission_runs_A_B1_B2_with_all_fixture_pairs_and_windows(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    result = evaluate_submission(freeze, submission, submission_dir=tmp_path / "method",
                                regimes=("A", "B1", "B2"))
    assert result["submission_audit"]["status"] == "PASS"
    assert result["A"]["mosaic_coverage"]["coverage"] == 1.0
    assert result["A"]["mosaic_coverage"]["holes"] == 0
    assert result["A"]["mosaic_coverage"]["extras"] == 0
    assert result["A"]["seam"]["status"] == "EVALUABLE"
    assert len(result["A"]["structure"]["structure_by_scene"]) == 13
    assert len(result["scene_structure"]) == 13
    assert result["B1"]["expected_pair_rows"] == 78
    assert result["B1"]["positive_overlap_pairs"] == 66
    assert result["B1"]["no_valid_overlap_pairs"] == 12
    assert set(result["B1"]["pair_rows"][0]["metrics"]["direct"]) >= {"mae", "rmse", "bias", "p95ae"}
    assert set(result["B1"]["pair_rows"][0]["metrics"]["distribution"]) >= {"mamd", "msdd", "rdd"}
    assert result["B2"]["window_count"] == 66
    assert len(result["B2"]["window_rows"]) == 66
    assert len(result["B2"]["structure_by_scene_window"]) == 132
    assert result["runtime"]["rss_status"] == "NOT_MEASURED"
    assert result["runtime"]["rss_measurement_source"] == "NOT_MEASURED"
    assert result["runtime"]["measurement_provenance_status"] == "INCOMPLETE"
    assert result["ranking_eligible"] is False


def test_missing_source_labels_are_explicitly_not_evaluable(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    submission["source_label_path"] = None
    result = evaluate_submission(freeze, submission, submission_dir=tmp_path / "method", regimes=("A",))
    assert result["A"]["seam"]["status"] == "NOT_EVALUABLE_NO_PROVENANCE"


def test_b3_without_truth_artifact_is_not_substituted_by_real_scene_metrics(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    result = evaluate_submission(freeze, submission, submission_dir=tmp_path / "method", regimes=("B3",))
    assert result["B3"]["status"] == "NOT_RUN_SYNTHETIC_METHOD_OUTPUT_REQUIRED"


def test_b3_status_does_not_require_real_scene_artifacts(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    submission.pop("corrected_scene_paths")
    submission.pop("output_mosaic_path")
    submission.pop("output_mosaic_mask_path")
    result = evaluate_submission(freeze, submission, submission_dir=tmp_path / "method", regimes=("B3",))
    assert result["B3"]["status"] == "NOT_RUN_SYNTHETIC_METHOD_OUTPUT_REQUIRED"


def test_partial_scene_support_is_reported_and_not_ranked(tmp_path):
    freeze, submission, masks = make_thirteen_scene_fixture(tmp_path)
    output_path = Path(submission["corrected_scene_paths"]["SCENE_00"])
    partial_mask = masks[0].copy()
    partial_mask[20:30, 20:30] = False
    values = np.full((128, 128), 100, dtype=np.float32)
    _write(output_path, values, partial_mask, Affine(14, 0, 633346, 0, -14, 3359874))
    result = evaluate_submission(freeze, submission, submission_dir=tmp_path / "method", regimes=("B1", "B2"))
    row = next(row for row in result["B1"]["pair_rows"] if row["pair_id"] == "00_01")
    assert row["status"] == "PARTIAL_OUTPUT"
    assert row["metrics"]["direct"] is None
    assert result["B1"]["status"] == "PARTIAL_OUTPUT"
    assert result["ranking_eligible"] is False


def test_b1_macro_excludes_no_overlap_pairs_but_retains_them_in_denominator(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    result = evaluate_submission(freeze, submission, submission_dir=tmp_path / "method", regimes=("B1",))
    assert result["B1"]["expected_pair_rows"] == 78
    assert result["B1"]["metric_summaries"]["mae"]["expected_pairs"] == 66
    assert result["B1"]["metric_summaries"]["mae"]["evaluable_pairs"] == 66


def test_resource_budget_preflight_stops_before_opening_method_rasters(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    submission["resource_budget"]["max_rss_bytes"] = 1
    submission["corrected_scene_paths"] = {"SCENE_00": "does-not-exist.tif"}
    result = evaluate_submission(freeze, submission, submission_dir=tmp_path / "method", regimes=("B1",))
    assert result["status"] == "RESOURCE_PREFLIGHT_FAILED"
    assert result["resource_preflight"]["status"] == "BUDGET_MISSING_OR_BELOW_ESTIMATE"


def test_missing_mosaic_is_an_explicit_artifact_status_not_an_exception(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    submission["output_mosaic_path"] = "missing-mosaic.tif"
    result = evaluate_submission(freeze, submission, submission_dir=tmp_path / "method", regimes=("A",))
    assert result["status"] == "MISSING_ARTIFACT"
    assert "MISSING_ARTIFACT" in result["error"]


def test_scoring_cli_writes_a_machine_readable_report(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    submission_path = tmp_path / "method" / "method_submission.json"
    submission_path.write_text(json.dumps(submission), encoding="utf-8")
    report_path = tmp_path / "method" / "evaluation.json"
    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "score_unified_v1_submission.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--freeze", str(freeze), "--method-submission", str(submission_path),
         "--regimes", "A", "B1", "B2", "--output", str(report_path)],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "EVALUATED"
    assert report["B1"]["expected_pair_rows"] == 78


def test_submission_enum_and_nonfinite_budget_are_rejected(tmp_path):
    freeze, submission, _ = make_thirteen_scene_fixture(tmp_path)
    submission["seam_type"] = "MYSTERY"
    result = evaluate_submission(freeze, submission, submission_dir=tmp_path / "method", regimes=("A",))
    assert result["status"] == "SUBMISSION_INVALID"
    assert "INVALID_SEAM_TYPE" in result["submission_audit"]["errors"]


def test_a_unique_source_structure_is_null_when_that_scene_output_is_partial(tmp_path):
    freeze, submission, masks = make_thirteen_scene_fixture(tmp_path)
    scene0 = Path(submission["corrected_scene_paths"]["SCENE_00"])
    partial = masks[0].copy()
    partial[:7, :7] = False
    _write(scene0, np.full((128, 128), 100, dtype=np.float32), partial,
           Affine(14, 0, 633346, 0, -14, 3359874))
    result = evaluate_submission(freeze, submission, submission_dir=tmp_path / "method", regimes=("A",))
    assert result["A"]["single_source_structure"]["status"] == "PARTIAL_OUTPUT"
    assert result["A"]["single_source_structure"]["structure"] is None
