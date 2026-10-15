from __future__ import annotations

import json

import pytest
from shapely.geometry import box, mapping

from benchmark_v1_radiometric_r15.report import build_aggregate_outputs
from benchmark_v1_radiometric_r15.storage import write_candidate_artifacts


PAIRS = ("00_01", "02_03", "05_07", "08_10")


def _feature(pair, cid, geometry, role):
    return {"type": "Feature", "geometry": mapping(geometry), "properties": {
        "pair_id": pair, "candidate_id": cid, "role": role,
        "grid_cell_id": "g0_0", "suggested_label": "UNKNOWN",
        "suggestion_confidence": "LOW", "review_status": "UNVERIFIED",
        "eligible_for_test": False, "valid_support_fraction": 1.0,
        "strong_edge_fraction": .05, "texture_strength": 4.0}}


def _run(root, pairs=PAIRS):
    run = root / "run"; run.mkdir()
    manifest = {"run_id": "r15_test", "config": {"window_px": 32},
                "generated_pairs": list(pairs), "protected_file_sha256": {"inventory.csv": "abc"},
                "pair_input_headers": {pair: {
                    "scene_i": {f"B{band}": {"usable": band <= 3} for band in range(1, 17)},
                    "scene_j": {f"B{band}": {"usable": band <= 3} for band in range(1, 17)},
                } for pair in pairs}}
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for pair in pairs:
        write_candidate_artifacts(run / f"{pair}.geojson", [_feature(pair, pair+"_c1", box(1050,1050,2000,2000), "DEVELOPMENT")])
        (run / f"{pair}.quality.json").write_text(json.dumps({"pair_id": pair, "diagnostics": {
            "windows_examined": 10, "quality_accepted_before_cap": 1, "rejected_windows": []}}), encoding="utf-8")
    return run


def test_aggregate_requires_exact_four_pair_set(tmp_path):
    run = _run(tmp_path, pairs=PAIRS[:3])
    with pytest.raises(ValueError, match="exactly the four"):
        build_aggregate_outputs(run, tmp_path / "docs")


def test_aggregate_reconciles_counts_map_and_unknown_unverified_labels(tmp_path):
    run = _run(tmp_path)
    out = build_aggregate_outputs(run, tmp_path / "docs")
    report = (out / "candidate_quality_report.md").read_text(encoding="utf-8")
    assert "Total automatic candidates: 4" in report
    assert "APPROVED: 0" in report
    assert "| Common V bands |" in report
    assert "| 3 |" in report
    assert "B4,B5,B6" in report
    assert (out / "candidate_spatial_distribution.png").stat().st_size > 1000
    assert (out / "radiometric_r1_5_auto_roi_usage.md").is_file()
    leakage = json.loads((out / "cross_pair_leakage_audit.json").read_text(encoding="utf-8"))
    assert len(leakage["overlaps"]) == 6


def test_aggregate_refuses_human_reviewed_or_mismatched_pair_output(tmp_path):
    run = _run(tmp_path)
    feature_path = run / "00_01.geojson"
    data = json.loads(feature_path.read_text(encoding="utf-8"))
    data["features"][0]["properties"]["review_status"] = "APPROVED"
    feature_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="automatically approved or human state"):
        build_aggregate_outputs(run, tmp_path / "docs")
