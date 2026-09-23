import numpy as np
import csv
import json

from src.multiscene_sift.local_affine_consistency_diagnostics import (
    LOCAL_AFFINE_EDGES,
    assign_points_to_overlap_regions,
    fit_affine_least_squares,
    load_local_affine_baseline,
)


def test_fixed_edge_set_is_exactly_three_edges():
    assert LOCAL_AFFINE_EDGES == {
        (0, 6): "GOOD_REFERENCE",
        (2, 5): "HIGH_INLIER_FALSE_GOOD",
        (0, 5): "LOW_SUPPORT_CONTROL",
    }


def test_exact_affine_recovery_with_normalized_coordinates():
    src = np.array([[x, y] for y in range(4) for x in range(4)], dtype=float)
    expected = np.array([[1.02, 0.03, 8.0], [-0.02, 0.98, -4.0], [0, 0, 1]])
    dst = (expected[:2, :2] @ src.T).T + expected[:2, 2]
    result = fit_affine_least_squares(src, dst)
    assert result["status"] == "OK"
    np.testing.assert_allclose(result["matrix_3x3"], expected, atol=1e-10)


def test_region_assignment_uses_final_boundary_in_last_cell():
    xy = np.array([[0, 0], [5, 5], [10, 10], [9.9, 0.1]])
    labels = assign_points_to_overlap_regions(xy, (0, 0, 10, 10), 2)
    np.testing.assert_array_equal(labels, [0, 3, 3, 1])


def test_degenerate_and_insufficient_affine_inputs_are_rejected():
    assert fit_affine_least_squares(np.zeros((11, 2)), np.zeros((11, 2)))["status"] == "INSUFFICIENT_POINTS"
    src = np.column_stack([np.arange(12), np.arange(12)])
    dst = src.copy()
    assert fit_affine_least_squares(src, dst)["status"] == "DEGENERATE"


def test_baseline_loader_freezes_required_edges_and_context(tmp_path):
    edge_dir = tmp_path / "edge_reliability"
    edge_dir.mkdir()
    baseline = {
        "edges": [
            {"edge": [0, 6], "group": "GOOD_REFERENCE", "inliers": 291},
            {"edge": [2, 5], "group": "HIGH_INLIER_FALSE_GOOD", "inliers": 3010},
            {"edge": [0, 5], "group": "LOW_SUPPORT_CONTROL", "inliers": 23},
        ]
    }
    (edge_dir / "00_edge_reliability_baseline.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )
    for name in (
        "01_inlier_reproduction_summary.json",
        "02_inlier_spatial_metrics.json",
        "07_direct_residual_summary.json",
        "08_residual_vs_control_support_summary.json",
        "11_good_vs_false_good_comparison.json",
        "12_edge_diagnoses.json",
        "13_edge_reliability_conclusion.json",
    ):
        (edge_dir / name).write_text("{}", encoding="utf-8")
    with (edge_dir / "01_inlier_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["edge_i", "edge_j", "ref_x", "ref_y", "tgt_x", "tgt_y", "residual_px"],
        )
        writer.writeheader()
        for i, j in ((0, 6), (2, 5), (0, 5)):
            writer.writerow({"edge_i": i, "edge_j": j, "ref_x": 1, "ref_y": 2,
                             "tgt_x": 3, "tgt_y": 4, "residual_px": 0.5})

    result = load_local_affine_baseline(edge_dir)

    assert result["edge_keys"] == ["0-6", "2-5", "0-5"]
    for key in result["edge_keys"]:
        record = result["edges"][key]
        assert record["ref_xy"].shape == (1, 2)
        assert record["tgt_xy"].shape == (1, 2)
        assert "direct_overlap" in record
    assert result["missing_critical_artifacts"] == []
