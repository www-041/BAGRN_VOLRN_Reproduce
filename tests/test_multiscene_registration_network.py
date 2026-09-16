import csv
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio


def make_pair(i, j, dx, dy, confidence=0.9, n_blocks=20, rmse=0.2):
    return {
        "idx_i": i,
        "idx_j": j,
        "shift_dx": float(dx),
        "shift_dy": float(dy),
        "confidence": float(confidence),
        "n_blocks": int(n_blocks),
        "rmse": float(rmse),
        "matches": [],
    }


def make_edge_scenes():
    """Create unequal-shaped scenes whose geospatial grids partially overlap."""
    return [
        __import__("scripts.five_image_pipeline", fromlist=["SceneData"]).SceneData(
            name="scene_0",
            path="scene_0.tif",
            array=np.arange(60, dtype=np.float32).reshape(10, 6),
            transform=rasterio.Affine(1, 0, 100, 0, -1, 110),
            crs="EPSG:4326",
            nodata=None,
        ),
        __import__("scripts.five_image_pipeline", fromlist=["SceneData"]).SceneData(
            name="scene_1",
            path="scene_1.tif",
            array=np.arange(55, dtype=np.float32).reshape(11, 5),
            transform=rasterio.Affine(1, 0, 102, 0, -1, 110),
            crs="EPSG:4326",
            nodata=None,
        ),
    ]


def test_sparse_edges_use_geospatial_overlap_fallback_for_zero_one_or_two_matches(
    monkeypatch,
):
    from scripts import five_image_pipeline as pipeline

    scenes = make_edge_scenes()
    fallback_calls = []

    def fake_overlap_shift(arr_ref, tr_ref, arr_tgt, tr_tgt, nd_ref, nd_tgt, **kwargs):
        fallback_calls.append((arr_ref, tr_ref, arr_tgt, tr_tgt, kwargs))
        return -2.0, 3.0, 0.8, {
            "available": True,
            "failure_reason": None,
            "screening": {"total": 12},
        }

    monkeypatch.setattr(pipeline, "compute_shifts_from_overlap", fake_overlap_shift)
    monkeypatch.setattr(
        pipeline,
        "phase_correlation",
        lambda *args, **kwargs: pytest.fail("full-scene phase fallback must not run"),
    )

    for sparse_matches in ([], [{"shift_dx": 1.0, "shift_dy": 2.0, "confidence": 0.8}], [
        {"shift_dx": 1.0, "shift_dy": 2.0, "confidence": 0.8},
        {"shift_dx": 1.1, "shift_dy": 1.9, "confidence": 0.9},
    ]):
        monkeypatch.setattr(
            pipeline,
            "collect_block_matches",
            lambda *args, matches=sparse_matches, **kwargs: (matches, {"total": 12}),
        )
        measurement, rejection = pipeline._measure_registration_edge(
            scenes[0], scenes[1], {"idx_i": 0, "idx_j": 1}
        )

        assert rejection is None
        assert measurement["method"] == "overlap_translation_fallback"
        assert measurement["shift_dx"] == 3.0
        assert measurement["shift_dy"] == -2.0
        assert measurement["n_blocks"] == len(sparse_matches)
        assert measurement["translation_stats"]["available"] is True

    assert len(fallback_calls) == 3
    assert fallback_calls[0][0] is scenes[0].array
    assert fallback_calls[0][1] == scenes[0].transform
    assert fallback_calls[0][2] is scenes[1].array
    assert fallback_calls[0][3] == scenes[1].transform
    assert fallback_calls[0][4]["max_global_shift"] == 40


def test_sparse_overlap_fallback_rejects_unavailable_translation_with_reason(monkeypatch):
    from scripts import five_image_pipeline as pipeline

    scenes = make_edge_scenes()
    monkeypatch.setattr(
        pipeline,
        "collect_block_matches",
        lambda *args, **kwargs: ([], {"total": 12, "low_valid": 12}),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_shifts_from_overlap",
        lambda *args, **kwargs: (
            0.0,
            0.0,
            0.0,
            {"available": False, "failure_reason": "no common valid pixels"},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "phase_correlation",
        lambda *args, **kwargs: pytest.fail("full-scene phase fallback must not run"),
    )

    measurement, rejection = pipeline._measure_registration_edge(
        scenes[0], scenes[1], {"idx_i": 0, "idx_j": 1}
    )

    assert measurement is None
    assert "no common valid pixels" in rejection["reason"]
    assert rejection["translation_stats"]["available"] is False


def test_three_or_more_blocks_use_joint_mad_inliers_for_edge_estimate(monkeypatch):
    from scripts import five_image_pipeline as pipeline

    scenes = make_edge_scenes()
    matches = [
        {"shift_dx": 1.0, "shift_dy": 2.0, "confidence": 0.9},
        {"shift_dx": 1.1, "shift_dy": 2.1, "confidence": 0.8},
        {"shift_dx": 0.9, "shift_dy": 1.9, "confidence": 1.0},
        {"shift_dx": 8.0, "shift_dy": 8.0, "confidence": 1.0},
    ]
    monkeypatch.setattr(
        pipeline,
        "collect_block_matches",
        lambda *args, **kwargs: (matches, {"total": 20, "accepted": 4}),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_shifts_from_overlap",
        lambda *args, **kwargs: pytest.fail("fallback is only for fewer than three blocks"),
    )

    measurement, rejection = pipeline._measure_registration_edge(
        scenes[0], scenes[1], {"idx_i": 0, "idx_j": 1}
    )

    assert rejection is None
    assert measurement["method"] == "block_match"
    assert measurement["n_blocks"] == 4
    assert measurement["inlier_count"] == 3
    assert measurement["shift_dx"] == pytest.approx(1.0, abs=0.05)
    assert measurement["shift_dy"] == pytest.approx(2.0, abs=0.05)
    assert measurement["rmse"] < 0.2
    assert measurement["p95"] < 0.2


def test_network_adjustment_anchors_reference_and_solves_chain():
    from src.coregistration import multi_image_network_adjustment

    pairs = [
        make_pair(0, 1, 1.0, 0.0),
        make_pair(1, 2, 2.0, 0.0),
        make_pair(2, 3, 3.0, 0.0),
    ]

    result = multi_image_network_adjustment(pairs, n_images=4, reference_idx=0)

    assert result["n_edges"] == 3
    np.testing.assert_allclose(
        result["global_shifts"],
        np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]]),
        atol=1e-8,
    )
    np.testing.assert_allclose(result["global_shifts"][0], [0.0, 0.0])


def test_network_adjustment_supports_nonzero_reference_index():
    from src.coregistration import multi_image_network_adjustment

    pairs = [
        make_pair(0, 1, 1.0, 0.0),
        make_pair(1, 2, 2.0, 0.0),
        make_pair(2, 3, 3.0, 0.0),
    ]

    result = multi_image_network_adjustment(pairs, n_images=4, reference_idx=1)

    np.testing.assert_allclose(
        result["global_shifts"],
        np.array([[-1.0, 0.0], [0.0, 0.0], [2.0, 0.0], [5.0, 0.0]]),
        atol=1e-8,
    )
    np.testing.assert_allclose(result["global_shifts"][1], [0.0, 0.0])


def test_all_geometric_overlap_edges_are_matched_not_only_reference_edges(monkeypatch):
    from scripts import five_image_pipeline as pipeline

    scenes = [
        pipeline.SceneData(
            name=f"scene_{index}",
            path=f"scene_{index}.tif",
            array=np.full((12, 12), index + 1, dtype=np.float32),
            transform=rasterio.Affine.identity(),
            crs="EPSG:4326",
            nodata=None,
        )
        for index in range(4)
    ]
    overlaps = [
        {"idx_i": 0, "idx_j": 1},
        {"idx_i": 1, "idx_j": 2},
        {"idx_i": 2, "idx_j": 3},
    ]
    calls = []

    def fake_collect(arr_i, tr_i, arr_j, tr_j, nd_i, nd_j, **kwargs):
        calls.append((int(arr_i[0, 0]), int(arr_j[0, 0])))
        return [
            {
                "ref_x": 2.0,
                "ref_y": 2.0,
                "shift_dx": 1.0,
                "shift_dy": 0.0,
                "confidence": 0.9,
            }
        ] * 3, {"total": 3}

    monkeypatch.setattr(pipeline, "collect_block_matches", fake_collect)

    measurements, rejected = pipeline.match_all_overlap_edges(scenes, overlaps)

    assert calls == [(1, 2), (2, 3), (3, 4)]
    assert [(item["idx_i"], item["idx_j"]) for item in measurements] == [
        (0, 1), (1, 2), (2, 3)
    ]
    assert rejected == []


def test_registration_graph_finds_indirect_reachability_and_unreachable_scene():
    from scripts.five_image_pipeline import build_registration_graph

    graph = build_registration_graph(
        [
            make_pair(0, 1, 1.0, 0.0),
            make_pair(1, 2, 1.0, 0.0),
            make_pair(2, 4, 1.0, 0.0),
        ],
        n_images=5,
        reference_idx=0,
    )

    assert graph["reachable"] == [0, 1, 2, 4]
    assert graph["unreachable"] == [3]
    assert graph["parent"] == {0: None, 1: 0, 2: 1, 4: 2}
    assert graph["spanning_tree_edges"] == [(0, 1), (1, 2), (2, 4)]


def test_registration_graph_does_not_use_rejected_or_unrelated_edges():
    from scripts.five_image_pipeline import build_registration_graph

    graph = build_registration_graph(
        [make_pair(1, 2, 1.0, 0.0)],
        n_images=4,
        reference_idx=0,
    )

    assert graph["reachable"] == [0]
    assert graph["unreachable"] == [1, 2, 3]
    assert graph["spanning_tree_edges"] == []


def test_network_solver_uses_all_reliable_edges_and_keeps_reference_anchor():
    from scripts.five_image_pipeline import solve_registration_network

    pairs = [
        make_pair(0, 1, 1.0, 0.0),
        make_pair(1, 2, 2.0, 0.0),
        make_pair(2, 3, 3.0, 0.0),
        make_pair(0, 2, 3.0, 0.0),
        make_pair(1, 3, 5.0, 0.0),
    ]

    result = solve_registration_network(pairs, n_images=4, reference_idx=0)

    assert result["n_edges"] == 5
    assert result["is_tree"] is False
    assert len(result["pair_results"]) == 5
    np.testing.assert_allclose(
        result["global_shifts"],
        np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]]),
        atol=1e-8,
    )


def test_reference_paths_report_indirect_registration_route():
    from scripts.five_image_pipeline import build_reference_paths

    paths = build_reference_paths(
        {0: None, 1: 0, 2: 1, 4: 2}, reference_idx=0, n_images=5
    )

    assert paths == {
        0: [0],
        1: [0, 1],
        2: [0, 1, 2],
        3: None,
        4: [0, 1, 2, 4],
    }


def test_reference_paths_reject_parent_cycle():
    from scripts.five_image_pipeline import build_reference_paths

    with pytest.raises(RuntimeError, match="cycle"):
        build_reference_paths({0: None, 1: 2, 2: 1}, 0, 3)


def test_final_warps_are_applied_once_from_original_scene_arrays(monkeypatch):
    from scripts import five_image_pipeline as pipeline

    original_arrays = [
        np.arange(25, dtype=np.float32).reshape(5, 5) + index * 100
        for index in range(3)
    ]
    scenes = [
        pipeline.SceneData(
            name=f"scene_{index}",
            path=f"scene_{index}.tif",
            array=array,
            transform=rasterio.Affine.identity(),
            crs="EPSG:4326",
            nodata=0.0,
        )
        for index, array in enumerate(original_arrays)
    ]
    shifts = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, -1.0]])
    calls = []

    def fake_warp(array, dx, dy, local_dx, local_dy, nodata):
        calls.append({"array": array, "dx": dx, "dy": dy})
        return array.astype(np.float64) + 1000.0

    monkeypatch.setattr(pipeline, "warp_with_displacement_field", fake_warp)

    registered, statuses = pipeline.apply_network_shifts_from_original(
        scenes, shifts, reference_idx=0
    )

    assert statuses == {
        0: "reference_anchor",
        1: "network_solution_zero",
        2: "network_adjusted",
    }
    assert len(calls) == 1
    assert calls[0]["array"] is original_arrays[2]
    assert calls[0]["dx"] == 2.0
    assert calls[0]["dy"] == -1.0
    np.testing.assert_array_equal(registered[0].array, original_arrays[0])
    np.testing.assert_array_equal(registered[1].array, original_arrays[1])
    np.testing.assert_array_equal(registered[2].array, original_arrays[2] + 1000.0)
    assert registered[0].array.dtype == np.float64


def test_parent_edge_local_controls_use_network_relative_shift():
    from scripts import five_image_pipeline as pipeline

    scenes = [
        pipeline.SceneData(
            name=f"scene_{index}",
            path=f"scene_{index}.tif",
            array=np.ones((20, 20), dtype=np.float32),
            transform=rasterio.Affine.identity(),
            crs="EPSG:4326",
            nodata=0.0,
        )
        for index in range(5)
    ]
    matches = [
        {
            "ref_x": 2.0,
            "ref_y": 2.0,
            "tgt_x": float(2 + index % 6),
            "tgt_y": float(2 + index // 6),
            "shift_dx": 8.5,
            "shift_dy": 4.25,
            "confidence": 0.9,
        }
        for index in range(24)
    ]
    pairs = [{
        "idx_i": 3,
        "idx_j": 4,
        "shift_dx": 8.5,
        "shift_dy": 4.25,
        "confidence": 0.9,
        "n_blocks": len(matches),
        "rmse": 0.1,
        "matches": matches,
    }]
    graph = {"parent_map": {0: None, 1: 0, 2: 1, 3: 2, 4: 3}}
    global_shifts = np.array([
        [0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [5.0, 1.0], [13.0, 5.0]
    ])

    fields, details = pipeline.build_network_local_corrections(
        scenes, pairs, global_shifts, graph, reference_idx=0
    )

    assert details[4]["parent_edge"] == [3, 4]
    assert details[4]["parent_relative_shift"] == {"dx": 8.0, "dy": 4.0}
    assert details[4]["control_points"] == 24
    np.testing.assert_array_equal(fields[4][0], np.zeros((20, 20)))


def test_indirect_scene_can_use_parent_edge_rbf(monkeypatch):
    from scripts import five_image_pipeline as pipeline

    scenes = [
        pipeline.SceneData(
            name=f"scene_{index}",
            path=f"scene_{index}.tif",
            array=np.ones((8, 9), dtype=np.float32),
            transform=rasterio.Affine.identity(),
            crs="EPSG:4326",
            nodata=0.0,
        )
        for index in range(5)
    ]
    graph = {"parent_map": {0: None, 1: 0, 2: 1, 3: 2, 4: 3}}
    global_shifts = np.zeros((5, 2), dtype=float)
    valid_controls = {
        "points_xy": np.column_stack([
            np.arange(30, dtype=float) % 6,
            np.arange(30, dtype=float) // 6,
        ]),
        "residual_dx": np.full(30, 0.5),
        "residual_dy": np.full(30, -0.25),
        "n_valid": 30,
    }
    pairs = [{
        "idx_i": 3,
        "idx_j": 4,
        "shift_dx": 0.0,
        "shift_dy": 0.0,
        "confidence": 0.9,
        "n_blocks": 30,
        "rmse": 0.1,
        "matches": [],
    }]
    requested_edges = []

    def fake_controls(image_idx, parent_idx, pair_measurements, shifts):
        requested_edges.append((image_idx, parent_idx))
        return valid_controls if image_idx == 4 and parent_idx == 3 else {
            "points_xy": np.empty((0, 2)),
            "residual_dx": np.array([]),
            "residual_dy": np.array([]),
            "n_valid": 0,
        }

    monkeypatch.setattr(pipeline, "build_parent_based_local_controls", fake_controls)
    monkeypatch.setattr(
        pipeline,
        "spatial_cross_validate",
        lambda *args, **kwargs: (
            {"translation": {"p95": 1.0}, "rbf": {"p95": 0.5}},
            np.zeros(30, dtype=int),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "fit_local_rbf",
        lambda *args, **kwargs: (
            lambda points: np.ones(len(points)) * 0.25,
            lambda points: np.ones(len(points)) * -0.1,
            (0.0, 0.0),
            (5.0, 4.0),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_build_local_fields",
        lambda shape, controls, rbf_dx, rbf_dy, coord_range: (
            np.full(shape, 0.25), np.full(shape, -0.1)
        ),
    )

    fields, details = pipeline.build_network_local_corrections(
        scenes, pairs, global_shifts, graph, reference_idx=0
    )

    assert (4, 3) in requested_edges
    assert details[4]["parent_edge"] == [3, 4]
    assert details[4]["model_used"] == "rbf"
    np.testing.assert_array_equal(fields[4][0], np.full((8, 9), 0.25))
    np.testing.assert_array_equal(fields[4][1], np.full((8, 9), -0.1))


def test_rbf_unavailable_keeps_global_only_warp_field():
    from scripts import five_image_pipeline as pipeline

    scenes = [
        pipeline.SceneData(
            name=f"scene_{index}",
            path=f"scene_{index}.tif",
            array=np.ones((4, 4), dtype=np.float32),
            transform=rasterio.Affine.identity(),
            crs="EPSG:4326",
            nodata=0.0,
        )
        for index in range(2)
    ]
    fields, details = pipeline.build_network_local_corrections(
        scenes,
        [{"idx_i": 0, "idx_j": 1, "matches": []}],
        np.array([[0.0, 0.0], [2.0, 1.0]]),
        {"parent_map": {0: None, 1: 0}},
        reference_idx=0,
    )

    assert details[1]["model_used"] == "translation"
    assert details[1]["control_points"] == 0
    assert not np.any(fields[1][0])
    assert not np.any(fields[1][1])


def test_final_global_plus_rbf_warp_uses_original_array_once(monkeypatch):
    from scripts import five_image_pipeline as pipeline

    original_arrays = [
        np.arange(16, dtype=np.float32).reshape(4, 4),
        np.arange(16, dtype=np.float32).reshape(4, 4) + 100,
    ]
    scenes = [
        pipeline.SceneData(
            name=f"scene_{index}",
            path=f"scene_{index}.tif",
            array=array,
            transform=rasterio.Affine.identity(),
            crs="EPSG:4326",
            nodata=0.0,
        )
        for index, array in enumerate(original_arrays)
    ]
    local_dx = np.full((4, 4), 0.25)
    local_dy = np.full((4, 4), -0.5)
    calls = []

    def fake_warp(array, dx, dy, passed_dx, passed_dy, nodata):
        calls.append((array, dx, dy, passed_dx, passed_dy))
        return array.astype(np.float64) + 1.0

    monkeypatch.setattr(pipeline, "warp_with_displacement_field", fake_warp)
    registered, statuses = pipeline.apply_network_shifts_from_original(
        scenes,
        np.array([[9.0, 8.0], [2.0, -1.0]]),
        reference_idx=0,
        local_fields={1: (local_dx, local_dy)},
    )

    assert statuses[0] == "reference_anchor"
    assert len(calls) == 1
    assert calls[0][0] is original_arrays[1]
    assert calls[0][1:3] == (2.0, -1.0)
    assert calls[0][3] is local_dx
    assert calls[0][4] is local_dy
    np.testing.assert_array_equal(registered[0].array, original_arrays[0])


def test_pipeline_rejects_mixed_crs_before_overlap_detection(tmp_path, monkeypatch):
    from scripts import five_image_pipeline as pipeline

    paths = [tmp_path / f"scene_{index}_B14.TIF" for index in range(5)]

    def fake_load_scene(path, band):
        index = int(Path(path).stem.split("_")[1])
        return pipeline.SceneData(
            name=f"scene_{index}",
            path=str(path),
            array=np.ones((4, 4), dtype=np.float32),
            transform=rasterio.Affine(1, 0, index * 4, 0, -1, 4),
            crs="EPSG:4326" if index == 0 else "EPSG:3857",
            nodata=0.0,
        )

    monkeypatch.setattr(pipeline, "load_scene", fake_load_scene)
    monkeypatch.setattr(
        pipeline,
        "detect_multi_overlap",
        lambda *args, **kwargs: pytest.fail("overlap detection must not run"),
    )

    with pytest.raises(ValueError, match="CRS mismatch"):
        pipeline.run_pipeline(paths, tmp_path / "output", band="B14")


def test_pipeline_rejects_mixed_pixel_resolution_before_overlap_detection(
    tmp_path, monkeypatch
):
    from scripts import five_image_pipeline as pipeline

    paths = [tmp_path / f"scene_{index}_B14.TIF" for index in range(5)]

    def fake_load_scene(path, band):
        index = int(Path(path).stem.split("_")[1])
        resolution = 1.0 if index == 0 else 2.0
        return pipeline.SceneData(
            name=f"scene_{index}",
            path=str(path),
            array=np.ones((4, 4), dtype=np.float32),
            transform=rasterio.Affine(resolution, 0, 0, 0, -resolution, 4),
            crs="EPSG:4326",
            nodata=0.0,
        )

    monkeypatch.setattr(pipeline, "load_scene", fake_load_scene)
    monkeypatch.setattr(
        pipeline,
        "detect_multi_overlap",
        lambda *args, **kwargs: pytest.fail("overlap detection must not run"),
    )

    with pytest.raises(ValueError, match="pixel resolution mismatch"):
        pipeline.run_pipeline(paths, tmp_path / "output", band="B14")



def test_network_diagnostics_separate_geometric_reliable_rejected_and_radiometric_edges():
    from scripts.five_image_pipeline import summarize_registration_edges

    geometric = [(0, 1), (1, 2), (2, 3), (0, 3)]
    reliable = [make_pair(0, 1, 1.0, 0.0), make_pair(1, 2, 1.0, 0.0)]
    rejected = [
        {"idx_i": 2, "idx_j": 3, "reason": "low confidence"},
        {"idx_i": 0, "idx_j": 3, "reason": "no overlap texture"},
    ]

    summary = summarize_registration_edges(
        geometric,
        reliable,
        rejected,
        radiometric_overlap_count=len(geometric),
    )

    assert summary == {
        "geometric_overlap_pairs": 4,
        "reliable_registration_edges": 2,
        "rejected_registration_edges": 2,
        "radiometric_overlap_pairs": 4,
    }


def test_registration_network_artifacts_preserve_indirect_paths_and_edge_rows(tmp_path):
    from scripts import five_image_pipeline as pipeline

    scenes = [
        pipeline.SceneData(
            name=f"scene_{index}",
            path=f"scene_{index}.tif",
            array=np.zeros((2, 2), dtype=np.float32),
            transform=rasterio.Affine.identity(),
            crs="EPSG:4326",
            nodata=None,
        )
        for index in range(4)
    ]
    overlaps = [
        {"idx_i": 0, "idx_j": 1, "pixel_count": 10},
        {"idx_i": 1, "idx_j": 2, "pixel_count": 10},
        {"idx_i": 2, "idx_j": 3, "pixel_count": 10},
        {"idx_i": 0, "idx_j": 3, "pixel_count": 10},
    ]
    pairs = [
        make_pair(0, 1, 1.0, 0.0),
        make_pair(1, 2, 2.0, 0.0),
        make_pair(2, 3, 3.0, 0.0),
    ]
    rejected = [{"idx_i": 0, "idx_j": 3, "reason": "test rejection"}]
    graph = pipeline.build_registration_graph(pairs, 4, reference_idx=0)
    network = pipeline.solve_registration_network(pairs, 4, reference_idx=0)
    paths = pipeline.build_reference_paths(graph["parent"], 0, 4)
    statuses = {
        0: "reference_anchor",
        1: "network_adjusted",
        2: "network_adjusted",
        3: "network_adjusted",
    }

    artifacts = pipeline.write_registration_network_artifacts(
        tmp_path,
        scenes,
        reference_idx=0,
        overlaps=overlaps,
        pair_measurements=pairs,
        rejected_edges=rejected,
        graph=graph,
        network_result=network,
        reference_paths=paths,
        registration_statuses=statuses,
    )

    data = json.loads(artifacts["json"].read_text(encoding="utf-8"))
    assert data["reference_scene_index"] == 0
    assert data["reference_scene_id"] == "scene_0"
    assert data["reference_role"] == "anchor_only"
    assert data["geometric_edges"] == [[0, 1], [1, 2], [2, 3], [0, 3]]
    assert data["reliable_edges"] == [[0, 1], [1, 2], [2, 3]]
    assert data["rejected_edges"] == [[0, 3]]
    assert data["connected"] is True
    assert data["unreachable_scene_indices"] == []
    assert data["reference_paths"]["3"] == [0, 1, 2, 3]
    assert data["reference_paths"]["3"] != [0, 3]
    assert np.isclose(data["global_shifts"][3]["dx"], 6.0)
    assert data["global_shifts"][3]["status"] == "network_adjusted"

    with artifacts["pairs_csv"].open(newline="", encoding="utf-8") as handle:
        pair_rows = list(csv.DictReader(handle))
    assert len(pair_rows) == 4
    assert set(pair_rows[0]) == {
        "idx_i", "scene_i", "idx_j", "scene_j", "geometric_overlap",
        "registration_available", "method", "shift_dx", "shift_dy",
        "confidence", "n_blocks", "rmse", "p95", "reject_reason",
    }
    rejected_row = next(row for row in pair_rows if row["idx_i"] == "0" and row["idx_j"] == "3")
    assert rejected_row["registration_available"] == "False"
    assert rejected_row["reject_reason"] == "test rejection"

    with artifacts["scenes_csv"].open(newline="", encoding="utf-8") as handle:
        scene_rows = list(csv.DictReader(handle))
    assert len(scene_rows) == 4
    assert set(scene_rows[0]) == {
        "scene_index", "scene_id", "is_reference", "reference_path",
        "global_dx_pixels", "global_dy_pixels", "global_magnitude_pixels",
        "registration_status",
    }
    scene_three = next(row for row in scene_rows if row["scene_index"] == "3")
    assert scene_three["reference_path"] == "0 -> 1 -> 2 -> 3"


def test_formal_pipeline_stops_before_normalization_when_graph_is_disconnected(
    tmp_path, monkeypatch
):
    from scripts import five_image_pipeline as pipeline

    scene_names = ["reference", "scene_1", "scene_2", "scene_3", "scene_4"]
    paths = [tmp_path / f"{name}_B14.TIF" for name in scene_names]
    for path in paths:
        path.touch()

    def fake_load_scene(path, band):
        index = scene_names.index(Path(path).stem.removesuffix("_B14"))
        return pipeline.SceneData(
            name=scene_names[index],
            path=str(path),
            array=np.full((8, 8), index + 1, dtype=np.float32),
            transform=rasterio.Affine.identity(),
            crs="EPSG:4326",
            nodata=None,
        )

    overlaps = [
        {"idx_i": 0, "idx_j": 1},
        {"idx_i": 1, "idx_j": 2},
        {"idx_i": 3, "idx_j": 4},
    ]
    pairs = [
        make_pair(0, 1, 1.0, 0.0),
        make_pair(1, 2, 1.0, 0.0),
        make_pair(3, 4, 1.0, 0.0),
    ]
    normalization_calls = []

    monkeypatch.setattr(pipeline, "load_scene", fake_load_scene)
    monkeypatch.setattr(pipeline, "detect_multi_overlap", lambda *args, **kwargs: overlaps)
    monkeypatch.setattr(
        pipeline,
        "match_all_overlap_edges",
        lambda *args, **kwargs: (pairs, []),
    )
    monkeypatch.setattr(
        pipeline,
        "bagrn_normalize",
        lambda *args, **kwargs: normalization_calls.append("bagrn"),
    )
    monkeypatch.setattr(
        pipeline,
        "volrn_normalize",
        lambda *args, **kwargs: normalization_calls.append("volrn"),
    )
    monkeypatch.setattr(
        pipeline,
        "create_mosaic",
        lambda *args, **kwargs: normalization_calls.append("mosaic"),
    )

    with pytest.raises(RuntimeError, match="disconnected"):
        pipeline.run_pipeline(paths, tmp_path / "output", band="B14")

    assert normalization_calls == []


def test_synthetic_acceptance_covers_all_planned_registration_network_topologies(
    monkeypatch,
):
    from scripts import five_image_pipeline as pipeline

    def solve(pairs, n_images):
        graph = pipeline.build_registration_graph(pairs, n_images, reference_idx=0)
        result = pipeline.solve_registration_network(pairs, n_images, reference_idx=0)
        return graph, result

    # Case 1: direct star graph.
    star_graph, star_result = solve(
        [
            make_pair(0, 1, 1.0, 0.0),
            make_pair(0, 2, 2.0, 0.0),
            make_pair(0, 3, 3.0, 0.0),
        ],
        4,
    )
    assert star_graph["unreachable"] == []
    np.testing.assert_allclose(star_result["global_shifts"][:, 0], [0, 1, 2, 3])

    # Case 2: chain graph; the last scene is only indirectly connected.
    chain_pairs = [make_pair(i, i + 1, i + 1.0, 0.0) for i in range(4)]
    chain_graph, chain_result = solve(chain_pairs, 5)
    assert chain_graph["reachable"] == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(chain_result["global_shifts"][4], [10.0, 0.0], atol=1e-8)

    # Case 3: rejected direct edge, but a valid indirect path remains.
    indirect_pairs = [make_pair(0, 1, 1.0, 0.0), make_pair(1, 2, 2.0, 0.0)]
    indirect_graph, indirect_result = solve(indirect_pairs, 3)
    assert indirect_graph["reachable"] == [0, 1, 2]
    np.testing.assert_allclose(indirect_result["global_shifts"][2], [3.0, 0.0], atol=1e-8)

    # Case 4: the reference has no direct edge to the last scene.
    last_scene_pairs = [
        make_pair(0, 1, 1.0, 0.0),
        make_pair(1, 2, 1.0, 0.0),
        make_pair(0, 3, 3.0, 0.0),
        make_pair(3, 4, 4.0, 0.0),
    ]
    last_graph, last_result = solve(last_scene_pairs, 5)
    paths = pipeline.build_reference_paths(last_graph["parent_map"], 0, 5)
    assert paths[4] == [0, 3, 4]
    np.testing.assert_allclose(last_result["global_shifts"][4], [7.0, 0.0], atol=1e-8)

    # Case 5: redundant cycle; all five edges participate in adjustment.
    cycle_pairs = [
        make_pair(0, 1, 1.0, 0.0),
        make_pair(1, 2, 2.0, 0.0),
        make_pair(2, 0, -3.0, 0.0),
        make_pair(2, 3, 3.0, 0.0),
        make_pair(1, 3, 5.0, 0.0),
    ]
    cycle_graph, cycle_result = solve(cycle_pairs, 4)
    assert cycle_graph["unreachable"] == []
    assert cycle_result["n_edges"] == 5
    np.testing.assert_allclose(cycle_result["global_shifts"][3], [6.0, 0.0], atol=1e-8)

    # Case 6: disconnected component; the formal stage stops before warping.
    scenes = [
        pipeline.SceneData(
            name=f"scene_{index}",
            path=f"scene_{index}.tif",
            array=np.ones((4, 4), dtype=np.float32),
            transform=rasterio.Affine.identity(),
            crs="EPSG:4326",
            nodata=0.0,
        )
        for index in range(5)
    ]
    disconnected_pairs = [
        make_pair(0, 1, 1.0, 0.0),
        make_pair(1, 2, 1.0, 0.0),
        make_pair(3, 4, 1.0, 0.0),
    ]
    monkeypatch.setattr(
        pipeline,
        "match_all_overlap_edges",
        lambda *args, **kwargs: (disconnected_pairs, []),
    )
    with pytest.raises(RuntimeError, match="disconnected"):
        pipeline.run_registration_stage(
            scenes,
            [{"idx_i": 0, "idx_j": 1}],
            reference_idx=0,
        )
