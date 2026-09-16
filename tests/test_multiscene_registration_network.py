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
