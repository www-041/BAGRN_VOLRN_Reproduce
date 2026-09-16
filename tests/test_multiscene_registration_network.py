import numpy as np
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
        ], {"total": 1}

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
