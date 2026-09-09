import numpy as np
from rasterio.transform import from_origin


def test_registration_fails_early_when_five_windows_are_physically_impossible(monkeypatch):
    from types import SimpleNamespace
    from src import coregistration, multiband_pipeline

    monkeypatch.setattr(
        multiband_pipeline,
        "_build_pair_holdout_context",
        lambda *args, **kwargs: {
            "available": False,
            "failure_reason": "insufficient independent validation geometry",
            "reserved_count": 1,
        },
    )

    def should_not_match(*args, **kwargs):
        raise AssertionError("registration matching must not start after early geometry failure")

    monkeypatch.setattr(coregistration, "collect_block_matches", should_not_match)

    pipeline = object.__new__(multiband_pipeline.MultibandPipeline)
    pipeline.registration_band_idx = 0
    pipeline.common_bands = ["B14"]
    pipeline.control_idx = 0
    pipeline.smoke = False
    pipeline.config = SimpleNamespace(registration_params={
        "enable_local_refinement": True,
        "enable_spatial_holdout": True,
        "final_min_blocks": 5,
    })
    arr = np.ones((1, 64, 64), dtype=float)
    scene_data = {
        "arrays": [arr, arr.copy()],
        "transforms": [from_origin(0, 64, 1, 1), from_origin(0, 64, 1, 1)],
        "nodata_values": [None, None],
        "scene_ids": ["a", "b"],
    }

    result = pipeline.register_scenes(scene_data, [{"idx_i": 0, "idx_j": 1}])

    assert result["status"] == "fail"
    assert "insufficient independent validation geometry" in result["failure"]["reason"]


def test_holdout_is_created_before_control_extraction():
    from src.coregistration import build_spatial_train_holdout_split

    common = np.ones((96, 160), dtype=bool)
    result = build_spatial_train_holdout_split(
        common, block_size=32, seed=42, holdout_fraction=0.20,
        min_holdout_cells=2,
    )

    assert result["holdout_mask"].any()
    assert result["train_mask"].dtype == bool
    assert result["holdout_mask"].dtype == bool
    assert result["train_mask"].shape == common.shape
    assert result["holdout_mask"].shape == common.shape
    assert not np.any(result["train_mask"] & result["holdout_buffer_mask"])


def test_training_blocks_never_overlap_holdout_buffer():
    from src.coregistration import build_spatial_train_holdout_split

    common = np.ones((128, 192), dtype=bool)
    result = build_spatial_train_holdout_split(
        common, block_size=32, seed=7, holdout_fraction=0.25,
        min_holdout_cells=2,
    )

    assert result["train_cells"]
    for row, col in result["train_cells"]:
        block = result["train_mask"][row:row + 32, col:col + 32]
        assert block.shape == (32, 32)
        assert block.all()


def test_holdout_cells_are_spatially_separated():
    from src.coregistration import build_spatial_train_holdout_split

    common = np.ones((192, 192), dtype=bool)
    result = build_spatial_train_holdout_split(
        common, block_size=32, seed=42, holdout_fraction=0.35,
        min_holdout_cells=3,
    )

    cells = result["holdout_cells"]
    assert len(cells) >= 3
    for row_a, col_a in cells:
        for row_b, col_b in cells:
            if (row_a, col_a) == (row_b, col_b):
                continue
            assert abs(row_a - row_b) > 1 or abs(col_a - col_b) > 1


def test_narrow_diagonal_overlap_still_produces_holdout():
    from src.coregistration import build_spatial_train_holdout_split

    yy, xx = np.mgrid[:192, :192]
    common = np.abs(xx - yy) <= 10
    result = build_spatial_train_holdout_split(
        common, block_size=16, seed=3, holdout_fraction=0.25,
        min_holdout_cells=2,
    )

    assert result["available"] is True
    assert len(result["holdout_cells"]) >= 2


def test_register_scenes_reserves_holdout_before_matching(monkeypatch):
    from types import SimpleNamespace
    from src import coregistration, multiband_pipeline

    seen = {}
    matches = [
        {"ref_x": 20.0, "ref_y": 20.0, "tgt_x": 20.0, "tgt_y": 20.0,
         "shift_dx": 0.0, "shift_dy": 0.0, "confidence": 0.9},
        {"ref_x": 44.0, "ref_y": 20.0, "tgt_x": 44.0, "tgt_y": 20.0,
         "shift_dx": 0.0, "shift_dy": 0.0, "confidence": 0.9},
        {"ref_x": 20.0, "ref_y": 44.0, "tgt_x": 20.0, "tgt_y": 44.0,
         "shift_dx": 0.0, "shift_dy": 0.0, "confidence": 0.9},
        {"ref_x": 44.0, "ref_y": 44.0, "tgt_x": 44.0, "tgt_y": 44.0,
         "shift_dx": 0.0, "shift_dy": 0.0, "confidence": 0.9},
    ]

    def fake_collect(*args, **kwargs):
        seen["allowed_mask"] = kwargs.get("allowed_mask")
        return matches, {"total": 4, "accepted": 4}

    monkeypatch.setattr(coregistration, "collect_block_matches", fake_collect)
    monkeypatch.setattr(
        coregistration, "build_robust_pair_measurement",
        lambda m, p: {"status": "pass", "shift_dx": 0.0, "shift_dy": 0.0,
                      "confidence": 0.9, "n_blocks_inlier": 4,
                      "n_blocks_total": 4, "rmse": 0.0, "p95": 0.0,
                      "matches": m, "screening": p["screening"]},
    )
    monkeypatch.setattr(
        coregistration, "multi_image_network_adjustment",
        lambda *args: {"global_shifts": np.zeros((2, 2)), "loop_errors": []},
    )
    monkeypatch.setattr(
        coregistration, "refine_global_residual_shifts_from_original",
        lambda *args, **kwargs: {"global_shifts": np.zeros((2, 2)),
                                  "history": [], "warnings": []},
    )
    def fake_validate(*args, **kwargs):
        seen["holdout_contexts"] = kwargs["holdout_contexts"]
        return ({"quality": "pass", "rmse": 0.0, "p95": 0.0,
                 "median": 0.0, "confidence": 0.9, "n_blocks": 5},
                {"edges": [], "overall": {"quality": "pass"}})
    monkeypatch.setattr(multiband_pipeline, "_validate_final_registration_arrays", fake_validate)

    pipeline = object.__new__(multiband_pipeline.MultibandPipeline)
    pipeline.registration_band_idx = 0
    pipeline.common_bands = ["B14"]
    pipeline.control_idx = 0
    pipeline.smoke = False
    pipeline.config = SimpleNamespace(registration_params={
        "enable_local_refinement": False,
        "enable_spatial_holdout": True,
        "global_block_size": 16,
        "holdout_block_size": 16,
        "holdout_fraction": 0.20,
        "min_holdout_cells": 2,
        "validation_block_size_candidates": [16],
    })
    arr = np.arange(64 * 64, dtype=float).reshape(1, 64, 64)
    scene_data = {
        "arrays": [arr, arr.copy()],
        "transforms": [from_origin(0, 64, 1, 1), from_origin(0, 64, 1, 1)],
        "nodata_values": [None, None],
        "scene_ids": ["a", "b"],
    }

    result = pipeline.register_scenes(scene_data, [{"idx_i": 0, "idx_j": 1}])

    assert seen["allowed_mask"] is not None
    assert seen["holdout_contexts"][(0, 1)]["available"] is True
    assert seen["holdout_contexts"][(0, 1)]["holdout_region_full_mask"].any()
    reservation = seen["holdout_contexts"][(0, 1)]["validation_reservation"]
    assert reservation["reserved_count"] >= 5
    assert reservation["reserved_windows"]
    assert result["status"] == "pass"
