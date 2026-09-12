"""N=2 KLT/TPS pipeline integration contracts."""

import numpy as np
import pytest
from rasterio.transform import Affine


def test_prepare_pair_holdout_contexts_preserves_reserved_windows(monkeypatch):
    import src.multiband_pipeline as pipeline

    seen = []
    def fake_builder(*args, **kwargs):
        seen.append(kwargs.get("reserved_windows_override"))
        return {
            "available": True,
            "patch_window_ref": (1, 9, 2, 10),
            "holdout_region_mask": np.zeros((8, 8), dtype=bool),
            "reserved_count": 5,
        }
    monkeypatch.setattr(pipeline, "_build_pair_holdout_context", fake_builder)
    arrays = [np.zeros((1, 12, 12)), np.zeros((1, 12, 12))]
    transforms = [Affine.identity(), Affine.identity()]
    result = pipeline._prepare_pair_holdout_contexts(
        arrays, transforms, [None, None], [{"idx_i": 0, "idx_j": 1}], 0,
        {"enable_spatial_holdout": True},
        holdout_reservation_overrides={(0, 1): [(1, 2, 5, 6)]},
    )
    assert (0, 1) in result
    assert seen == [[(1, 2, 5, 6)]]
    assert result[(0, 1)]["holdout_region_full_mask"].shape == (12, 12)


def _dummy_pipeline(backend="klt_tps", bands=("B12", "B14")):
    from src.multiband_pipeline import MultibandPipeline
    pipe = MultibandPipeline.__new__(MultibandPipeline)
    pipe.common_bands = list(bands)
    pipe.n_bands = len(bands)
    pipe.registration_band_idx = 0
    pipe.control_idx = 0
    pipe.config = type("Config", (), {
        "registration_params": {
            "registration_backend": backend,
            "enable_spatial_holdout": False,
            "required_quality": "pass",
        }
    })()
    return pipe


def test_klt_tps_backend_rejects_more_than_two_scenes():
    pipe = _dummy_pipeline()
    arrays = [np.zeros((2, 64, 64), dtype=np.float32) for _ in range(3)]
    scene_data = {"arrays": arrays, "transforms": [Affine.identity()] * 3, "nodata_values": [None] * 3}
    with pytest.raises(ValueError, match="exactly N=2"):
        pipe.register_scenes(scene_data, [{"idx_i": 0, "idx_j": 1}])


def _fake_estimation(shape=(48, 56)):
    p = np.asarray([[8, 8], [40, 8], [8, 32], [40, 32]], dtype=float)
    q = p + np.asarray([1.25, -0.5])
    flow = np.zeros((shape[0], shape[1], 2), dtype=np.float32)
    flow[..., 0] = 1.25
    flow[..., 1] = -0.5
    return {
        "available": True,
        "failure_reason": None,
        "overlap_context": {"overlap_transform": Affine.identity(), "resampled_target": False},
        "initial_corner_count": 40,
        "accepted_point_count": 40,
        "reference_points_overlap_xy": np.tile(p, (10, 1)),
        "moving_points_overlap_xy": np.tile(q, (10, 1)),
        "forward_backward_error": np.zeros(40),
        "control_points_moving_xy": np.tile(p, (10, 1)),
        "source_points_moving_xy": np.tile(q, (10, 1)),
        "displacement_xy": np.tile(q - p, (10, 1)),
        "flow": flow,
        "geometry": {"jacobian_min": 1.0, "jacobian_max": 1.0, "fold_pixels": 0, "max_displacement_pixels": 1.35},
    }


def _run_fake_backend(monkeypatch, pipe, registration_band_idx=0, validation_band="B14", estimate_fn=None):
    import src.multiband_pipeline as pipeline
    import src.klt_tps_registration as klt

    arrays = [
        np.stack([np.ones((48, 56)), np.ones((48, 56)) * 2]),
        np.stack([np.ones((48, 56)) * 3, np.ones((48, 56)) * 4]),
    ]
    scene_data = {
        "arrays": arrays,
        "transforms": [Affine.identity(), Affine.identity()],
        "nodata_values": [None, None],
        "scene_ids": ["ref", "moving"],
    }
    seen = {}
    estimation = _fake_estimation()
    monkeypatch.setattr(klt, "estimate_klt_tps_pair", estimate_fn or (lambda *args: estimation))
    def fake_warp(source, flow, nodata):
        seen["source"] = source.copy()
        return source.copy(), np.ones(source.shape[1:], dtype=bool)
    monkeypatch.setattr(klt, "warp_multiband_with_tps_flow", fake_warp)
    monkeypatch.setattr(pipeline, "_validate_final_registration_arrays", lambda *args, **kwargs: (
        {"quality": "pass", "rmse": 0.1, "p95": 0.2, "median": 0.1, "confidence": 0.9, "n_blocks": 5},
        {"edges": [{"idx_i": 0, "idx_j": 1}], "overall": {"quality": "pass"}},
    ))
    result = pipe.register_scenes(
        scene_data, [{"idx_i": 0, "idx_j": 1}],
        registration_band_idx=registration_band_idx,
        diagnostic_validation_band=validation_band,
    )
    return result, scene_data, seen


def test_klt_tps_backend_uses_selected_registration_band(monkeypatch):
    import src.klt_tps_registration as klt

    pipe = _dummy_pipeline(bands=("B14", "B12"))
    pipe.config.registration_params["registration_backend"] = "klt_tps"
    seen = {}
    def fake_estimate(ref, *args):
        seen["ref"] = ref
        return _fake_estimation()
    monkeypatch.setattr(klt, "estimate_klt_tps_pair", fake_estimate)
    result, scene_data, _ = _run_fake_backend(monkeypatch, pipe, registration_band_idx=1, estimate_fn=fake_estimate)
    assert np.all(seen["ref"] == scene_data["arrays"][0][1])
    assert np.all(seen["ref"] == 2)
    assert result["registration_band_name"] == "B12"


def test_klt_tps_backend_applies_one_flow_to_all_bands(monkeypatch):
    pipe = _dummy_pipeline()
    result, scene_data, seen = _run_fake_backend(monkeypatch, pipe)
    assert seen["source"].shape == scene_data["arrays"][1].shape
    assert result["registered_arrays"][1].shape == scene_data["arrays"][1].shape
    assert result["transform_model"] == "dense_klt_tps"


def test_klt_tps_backend_preserves_original_scene_transforms(monkeypatch):
    pipe = _dummy_pipeline()
    result, scene_data, _ = _run_fake_backend(monkeypatch, pipe)
    assert scene_data["transforms"] == [Affine.identity(), Affine.identity()]
    assert result["registered_arrays"][0] is not scene_data["arrays"][0]


def test_klt_tps_backend_reuses_holdout_override_for_final_validation(monkeypatch):
    import src.multiband_pipeline as pipeline
    import src.klt_tps_registration as klt

    pipe = _dummy_pipeline()
    pipe.config.registration_params["enable_spatial_holdout"] = True
    context = {"available": True, "reserved_count": 5, "holdout_region_full_mask": np.zeros((48, 56), bool), "validation_reservation": {}}
    seen = {}
    monkeypatch.setattr(pipeline, "_prepare_pair_holdout_contexts", lambda *args, **kwargs: {(0, 1): context})
    monkeypatch.setattr(klt, "estimate_klt_tps_pair", lambda *args: _fake_estimation())
    monkeypatch.setattr(klt, "warp_multiband_with_tps_flow", lambda source, flow, nodata: (source.copy(), np.ones(source.shape[1:], bool)))
    def fake_validate(*args, **kwargs):
        seen["contexts"] = kwargs.get("holdout_contexts")
        return ({"quality": "pass", "rmse": 0.1, "p95": 0.2, "median": 0.1, "confidence": 0.9, "n_blocks": 5}, {"edges": [], "overall": {"quality": "pass"}})
    monkeypatch.setattr(pipeline, "_validate_final_registration_arrays", fake_validate)
    pipe.register_scenes({"arrays": [np.ones((2, 48, 56))] * 2, "transforms": [Affine.identity()] * 2, "nodata_values": [None] * 2}, [{"idx_i": 0, "idx_j": 1}])
    assert seen["contexts"][(0, 1)] is context


def test_klt_tps_backend_does_not_fallback_to_legacy_when_estimation_fails(monkeypatch):
    import src.multiband_pipeline as pipeline
    import src.klt_tps_registration as klt

    pipe = _dummy_pipeline()
    monkeypatch.setattr(klt, "estimate_klt_tps_pair", lambda *args: {"available": False, "failure_reason": "insufficient KLT corners"})
    result = pipe.register_scenes(
        {"arrays": [np.ones((2, 48, 56))] * 2, "transforms": [Affine.identity()] * 2, "nodata_values": [None] * 2},
        [{"idx_i": 0, "idx_j": 1}],
    )
    assert result["registration_backend"] == "klt_tps"
    assert result["status"] == "fail"
    assert "insufficient KLT corners" in result["failure"]["reason"]


def test_legacy_backend_dispatch_still_uses_existing_path(monkeypatch):
    pipe = _dummy_pipeline(backend="legacy")
    pipe._register_scenes_klt_tps_n2 = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("legacy dispatch entered KLT/TPS path")
    )
    result = pipe.register_scenes(
        {"arrays": [np.ones((2, 48, 56))] * 2, "transforms": [Affine.identity()] * 2, "nodata_values": [None] * 2},
        [],
    )
    assert result["status"] == "fail"
    assert result.get("registration_backend") != "klt_tps"


def test_dz01_klt_tps_n2_config_is_two_scene_b12_b14():
    from src.experiment_config import load_config, validate_config

    cfg = load_config("configs/dz01_klt_tps_n2_b12.yaml")
    assert [scene["id"] for scene in cfg.scenes] == ["scene_20251114", "scene_20251120"]
    assert cfg.registration_band == "B12"
    assert cfg.selected_bands == ["B12", "B14"]
    assert cfg.registration_params["registration_backend"] == "klt_tps"
    assert cfg.registration_params["enable_local_refinement"] is False
    assert validate_config(cfg, skip_file_check=True) == []


def test_klt_tps_n2_synthetic_end_to_end_uses_actual_backend_dispatch(monkeypatch):
    pipe = _dummy_pipeline()
    result, scene_data, _ = _run_fake_backend(monkeypatch, pipe)
    assert result["registration_backend"] == "klt_tps"
    assert result["transform_model"] == "dense_klt_tps"
    assert len(result["registered_arrays"]) == 2
    assert result["registered_arrays"][0].shape == scene_data["arrays"][0].shape
    assert result["registered_arrays"][1].shape == scene_data["arrays"][1].shape
    assert result["final_validation"]["overall"]["quality"] == "pass"
    assert result["klt_tps"]["accepted_point_count"] >= 30
