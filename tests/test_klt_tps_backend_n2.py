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
    monkeypatch.setattr(klt, "estimate_klt_tps_pair", estimate_fn or (lambda *args, **kwargs: estimation))
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
    def fake_estimate(ref, *args, **kwargs):
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
    context = {
        "available": True, "reserved_count": 5,
        "train_sampling_mask": np.ones((48, 56), bool),
        "holdout_region_full_mask": np.zeros((48, 56), bool),
        "validation_reservation": {},
    }
    seen = {}
    monkeypatch.setattr(pipeline, "_prepare_pair_holdout_contexts", lambda *args, **kwargs: {(0, 1): context})
    monkeypatch.setattr(klt, "estimate_klt_tps_pair", lambda *args, **kwargs: _fake_estimation())
    monkeypatch.setattr(klt, "warp_multiband_with_tps_flow", lambda source, flow, nodata: (source.copy(), np.ones(source.shape[1:], bool)))
    def fake_validate(*args, **kwargs):
        seen["contexts"] = kwargs.get("holdout_contexts")
        return ({"quality": "pass", "rmse": 0.1, "p95": 0.2, "median": 0.1, "confidence": 0.9, "n_blocks": 5}, {"edges": [], "overall": {"quality": "pass"}})
    monkeypatch.setattr(pipeline, "_validate_final_registration_arrays", fake_validate)
    pipe.register_scenes({"arrays": [np.ones((2, 48, 56))] * 2, "transforms": [Affine.identity()] * 2, "nodata_values": [None] * 2}, [{"idx_i": 0, "idx_j": 1}])
    assert seen["contexts"][(0, 1)] is context


def test_klt_tps_backend_passes_holdout_train_mask_to_estimator(monkeypatch):
    import src.klt_tps_registration as klt
    import src.multiband_pipeline as pipeline

    pipe = _dummy_pipeline()
    pipe.config.registration_params["enable_spatial_holdout"] = True
    train_mask = np.ones((48, 56), dtype=bool)
    context = {
        "available": True,
        "reserved_count": 5,
        "train_sampling_mask": train_mask,
        "holdout_region_full_mask": np.zeros((48, 56), dtype=bool),
        "validation_reservation": {},
    }
    seen = {}
    monkeypatch.setattr(
        pipeline, "_prepare_pair_holdout_contexts",
        lambda *args, **kwargs: {(0, 1): context},
    )

    def fake_estimate(*args, **kwargs):
        seen["training_mask"] = kwargs.get("training_mask")
        return _fake_estimation()

    monkeypatch.setattr(klt, "estimate_klt_tps_pair", fake_estimate)
    monkeypatch.setattr(
        klt, "warp_multiband_with_tps_flow",
        lambda source, flow, nodata: (source.copy(), np.ones(source.shape[1:], bool)),
    )
    monkeypatch.setattr(
        pipeline, "_validate_final_registration_arrays",
        lambda *args, **kwargs: (
            {"quality": "pass", "rmse": 0.1, "p95": 0.2, "median": 0.1,
             "confidence": 0.9, "n_blocks": 5},
            {"edges": [], "overall": {"quality": "pass"}},
        ),
    )

    pipe.register_scenes(
        {"arrays": [np.ones((2, 48, 56))] * 2,
         "transforms": [Affine.identity()] * 2,
         "nodata_values": [None, None]},
        [{"idx_i": 0, "idx_j": 1}],
    )

    assert seen["training_mask"] is train_mask


def test_klt_tps_backend_does_not_fallback_to_legacy_when_estimation_fails(monkeypatch):
    import src.multiband_pipeline as pipeline
    import src.klt_tps_registration as klt

    pipe = _dummy_pipeline()
    monkeypatch.setattr(klt, "estimate_klt_tps_pair", lambda *args, **kwargs: {"available": False, "failure_reason": "insufficient KLT corners"})
    result = pipe.register_scenes(
        {"arrays": [np.ones((2, 48, 56))] * 2, "transforms": [Affine.identity()] * 2, "nodata_values": [None] * 2},
        [{"idx_i": 0, "idx_j": 1}],
    )
    assert result["registration_backend"] == "klt_tps"
    assert result["status"] == "fail"
    assert "insufficient KLT corners" in result["failure"]["reason"]


def test_klt_tps_geometry_failure_exposes_compact_failure_diagnostics(monkeypatch):
    import src.klt_tps_registration as klt

    pipe = _dummy_pipeline()
    estimation = {
        "available": False,
        "failure_reason": "KLT/TPS geometry rejected: TPS flow contains fold pixels: 7",
        "failure_diagnostics": {"stage": "tps_geometry_gate", "fold_support": {"fold_pixels_total": 7}},
        "_failure_diagnostic_arrays": {
            "flow": np.zeros((48, 56, 2), dtype=np.float32),
            "fold_mask": np.zeros((48, 56), dtype=bool),
        },
    }
    monkeypatch.setattr(klt, "estimate_klt_tps_pair", lambda *args, **kwargs: estimation)

    result = pipe.register_scenes(
        {"arrays": [np.ones((2, 48, 56))] * 2,
         "transforms": [Affine.identity()] * 2,
         "nodata_values": [None, None]},
        [{"idx_i": 0, "idx_j": 1}],
    )

    compact = result["diagnostics"]["klt_tps"]
    assert compact["failure_diagnostics"]["stage"] == "tps_geometry_gate"
    assert "_failure_diagnostic_arrays" not in compact
    assert result["klt_tps"]["_failure_diagnostic_arrays"]["flow"].shape == (48, 56, 2)


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


def _c1_context(shape=(48, 56)):
    return {
        "available": True,
        "reserved_count": 7,
        "train_sampling_mask": np.ones(shape, dtype=bool),
        "holdout_region_full_mask": np.zeros(shape, dtype=bool),
        "holdout_exclusion_mask": np.ones(shape, dtype=bool),
        "validation_reservation": {
            "reserved_windows": [
                {"row": 4, "col": 4, "height": 384, "width": 384}
                for _ in range(7)
            ],
        },
    }


def _c1_scene_data(shape=(48, 56)):
    arrays = [
        np.stack([np.ones(shape), np.ones(shape) * 2]),
        np.stack([np.ones(shape) * 3, np.ones(shape) * 4]),
    ]
    return {
        "arrays": arrays,
        "transforms": [Affine.identity(), Affine.identity()],
        "nodata_values": [None, None],
        "scene_ids": ["ref", "moving"],
    }


def _patch_c1_dependencies(monkeypatch, *, estimation=None, validate=None):
    import src.klt_tps_registration as klt
    import src.multiband_pipeline as pipeline

    context = _c1_context()
    monkeypatch.setattr(
        pipeline, "_prepare_pair_holdout_contexts",
        lambda *args, **kwargs: {(0, 1): context},
    )
    monkeypatch.setattr(
        klt, "estimate_klt_tps_pair",
        lambda *args, **kwargs: estimation or _fake_estimation(),
    )
    validations = []
    if validate is None:
        validate = lambda *args, **kwargs: (
            {"quality": "pass", "rmse": 0.1, "p95": 0.2, "median": 0.1,
             "confidence": 0.9, "n_blocks": 5},
            {"edges": [{"idx_i": 0, "idx_j": 1, "blocks": []}],
             "overall": {"quality": "pass"}},
        )

    def fake_validate(*args, **kwargs):
        validations.append(kwargs.get("holdout_contexts"))
        return validate(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_validate_final_registration_arrays", fake_validate)
    return context, validations


def test_tps_support_c1_calls_klt_tps_estimator_exactly_once(monkeypatch):
    import src.klt_tps_registration as klt

    pipe = _dummy_pipeline()
    calls = []
    estimation = _fake_estimation()
    _patch_c1_dependencies(monkeypatch, estimation=estimation)

    def fake_estimate(*args, **kwargs):
        calls.append((args, kwargs))
        return estimation

    monkeypatch.setattr(klt, "estimate_klt_tps_pair", fake_estimate)
    pipe.run_klt_tps_support_c1_n2(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], 0,
        diagnostic_validation_band="B14",
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    assert len(calls) == 1


def test_tps_support_c1_can_use_raw_flow_from_geometry_gate_rejection(monkeypatch):
    pipe = _dummy_pipeline()
    estimation = _fake_estimation()
    estimation.update({
        "available": False,
        "failure_reason": "TPS flow contains fold pixels",
        "failure_diagnostics": {"stage": "tps_geometry_gate"},
        "_failure_diagnostic_arrays": {
            "flow": np.full_like(estimation["flow"], [51.0, 0.0]),
        },
    })
    _patch_c1_dependencies(monkeypatch, estimation=estimation)

    result = pipe.run_klt_tps_support_c1_n2(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], 0,
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    causal = result["tps_support_causal"]
    assert causal["available"] is True
    assert causal["raw_geometry"]["geometry_safe"] is False
    assert causal["integrity"]["raw_fit_count"] == 1


def test_tps_support_c1_rejects_tps_fit_failure_without_counterfactual(monkeypatch):
    pipe = _dummy_pipeline()
    estimation = {
        "available": False,
        "failure_reason": "TPS fit failed",
        "failure_diagnostics": {"stage": "tps_fit"},
        "_failure_diagnostic_arrays": {},
    }
    _patch_c1_dependencies(monkeypatch, estimation=estimation)

    result = pipe.run_klt_tps_support_c1_n2(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], 0,
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    assert result["tps_support_causal"]["available"] is False
    assert result["tps_support_causal"]["raw_geometry"] is None
    assert result["tps_support_causal"]["translation_validation"] is None


def test_tps_support_c1_warps_translation_and_supported_from_original(monkeypatch):
    import src.klt_tps_registration as klt

    pipe = _dummy_pipeline()
    scene_data = _c1_scene_data()
    _patch_c1_dependencies(monkeypatch)
    sources = []
    flows = []

    def fake_warp(source, flow, nodata):
        sources.append(source)
        flows.append(flow)
        return source.copy(), np.ones(source.shape[1:], bool)

    monkeypatch.setattr(klt, "warp_multiband_with_tps_flow", fake_warp)
    result = pipe.run_klt_tps_support_c1_n2(
        scene_data, [{"idx_i": 0, "idx_j": 1}], 0,
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    assert result["tps_support_causal"]["available"] is True
    assert len(sources) == 2
    assert all(source is scene_data["arrays"][1] for source in sources)
    assert not np.shares_memory(flows[0], flows[1])


def test_tps_support_c1_reuses_same_holdout_context_for_both_validations(monkeypatch):
    pipe = _dummy_pipeline()
    context, validations = _patch_c1_dependencies(monkeypatch)

    pipe.run_klt_tps_support_c1_n2(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], 0,
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    assert len(validations) == 2
    assert validations[0] is validations[1]
    assert validations[0][(0, 1)] is context


def test_tps_support_c1_does_not_warp_supported_when_geometry_unsafe(monkeypatch):
    import src.klt_tps_registration as klt

    pipe = _dummy_pipeline()
    estimation = _fake_estimation()
    estimation["flow"][..., 0] = 1000.0
    _patch_c1_dependencies(monkeypatch, estimation=estimation)
    calls = []
    monkeypatch.setattr(
        klt, "warp_multiband_with_tps_flow",
        lambda source, flow, nodata: (
            calls.append(flow) or (source.copy(), np.ones(source.shape[1:], bool))
        ),
    )

    result = pipe.run_klt_tps_support_c1_n2(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], 0,
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    assert len(calls) == 1
    assert result["tps_support_causal"]["supported_geometry_safe"] is False
    assert result["tps_support_causal"]["supported_validation"] is None


def test_tps_support_c1_records_fold_d2_when_supported_geometry_is_unsafe(monkeypatch):
    pipe = _dummy_pipeline()
    estimation = _fake_estimation()
    estimation["flow"][..., 0] = 1000.0
    _patch_c1_dependencies(monkeypatch, estimation=estimation)

    result = pipe.run_klt_tps_support_c1_n2(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], 0,
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    causal = result["tps_support_causal"]
    assert causal["fold_d2"]["available"] is True
    assert causal["supported_geometry_safe"] is False


def test_tps_support_c1_fold_d2_does_not_allow_unsafe_supported_warp(monkeypatch):
    import src.klt_tps_registration as klt

    pipe = _dummy_pipeline()
    estimation = _fake_estimation()
    estimation["flow"][..., 0] = 1000.0
    _patch_c1_dependencies(monkeypatch, estimation=estimation)
    calls = []
    monkeypatch.setattr(
        klt, "warp_multiband_with_tps_flow",
        lambda source, flow, nodata: (
            calls.append(flow) or (source.copy(), np.ones(source.shape[1:], bool))
        ),
    )

    result = pipe.run_klt_tps_support_c1_n2(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], 0,
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    assert result["tps_support_causal"]["fold_d2"]["available"] is True
    assert result["tps_support_causal"]["supported_geometry_safe"] is False
    assert result["_tps_support_causal_arrays"]["supported_registered"] is None
    assert len(calls) == 1


def _fake_density_d3_result(shape=(48, 56)):
    coarse_shape = (3, 3)
    return {
        "available": True,
        "field_step": 4,
        "requested_neighbor_count": 80,
        "k_used": 4,
        "local_radius_cells": 2,
        "percentile_definition": "100 * mean(reference <= value)",
        "global_density": {
            "coarse_hull_sample_count": 1,
            "d1": {"count": 1, "min": 1.0, "p01": 1.0,
                    "p05": 1.0, "median": 1.0, "p75": 1.0,
                    "p90": 1.0, "p95": 1.0, "p99": 1.0, "max": 1.0},
            "dk": {"count": 1, "min": 2.0, "p01": 2.0,
                   "p05": 2.0, "median": 2.0, "p75": 2.0,
                   "p90": 2.0, "p95": 2.0, "p99": 2.0, "max": 2.0},
        },
        "fold_pixels": [],
        "neighbor_set_baseline": {
            "candidate_pair_count": 0,
            "sampled_pair_count": 0,
            "jaccard": {"count": 0},
            "replaced_neighbor_count": {"count": 0},
        },
        "component_patches": [],
        "_arrays": {
            "coarse_d1": np.zeros(coarse_shape),
            "coarse_dk": np.zeros(coarse_shape),
            "coarse_inside_hull": np.ones(coarse_shape, dtype=bool),
            "coarse_x": np.zeros(coarse_shape),
            "coarse_y": np.zeros(coarse_shape),
        },
    }


def test_tps_support_c1_records_density_d3_when_supported_is_unsafe(monkeypatch):
    import src.klt_tps_support_c1 as support

    pipe = _dummy_pipeline()
    estimation = _fake_estimation()
    estimation["flow"][..., 0] = 1000.0
    _patch_c1_dependencies(monkeypatch, estimation=estimation)
    seen = []

    def fake_d3(**kwargs):
        seen.append(kwargs)
        return _fake_density_d3_result()

    monkeypatch.setattr(support, "diagnose_tps_control_density", fake_d3)

    result = pipe.run_klt_tps_support_c1_n2(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], 0,
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    causal = result["tps_support_causal"]
    assert seen
    assert causal["density_d3"]["available"] is True
    assert causal["supported_geometry_safe"] is False
    assert causal["supported_validation"] is None
    assert "_arrays" not in causal["density_d3"]
    assert result["_tps_support_causal_arrays"]["density_d3_coarse_d1"].shape == (3, 3)


def test_tps_density_d3_receives_configured_field_step_and_neighbor_count(monkeypatch):
    import src.klt_tps_support_c1 as support

    pipe = _dummy_pipeline()
    pipe.config.registration_params["klt_tps_field_step"] = 7
    pipe.config.registration_params["klt_tps_neighbors"] = 13
    _patch_c1_dependencies(monkeypatch)
    seen = []

    def fake_d3(**kwargs):
        seen.append(kwargs)
        return _fake_density_d3_result()

    monkeypatch.setattr(support, "diagnose_tps_control_density", fake_d3)

    pipe.run_klt_tps_support_c1_n2(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], 0,
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    assert seen[0]["field_step"] == 7
    assert seen[0]["neighbor_count"] == 13


def test_tps_density_d3_does_not_change_supported_geometry_gate(monkeypatch):
    import src.klt_tps_support_c1 as support

    pipe = _dummy_pipeline()
    estimation = _fake_estimation()
    estimation["flow"][..., 0] = 1000.0
    _patch_c1_dependencies(monkeypatch, estimation=estimation)
    monkeypatch.setattr(
        support, "diagnose_tps_control_density",
        lambda **kwargs: _fake_density_d3_result(),
    )

    result = pipe.run_klt_tps_support_c1_n2(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], 0,
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    assert result["tps_support_causal"]["density_d3"]["available"] is True
    assert result["tps_support_causal"]["supported_geometry_safe"] is False
    assert result["_tps_support_causal_arrays"]["supported_registered"] is None


def test_tps_support_c1_passes_configured_tps_neighbor_count_to_d2(monkeypatch):
    import src.klt_tps_support_c1 as support

    pipe = _dummy_pipeline()
    pipe.config.registration_params["klt_tps_neighbors"] = 17
    _patch_c1_dependencies(monkeypatch)
    seen = []

    def fake_d2(**kwargs):
        seen.append(kwargs["neighbor_count"])
        return {
            "available": True,
            "fold_pixel_count": 0,
            "classification_counts": {
                "outside_hull": 0, "taper": 0, "deep_inside": 0,
            },
            "weight_class_counts": {"zero": 0, "partial": 0, "one": 0},
            "component_count": 0,
            "components": [],
            "pixels": [],
        }

    monkeypatch.setattr(support, "diagnose_supported_fold_pixels", fake_d2)

    pipe.run_klt_tps_support_c1_n2(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], 0,
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    assert seen == [17]


def test_tps_support_c1_records_raw_fit_count_one(monkeypatch):
    pipe = _dummy_pipeline()
    _patch_c1_dependencies(monkeypatch)

    result = pipe.run_klt_tps_support_c1_n2(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], 0,
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    assert result["tps_support_causal"]["integrity"]["raw_fit_count"] == 1


def test_tps_support_c1_integrity_requires_available_holdout_comparison(monkeypatch):
    pipe = _dummy_pipeline()
    calls = []

    def validate(*args, **kwargs):
        calls.append(True)
        blocks = [{
            "validation_row": 4,
            "validation_col": 8,
            "block_size": 384,
            "residual_magnitude": 1.0 if len(calls) == 1 else None,
            "accepted": True,
        }]
        return (
            {"quality": "pass", "rmse": 0.1, "p95": 0.2, "median": 0.1,
             "confidence": 0.9, "n_blocks": 1},
            {"edges": [{"idx_i": 0, "idx_j": 1, "blocks": blocks}],
             "overall": {"quality": "pass"}},
        )

    _patch_c1_dependencies(monkeypatch, validate=validate)
    result = pipe.run_klt_tps_support_c1_n2(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], 0,
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    integrity = result["tps_support_causal"]["integrity"]
    assert integrity["comparison_available"] is False
    assert integrity["n_paired_blocks"] == 0
    assert integrity["integrity_pass"] is False


def test_production_klt_tps_register_scenes_does_not_enable_support_c1(monkeypatch):
    pipe = _dummy_pipeline()
    called = []
    monkeypatch.setattr(
        pipe, "_register_scenes_klt_tps_n2",
        lambda *args, **kwargs: called.append(kwargs) or {"production": True},
    )
    pipe.run_klt_tps_support_c1_n2 = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("support C1 must not be enabled by production dispatch")
    )

    result = pipe.register_scenes(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], registration_band_idx=0,
    )

    assert result == {"production": True}
    assert called


def test_tps_support_c1_does_not_change_klt_or_tps_parameters(monkeypatch):
    pipe = _dummy_pipeline()
    fixed = {
        "klt_tps_window": 9,
        "klt_tps_pyramid_level": 3,
        "klt_tps_max_corners": 4000,
        "klt_tps_min_corner_distance": 5.0,
        "klt_tps_fb_threshold": 0.5,
        "klt_tps_min_points": 30,
        "klt_tps_neighbors": 80,
        "klt_tps_smoothing": 3.0,
        "klt_tps_field_step": 4,
        "klt_tps_max_shift": 50.0,
    }
    pipe.config.registration_params.update(fixed)
    before = dict(pipe.config.registration_params)
    _patch_c1_dependencies(monkeypatch)

    pipe.run_klt_tps_support_c1_n2(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], 0,
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    assert pipe.config.registration_params == before


def test_tps_support_c1_does_not_change_quality_thresholds(monkeypatch):
    pipe = _dummy_pipeline()
    thresholds = {
        "required_quality": "pass",
        "final_min_blocks": 5,
        "validation_confidence_threshold": 0.45,
        "validation_max_residual_shift": 3.0,
    }
    pipe.config.registration_params.update(thresholds)
    before = dict(pipe.config.registration_params)
    _patch_c1_dependencies(monkeypatch)

    pipe.run_klt_tps_support_c1_n2(
        _c1_scene_data(), [{"idx_i": 0, "idx_j": 1}], 0,
        holdout_reservation_overrides={(0, 1): [(4, 4, 384, 384)] * 7},
    )

    assert pipe.config.registration_params == before


def test_legacy_backend_dispatch_remains_unchanged(monkeypatch):
    pipe = _dummy_pipeline(backend="legacy")
    pipe._register_scenes_klt_tps_n2 = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("legacy dispatch entered KLT/TPS path")
    )

    result = pipe.register_scenes(
        _c1_scene_data(), [], registration_band_idx=0,
    )

    assert result["status"] == "fail"
    assert result.get("registration_backend") != "klt_tps"
