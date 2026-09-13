import copy
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "dz01_model_c1_holdout_manifest.json"


def _config():
    return {
        "registration_band": "B12",
        "selected_bands": ["B12", "B14"],
        "scenes": [
            {"id": "scene_20251114"},
            {"id": "scene_20251120"},
        ],
        "registration_params": {"registration_backend": "klt_tps"},
    }


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_tps_support_c1_protocol_accepts_frozen_b14_manifest():
    from src.klt_tps_support_c1 import validate_tps_support_c1_protocol

    result = validate_tps_support_c1_protocol(_config(), _manifest())

    assert result["valid"] is True
    assert result["errors"] == []
    assert result["registration_band"] == "B12"
    assert result["validation_band"] == "B14"
    assert result["reserved_count"] == 7
    assert result["selected_block_size"] == 384
    assert result["taper_pixels"] == 64


def test_tps_support_c1_protocol_requires_klt_tps_backend():
    from src.klt_tps_support_c1 import validate_tps_support_c1_protocol

    config = _config()
    config["registration_params"]["registration_backend"] = "legacy"

    result = validate_tps_support_c1_protocol(config, _manifest())

    assert result["valid"] is False
    assert any("klt_tps" in error for error in result["errors"])


def test_tps_support_c1_protocol_requires_b12_registration_and_b14_validation():
    from src.klt_tps_support_c1 import validate_tps_support_c1_protocol

    config = _config()
    config["registration_band"] = "B14"
    config["selected_bands"] = ["B12", "B14"]

    result = validate_tps_support_c1_protocol(
        config, _manifest(), validation_band="B12"
    )

    assert result["valid"] is False
    assert any("registration band" in error.lower() for error in result["errors"])
    assert any("validation band" in error.lower() for error in result["errors"])


def test_tps_support_c1_protocol_rejects_empty_pair_windows():
    from src.klt_tps_support_c1 import validate_tps_support_c1_protocol

    manifest = _manifest()
    manifest["pairs"]["0-1"]["reserved_windows"] = []

    result = validate_tps_support_c1_protocol(_config(), manifest)

    assert result["valid"] is False
    assert any("window" in error.lower() for error in result["errors"])


def test_tps_support_c1_protocol_requires_exactly_seven_384_windows():
    from src.klt_tps_support_c1 import validate_tps_support_c1_protocol

    manifest = _manifest()
    manifest["pairs"]["0-1"]["reserved_windows"].append(
        {"row": 10, "col": 10, "height": 384, "width": 384}
    )

    result = validate_tps_support_c1_protocol(_config(), manifest)

    assert result["valid"] is False
    assert any("seven" in error.lower() for error in result["errors"])


def test_tps_support_c1_protocol_rejects_scene_id_mismatch():
    from src.klt_tps_support_c1 import validate_tps_support_c1_protocol

    config = _config()
    config["scenes"][1]["id"] = "other_scene"

    result = validate_tps_support_c1_protocol(config, _manifest())

    assert result["valid"] is False
    assert any("scene" in error.lower() for error in result["errors"])


def test_tps_support_c1_array_sha256_is_shape_dtype_and_content_sensitive():
    from src.klt_tps_support_c1 import array_sha256

    base = np.array([[1, 2], [3, 4]], dtype=np.int16)

    assert array_sha256(base) != array_sha256(base.astype(np.float32))
    assert array_sha256(base) != array_sha256(base.reshape(1, 4))
    changed = base.copy()
    changed[0, 0] = 9
    assert array_sha256(base) != array_sha256(changed)
    assert array_sha256(base) == array_sha256(np.ascontiguousarray(base))


def test_validation_block_keys_are_sorted_and_ignore_acceptance_status():
    from src.klt_tps_support_c1 import validation_block_keys

    validation = {
        "edges": [
            {
                "idx_i": 0,
                "idx_j": 1,
                "validation_block_size_selected": 384,
                "blocks": [
                    {
                        "validation_row": 512,
                        "validation_col": 64,
                        "block_size": 384,
                        "accepted": False,
                    },
                    {
                        "validation_row": 256,
                        "validation_col": 640,
                        "block_size": 384,
                        "accepted": True,
                    },
                ],
            }
        ]
    }

    assert validation_block_keys(validation) == [
        (0, 1, 256, 640, 384),
        (0, 1, 512, 64, 384),
    ]


def _field_inputs(shape=(64, 72)):
    controls = np.asarray([
        [10.0, 10.0], [60.0, 10.0], [60.0, 54.0], [10.0, 54.0],
    ])
    displacements = np.asarray([
        [2.0, -1.0], [2.1, -0.9], [1.9, -1.1], [2.0, -1.0],
    ])
    y, x = np.indices(shape)
    raw = np.stack([2.0 + 0.01 * x, -1.0 + 0.005 * y], axis=-1).astype(
        np.float32
    )
    return raw, controls, displacements


def test_tps_support_field_bundle_uses_one_raw_flow_without_refit():
    from src.klt_tps_support_c1 import build_tps_support_c1_fields

    raw, controls, displacements = _field_inputs()
    fields = build_tps_support_c1_fields(
        raw, controls, displacements, max_shift=50.0, taper_pixels=8,
    )

    assert fields["raw_flow_sha256"]
    np.testing.assert_array_equal(fields["_arrays"]["raw_flow"], raw)
    assert fields["_arrays"]["raw_flow"] is raw


def test_tps_support_field_bundle_preserves_distance_inside_for_diagnostics():
    from src.klt_tps_registration import build_inside_hull_taper_weight
    from src.klt_tps_support_c1 import build_tps_support_c1_fields

    raw, controls, displacements = _field_inputs()
    fields = build_tps_support_c1_fields(
        raw, controls, displacements, max_shift=50.0, taper_pixels=8,
    )
    support = build_inside_hull_taper_weight(controls, raw.shape[:2], 8)

    assert fields["_arrays"]["support_distance_inside"].shape == raw.shape[:2]
    np.testing.assert_array_equal(
        fields["_arrays"]["support_distance_inside"],
        support["distance_inside"],
    )


def test_adding_d2_private_array_does_not_change_supported_flow():
    from src.klt_tps_registration import (
        build_inside_hull_taper_weight,
        compose_supported_tps_flow,
    )
    from src.klt_tps_support_c1 import build_tps_support_c1_fields

    raw, controls, displacements = _field_inputs()
    fields = build_tps_support_c1_fields(
        raw, controls, displacements, max_shift=50.0, taper_pixels=8,
    )
    support = build_inside_hull_taper_weight(controls, raw.shape[:2], 8)
    expected = compose_supported_tps_flow(
        raw, fields["translation_xy"], support["weight"],
    )

    np.testing.assert_array_equal(fields["_arrays"]["supported_flow"], expected)


def test_tps_support_field_bundle_records_raw_translation_supported_geometry():
    from src.klt_tps_support_c1 import build_tps_support_c1_fields

    raw, controls, displacements = _field_inputs()
    fields = build_tps_support_c1_fields(
        raw, controls, displacements, max_shift=50.0, taper_pixels=8,
    )

    for name in ("raw_geometry", "translation_geometry", "supported_geometry"):
        assert "geometry_safe" in fields[name]
        assert "fold_pixels" in fields[name]
        assert "max_displacement_pixels" in fields[name]
    np.testing.assert_allclose(fields["translation_xy"], [2.0, -1.0])
    assert fields["taper_pixels"] == 8


def test_tps_support_field_integrity_requires_outside_translation_match():
    from src.klt_tps_support_c1 import (
        build_tps_support_c1_fields,
        summarize_tps_support_field_integrity,
    )

    raw, controls, displacements = _field_inputs()
    fields = build_tps_support_c1_fields(
        raw, controls, displacements, max_shift=50.0, taper_pixels=8,
    )
    arrays = fields["_arrays"]
    arrays["supported_flow"][0, 0, 0] += 0.1

    integrity = summarize_tps_support_field_integrity(fields)

    assert integrity["outside_max_abs_supported_minus_translation"] > 1e-6
    assert integrity["support_formula_pass"] is False


def test_tps_support_field_integrity_requires_deep_inside_raw_match():
    from src.klt_tps_support_c1 import (
        build_tps_support_c1_fields,
        summarize_tps_support_field_integrity,
    )

    raw, controls, displacements = _field_inputs()
    fields = build_tps_support_c1_fields(
        raw, controls, displacements, max_shift=50.0, taper_pixels=8,
    )
    arrays = fields["_arrays"]
    deep = arrays["deep_inside_mask"]
    assert deep.any()
    row, col = np.argwhere(deep)[0]
    arrays["supported_flow"][row, col, 1] += 0.1

    integrity = summarize_tps_support_field_integrity(fields)

    assert integrity["deep_inside_max_abs_supported_minus_raw"] > 1e-6
    assert integrity["support_formula_pass"] is False


def test_tps_support_field_bundle_reports_supported_unsafe_without_warping():
    from src.klt_tps_support_c1 import build_tps_support_c1_fields

    raw, controls, displacements = _field_inputs()
    raw[..., 0] = 60.0
    fields = build_tps_support_c1_fields(
        raw, controls, displacements, max_shift=50.0, taper_pixels=8,
    )

    assert fields["translation_geometry"]["geometry_safe"] is True
    assert fields["supported_geometry"]["geometry_safe"] is False
    assert fields["supported_geometry"]["rejection_reason"]


def _d2_inputs():
    shape = (6, 7)
    raw_flow = np.zeros((*shape, 2), dtype=float)
    translation_flow = np.zeros_like(raw_flow)
    translation_flow[..., 0] = 1.0
    translation_flow[..., 1] = -2.0
    supported_flow = translation_flow.copy()
    support_weight = np.zeros(shape, dtype=float)
    support_hull_mask = np.zeros(shape, dtype=bool)
    deep_inside_mask = np.zeros(shape, dtype=bool)
    distance_inside = np.zeros(shape, dtype=float)
    supported_fold_mask = np.zeros(shape, dtype=bool)
    supported_fold_mask[0, 6] = True
    supported_fold_mask[2, 2] = True
    supported_fold_mask[3, 3] = True
    support_hull_mask[2, 2] = True
    support_hull_mask[3, 3] = True
    support_weight[2, 2] = 0.5
    support_weight[3, 3] = 1.0
    deep_inside_mask[3, 3] = True
    distance_inside[2, 2] = 4.0
    distance_inside[3, 3] = 64.0
    return {
        "raw_flow": raw_flow,
        "translation_flow": translation_flow,
        "supported_flow": supported_flow,
        "support_weight": support_weight,
        "support_hull_mask": support_hull_mask,
        "deep_inside_mask": deep_inside_mask,
        "distance_inside": distance_inside,
        "supported_fold_mask": supported_fold_mask,
        "supported_jacobian": np.ones(shape, dtype=float),
        "control_points_xy": np.asarray([
            [0.0, 0.0], [6.0, 0.0], [6.0, 5.0], [0.0, 5.0],
        ]),
        "displacement_xy": np.asarray([
            [0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 4.0],
        ]),
        "neighbor_count": 80,
    }


def _diagnose_d2(**updates):
    from src.klt_tps_support_c1 import diagnose_supported_fold_pixels

    data = _d2_inputs()
    data.update(updates)
    return diagnose_supported_fold_pixels(**data)


def test_fold_d2_classifies_outside_hull_pixel():
    result = _diagnose_d2()

    pixel = next(item for item in result["pixels"] if item["row"] == 0)
    assert pixel["region"] == "outside_hull"
    assert pixel["weight_class"] == "zero"


def test_fold_d2_classifies_taper_pixel():
    result = _diagnose_d2()

    pixel = next(item for item in result["pixels"] if item["row"] == 2)
    assert pixel["region"] == "taper"
    assert pixel["weight_class"] == "partial"


def test_fold_d2_classifies_deep_inside_pixel():
    result = _diagnose_d2()

    pixel = next(item for item in result["pixels"] if item["row"] == 3)
    assert pixel["region"] == "deep_inside"
    assert pixel["weight_class"] == "one"


def test_fold_d2_weight_classes_zero_partial_one():
    result = _diagnose_d2()

    assert result["weight_class_counts"] == {"zero": 1, "partial": 1, "one": 1}
    assert result["classification_counts"] == {
        "outside_hull": 1, "taper": 1, "deep_inside": 1,
    }


def test_fold_d2_uses_xy_control_coordinates_for_row_col_query():
    data = _d2_inputs()
    data["supported_fold_mask"][:] = False
    data["supported_fold_mask"][1, 4] = True
    data["support_hull_mask"][1, 4] = True
    data["support_weight"][1, 4] = 0.5
    data["control_points_xy"] = np.asarray([
        [4.0, 1.0], [0.0, 0.0], [6.0, 5.0],
    ])
    data["displacement_xy"] = np.asarray([
        [9.0, 8.0], [0.0, 0.0], [3.0, 4.0],
    ])

    pixel = _diagnose_d2(**data)["pixels"][0]

    assert pixel["nearest_control_index"] == 0
    assert pixel["nearest_control_x"] == pytest.approx(4.0)
    assert pixel["nearest_control_y"] == pytest.approx(1.0)
    assert pixel["nearest_control_distance_pixels"] == pytest.approx(0.0)


def test_fold_d2_reports_nearest_control_distance():
    pixel = _diagnose_d2()["pixels"][1]

    assert pixel["nearest_control_index"] == 0
    assert pixel["nearest_control_distance_pixels"] == pytest.approx(
        np.sqrt(8.0)
    )


def test_fold_d2_uses_requested_neighbor_count():
    pixel = _diagnose_d2(neighbor_count=2)["pixels"][0]

    assert pixel["local_neighbor_count"] == 2


def test_fold_d2_reports_local_displacement_spread():
    pixel = _diagnose_d2()["pixels"][1]

    assert pixel["local_dx_max"] > pixel["local_dx_min"]
    assert pixel["local_dy_max"] > pixel["local_dy_min"]
    assert pixel["local_vector_deviation_median"] >= 0.0
    assert pixel["local_vector_deviation_p95"] >= pixel[
        "local_vector_deviation_median"
    ]
    assert pixel["local_vector_deviation_max"] >= pixel[
        "local_vector_deviation_p95"
    ]


def test_fold_d2_labels_8_connected_components():
    data = _d2_inputs()
    data["supported_fold_mask"][:] = False
    data["supported_fold_mask"][1, 1] = True
    data["supported_fold_mask"][2, 2] = True
    data["supported_fold_mask"][4, 5] = True
    data["support_hull_mask"][1, 1] = True
    data["support_hull_mask"][2, 2] = True
    data["support_hull_mask"][4, 5] = True
    data["support_weight"][1, 1] = 1.0
    data["support_weight"][2, 2] = 1.0
    data["support_weight"][4, 5] = 1.0
    data["deep_inside_mask"][1, 1] = True
    data["deep_inside_mask"][2, 2] = True
    data["deep_inside_mask"][4, 5] = True

    result = _diagnose_d2(**data)

    assert result["component_count"] == 2
    assert result["components"][0]["pixel_count"] == 2
    assert result["components"][0]["n_deep_inside"] == 2
    assert result["components"][1]["pixel_count"] == 1


def test_fold_d2_handles_zero_fold_pixels():
    data = _d2_inputs()
    data["supported_fold_mask"][:] = False

    result = _diagnose_d2(**data)

    assert result == {
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


def test_fold_d2_does_not_mutate_any_input_array():
    data = _d2_inputs()
    before = {name: value.copy() for name, value in data.items()
              if isinstance(value, np.ndarray)}

    _diagnose_d2(**data)

    for name, value in before.items():
        np.testing.assert_array_equal(data[name], value)


def _density_d3_fixture(*, shape=(9, 10), step=4, fold_pixels=None):
    height, width = shape
    raw_flow = np.zeros((height, width, 2), dtype=float)
    yy, xx = np.indices(shape, dtype=float)
    raw_flow[..., 0] = xx
    raw_flow[..., 1] = 2.0 * yy
    hull = np.ones(shape, dtype=bool)
    if fold_pixels is None:
        fold_pixels = [{"row": 3, "col": 3, "region": "deep_inside",
                        "weight_class": "one"}]
    components = []
    for index, pixel in enumerate(fold_pixels, start=1):
        row = int(pixel["row"])
        col = int(pixel["col"])
        components.append({
            "component_id": index,
            "pixel_count": 1,
            "row_min": row,
            "row_max": row,
            "col_min": col,
            "col_max": col,
            "n_outside_hull": 0,
            "n_taper": int(pixel.get("region") == "taper"),
            "n_deep_inside": int(pixel.get("region") == "deep_inside"),
        })
    return {
        "raw_flow": raw_flow,
        "control_points_xy": np.asarray([
            [0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0],
        ]),
        "support_hull_mask": hull,
        "fold_d2": {
            "available": True,
            "pixels": copy.deepcopy(fold_pixels),
            "components": components,
        },
        "field_step": step,
        "neighbor_count": 2,
    }


def _run_density_d3(fixture=None, **overrides):
    from src.klt_tps_support_c1 import diagnose_tps_control_density

    data = _density_d3_fixture() if fixture is None else fixture
    data = {**data, **overrides}
    return diagnose_tps_control_density(**data)


def test_density_d3_uses_same_query_grid_semantics_as_raw_tps():
    result = _run_density_d3()
    arrays = result.pop("_arrays")

    assert np.array_equal(arrays["coarse_x"][0], [0, 4, 8, 12])
    assert np.array_equal(arrays["coarse_y"][:, 0], [0, 4, 8, 12])
    assert arrays["coarse_inside_hull"].shape == (4, 4)
    assert not arrays["coarse_inside_hull"][2, 3]
    assert result["global_density"]["coarse_hull_sample_count"] == 9


def test_density_d3_global_baseline_uses_only_scene_interior_hull_queries():
    fixture = _density_d3_fixture()
    fixture["support_hull_mask"][:] = False
    fixture["support_hull_mask"][:5, :5] = True

    result = _run_density_d3(fixture)

    assert result["global_density"]["coarse_hull_sample_count"] == 4
    assert result["_arrays"]["coarse_d1"][0, 0] == 0.0
    assert np.isnan(result["_arrays"]["coarse_d1"][2, 2])


def test_density_d3_reports_exact_fold_d1_and_dk():
    fixture = _density_d3_fixture(shape=(12, 12), step=4,
                                  fold_pixels=[{
                                      "row": 0, "col": 0,
                                      "region": "deep_inside",
                                      "weight_class": "one",
                                  }])
    result = _run_density_d3(fixture, neighbor_count=2)

    fold = result["fold_pixels"][0]
    assert fold["d1_pixels"] == 0.0
    assert fold["dk_pixels"] == 10.0


def test_density_d3_reports_fold_density_percentiles():
    result = _run_density_d3()
    fold = result["fold_pixels"][0]
    coarse = result["_arrays"]["coarse_d1"][
        result["_arrays"]["coarse_inside_hull"]
    ]

    assert fold["d1_percentile"] == pytest.approx(
        100.0 * np.mean(coarse <= fold["d1_pixels"])
    )
    assert 0.0 <= fold["dk_percentile"] <= 100.0


def test_density_d3_fold_percentiles_are_relative_to_coarse_hull_baseline():
    fixture = _density_d3_fixture(shape=(9, 10), step=4,
                                  fold_pixels=[{
                                      "row": 3, "col": 3,
                                      "region": "deep_inside",
                                      "weight_class": "one",
                                  }])
    fixture["control_points_xy"] = np.asarray([[0.0, 0.0]])
    result = _run_density_d3(fixture, neighbor_count=1)
    fold = result["fold_pixels"][0]
    coarse = result["_arrays"]["coarse_d1"][
        result["_arrays"]["coarse_inside_hull"]
    ]

    assert fold["d1_percentile"] == pytest.approx(
        100.0 * np.mean(coarse <= fold["d1_pixels"])
    )
    dense_distance = np.linalg.norm(
        fixture["control_points_xy"][:1] - np.asarray([[3.0, 3.0]]), axis=1,
    )[0]
    dense_xy = np.stack(np.indices(fixture["raw_flow"].shape[:2])[::-1], axis=-1)
    dense_baseline = np.linalg.norm(
        dense_xy - fixture["control_points_xy"][:1], axis=2,
    )
    dense_percentile = 100.0 * np.mean(dense_baseline <= dense_distance)
    assert fold["d1_percentile"] != pytest.approx(dense_percentile)


def test_density_d3_neighbor_jaccard_identical_sets_is_one():
    result = _run_density_d3(neighbor_count=4)

    assert result["neighbor_set_baseline"]["jaccard"]["min"] == 1.0
    assert result["neighbor_set_baseline"]["jaccard"]["max"] == 1.0


def test_density_d3_neighbor_jaccard_detects_one_replaced_neighbor():
    from src.klt_tps_support_c1 import _density_d3_neighbor_metrics

    result = _density_d3_neighbor_metrics(
        np.asarray([1, 2, 3, 4]), np.asarray([1, 2, 3, 5]), 4,
    )

    assert result == {
        "intersection_count": 3,
        "jaccard": pytest.approx(0.6),
        "replaced_neighbor_count": 1,
    }


def test_density_d3_global_neighbor_baseline_is_deterministic():
    first = _run_density_d3()
    second = _run_density_d3()
    first_arrays = first.pop("_arrays")
    second_arrays = second.pop("_arrays")

    assert first == second
    for key in first_arrays:
        assert np.array_equal(first_arrays[key], second_arrays[key], equal_nan=True)


def test_density_d3_local_patch_uses_coarse_field_step():
    result = _run_density_d3()

    points = result["component_patches"][0]["query_points"]
    assert {point["query_x"] % 4 for point in points} <= {0}
    assert {point["query_y"] % 4 for point in points} <= {0}
    assert all(
        pair["x1"] - pair["x0"] == 4
        or pair["y1"] - pair["y0"] == 4
        for pair in result["component_patches"][0]["adjacent_pairs"]
    )


def test_density_d3_local_patch_reports_raw_flow_change():
    result = _run_density_d3()

    pair = next(
        pair for pair in result["component_patches"][0]["adjacent_pairs"]
        if pair["x1"] - pair["x0"] == 4
    )
    assert pair["raw_dx_delta"] == pytest.approx(4.0)
    assert pair["raw_dy_delta"] == pytest.approx(0.0)
    assert pair["raw_displacement_delta_magnitude"] == pytest.approx(4.0)


def test_density_d3_does_not_mutate_inputs():
    fixture = _density_d3_fixture()
    original = {
        "raw_flow": fixture["raw_flow"].copy(),
        "control_points_xy": fixture["control_points_xy"].copy(),
        "support_hull_mask": fixture["support_hull_mask"].copy(),
        "fold_d2": copy.deepcopy(fixture["fold_d2"]),
    }

    _run_density_d3(fixture)

    assert np.array_equal(fixture["raw_flow"], original["raw_flow"])
    assert np.array_equal(
        fixture["control_points_xy"], original["control_points_xy"]
    )
    assert np.array_equal(
        fixture["support_hull_mask"], original["support_hull_mask"]
    )
    assert fixture["fold_d2"] == original["fold_d2"]


def test_density_d3_handles_zero_fold_pixels():
    fixture = _density_d3_fixture(fold_pixels=[])
    fixture["fold_d2"] = {"available": True, "pixels": [], "components": []}

    result = _run_density_d3(fixture)

    assert result["available"] is True
    assert result["global_density"]["coarse_hull_sample_count"] > 0
    assert result["fold_pixels"] == []
    assert result["component_patches"] == []


def _validation_fixture(*, supported=False, include_second=True):
    first = {
        "validation_row": 10,
        "validation_col": 20,
        "block_size": 384,
        "residual_magnitude": 2.0 if supported else 3.0,
        "accepted": True if supported else False,
        "reject_reason": None if supported else "low_confidence",
    }
    blocks = [first]
    if include_second:
        blocks.append({
            "validation_row": 40,
            "validation_col": 50,
            "block_size": 384,
            "residual_magnitude": 4.0 if supported else 5.0,
            "accepted": True,
            "reject_reason": None,
        })
    return {
        "edges": [{
            "idx_i": 0,
            "idx_j": 1,
            "validation_block_size_selected": 384,
            "blocks": blocks,
        }],
        "overall": {
            "median": 3.0 if supported else 4.0,
            "rmse": 3.0 if supported else 4.0,
            "p95": 4.0 if supported else 5.0,
        },
    }


def test_compare_tps_support_validations_pairs_identical_fixed_windows():
    from src.klt_tps_support_c1 import compare_tps_support_validations

    comparison = compare_tps_support_validations(
        _validation_fixture(), _validation_fixture(supported=True)
    )

    assert comparison["holdout_keys_match"] is True
    assert len(comparison["paired_blocks"]) == 2
    assert comparison["paired_blocks"][0]["key"] == [0, 1, 10, 20, 384]


def test_compare_tps_support_validations_preserves_rejected_blocks():
    from src.klt_tps_support_c1 import compare_tps_support_validations

    comparison = compare_tps_support_validations(
        _validation_fixture(include_second=False),
        _validation_fixture(supported=True, include_second=False),
    )

    block = comparison["paired_blocks"][0]
    assert block["translation_accepted"] is False
    assert block["translation_reject_reason"] == "low_confidence"
    assert block["translation_residual"] == pytest.approx(3.0)
    assert block["supported_residual"] == pytest.approx(2.0)
    assert block["improvement"] == pytest.approx(1.0)


def test_compare_tps_support_validations_reports_translation_minus_supported_improvement():
    from src.klt_tps_support_c1 import compare_tps_support_validations

    comparison = compare_tps_support_validations(
        _validation_fixture(), _validation_fixture(supported=True)
    )

    assert comparison["rmse_improvement"] == pytest.approx(
        np.sqrt(17.0) - np.sqrt(10.0)
    )
    assert comparison["p95_improvement"] == pytest.approx(1.0)
    assert comparison["median_improvement"] == pytest.approx(1.0)
    assert all(block["improvement"] > 0 for block in comparison["paired_blocks"])


def test_compare_tps_support_validations_uses_same_paired_blocks_for_rmse():
    from src.klt_tps_support_c1 import compare_tps_support_validations

    translation_validation = {
        "edges": [{"idx_i": 0, "idx_j": 1, "blocks": [
            {"validation_row": 10, "validation_col": 20,
             "block_size": 384, "residual_magnitude": 1.0,
             "accepted": True},
            {"validation_row": 30, "validation_col": 40,
             "block_size": 384, "residual_magnitude": 5.0,
             "accepted": False},
        ]}],
        "overall": {"median": 101.0, "rmse": 102.0, "p95": 103.0},
    }
    supported_validation = {
        "edges": [{"idx_i": 0, "idx_j": 1, "blocks": [
            {"validation_row": 10, "validation_col": 20,
             "block_size": 384, "residual_magnitude": 2.0,
             "accepted": True},
            {"validation_row": 30, "validation_col": 40,
             "block_size": 384, "residual_magnitude": 3.0,
             "accepted": True},
        ]}],
        "overall": {"median": 201.0, "rmse": 202.0, "p95": 203.0},
    }

    comparison = compare_tps_support_validations(
        translation_validation, supported_validation
    )

    assert comparison["all_measurable"]["translation"] == {
        "n": 2, "median": 3.0, "rmse": pytest.approx(np.sqrt(13.0)),
        "p95": pytest.approx(4.8),
    }
    assert comparison["all_measurable"]["supported"] == {
        "n": 2, "median": 2.5, "rmse": pytest.approx(np.sqrt(6.5)),
        "p95": pytest.approx(2.95),
    }
    assert comparison["rmse_improvement"] == pytest.approx(
        np.sqrt(13.0) - np.sqrt(6.5)
    )
    assert comparison["p95_improvement"] == pytest.approx(1.85)
    assert comparison["median_improvement"] == pytest.approx(0.5)
    assert comparison["stage_quality"]["translation"] == {
        "median": 101.0, "rmse": 102.0, "p95": 103.0,
    }
    assert comparison["paired_blocks"][1]["translation_accepted"] is False


def test_compare_tps_support_validations_marks_holdout_mismatch_unavailable():
    from src.klt_tps_support_c1 import compare_tps_support_validations

    comparison = compare_tps_support_validations(
        _validation_fixture(),
        _validation_fixture(supported=True, include_second=False),
    )

    assert comparison["holdout_keys_match"] is False
    assert comparison["available"] is False
    assert comparison["paired_blocks"]
