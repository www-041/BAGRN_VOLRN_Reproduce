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

    assert comparison["rmse_improvement"] == pytest.approx(1.0)
    assert comparison["p95_improvement"] == pytest.approx(1.0)
    assert comparison["median_improvement"] == pytest.approx(1.0)
    assert all(block["improvement"] > 0 for block in comparison["paired_blocks"])


def test_compare_tps_support_validations_marks_holdout_mismatch_unavailable():
    from src.klt_tps_support_c1 import compare_tps_support_validations

    comparison = compare_tps_support_validations(
        _validation_fixture(),
        _validation_fixture(supported=True, include_second=False),
    )

    assert comparison["holdout_keys_match"] is False
    assert comparison["available"] is False
    assert comparison["paired_blocks"]
