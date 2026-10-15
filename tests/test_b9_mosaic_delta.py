import importlib


delta = importlib.import_module("scripts.audit_b9_mosaic_delta")


def test_translation_delta_uses_explicit_sign_convention():
    mst = {
        "ZNCC_intensity": 0.80,
        "gradient_magnitude_NCC": 0.70,
        "gradient_orientation_cosine": 0.60,
        "intensity_MAE": 10.0,
        "valid_overlap_pixels": 100,
    }
    translation = {
        "ZNCC_intensity": 0.90,
        "gradient_magnitude_NCC": 0.75,
        "gradient_orientation_cosine": 0.65,
        "intensity_MAE": 8.0,
        "valid_overlap_pixels": 98,
    }

    result = delta.pair_delta(mst, translation)

    assert result["delta_overlap_ZNCC"] == 0.10
    assert result["delta_gradient_magnitude_NCC"] == 0.05
    assert result["delta_gradient_orientation_cosine"] == 0.05
    assert result["delta_aux_intensity_MAE"] == -2.0
    assert result["common_valid_pixels"] == 98
