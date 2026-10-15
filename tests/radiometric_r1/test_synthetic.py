import numpy as np
import pytest

from src.benchmark_v1_radiometric_r1.synthetic import generate_case


def test_s1_global_gain_offset_has_exact_known_inverse_and_does_not_mutate_input():
    source = np.arange(12, dtype=np.float64).reshape(3, 4)
    original = source.copy()
    result = generate_case(source, np.ones(source.shape, dtype=bool), {"case": "S1", "gain": 1.25, "offset": -3.0}, seed=31)
    assert np.allclose(result.observed, source * 1.25 - 3.0)
    assert np.allclose((result.observed - (-3.0)) / 1.25, result.clean_target)
    assert np.array_equal(source, original)
    assert not result.saturation_mask.any()


def test_s2_spatial_field_is_deterministic_and_reported_as_truth():
    source = np.ones((4, 5), dtype=np.float32)
    config = {"case": "S2", "gain_base": 1.0, "gain_gradient_x": 0.2, "offset_base": 2.0, "offset_gradient_y": 4.0}
    a = generate_case(source, np.ones(source.shape, bool), config, seed=7)
    b = generate_case(source, np.ones(source.shape, bool), config, seed=7)
    assert np.array_equal(a.observed, b.observed)
    assert np.array_equal(a.perturbation["gain_field"], b.perturbation["gain_field"])
    assert np.allclose(a.observed, source * a.perturbation["gain_field"] + a.perturbation["offset_field"])


def test_s3_keeps_per_band_truth_without_mixing_scales():
    source = np.ones((2, 3, 4), dtype=np.float64)
    result = generate_case(source, np.ones((3, 4), bool), {"case": "S3", "gains": [2.0, 0.5], "offsets": [1.0, -2.0]}, seed=1)
    assert np.allclose(result.observed[0], 3.0)
    assert np.allclose(result.observed[1], -1.5)
    assert result.perturbation["gain_per_band"].tolist() == [2.0, 0.5]
    assert result.perturbation["offset_per_band"].tolist() == [1.0, -2.0]


def test_s4_retains_change_truth_and_observation_perturbation_separately():
    clean = np.full((6, 8), 10.0)
    mask = np.ones(clean.shape, dtype=bool)
    config = {"case": "S4", "gain": 2.0, "offset": 1.0, "change_region": [1, 4, 2, 6], "change_amplitude": 3.0}
    result = generate_case(clean, mask, config, seed=9)
    assert result.change_mask.sum() == 12
    assert np.all(result.change_amplitude[result.change_mask] == 3.0)
    assert np.all(result.clean_target == 10.0)
    assert np.all(result.observed[~result.change_mask] == 21.0)
    assert np.all(result.observed[result.change_mask] == 27.0)


def test_valid_mask_nan_and_clipping_are_explicit():
    source = np.array([[0.0, 1.0], [np.nan, 4.0]])
    valid = np.array([[True, True], [True, False]])
    result = generate_case(source, valid, {"case": "S1", "gain": 2.0, "offset": 0.0, "clip_min": 0.0, "clip_max": 1.5}, seed=0)
    assert result.valid_mask.tolist() == [[True, True], [False, False]]
    assert result.observed[0, 1] == 1.5
    assert result.saturation_mask[0, 1]
    assert result.observed[1, 0] == 0 and result.observed[1, 1] == 0


def test_invalid_case_or_band_parameter_count_fails_explicitly():
    with pytest.raises(ValueError, match="case"):
        generate_case(np.ones((2, 2)), np.ones((2, 2), bool), {"case": "S9"}, seed=0)
    with pytest.raises(ValueError, match="bands"):
        generate_case(np.ones((2, 2, 2)), np.ones((2, 2), bool), {"case": "S3", "gains": [1], "offsets": [0]}, seed=0)
