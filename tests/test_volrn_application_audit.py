import numpy as np


def test_application_audit_reports_per_scene_changes_and_coefficients():
    from src.multiscene_sift.radiometric_runner import build_volrn_application_audit

    before = [np.array([[[1.0, 2.0], [3.0, np.nan]]]), np.array([[[4.0, 5.0], [6.0, np.nan]]])]
    after = [np.array([[[1.0, 2.5], [3.0, np.nan]]]), np.array([[[4.0, 4.0], [6.0, np.nan]]])]
    valid = [np.array([[True, True], [True, False]]), np.array([[True, True], [True, False]])]
    coeffs = np.array([[[1.0, 0.0], [1.2, -0.5]]])

    result = build_volrn_application_audit(before, after, valid, coeffs)

    assert len(result["scenes"]) == 2
    assert result["scenes"][0]["mean_abs_difference"] == 1.0 / 6.0
    assert result["scenes"][0]["max_difference"] == 0.5
    assert result["scenes"][0]["changed_fraction"] == 1.0 / 3.0
    assert result["coefficient_range"]["a_min"] == 1.0
    assert result["coefficient_range"]["a_max"] == 1.2
    assert result["coefficient_range"]["b_min"] == -0.5
