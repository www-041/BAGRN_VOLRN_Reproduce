import numpy as np

from src.coregistration import build_robust_pair_measurement


def test_robust_pair_measurement_rejects_joint_xy_outliers():
    matches = [{"shift_dx": 3.0 + dx, "shift_dy": -2.0 + dy, "confidence": 0.9}
               for dx, dy in [(0.0, 0.0), (0.1, -0.1), (-0.1, 0.1),
                              (0.0, 0.1), (0.1, 0.0), (18.0, 15.0), (-20.0, 12.0)]]
    result = build_robust_pair_measurement(
        matches, {"global_confidence_threshold": 0.5, "robust_mad_scale": 3.0,
                  "robust_residual_floor": 0.75, "robust_min_inliers": 5,
                  "robust_min_inlier_ratio": 0.35})
    assert result["status"] == "pass"
    assert result["n_blocks_total"] == 7
    assert result["n_blocks_inlier"] == 5
    assert np.isclose(result["shift_dx"], 3.0, atol=0.1)
    assert np.isclose(result["shift_dy"], -2.0, atol=0.1)
    assert result["rmse"] < 0.2


def test_n2_pair_uses_block_samples_not_one_pair_sample():
    matches = [{"shift_dx": 4.0 + dx, "shift_dy": 1.5 + dy, "confidence": 0.8}
               for dx, dy in [(0.0, 0.0), (0.1, 0.0), (-0.1, 0.0),
                              (0.0, 0.1), (0.0, -0.1)]]
    result = build_robust_pair_measurement(
        matches, {"global_confidence_threshold": 0.5, "robust_min_inliers": 5,
                  "robust_min_inlier_ratio": 0.35})
    assert result["n_blocks_total"] == 5
    assert result["n_blocks_inlier"] == 5
    assert result["shift_dx"] != matches[0]["shift_dx"] or result["shift_dy"] != matches[0]["shift_dy"]
