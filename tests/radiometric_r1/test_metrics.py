import numpy as np
import pytest

from src.benchmark_v1_radiometric_r1.metrics import (
    aggregate_by_roi_pair_band_time,
    change_preservation_metrics,
    multiband_metrics,
    pair_metrics,
    structure_metrics,
)


def test_pair_metrics_exact_offset_formulas_and_distribution_errors():
    reference = np.array([[1.0, 2.0, 3.0]])
    corrected = reference + 1.0
    result = pair_metrics(reference, corrected, np.ones(reference.shape, bool))
    assert result["status"] == "EVALUABLE"
    assert result["shared_pixels"] == 3
    assert result["MAMD"] == pytest.approx(1.0)
    assert result["MSDD"] == pytest.approx(0.0)
    assert result["RDD"] == pytest.approx(1.0)
    assert result["MAE"] == pytest.approx(1.0)
    assert result["RMSE"] == pytest.approx(1.0)
    assert result["P95"] == pytest.approx(1.0)
    assert result["MAX_ABS_ERROR"] == pytest.approx(1.0)


def test_pair_metrics_gain_and_excludes_nodata_nan_and_infinity():
    reference = np.array([[1.0, 2.0, np.nan], [4.0, np.inf, 8.0]])
    corrected = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 16.0]])
    mask = np.array([[True, True, True], [True, True, False]])
    result = pair_metrics(reference, corrected, mask)
    assert result["shared_pixels"] == 3
    assert result["MAE"] == pytest.approx(7 / 3)
    assert result["MSDD"] == pytest.approx(abs(np.std([1, 2, 4]) - np.std([2, 4, 8])))


def test_empty_support_is_explicit_not_zero():
    image = np.ones((2, 3))
    result = pair_metrics(image, image, np.zeros(image.shape, bool))
    assert result["status"] == "NO_VALID_SUPPORT"
    assert result["MAE"] is None and result["RDD"] is None


def test_change_preservation_requires_independently_approved_truth():
    truth = np.array([[0.0, 2.0], [0.0, 3.0]])
    predicted = np.array([[0.0, 1.0], [0.0, 4.0]])
    mask = np.array([[False, True], [False, True]])
    pending = change_preservation_metrics(truth, predicted, mask, truth_status="UNVERIFIED")
    assert pending["status"] == "CONDITIONAL_NOT_EVALUABLE"
    approved = change_preservation_metrics(truth, predicted, mask, truth_status="APPROVED", output_change_mask=mask)
    assert approved["status"] == "EVALUABLE"
    assert approved["change_magnitude_MAE"] == pytest.approx(1.0)
    assert approved["change_F1"] == pytest.approx(1.0)
    assert approved["change_IoU"] == pytest.approx(1.0)


def test_multiband_outputs_per_band_and_scale_normalized_macro_not_raw_pool():
    reference = np.stack([np.ones((4, 4)), np.ones((4, 4)) * 100])
    corrected = reference.copy(); corrected[0] += 1; corrected[1] += 10
    result = multiband_metrics(reference, corrected, np.ones((4, 4), bool), {"name": "frozen_demo_ranges", "scales": [2.0, 20.0]})
    assert [row["MAE"] for row in result["per_band"]] == [1.0, 10.0]
    assert result["normalized_MAE_macro_mean"] == pytest.approx(0.5)
    assert result["raw_MAE_pooled"] is None
    assert result["SAM"]["status"] == "CONDITIONAL_NOT_EVALUABLE"


def test_structure_metrics_require_confirmed_common_grid_and_numeric_scale():
    image = np.tile(np.arange(16, dtype=float), (16, 1))
    pending = structure_metrics(image, image, np.ones(image.shape, bool), co_registered=False, scale_range=None)
    assert pending["status"] == "CONDITIONAL_NOT_EVALUABLE"
    evaluated = structure_metrics(image, image, np.ones(image.shape, bool), co_registered=True, scale_range=15.0)
    assert evaluated["status"] == "EVALUABLE"
    assert evaluated["gradient_NCC"] == pytest.approx(1.0)
    assert evaluated["SSIM"] == pytest.approx(1.0)


def test_aggregation_is_macro_by_pair_and_keeps_failures_and_distributions():
    rows = [
        {"pair_id": "a", "roi_id": "r1", "band": "B9", "time_bin": "T1", "status": "EVALUABLE", "MAE": 1.0, "valid_pixels": 1000},
        {"pair_id": "b", "roi_id": "r2", "band": "B9", "time_bin": "T3", "status": "EVALUABLE", "MAE": 3.0, "valid_pixels": 1},
        {"pair_id": "c", "roi_id": "r3", "band": "B9", "time_bin": "T3", "status": "NO_VALID_SUPPORT", "MAE": None, "valid_pixels": 0},
    ]
    result = aggregate_by_roi_pair_band_time(rows)
    assert result["pair_macro_mean_MAE"] == pytest.approx(2.0)
    assert result["evaluable_pairs"] == 2 and result["nonevaluable_pairs"] == 1
    assert result["pair_MAE_distribution"] == [1.0, 3.0]
