"""Truth isolation and status-preserving Phase5 DEVELOPMENT scorer tests."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_phase2_core import pair_metrics
from benchmark_v1_phase5.evaluation import (
    load_approved_development_truth, score_development, score_georef_only,
    summarize_development,
)
from benchmark_v1_phase5.inputs import MatchView, PairInput, ViewProvenance
from affine import Affine


def _points(pair="00_01", split="DEVELOPMENT"):
    return [
        {"point_id": f"{pair}_d1", "pair_id": pair, "split": split,
         "human_review_status": "APPROVED",
         "row_i": 10.0, "col_i": 20.0, "row_j": 11.0, "col_j": 22.0},
        {"point_id": f"{pair}_d2", "pair_id": pair, "split": split,
         "human_review_status": "APPROVED",
         "row_i": 40.0, "col_i": 50.0, "row_j": 39.0, "col_j": 49.0},
    ]


def _result(status="SUCCESS", delta=(1.0, 2.0), pair="00_01", method="SIFT"):
    return {"method": method, "pair_id": pair, "status": status,
            "delta_row_col": delta, "direction": "scene_i_to_scene_j",
            "frame": "native", "translation_semantics": "full_native_i_to_j",
            "elapsed_s": 1.0, "reason": None}


def _pair():
    from benchmark_v1_phase5.inputs import PairInput
    i = Affine(14, 0, 100, 0, -14, 300)
    j = Affine(14, 0, 128, 0, -14, 286)
    view = MatchView(np.ones((8, 8), np.float32), np.ones((8, 8), np.float32),
                     np.ones((8, 8), bool), np.ones((8, 8), bool), np.ones((8, 8), bool),
                     ViewProvenance(i, j, i, 1, 1, (2, 98), (0, 1), (0, 1),
                                    "bilinear", "nearest"))
    return PairInput("00_01", "READY", view, i, j, 64, 64)


def test_development_metrics_match_phase4_pair_metrics_exactly():
    result = score_development(_result(), _points())
    expected_errors = [math.hypot(0, 0), math.hypot(2, 3)]
    assert result["execution_status"] == "SUCCESS"
    assert result["evaluation_status"] == "SCORED"
    assert result["metrics"] == pair_metrics(expected_errors)
    assert result["independent_point_count"] == 2


@pytest.mark.parametrize("split", ["TEST", "UNASSIGNED", "APPROVED_UNASSIGNED"])
def test_non_development_truth_is_rejected(split):
    with pytest.raises(ValueError, match="DEVELOPMENT"):
        score_development(_result(), _points(split=split))


@pytest.mark.parametrize("review_status", [None, "UNVERIFIED", "REJECTED"])
def test_truth_without_explicit_approved_first_review_is_rejected(review_status):
    point = _points()[0].copy()
    point["human_review_status"] = review_status
    with pytest.raises(ValueError, match="APPROVED"):
        score_development(_result(), [point])


@pytest.mark.parametrize("status", ["ENV_UNAVAILABLE", "WEIGHTS_UNAVAILABLE",
                                      "NO_VALID_OVERLAP", "UNSUPPORTED_GRID",
                                      "ALGORITHM_FAIL", "NOT_ATTEMPTED"])
def test_non_success_status_is_preserved_with_null_metrics(status):
    scored = score_development(_result(status=status, delta=None), _points())
    assert scored["execution_status"] == status
    assert scored["evaluation_status"] == "NOT_SCORED"
    assert scored["metrics"] is None


def test_success_without_truth_is_not_relabelled_as_algorithm_failure():
    scored = score_development(_result(), [])
    assert scored["execution_status"] == "SUCCESS"
    assert scored["evaluation_status"] == "NO_INDEPENDENT_TRUTH"
    assert scored["metrics"] is None


def test_success_requires_native_full_translation_and_pair_consistency():
    with pytest.raises(ValueError, match="native"):
        score_development({**_result(), "frame": "view"}, _points())
    with pytest.raises(ValueError, match="direction"):
        score_development({**_result(), "direction": "scene_j_to_scene_i"}, _points())
    with pytest.raises(ValueError, match="finite"):
        score_development({**_result(), "delta_row_col": (float("nan"), 0)}, _points())
    with pytest.raises(ValueError, match="pair"):
        score_development(_result(pair="02_03"), _points())


def test_macro_summary_keeps_full_distribution_and_missing_attempts():
    scored = [score_development(_result(), _points())]
    summary = summarize_development(scored, expected_pairs=("00_01", "02_03"))
    method = summary["methods"]["SIFT"]
    assert method["attempted_pairs"] == 1
    assert method["execution_status_counts"] == {"NOT_ATTEMPTED": 1, "SUCCESS": 1}
    assert method["scored_pair_count"] == 1
    assert method["macro_rmse_success_only"] == pytest.approx(scored[0]["metrics"]["rmse_px"])
    assert method["pair_rmse_distribution"] == [scored[0]["metrics"]["rmse_px"]]
    empty = summarize_development([], expected_pairs=("00_01",))
    assert set(empty["methods"]) == {"SIFT", "KARIOS", "AROSICS_GLOBAL", "LoFTR",
                                      "EfficientLoFTR", "LightGlue+DISK"}
    assert all(row["execution_status_counts"] == {"NOT_ATTEMPTED": 1}
               for row in empty["methods"].values())
    assert set(empty["diagnostic_baselines"]) == {"GEOREF_ONLY"}
    georef_summary = empty["diagnostic_baselines"]["GEOREF_ONLY"]
    assert georef_summary["diagnostic_only"] is True
    assert georef_summary["execution_status_counts"] == {"NOT_ATTEMPTED": 1}
    assert "macro_rmse_success_only" not in georef_summary


def test_summary_rejects_metrics_attached_to_failed_execution():
    invalid = {"method": "SIFT", "pair_id": "00_01", "execution_status": "ALGORITHM_FAIL",
               "evaluation_status": "SCORED", "metrics": pair_metrics([0.0])}
    with pytest.raises(ValueError, match="Failed execution"):
        summarize_development([invalid], expected_pairs=("00_01",))


def test_georef_only_uses_same_independent_scorer_and_is_marked_diagnostic():
    # GEOREF_ONLY is A_j^-1(t_i-t_j): here (-1 row, -2 cols).
    scored = score_georef_only(_pair(), _points())
    assert scored["method"] == "GEOREF_ONLY"
    assert scored["diagnostic_only"] is True
    assert scored["execution_status"] == "SUCCESS"
    assert scored["metrics"] == pair_metrics([math.sqrt(20), 1.0])


def test_approved_truth_adapter_connects_phase4_loader_and_scored_georef_summary():
    from benchmark_phase4_freeze import load_frozen_trial
    snapshot = load_frozen_trial(ROOT)
    truth = load_approved_development_truth(snapshot)
    assert {pair: len(points) for pair, points in truth.items()} == {
        "00_01": 11, "02_03": 9, "05_07": 9, "08_10": 14}
    assert all(point["human_review_status"] == "APPROVED"
               for points in truth.values() for point in points)
    result = score_georef_only(_pair(), truth["00_01"])
    summary = summarize_development([result])
    georef = summary["diagnostic_baselines"]["GEOREF_ONLY"]
    assert georef["scored_pair_count"] == 1
    assert len(georef["pair_rmse_distribution"]) == 1
    assert "macro_rmse_success_only" not in georef
