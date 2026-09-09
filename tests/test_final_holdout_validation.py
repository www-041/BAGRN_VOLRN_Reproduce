import numpy as np
from rasterio.transform import from_origin


def test_final_validation_reads_only_reserved_holdout(monkeypatch):
    from src import coregistration

    monkeypatch.setattr(
        coregistration, "phase_correlation",
        lambda *args, **kwargs: (0.0, 0.0, 0.9),
    )
    arr = np.arange(64 * 64, dtype=float).reshape(64, 64)
    holdout = np.zeros_like(arr, dtype=bool)
    holdout[:32, :32] = True

    result = coregistration.validate_registration_independent_grid(
        arr, from_origin(0, 64, 1, 1), arr, from_origin(0, 64, 1, 1),
        None, None, np.array([[8.0, 8.0]]), block_size=16, step=16,
        offset_row=0, offset_col=0, min_distance_from_training=256,
        confidence_threshold=0.5, max_residual_shift=3, min_accepted=1,
        reserved_holdout_mask=holdout,
        validation_block_size_candidates=[16],
        required_candidate_count=1,
    )

    accepted = [block for block in result["blocks"] if block["accepted"]]
    assert accepted
    assert all(block["validation_row"] < 32 for block in accepted)
    assert all(block["validation_col"] < 32 for block in accepted)
    assert result["stats"]["n_rejected_near_training"] == 0


def test_bad_holdout_rmse_or_p95_remains_fail():
    from src.coregistration import aggregate_final_validation_quality

    result = aggregate_final_validation_quality(
        [{
            "idx_i": 0, "idx_j": 1, "failure_reason": None,
            "blocks": [{
                "accepted": True, "residual_magnitude": 2.0,
                "confidence": 0.9,
            }],
            "stats": {"n_accepted": 1, "rmse": 2.0, "p95": 2.0,
                      "median": 2.0, "mean_confidence": 0.9},
        }],
        {"final_min_blocks": 1, "pass_min_mean_confidence": 0.5,
         "pass_max_median": 0.35, "pass_max_rmse": 0.6,
         "pass_max_p95": 1.0},
        required_edges=[(0, 1)],
    )

    assert result["quality"] == "fail"
