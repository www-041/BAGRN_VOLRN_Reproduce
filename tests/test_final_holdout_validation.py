import numpy as np
from rasterio.transform import from_origin


def _different_shape_overlap_pair():
    """Build two full rasters whose overlap is not their top-left crop."""
    ref_h, ref_w = 100, 120
    tgt_h, tgt_w = 110, 90
    ref_transform = from_origin(0, 100, 1, 1)
    tgt_transform = from_origin(30, 110, 1, 1)

    ref_rows, ref_cols = np.indices((ref_h, ref_w), dtype=float)
    tgt_rows, tgt_cols = np.indices((tgt_h, tgt_w), dtype=float)

    def field(rows, cols):
        return (
            np.sin((cols + 0.5) * 0.17)
            + np.cos((rows + 0.5) * 0.23)
            + 0.002 * rows * cols
        )

    ref = field(ref_rows, ref_cols)
    # The target starts 30 columns right and 10 rows north in geographic space.
    # Its local row 10 / col 0 is therefore reference row 0 / col 30.
    tgt = field(tgt_rows - 10, tgt_cols + 30)
    return ref, ref_transform, tgt, tgt_transform


def test_final_validation_handles_different_full_scene_shapes(monkeypatch):
    from src import coregistration, multiband_pipeline

    ref, ref_transform, tgt, tgt_transform = _different_shape_overlap_pair()
    monkeypatch.setattr(
        coregistration,
        "phase_correlation",
        lambda *args, **kwargs: (1.0, -2.0, 0.9),
    )

    holdout_full = np.zeros_like(ref, dtype=bool)
    holdout_full[:64, 30:94] = True
    quality, validation = multiband_pipeline._validate_final_registration_arrays(
        [ref[None, ...], tgt[None, ...]], 0,
        [ref_transform, tgt_transform], [None, None], [(0, 1)], [],
        {
            "enable_spatial_holdout": True,
            "validation_block_size": 16,
            "validation_step": 16,
            "validation_offset_row": 0,
            "validation_offset_col": 0,
            "validation_block_size_candidates": [16],
            "validation_required_candidate_count": 1,
            "validation_confidence_threshold": 0.5,
            "validation_max_residual_shift": 3.0,
            "final_min_blocks": 1,
        },
        holdout_contexts={
            (0, 1): {
                "available": True,
                "holdout_region_full_mask": holdout_full,
            },
        },
    )

    accepted = [block for block in validation["edges"][0]["blocks"] if block["accepted"]]
    assert accepted
    assert quality["rmse"] == np.hypot(1.0, -2.0)


def test_overlap_alignment_uses_geospatial_windows_not_top_left_crop():
    from src.coregistration import build_pair_overlap_context

    ref, ref_transform, tgt, tgt_transform = _different_shape_overlap_pair()
    context = build_pair_overlap_context(
        ref, ref_transform, tgt, tgt_transform, None, None,
    )

    assert context["ref_overlap"].shape == context["tgt_overlap"].shape
    assert context["common_valid_mask"].shape == context["ref_overlap"].shape
    assert context["shape"] == context["common_valid_mask"].shape
    assert context["ref_window"] == (0, 100, 30, 120)
    assert context["tgt_window"] == (10, 110, 0, 90)
    np.testing.assert_allclose(context["ref_overlap"], context["tgt_overlap"])
    assert not np.allclose(ref[:, :90], tgt[:100, :90])


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


def test_final_validation_reuses_exact_reserved_windows(monkeypatch):
    from src import coregistration

    monkeypatch.setattr(
        coregistration,
        "phase_correlation",
        lambda *args, **kwargs: (0.5, -0.25, 0.9),
    )
    rng = np.random.RandomState(7)
    arr = rng.normal(size=(96, 96)).astype(float)
    reserved = [(5, 7, 16, 16), (51, 43, 16, 16)]

    result = coregistration.validate_registration_independent_grid(
        arr, from_origin(0, 96, 1, 1), arr.copy(), from_origin(0, 96, 1, 1),
        None, None, [], block_size=32, step=32, offset_row=0, offset_col=0,
        confidence_threshold=0.5, max_residual_shift=3, min_accepted=2,
        reserved_validation_windows=reserved,
    )

    accepted = [block for block in result["blocks"] if block["accepted"]]
    assert {(block["validation_row"], block["validation_col"]) for block in accepted} == {
        (5, 7), (51, 43),
    }


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
