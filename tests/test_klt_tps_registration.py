"""Unit tests for the opt-in KLT/TPS registration backend."""

import cv2
import numpy as np
import pytest
from rasterio.transform import Affine


def test_opencv_dependency_is_available():
    assert cv2.__version__


def test_normalize_klt_image_returns_uint8_and_masks_invalid():
    from src.klt_tps_registration import normalize_klt_image

    y, x = np.mgrid[0:96, 0:96]
    image = (x + 2 * y).astype(np.float32)
    valid = np.ones_like(image, dtype=bool)
    valid[:8, :] = False
    result = normalize_klt_image(image, valid)
    assert result.dtype == np.uint8
    assert result.shape == image.shape
    assert np.isfinite(result).all()
    assert result.min() >= 0
    assert result.max() <= 255


def test_build_klt_interior_mask_erodes_boundary():
    from src.klt_tps_registration import build_klt_interior_mask

    valid = np.ones((25, 25), dtype=bool)
    mask = build_klt_interior_mask(valid, 3)
    assert mask.dtype == np.uint8
    assert mask[0, 0] == 0
    assert mask[12, 12] == 1


def _textured_scene(shape=(192, 192)):
    image = np.zeros(shape, dtype=np.float32)
    for cy in range(18, shape[0] - 10, 28):
        for cx in range(18, shape[1] - 10, 28):
            cv2.circle(image, (cx, cy), 7, 80 + ((cx + cy) % 100), -1)
            cv2.rectangle(image, (cx - 11, cy + 10), (cx + 5, cy + 17), 35, -1)
    cv2.line(image, (8, 80), (shape[1] - 8, shape[0] - 24), 210, 2)
    cv2.line(image, (12, shape[0] - 20), (shape[1] - 20, 12), 150, 2)
    return cv2.GaussianBlur(image, (3, 3), 0)


def _klt_params(**overrides):
    params = {
        "klt_tps_window": 9,
        "klt_tps_pyramid_level": 3,
        "klt_tps_max_corners": 4000,
        "klt_tps_min_corner_distance": 5.0,
        "klt_tps_fb_threshold": 0.5,
        "klt_tps_min_points": 30,
        "klt_tps_threads": 1,
    }
    params.update(overrides)
    return params


def test_bidirectional_klt_recovers_synthetic_subpixel_translation():
    from src.klt_tps_registration import match_bidirectional_klt

    reference = _textured_scene()
    expected = np.array([2.25, -1.50])
    moving = cv2.warpAffine(
        reference, np.array([[1, 0, expected[0]], [0, 1, expected[1]]], dtype=np.float32),
        (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    valid = np.ones(reference.shape, dtype=bool)
    result = match_bidirectional_klt(reference, moving, valid, valid, _klt_params())
    assert result["initial_corner_count"] >= 30
    assert result["accepted_point_count"] >= 30
    median = np.median(result["displacement_xy"], axis=0)
    assert np.linalg.norm(median - expected) < 0.35
    assert np.percentile(result["forward_backward_error"], 95) < 0.5


def test_klt_training_mask_excludes_reserved_holdout_corners():
    from src.klt_tps_registration import match_bidirectional_klt

    reference = _textured_scene()
    expected = np.array([2.25, -1.50])
    moving = cv2.warpAffine(
        reference, np.array([[1, 0, expected[0]], [0, 1, expected[1]]], dtype=np.float32),
        (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    valid = np.ones(reference.shape, dtype=bool)
    training_mask = np.ones(reference.shape, dtype=bool)
    training_mask[:, 112:] = False
    result = match_bidirectional_klt(
        reference, moving, valid, valid, _klt_params(), training_mask=training_mask,
    )
    for points in (result["reference_points_xy"], result["moving_points_xy"]):
        rows = np.rint(points[:, 1]).astype(int)
        cols = np.rint(points[:, 0]).astype(int)
        assert np.all(training_mask[rows, cols])


def test_klt_training_mask_rejects_wrong_shape():
    from src.klt_tps_registration import match_bidirectional_klt

    image = _textured_scene()
    valid = np.ones(image.shape, dtype=bool)
    with pytest.raises(ValueError, match="KLT training mask must match overlap image shape"):
        match_bidirectional_klt(image, image, valid, valid, _klt_params(), training_mask=np.ones((96, 96), bool))


def test_klt_training_mask_applies_to_reference_and_moving_support():
    from src.klt_tps_registration import match_bidirectional_klt

    reference = _textured_scene()
    moving = cv2.warpAffine(
        reference, np.array([[1, 0, 1.5], [0, 1, -0.75]], dtype=np.float32),
        (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    valid = np.ones(reference.shape, dtype=bool)
    training_mask = np.ones(reference.shape, dtype=bool)
    training_mask[:28, :] = False
    result = match_bidirectional_klt(
        reference, moving, valid, valid, _klt_params(), training_mask=training_mask,
    )
    assert np.all(np.rint(result["reference_points_xy"][:, 1]).astype(int) >= 28)
    assert np.all(np.rint(result["moving_points_xy"][:, 1]).astype(int) >= 28)


def test_estimate_klt_tps_pair_passes_training_mask_to_matcher(monkeypatch):
    from src import klt_tps_registration as module

    image = _textured_scene((64, 64))
    valid = np.ones(image.shape, dtype=bool)
    training_mask = np.ones(image.shape, dtype=bool)
    captured = {}
    p = np.asarray([[x, y] for y in (10, 20, 30, 40, 50) for x in (10, 15, 20, 25, 30, 35, 40, 50)], dtype=float)
    def fake_match(*args, **kwargs):
        captured["training_mask"] = kwargs.get("training_mask")
        return {
            "initial_corner_count": 40, "accepted_point_count": 40,
                "reference_points_xy": p,
                "moving_points_xy": p + [1, 0],
            "displacement_xy": np.tile([1, 0], (40, 1)),
            "forward_backward_error": np.zeros(40),
        }
    monkeypatch.setattr(module, "match_bidirectional_klt", fake_match)
    result = module.estimate_klt_tps_pair(
        image, Affine.identity(), image, Affine.identity(), None, None,
        {**_klt_params(), "klt_tps_smoothing": 0.0}, training_mask=training_mask,
    )
    assert result["available"] is True
    np.testing.assert_array_equal(captured["training_mask"], training_mask)


def test_bidirectional_klt_rejects_too_few_corners():
    from src.klt_tps_registration import match_bidirectional_klt

    image = np.zeros((96, 96), dtype=np.float32)
    valid = np.ones(image.shape, dtype=bool)
    with pytest.raises(ValueError, match="insufficient"):
        match_bidirectional_klt(image, image, valid, valid, _klt_params())


def test_bidirectional_klt_rejects_collinear_controls(monkeypatch):
    from src import klt_tps_registration as module

    points = np.column_stack([np.arange(40, dtype=np.float32) + 20, np.full(40, 40, dtype=np.float32)])
    monkeypatch.setattr(module.cv2, "goodFeaturesToTrack", lambda *args, **kwargs: points.reshape(-1, 1, 2))
    monkeypatch.setattr(module.cv2, "calcOpticalFlowPyrLK", lambda *args, **kwargs: (points.reshape(-1, 1, 2), np.ones((40, 1), dtype=np.uint8), None))
    image = _textured_scene((96, 96))
    valid = np.ones(image.shape, dtype=bool)
    with pytest.raises(ValueError, match="collinear"):
        module.match_bidirectional_klt(image, image, valid, valid, _klt_params())


def test_map_klt_matches_identity_grid_preserves_displacement():
    from src.klt_tps_registration import map_klt_matches_to_moving_grid

    p = np.array([[10.0, 20.0], [30.0, 40.0]])
    q = p + np.array([2.5, -1.25])
    mapped = map_klt_matches_to_moving_grid(
        p, q, Affine(14, 0, 1000, 0, -14, 2000),
        Affine(14, 0, 1000, 0, -14, 2000),
    )
    np.testing.assert_allclose(mapped["control_points_xy"], p, atol=1e-9)
    np.testing.assert_allclose(mapped["source_points_xy"], q, atol=1e-9)
    np.testing.assert_allclose(mapped["displacement_xy"], q - p, atol=1e-9)


def test_map_klt_matches_translated_overlap_to_moving_native_grid():
    from src.klt_tps_registration import map_klt_matches_to_moving_grid

    overlap_transform = Affine(14, 0, 696416, 0, -14, 3375974)
    moving_transform = Affine(14, 0, 659722, 0, -14, 3371634)
    p = np.array([[20.0, 40.0], [300.0, 500.0]])
    q = p + np.array([2.5, -1.25])
    mapped = map_klt_matches_to_moving_grid(
        p, q, overlap_transform, moving_transform,
    )
    assert not np.allclose(mapped["control_points_xy"], p)
    np.testing.assert_allclose(mapped["displacement_xy"], q - p, atol=1e-9)


def test_estimate_pair_uses_existing_geographic_overlap_context(monkeypatch):
    from src import klt_tps_registration as module

    ref = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    moving = ref.copy()
    transform = Affine(2, 0, 100, 0, -2, 200)
    overlap_transform = Affine(2, 0, 500, 0, -2, 800)
    valid = np.ones((32, 32), dtype=bool)
    monkeypatch.setattr(module, "build_pair_overlap_context", lambda *args: {
        "available": True,
        "ref_window": (10, 42, 12, 44),
        "tgt_window": (5, 37, 7, 39),
        "shape": (32, 32),
        "overlap_transform": overlap_transform,
        "ref_overlap": np.ones((32, 32), dtype=np.float32),
        "tgt_overlap": np.ones((32, 32), dtype=np.float32),
        "ref_valid": valid,
        "tgt_valid": valid,
        "resampled_target": True,
    })
    p = np.array([[4.0, 5.0], [20.0, 25.0], [25.0, 8.0]])
    q = p + np.array([1.5, -0.75])
    monkeypatch.setattr(module, "match_bidirectional_klt", lambda *args, **kwargs: {
        "initial_corner_count": 3,
        "accepted_point_count": 3,
        "reference_points_xy": p,
        "moving_points_xy": q,
        "displacement_xy": q - p,
        "forward_backward_error": np.zeros(3),
    })
    result = module.estimate_klt_tps_pair(
        ref, transform, moving, transform, None, None, _klt_params()
    )
    expected = module.map_klt_matches_to_moving_grid(
        p, q, overlap_transform, transform,
    )
    np.testing.assert_allclose(result["control_points_moving_xy"], expected["control_points_xy"])
    np.testing.assert_allclose(result["source_points_moving_xy"], expected["source_points_xy"])
    assert result["overlap_context"]["resampled_target"] is True


def _geometry_rejection_case(monkeypatch):
    from src import klt_tps_registration as module

    image = _textured_scene((64, 64))
    points = np.asarray(
        [[12.0, 12.0], [50.0, 12.0], [12.0, 50.0], [50.0, 50.0], [32.0, 32.0]],
    )
    monkeypatch.setattr(module, "match_bidirectional_klt", lambda *args, **kwargs: {
        "initial_corner_count": 40,
        "accepted_point_count": len(points),
        "reference_points_xy": points,
        "moving_points_xy": points + [1.0, -0.5],
        "displacement_xy": np.tile([1.0, -0.5], (len(points), 1)),
        "forward_backward_error": np.zeros(len(points)),
    })
    return module, image


def _assert_structured_geometry_rejection(result, reason):
    assert result["available"] is False
    assert result["flow"] is None
    assert result["geometry"] is None
    assert reason in result["failure_reason"]
    assert result["overlap_context"] is not None
    assert result["accepted_point_count"] == 5
    assert result["reference_points_overlap_xy"].shape == (5, 2)


def test_estimate_klt_tps_pair_returns_unavailable_when_tps_fit_fails(monkeypatch):
    module, image = _geometry_rejection_case(monkeypatch)
    monkeypatch.setattr(
        module, "build_tps_dense_flow",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("TPS fit failed")),
    )
    result = module.estimate_klt_tps_pair(
        image, Affine.identity(), image, Affine.identity(), None, None, _klt_params(),
    )
    _assert_structured_geometry_rejection(result, "TPS fit failed")


def test_estimate_klt_tps_pair_returns_unavailable_when_flow_folds(monkeypatch):
    module, image = _geometry_rejection_case(monkeypatch)
    flow = np.zeros((64, 64, 2), dtype=np.float32)
    flow[..., 0] = -2 * np.indices(flow.shape[:2])[1]
    monkeypatch.setattr(module, "build_tps_dense_flow", lambda *args, **kwargs: flow)
    result = module.estimate_klt_tps_pair(
        image, Affine.identity(), image, Affine.identity(), None, None, _klt_params(),
    )
    _assert_structured_geometry_rejection(result, "fold")


def test_estimate_klt_tps_pair_returns_unavailable_when_max_shift_exceeded(monkeypatch):
    module, image = _geometry_rejection_case(monkeypatch)
    flow = np.zeros((64, 64, 2), dtype=np.float32)
    flow[..., 0] = 51
    monkeypatch.setattr(module, "build_tps_dense_flow", lambda *args, **kwargs: flow)
    result = module.estimate_klt_tps_pair(
        image, Affine.identity(), image, Affine.identity(), None, None,
        _klt_params(klt_tps_max_shift=50.0),
    )
    _assert_structured_geometry_rejection(result, "exceeds")


def test_estimate_klt_tps_pair_preserves_failure_reason(monkeypatch):
    module, image = _geometry_rejection_case(monkeypatch)
    monkeypatch.setattr(
        module, "build_tps_dense_flow",
        lambda *args, **kwargs: (_ for _ in ()).throw(np.linalg.LinAlgError("singular TPS")),
    )
    result = module.estimate_klt_tps_pair(
        image, Affine.identity(), image, Affine.identity(), None, None, _klt_params(),
    )
    _assert_structured_geometry_rejection(result, "singular TPS")


def test_tps_dense_flow_recovers_constant_control_displacement():
    from src.klt_tps_registration import build_tps_dense_flow

    points = np.array([[10, 10], [118, 10], [10, 86], [118, 86], [64, 48]], dtype=float)
    displacement = np.tile([1.75, -0.50], (len(points), 1))
    flow = build_tps_dense_flow(
        points, displacement, (96, 128),
        {"klt_tps_smoothing": 0.0, "klt_tps_neighbors": 80, "klt_tps_field_step": 4},
    )
    assert flow.shape == (96, 128, 2)
    assert flow.dtype == np.float32
    np.testing.assert_allclose(np.median(flow.reshape(-1, 2), axis=0), displacement[0], atol=0.05)
    np.testing.assert_allclose(flow[48, 64], displacement[0], atol=0.05)


def test_inspect_tps_dense_flow_accepts_safe_translation():
    from src.klt_tps_registration import inspect_tps_dense_flow

    flow = np.zeros((32, 40, 2), dtype=np.float32)
    flow[..., 0] = 1.25
    flow[..., 1] = -0.5
    result = inspect_tps_dense_flow(flow, 50.0)
    assert result["fold_pixels"] == 0
    assert result["max_displacement_pixels"] == pytest.approx(1.3462912, abs=1e-5)


def test_inspect_tps_dense_flow_rejects_folding():
    from src.klt_tps_registration import inspect_tps_dense_flow

    flow = np.zeros((32, 40, 2), dtype=np.float32)
    flow[..., 0] = -2 * np.indices(flow.shape[:2])[1]
    with pytest.raises(ValueError, match="fold"):
        inspect_tps_dense_flow(flow, 50.0)


def test_inspect_tps_dense_flow_rejects_excessive_shift():
    from src.klt_tps_registration import inspect_tps_dense_flow

    flow = np.zeros((32, 40, 2), dtype=np.float32)
    flow[..., 0] = 51
    with pytest.raises(ValueError, match="shift"):
        inspect_tps_dense_flow(flow, 50.0)


def test_inspect_tps_dense_flow_rejects_nonfinite_values():
    from src.klt_tps_registration import inspect_tps_dense_flow

    flow = np.zeros((32, 40, 2), dtype=np.float32)
    flow[3, 4, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        inspect_tps_dense_flow(flow, 50.0)


def test_warp_multiband_tps_flow_uses_output_to_source_sign():
    from src.klt_tps_registration import warp_multiband_with_tps_flow

    source = np.zeros((1, 24, 32), dtype=np.float32)
    source[0, 12, 16] = 100.0
    flow = np.zeros((24, 32, 2), dtype=np.float32)
    flow[..., 0] = 2.0
    warped, valid = warp_multiband_with_tps_flow(source, flow, None)
    assert valid[12, 14]
    assert warped[0, 12, 14] > 50
    assert warped[0, 12, 16] == pytest.approx(0, abs=1e-5)


def test_warp_multiband_tps_flow_applies_identical_geometry_to_every_band():
    from src.klt_tps_registration import warp_multiband_with_tps_flow

    y, x = np.mgrid[0:32, 0:40]
    band = (x + 2 * y).astype(np.float32)
    source = np.stack([band, band * 10], axis=0)
    flow = np.zeros((32, 40, 2), dtype=np.float32)
    flow[..., 0] = 0.75
    flow[..., 1] = -0.5
    warped, valid = warp_multiband_with_tps_flow(source, flow, None)
    assert valid.any()
    np.testing.assert_allclose(warped[1, valid], warped[0, valid] * 10, atol=1e-3)


def test_warp_multiband_tps_flow_rejects_cubic_neighbourhood_touching_nodata():
    from src.klt_tps_registration import warp_multiband_with_tps_flow

    source = np.ones((1, 32, 32), dtype=np.float32)
    source[0, 15, 15] = -999.0
    flow = np.zeros((32, 32, 2), dtype=np.float32)
    warped, valid = warp_multiband_with_tps_flow(source, flow, -999.0)
    assert not valid[15, 15]
    assert warped[0, 15, 15] == -999.0


def test_analyze_tps_dense_flow_reports_safe_translation():
    from src.klt_tps_registration import analyze_tps_dense_flow

    flow = np.zeros((32, 40, 2), dtype=np.float32)
    flow[..., 0] = 1.25
    flow[..., 1] = -0.50
    analysis = analyze_tps_dense_flow(flow)

    summary = analysis["summary"]
    assert summary["fold_pixels"] == 0
    assert summary["jacobian_min"] == pytest.approx(1.0)
    assert summary["jacobian_max"] == pytest.approx(1.0)
    assert summary["displacement_median"] == pytest.approx(1.3462912, abs=1e-5)
    assert analysis["fold_mask"].shape == flow.shape[:2]
    assert analysis["jacobian_determinant"].shape == flow.shape[:2]


def test_analyze_tps_dense_flow_reports_fold_mask():
    from src.klt_tps_registration import analyze_tps_dense_flow

    flow = np.zeros((32, 40, 2), dtype=np.float32)
    flow[..., 0] = -2 * np.indices(flow.shape[:2])[1]
    analysis = analyze_tps_dense_flow(flow)

    assert not np.any(analysis["fold_mask"] == False)
    assert analysis["summary"]["fold_pixels"] > 0
    np.testing.assert_array_equal(
        analysis["fold_mask"], analysis["jacobian_determinant"] <= 0,
    )


def test_analyze_tps_dense_flow_reports_jacobian_percentiles():
    from src.klt_tps_registration import analyze_tps_dense_flow

    flow = np.zeros((4, 5, 2), dtype=np.float64)
    flow[..., 0] = np.indices(flow.shape[:2])[1] * 0.5
    analysis = analyze_tps_dense_flow(flow)
    determinant = analysis["jacobian_determinant"]
    summary = analysis["summary"]

    for percentile, key in ((1, "jacobian_p01"), (5, "jacobian_p05"),
                            (50, "jacobian_median"), (95, "jacobian_p95"),
                            (99, "jacobian_p99")):
        assert summary[key] == pytest.approx(np.percentile(determinant, percentile))


def test_inspect_tps_dense_flow_still_rejects_same_folding_after_analysis_refactor():
    from src.klt_tps_registration import inspect_tps_dense_flow

    flow = np.zeros((32, 40, 2), dtype=np.float32)
    flow[..., 0] = -2 * np.indices(flow.shape[:2])[1]
    with pytest.raises(ValueError, match=r"TPS flow contains fold pixels: \d+"):
        inspect_tps_dense_flow(flow, 50.0)


def test_inspect_tps_dense_flow_still_rejects_same_max_shift_after_analysis_refactor():
    from src.klt_tps_registration import inspect_tps_dense_flow

    flow = np.zeros((32, 40, 2), dtype=np.float32)
    flow[..., 0] = 51
    with pytest.raises(ValueError, match=r"TPS flow max shift .* exceeds 50.000"):
        inspect_tps_dense_flow(flow, 50.0)


def test_summarize_control_displacements_reports_expected_percentiles():
    from src.klt_tps_registration import summarize_control_displacements

    displacement = np.asarray([[3.0, 4.0], [-3.0, 4.0], [0.0, 0.0], [10.0, 0.0]])
    summary = summarize_control_displacements(displacement)

    assert summary["count"] == 4
    assert summary["dx"]["min"] == -3.0
    assert summary["dx"]["max"] == 10.0
    assert summary["dy"]["median"] == 2.0
    assert summary["magnitude"]["max"] == 10.0
    assert summary["magnitude"]["p95"] == pytest.approx(
        np.percentile(np.hypot(displacement[:, 0], displacement[:, 1]), 95),
    )


def test_summarize_control_displacements_does_not_filter_extreme_values():
    from src.klt_tps_registration import summarize_control_displacements

    displacement = np.asarray([[0.0, 0.0], [1000.0, 0.0], [-1.0, 0.0]])
    summary = summarize_control_displacements(displacement)

    assert summary["dx"]["max"] == 1000.0
    assert summary["magnitude"]["max"] == 1000.0


def test_control_hull_mask_contains_control_interior():
    from src.klt_tps_registration import build_control_hull_mask

    points = np.asarray([[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]])
    result = build_control_hull_mask(points, (100, 120))

    assert result["available"] is True
    assert result["mask"].dtype == bool
    assert result["mask"][30, 30]
    assert not result["mask"][60, 60]
    assert result["pixel_count"] == int(result["mask"].sum())
    assert result["fraction"] == pytest.approx(result["pixel_count"] / (100 * 120))


def test_target_overlap_bbox_mask_uses_context_tgt_window():
    from src.klt_tps_registration import build_target_overlap_bbox_mask

    result = build_target_overlap_bbox_mask(
        {"tgt_window": (10, 40, 20, 60)}, (100, 120),
    )

    assert result["available"] is True
    assert result["window"] == (10, 40, 20, 60)
    assert result["mask"][10, 20]
    assert result["mask"][39, 59]
    assert not result["mask"][40, 20]
    assert result["pixel_count"] == 30 * 40


def test_fold_support_summary_distinguishes_inside_and_outside_hull():
    from src.klt_tps_registration import summarize_fold_support

    fold = np.zeros((10, 12), dtype=bool)
    fold[3, 3] = True
    fold[1, 1] = True
    hull = np.zeros_like(fold)
    hull[2:6, 2:7] = True
    result = summarize_fold_support(fold, hull, None)

    assert result["fold_pixels_total"] == 2
    assert result["control_hull_pixels"] == 20
    assert result["fold_pixels_inside_control_hull"] == 1
    assert result["fold_pixels_outside_control_hull"] == 1
    assert result["fold_fraction_inside_control_hull"] == pytest.approx(1 / 20)
    assert result["fold_fraction_outside_control_hull"] == pytest.approx(1 / (120 - 20))


def test_fold_support_summary_distinguishes_inside_and_outside_overlap_bbox():
    from src.klt_tps_registration import summarize_fold_support

    fold = np.zeros((10, 12), dtype=bool)
    fold[3, 3] = True
    fold[1, 1] = True
    overlap = np.zeros_like(fold)
    overlap[2:6, 2:7] = True
    result = summarize_fold_support(fold, None, overlap)

    assert result["overlap_bbox_pixels"] == 20
    assert result["fold_pixels_inside_overlap_bbox"] == 1
    assert result["fold_pixels_outside_overlap_bbox"] == 1
    assert result["fold_fraction_inside_overlap_bbox"] == pytest.approx(1 / 20)
    assert result["fold_fraction_outside_overlap_bbox"] == pytest.approx(1 / (120 - 20))
