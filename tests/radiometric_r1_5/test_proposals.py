from __future__ import annotations

import numpy as np

from benchmark_v1_radiometric_r15.proposals import recommend_label, window_quality


def test_partial_nodata_below_support_threshold_is_rejected():
    a = np.ones((32, 32), dtype=bool)
    b = a.copy()
    b[:3, :] = False
    result = window_quality(a, b, np.zeros_like(a), np.zeros_like(a))
    assert result["valid_fraction"] == 29 / 32
    assert result["accepted"] is False
    assert result["exclusion_reason"] == "B9_VALID_SUPPORT_BELOW_0.95"


def test_edge_contamination_is_rejected_without_relaxing_threshold():
    valid = np.ones((32, 32), dtype=bool)
    edges = np.zeros_like(valid)
    edges[:, :5] = True
    result = window_quality(valid, valid, edges, edges)
    assert result["edge_fraction"] == 7 / 32
    assert result["accepted"] is False
    assert result["exclusion_reason"] == "STRONG_EDGE_FRACTION_ABOVE_0.10"


def test_uniform_low_texture_is_unknown_not_a_stable_sample():
    result = recommend_label(days_apart=0, gradient_correlation=0.0,
                             spectral_shape_correlation=1.0, geometry_verified=True,
                             band_coreg_verified=True, texture_strength=0.0)
    assert result["suggested_label"] == "UNKNOWN"
    assert result["suggestion_confidence"] == "LOW"


def test_brightness_similarity_alone_never_suggests_stable():
    result = recommend_label(days_apart=0, gradient_correlation=None,
                             spectral_shape_correlation=None, geometry_verified=True,
                             band_coreg_verified=True, texture_strength=0.5,
                             brightness_similarity=0.999)
    assert result["suggested_label"] != "STABLE"
    assert "brightness" in result["uncertainty_reason"].lower()


def test_cross_date_change_is_suppressed_without_independent_geometry():
    result = recommend_label(days_apart=35, gradient_correlation=0.1,
                             spectral_shape_correlation=0.05, geometry_verified=False,
                             band_coreg_verified=False, texture_strength=0.7)
    assert result["suggested_label"] == "UNKNOWN"
    assert result["uncertainty_reason"] == "GEOMETRY_OR_COREG_UNVERIFIED"


def test_dark_or_bright_ambiguous_regions_do_not_become_shadow_or_cloud():
    dark = recommend_label(days_apart=0, gradient_correlation=None,
                           spectral_shape_correlation=None, geometry_verified=False,
                           band_coreg_verified=False, texture_strength=0.4,
                           brightness_similarity=0.2)
    bright = recommend_label(days_apart=0, gradient_correlation=None,
                             spectral_shape_correlation=None, geometry_verified=False,
                             band_coreg_verified=False, texture_strength=0.4,
                             brightness_similarity=0.9)
    assert dark["suggested_label"] == bright["suggested_label"] == "UNKNOWN"


def test_spatial_selection_is_not_created_from_highest_score_only():
    from benchmark_v1_radiometric_r15.proposals import select_spatially

    candidates = [{"candidate_id": f"c{i}", "grid_cell_id": cell, "score": score}
                  for i, (cell, score) in enumerate([
                      ("a", 0.99), ("a", 0.98), ("a", 0.97), ("b", 0.50), ("c", 0.40)])]
    selected = select_spatially(candidates, maximum=3)
    assert [row["candidate_id"] for row in selected] == ["c0", "c3", "c4"]


def test_multiband_stats_are_diagnostic_only_and_units_remain_unknown(tmp_path):
    from test_inputs import _fixture
    from benchmark_v1_radiometric_r15.inputs import load_pair_inputs
    from benchmark_v1_radiometric_r15.proposals import _window_band_support

    _fixture(tmp_path)
    pair = load_pair_inputs("00_01", tmp_path)
    features = _window_band_support(pair, [0, 0, 8, 8])
    assert features["B9"]["scene_i_pixel_median_diagnostic"] == 1.0
    assert features["B9"]["scene_j_pixel_mad_diagnostic"] == 0.0
    assert features["B9"]["unit"] == "UNKNOWN"
    assert features["_diagnostic_only"] is True
    assert features["_cross_scene_relative_spectral_shape_similarity_diagnostic"] == 1.0
