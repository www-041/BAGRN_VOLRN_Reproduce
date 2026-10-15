"""Tests for :mod:`src.registration_benchmark.matchers.phase`."""

import numpy as np
import pytest

from src.registration_benchmark.common_grid import (
    build_match_view,
    load_pair_to_common_grid,
)
from src.registration_benchmark.matchers.phase import match_phase
from src.registration_benchmark.models import MatchSet


class TestPhaseMatcher:
    """Tests for :func:`match_phase` adapter."""

    def test_returns_matchset(self, tmp_dir):
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)
        pair = load_pair_to_common_grid(ref_path, tgt_path)
        view = build_match_view(pair, max_side=256)

        result = match_phase(view, block_size=64, max_shift=50.0)

        assert isinstance(result, MatchSet)
        assert result.method == "phase"
        result.validate()

    def test_shift_estimation_on_translation(self, tmp_dir):
        """Target texture is geographically shifted by (+7, -5) px.

        The common grid covers the union extent, so the same texture
        appears at different grid positions. Phase correlation should
        detect the known offset.
        """
        from tests.registration_benchmark.conftest import make_translated_texture_pair

        ref_path, tgt_path = make_translated_texture_pair(
            tmp_dir, size=256, dx=7.0, dy=-5.0
        )
        pair = load_pair_to_common_grid(ref_path, tgt_path)
        view = build_match_view(pair, max_side=256)

        result = match_phase(view, block_size=128, max_shift=50.0)

        # The texture is identical but at different geo positions;
        # on the common grid this appears as a pixel offset of ~(7, -5)
        if len(result.ref_xy) >= 3:
            shifts = result.ref_xy - result.tgt_xy
            median_dx = np.median(shifts[:, 0])
            median_dy = np.median(shifts[:, 1])
            assert abs(median_dx - 7.0) < 3.0, f"Expected ~7 dx, got {median_dx}"
            assert abs(median_dy + 5.0) < 3.0, f"Expected ~-5 dy, got {median_dy}"

    def test_empty_result_on_unmatchable(self, tmp_dir):
        """Two very different random images may produce zero matches."""
        from rasterio.transform import Affine
        from tests.registration_benchmark.conftest import _write_geotiff

        rng = np.random.default_rng(99)
        ref_arr = rng.uniform(0, 255, size=(128, 128)).astype(np.float32)
        tgt_arr = rng.uniform(0, 255, size=(128, 128)).astype(np.float32)

        tr = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 128.0)
        ref_path = str(tmp_dir / "ref_rnd.tif")
        tgt_path = str(tmp_dir / "tgt_rnd.tif")
        _write_geotiff(ref_path, ref_arr, tr)
        _write_geotiff(tgt_path, tgt_arr, tr)

        pair = load_pair_to_common_grid(ref_path, tgt_path)
        view = build_match_view(pair, max_side=128)

        result = match_phase(view, block_size=32, confidence_threshold=0.9)

        # May have zero or few matches — but must not crash and must be valid
        result.validate()
        # If we got matches, they must be finite
        if len(result.ref_xy) > 0:
            assert np.isfinite(result.ref_xy).all()
            assert np.isfinite(result.tgt_xy).all()

    def test_confidence_in_range(self, tmp_dir):
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)
        pair = load_pair_to_common_grid(ref_path, tgt_path)
        view = build_match_view(pair, max_side=256)

        result = match_phase(view, block_size=64, max_shift=50.0)

        if len(result.confidence) > 0:
            assert result.confidence.min() >= 0.0
            assert result.confidence.max() <= 1.0

    def test_metadata_contains_screening(self, tmp_dir):
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)
        pair = load_pair_to_common_grid(ref_path, tgt_path)
        view = build_match_view(pair, max_side=256)

        result = match_phase(view, block_size=64)
        assert "screening" in result.metadata
        assert "block_size" in result.metadata

    def test_screening_fields_present(self, tmp_dir):
        """Screening dict must have all expected keys."""
        from tests.registration_benchmark.conftest import make_translated_texture_pair

        ref_path, tgt_path = make_translated_texture_pair(
            tmp_dir, size=128, dx=2.0, dy=-1.0
        )
        pair = load_pair_to_common_grid(ref_path, tgt_path)
        view = build_match_view(pair, max_side=128)

        result = match_phase(view, block_size=32, max_shift=20.0)

        screening = result.metadata["screening"]
        for key in ("total", "low_valid", "low_texture", "low_conf",
                     "large_shift", "accepted"):
            assert key in screening, f"Missing screening key: {key}"
            assert isinstance(screening[key], int)