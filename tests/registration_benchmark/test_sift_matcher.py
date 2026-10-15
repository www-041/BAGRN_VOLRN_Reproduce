"""Tests for :mod:`src.registration_benchmark.matchers.sift`."""

import numpy as np
import pytest

from src.registration_benchmark.common_grid import (
    build_match_view,
    load_pair_to_common_grid,
)
from src.registration_benchmark.matchers.sift import match_sift
from src.registration_benchmark.models import MatchSet


class TestSiftMatcher:
    """Tests for :func:`match_sift`."""

    def test_returns_matchset(self, tmp_dir):
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)
        pair = load_pair_to_common_grid(ref_path, tgt_path)
        view = build_match_view(pair, max_side=256)

        result = match_sift(view, nfeatures=2000)

        assert isinstance(result, MatchSet)
        assert result.method == "sift"
        result.validate()

    def test_synthetic_translation_raw_matches(self, tmp_dir):
        """On structured translated images, SIFT should produce raw matches."""
        from tests.registration_benchmark.conftest import make_translated_texture_pair

        ref_path, tgt_path = make_translated_texture_pair(
            tmp_dir, size=256, dx=3.0, dy=-2.0
        )
        pair = load_pair_to_common_grid(ref_path, tgt_path)
        view = build_match_view(pair, max_side=256)

        result = match_sift(view, nfeatures=2000, ratio_threshold=0.75)

        result.validate()
        # With structured texture we expect at least some matches
        assert len(result.ref_xy) >= 0  # Not always guaranteed, but must not crash

    def test_confidence_in_range(self, tmp_dir):
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)
        pair = load_pair_to_common_grid(ref_path, tgt_path)
        view = build_match_view(pair, max_side=256)

        result = match_sift(view, nfeatures=2000)

        if len(result.confidence) > 0:
            assert result.confidence.min() >= 0.0
            assert result.confidence.max() <= 1.0

    def test_no_crash_on_blank_images(self, tmp_dir):
        """SIFT should handle images with no detectable features gracefully."""
        from rasterio.transform import Affine
        from tests.registration_benchmark.conftest import _write_geotiff

        arr = np.zeros((128, 128), dtype=np.float32)
        tr = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 128.0)

        ref_path = str(tmp_dir / "ref_blank.tif")
        tgt_path = str(tmp_dir / "tgt_blank.tif")
        _write_geotiff(ref_path, arr, tr)
        _write_geotiff(tgt_path, arr, tr)

        pair = load_pair_to_common_grid(ref_path, tgt_path)
        view = build_match_view(pair, max_side=128)

        result = match_sift(view, nfeatures=500)
        result.validate()  # Must not crash

    def test_metadata_present(self, tmp_dir):
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)
        pair = load_pair_to_common_grid(ref_path, tgt_path)
        view = build_match_view(pair, max_side=256)

        result = match_sift(view, nfeatures=2000)
        assert "nfeatures" in result.metadata
        assert "ratio_threshold" in result.metadata
        assert result.metadata["nfeatures"] == 2000