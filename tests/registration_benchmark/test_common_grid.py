"""Tests for :mod:`src.registration_benchmark.common_grid`."""

import numpy as np
import pytest

from src.registration_benchmark.common_grid import (
    build_match_view,
    load_pair_to_common_grid,
)
from src.registration_benchmark.models import CommonGridPair, MatchView


class TestLoadPairToCommonGrid:
    """Integration tests using synthetic GeoTIFFs."""

    def test_basic_load_same_shape(self, tmp_dir):
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)

        pair = load_pair_to_common_grid(ref_path, tgt_path, band=1)

        assert isinstance(pair, CommonGridPair)
        assert pair.ref_raw.shape == pair.tgt_raw.shape
        assert pair.ref_valid.shape == pair.tgt_valid.shape
        assert pair.ref_raw.shape == pair.ref_valid.shape

    def test_overlap_width(self, tmp_dir):
        """512×512 images, tgt shifted 128 px right → overlap ≈ 384 px."""
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)

        pair = load_pair_to_common_grid(ref_path, tgt_path, band=1)

        row_start, row_end, col_start, col_end = pair.overlap_window
        overlap_width = col_end - col_start
        overlap_height = row_end - row_start

        # Expected: both are 512 px tall in overlap region
        assert 380 <= overlap_width <= 388, f"overlap width={overlap_width}"
        assert 508 <= overlap_height <= 515, f"overlap height={overlap_height}"

    def test_valid_masks_exist(self, tmp_dir):
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)

        pair = load_pair_to_common_grid(ref_path, tgt_path, band=1)

        assert pair.ref_valid.any()
        assert pair.tgt_valid.any()
        joint = pair.ref_valid & pair.tgt_valid
        assert joint.any(), "Overlap region should have valid pixels"

    def test_crs_mismatch_raises(self, tmp_dir):
        from tests.registration_benchmark.conftest import _write_geotiff

        ref_path = str(tmp_dir / "ref_crs1.tif")
        tgt_path = str(tmp_dir / "tgt_crs2.tif")

        from rasterio.transform import Affine

        _write_geotiff(
            ref_path,
            np.zeros((100, 100), dtype=np.float32),
            Affine(1, 0, 0, 0, -1, 100),
            crs="EPSG:4326",
        )
        _write_geotiff(
            tgt_path,
            np.zeros((100, 100), dtype=np.float32),
            Affine(1, 0, 0, 0, -1, 100),
            crs="EPSG:32650",
        )

        with pytest.raises(ValueError, match="CRS mismatch"):
            load_pair_to_common_grid(ref_path, tgt_path, band=1)


class TestBuildMatchView:
    """Tests for :func:`build_match_view`."""

    def test_produces_valid_view(self, tmp_dir):
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)
        pair = load_pair_to_common_grid(ref_path, tgt_path)

        view = build_match_view(pair, max_side=256)

        assert isinstance(view, MatchView)
        assert view.ref.shape == view.tgt.shape
        assert view.ref.dtype == np.float32
        assert view.tgt.dtype == np.float32
        # Values should be in [0, 1]
        assert 0.0 <= view.ref.max() <= 1.0
        assert 0.0 <= view.tgt.max() <= 1.0

    def test_max_side_respected(self, tmp_dir):
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)
        pair = load_pair_to_common_grid(ref_path, tgt_path)

        view = build_match_view(pair, max_side=200)

        assert max(view.ref.shape) <= 200
        assert max(view.tgt.shape) <= 200

    def test_ref_tgt_same_shape(self, tmp_dir):
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)
        pair = load_pair_to_common_grid(ref_path, tgt_path)

        view = build_match_view(pair, max_side=300)

        assert view.ref.shape == view.tgt.shape

    def test_to_canvas_maps_correctly(self, tmp_dir):
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)
        pair = load_pair_to_common_grid(ref_path, tgt_path)

        view = build_match_view(pair, max_side=256)

        # Map a point at the view origin → common-grid origin
        canvas = view.to_canvas(np.array([[0.0, 0.0]]))
        row_start, _row_end, col_start, _col_end = pair.overlap_window

        assert canvas[0, 0] == pytest.approx(float(col_start))
        assert canvas[0, 1] == pytest.approx(float(row_start))

    def test_scale_consistent(self, tmp_dir):
        """Verify scale_x == scale_y."""
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)
        pair = load_pair_to_common_grid(ref_path, tgt_path)

        view = build_match_view(pair, max_side=256)

        assert view.scale_x == pytest.approx(view.scale_y)

    def test_valid_masks_present(self, tmp_dir):
        from tests.registration_benchmark.conftest import make_synthetic_pair

        ref_path, tgt_path = make_synthetic_pair(tmp_dir)
        pair = load_pair_to_common_grid(ref_path, tgt_path)

        view = build_match_view(pair, max_side=256)

        assert view.ref_valid.any()
        assert view.tgt_valid.any()
        assert view.ref_valid.shape == view.ref.shape
        assert view.tgt_valid.shape == view.tgt.shape