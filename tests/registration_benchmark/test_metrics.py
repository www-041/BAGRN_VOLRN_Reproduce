"""Tests for :mod:`src.registration_benchmark.metrics`."""

import numpy as np
import pytest

from src.registration_benchmark.metrics import (
    gradient_ncc,
    phase_verification,
    spatial_coverage_ratio,
)


class TestSpatialCoverage:
    """Tests for :func:`spatial_coverage_ratio`."""

    def test_empty_points_returns_zero(self):
        c = spatial_coverage_ratio(
            np.empty((0, 2)), (0, 100, 0, 100)
        )
        assert c == 0.0

    def test_all_cells_occupied(self):
        """Place one point in every cell of an 8×8 grid."""
        overlap = (0, 80, 0, 80)  # 10×10 px per cell
        pts = []
        for r in range(8):
            for c in range(8):
                pts.append([c * 10 + 5, r * 10 + 5])  # center of each cell
        c = spatial_coverage_ratio(np.array(pts), overlap)
        assert c == pytest.approx(1.0)

    def test_single_cell(self):
        """All points in the same cell → 1/64."""
        overlap = (0, 80, 0, 80)
        pts = np.array([[5.0, 5.0], [8.0, 7.0], [3.0, 9.0]])  # all in cell (0,0)
        c = spatial_coverage_ratio(pts, overlap)
        assert c == pytest.approx(1.0 / 64.0)

    def test_half_occupied(self):
        """Points in 32 cells out of 64."""
        overlap = (0, 80, 0, 80)
        pts = []
        for r in range(4):  # only first 4 rows
            for c in range(8):
                pts.append([c * 10 + 5, r * 10 + 5])
        c = spatial_coverage_ratio(np.array(pts), overlap)
        assert c == pytest.approx(32.0 / 64.0)

    def test_points_outside_window_clipped(self):
        """Points far outside the window should be clipped."""
        overlap = (0, 80, 0, 80)
        pts = np.array([
            [-100.0, -100.0],
            [200.0, 200.0],
            [10.0, 10.0],  # in cell (1,1)
        ])
        c = spatial_coverage_ratio(pts, overlap)
        # The first two points clip to (0,0) and (7,7), the third is (1,1)
        # So at most 3 cells occupied
        assert 0.0 < c <= 3.0 / 64.0

    def test_zero_size_window(self):
        c = spatial_coverage_ratio(
            np.array([[10.0, 10.0]]), (0, 0, 0, 0)
        )
        assert c == 0.0


class TestGradientNCC:
    """Tests for :func:`gradient_ncc`."""

    def test_identical_images_ncc_one(self):
        """NCC of identical images should be ~1."""
        img = np.sin(np.linspace(0, 4 * np.pi, 100)).reshape(10, 10)
        img = img.astype(np.float64)
        ncc = gradient_ncc(img, img)
        assert ncc is not None
        assert not np.isnan(ncc)
        assert ncc > 0.95  # Near 1

    def test_ncc_lower_for_shifted(self):
        """NCC of misaligned images should be lower."""
        rng = np.random.default_rng(42)
        ref = rng.uniform(0, 1, size=(64, 64)).astype(np.float64)
        tgt = np.roll(ref, shift=5, axis=1)

        ncc_aligned = gradient_ncc(ref, ref)
        ncc_shifted = gradient_ncc(ref, tgt)

        assert ncc_aligned > ncc_shifted

    def test_nan_on_constant_image(self):
        """A uniform image has zero gradient variance → NaN NCC."""
        img = np.ones((32, 32), dtype=np.float64)
        ncc = gradient_ncc(img, img)
        assert np.isnan(ncc)

    def test_with_valid_mask(self):
        """Valid mask should exclude regions."""
        img = np.sin(np.linspace(0, 4 * np.pi, 256)).reshape(16, 16).astype(np.float64)
        valid = np.ones((16, 16), dtype=bool)
        valid[0:4, :] = False

        ncc = gradient_ncc(img, img, valid=valid)
        assert not np.isnan(ncc)
        assert ncc > 0.9


class TestPhaseVerification:
    """Tests for :func:`phase_verification`."""

    def test_aligned_images_near_zero_shift(self):
        """Phase verification on identical images should report ~0 shift."""
        rng = np.random.default_rng(123)
        ref = rng.uniform(50, 200, size=(128, 128)).astype(np.float64)
        tgt = ref.copy()

        valid_ref = np.ones((128, 128), dtype=bool)
        valid_tgt = np.ones((128, 128), dtype=bool)

        result = phase_verification(ref, tgt, valid_ref, valid_tgt)

        assert result["status"] == "OK"
        assert abs(result["dx"]) < 1.0
        assert abs(result["dy"]) < 1.0
        assert result["magnitude"] < 1.5
        assert not np.isnan(result["confidence"])

    def test_known_shift_detected(self):
        """Phase verification should detect a known 5-px shift."""
        rng = np.random.default_rng(456)
        ref = rng.uniform(50, 200, size=(128, 128)).astype(np.float64)
        tgt = np.roll(ref, shift=5, axis=1)  # 5 px right

        valid_ref = np.ones((128, 128), dtype=bool)
        valid_tgt = np.ones((128, 128), dtype=bool)

        result = phase_verification(ref, tgt, valid_ref, valid_tgt)

        assert result["status"] == "OK"
        assert result["dx"] == pytest.approx(-5.0, abs=2.0)  # tgt shifted right → dx ~ -5
        assert result["magnitude"] > 1.0

    def test_too_few_valid_pixels(self):
        """Degenerate case with almost no valid pixels."""
        ref = np.ones((16, 16), dtype=np.float64)
        tgt = np.ones((16, 16), dtype=np.float64)
        valid = np.zeros((16, 16), dtype=bool)
        valid[0, 0] = True  # Only 1 valid pixel

        result = phase_verification(ref, tgt, valid, valid)

        assert result["status"] != "OK"
        assert np.isnan(result["dx"])

    def test_fully_aligned_checkerboard_zero_shift(self):
        """Deterministic checkerboard: identical images should report exact zero shift."""
        # Build an 8×8 checkerboard of 16 px checks (128×128 total)
        size = 128
        check = 16
        y, x = np.mgrid[0:size, 0:size]
        row_check = (y // check).astype(int)
        col_check = (x // check).astype(int)
        board = np.where((row_check + col_check) % 2 == 0, 200.0, 50.0).astype(np.float64)
        # Tiny noise so constant-region gradient variance is non-zero
        rng = np.random.default_rng(42)
        board += rng.uniform(-0.5, 0.5, size=(size, size))

        tgt = board.copy()
        valid = np.ones((size, size), dtype=bool)

        result = phase_verification(board, tgt, valid, valid)

        assert result["status"] == "OK"
        assert result["dx"] == pytest.approx(0.0, abs=1e-3)
        assert result["dy"] == pytest.approx(0.0, abs=1e-3)
        assert result["magnitude"] == pytest.approx(0.0, abs=1e-3)
        assert result["confidence"] == pytest.approx(1.0, abs=0.05)

    def test_known_translation_checkerboard_exact(self):
        """Deterministic checkerboard shifted by (5, -3) px via np.roll."""
        size = 128
        check = 16
        y, x = np.mgrid[0:size, 0:size]
        row_check = (y // check).astype(int)
        col_check = (x // check).astype(int)
        board = np.where((row_check + col_check) % 2 == 0, 200.0, 50.0).astype(np.float64)
        rng = np.random.default_rng(42)
        board += rng.uniform(-0.5, 0.5, size=(size, size))

        # np.roll shifts RIGHT (positive axis=1) and DOWN (positive axis=0).
        # To align tgt back to ref, phase_cross_correlation reports
        # dx = -shift_x, dy = -shift_y
        shift_y, shift_x = -3, 5
        tgt = np.roll(board, shift=(shift_y, shift_x), axis=(0, 1))

        valid = np.ones((size, size), dtype=bool)

        result = phase_verification(board, tgt, valid, valid)

        assert result["status"] == "OK"
        assert result["dx"] == pytest.approx(-float(shift_x), abs=1e-3)
        assert result["dy"] == pytest.approx(-float(shift_y), abs=1e-3)
        assert result["magnitude"] > 4.0
        assert result["confidence"] == pytest.approx(1.0, abs=0.05)