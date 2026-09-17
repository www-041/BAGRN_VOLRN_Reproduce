"""Tests for :mod:`src.registration_benchmark.matchers.loftr`.

Unit tests use mocks; integration (smoke) test runs only when kornia LoFTR
is available and can download weights.
"""

from __future__ import annotations
from unittest import mock

import numpy as np
import pytest

from src.registration_benchmark.models import MatchSet, MatchView


class TestLoFTROutputParsing:
    """Tests that verify the adapter correctly processes LoFTR outputs.

    These are logical tests — they exercise the adapter through a mock to
    check coordinate mapping, confidence filtering, and max_matches capping.
    """

    @pytest.fixture
    def view(self) -> MatchView:
        return MatchView(
            ref=np.zeros((64, 64), dtype=np.float32),
            tgt=np.zeros((64, 64), dtype=np.float32),
            ref_valid=np.ones((64, 64), dtype=bool),
            tgt_valid=np.ones((64, 64), dtype=bool),
            origin_x=100.0,
            origin_y=50.0,
            scale_x=0.5,
            scale_y=0.5,
        )

    def _patch_loftr(self, view, n_keypoints=100, confidence_mean=0.5):
        """Create a mock LoFTR that returns synthetic output."""
        rng = np.random.default_rng(42)
        kpts0 = rng.uniform(0, 64, size=(n_keypoints, 2)).astype(np.float32)
        kpts1 = rng.uniform(0, 64, size=(n_keypoints, 2)).astype(np.float32)
        conf = rng.uniform(0, 1, size=n_keypoints).astype(np.float32)

        mock_output = {
            "keypoints0": torch.from_numpy(kpts0),
            "keypoints1": torch.from_numpy(kpts1),
            "confidence": torch.from_numpy(conf),
        }

        mock_model = mock.MagicMock()
        mock_model.return_value = mock_output

        return mock_model, kpts0, kpts1, conf

    def test_confidence_filtering(self, view):
        """Matches below confidence_threshold should be removed."""
        from src.registration_benchmark.matchers.loftr import match_loftr

        # Patch LoFTR availability
        with (
            mock.patch("src.registration_benchmark.matchers.loftr._LOFTR_AVAILABLE", True),
            mock.patch("src.registration_benchmark.matchers.loftr.KF") as mock_kf,
        ):
            n_kp = 100
            rng = np.random.default_rng(99)
            kpts0 = rng.uniform(0, 64, size=(n_kp, 2)).astype(np.float32)
            kpts1 = rng.uniform(0, 64, size=(n_kp, 2)).astype(np.float32)
            conf = np.zeros(n_kp, dtype=np.float32)
            conf[:20] = 0.8   # above threshold
            conf[20:] = 0.05  # below threshold

            mock_model = mock.MagicMock()
            mock_model.return_value = {
                "keypoints0": torch.from_numpy(kpts0),
                "keypoints1": torch.from_numpy(kpts1),
                "confidence": torch.from_numpy(conf),
            }
            mock_kf.LoFTR.return_value.eval.return_value.to.return_value = mock_model

            result = match_loftr(view, device="cpu", confidence_threshold=0.2)

            assert len(result.ref_xy) == 20
            result.validate()

    def test_max_matches_cap(self, view):
        """When output exceeds max_matches, only top-N by confidence are kept."""
        from src.registration_benchmark.matchers.loftr import match_loftr

        with (
            mock.patch("src.registration_benchmark.matchers.loftr._LOFTR_AVAILABLE", True),
            mock.patch("src.registration_benchmark.matchers.loftr.KF") as mock_kf,
        ):
            n_kp = 200
            rng = np.random.default_rng(100)
            kpts0 = rng.uniform(0, 64, size=(n_kp, 2)).astype(np.float32)
            kpts1 = rng.uniform(0, 64, size=(n_kp, 2)).astype(np.float32)
            conf = rng.uniform(0, 1, size=n_kp).astype(np.float32)

            mock_model = mock.MagicMock()
            mock_model.return_value = {
                "keypoints0": torch.from_numpy(kpts0),
                "keypoints1": torch.from_numpy(kpts1),
                "confidence": torch.from_numpy(conf),
            }
            mock_kf.LoFTR.return_value.eval.return_value.to.return_value = mock_model

            result = match_loftr(view, device="cpu", max_matches=50, confidence_threshold=0.0)

            assert len(result.ref_xy) == 50
            result.validate()

    def test_to_canvas_coordinate_mapping(self, view):
        """Verify that output coordinates are mapped through to_canvas."""
        from src.registration_benchmark.matchers.loftr import match_loftr

        with (
            mock.patch("src.registration_benchmark.matchers.loftr._LOFTR_AVAILABLE", True),
            mock.patch("src.registration_benchmark.matchers.loftr.KF") as mock_kf,
        ):
            kpts0 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
            kpts1 = np.array([[15.0, 22.0], [33.0, 44.0]], dtype=np.float32)
            conf = np.array([0.9, 0.8], dtype=np.float32)

            mock_model = mock.MagicMock()
            mock_model.return_value = {
                "keypoints0": torch.from_numpy(kpts0),
                "keypoints1": torch.from_numpy(kpts1),
                "confidence": torch.from_numpy(conf),
            }
            mock_kf.LoFTR.return_value.eval.return_value.to.return_value = mock_model

            result = match_loftr(view, device="cpu", confidence_threshold=0.5)

            # Canvas mapping: canvas_x = origin_x + view_x / scale_x
            # For (10, 20): canvas_x = 100 + 10/0.5 = 120, canvas_y = 50 + 20/0.5 = 90
            assert result.ref_xy[0, 0] == pytest.approx(100.0 + 10.0 / 0.5)
            assert result.ref_xy[0, 1] == pytest.approx(50.0 + 20.0 / 0.5)


class TestLoFTRIntegration:
    """Integration tests that run only when kornia LoFTR is actually available."""

    @pytest.fixture
    def view(self) -> MatchView:
        return MatchView(
            ref=np.random.default_rng(1).uniform(0, 1, (128, 128)).astype(np.float32),
            tgt=np.random.default_rng(2).uniform(0, 1, (128, 128)).astype(np.float32),
            ref_valid=np.ones((128, 128), dtype=bool),
            tgt_valid=np.ones((128, 128), dtype=bool),
            origin_x=0.0,
            origin_y=0.0,
            scale_x=1.0,
            scale_y=1.0,
        )

    def test_smoke_on_small_image(self, view):
        """Run LoFTR on 128×128 — download weights if needed."""
        from src.registration_benchmark.matchers.loftr import (
            _LOFTR_AVAILABLE,
            match_loftr,
        )

        if not _LOFTR_AVAILABLE:
            pytest.skip("LoFTR (kornia) not available")

        try:
            result = match_loftr(view, device="cpu", max_matches=100)
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("ssl", "certificate", "download",
                                       "connection", "network", "bad gateway",
                                       "502", "timeout", "http error")):
                pytest.skip(f"LoFTR weights unavailable (network): {e}")
            raise

        assert isinstance(result, MatchSet)
        assert result.method == "loftr"
        result.validate()

    def test_confidence_is_strictly_positive(self, view):
        """All returned matches should have confidence >= threshold."""
        from src.registration_benchmark.matchers.loftr import (
            _LOFTR_AVAILABLE,
            match_loftr,
        )

        if not _LOFTR_AVAILABLE:
            pytest.skip("LoFTR (kornia) not available")

        try:
            result = match_loftr(
                view, device="cpu", max_matches=100, confidence_threshold=0.3
            )
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("ssl", "certificate", "download",
                                       "connection", "network", "bad gateway",
                                       "502", "timeout", "http error")):
                pytest.skip(f"LoFTR weights unavailable (network): {e}")
            raise

        if len(result.confidence) > 0:
            assert result.confidence.min() >= 0.3


try:
    import torch
except ImportError:
    torch = None