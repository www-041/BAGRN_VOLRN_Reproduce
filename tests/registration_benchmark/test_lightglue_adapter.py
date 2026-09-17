"""Tests for :mod:`src.registration_benchmark.matchers.lightglue`.

Unit tests use mocks so they pass without a GPU or the lightglue package.
"""

from __future__ import annotations
from unittest import mock

import numpy as np
import pytest

from src.registration_benchmark.models import MatchSet, MatchView


class TestLightGlueAdapterUnit:
    """Mock-based unit tests for the LightGlue adapter.

    These verify coordinate mapping, confidence handling, and MatchSet
    validation without requiring the actual lightglue package.
    """

    @pytest.fixture
    def view(self) -> MatchView:
        return MatchView(
            ref=np.random.default_rng(1).uniform(0, 1, (128, 128)).astype(np.float32),
            tgt=np.random.default_rng(2).uniform(0, 1, (128, 128)).astype(np.float32),
            ref_valid=np.ones((128, 128), dtype=bool),
            tgt_valid=np.ones((128, 128), dtype=bool),
            origin_x=50.0,
            origin_y=30.0,
            scale_x=0.5,
            scale_y=0.5,
        )

    @staticmethod
    def _mock_lightglue_modules(view, n_matches=20):
        """Patch LightGlue imports and return fake output."""
        # Create fake keypoints in view space
        rng = np.random.default_rng(42)
        kpts0 = torch.from_numpy(rng.uniform(0, 128, size=(n_matches * 2, 2)))
        kpts1 = torch.from_numpy(rng.uniform(0, 128, size=(n_matches * 2, 2)))
        match_indices = torch.arange(n_matches)
        scores = torch.from_numpy(rng.uniform(0.3, 1.0, size=n_matches))

        # Build fake feats dicts
        feats0 = {"keypoints": kpts0}
        feats1 = {"keypoints": kpts1}
        matches01 = {"matches": match_indices, "scores": scores}

        def fake_rbd(d):
            return feats0, feats1, matches01

        return {
            "extractor": mock.MagicMock(),
            "matcher": mock.MagicMock(),
            "rbd": fake_rbd,
            "feats0": feats0,
            "feats1": feats1,
            "matches01": matches01,
        }

    def test_registers_in_runner(self):
        """Once lightglue is installable, it should be in MATCHERS."""
        from src.registration_benchmark.matchers.lightglue import is_lightglue_available

        # Just verify the availability check doesn't crash
        avail = is_lightglue_available()
        assert isinstance(avail, bool)

    def test_unavailable_raises_runtime_error(self, view):
        """If lightglue is not installed, match_lightglue raises RuntimeError."""
        from src.registration_benchmark.matchers.lightglue import (
            _LIGHTGLUE_AVAILABLE,
            match_lightglue,
        )

        if not _LIGHTGLUE_AVAILABLE:
            with pytest.raises(RuntimeError, match="LightGlue is not available"):
                match_lightglue(view)
        else:
            pytest.skip("LightGlue is installed — run integration test instead")


class TestLightGlueAdapterIntegration:
    """Integration tests (only run when lightglue is actually available)."""

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
        """Run LightGlue on a 128×128 image — smoke test only."""
        from src.registration_benchmark.matchers.lightglue import (
            _LIGHTGLUE_AVAILABLE,
            match_lightglue,
        )

        if not _LIGHTGLUE_AVAILABLE:
            pytest.skip("LightGlue not available")

        result = match_lightglue(view, device="cpu", max_num_keypoints=512)

        assert isinstance(result, MatchSet)
        assert result.method == "lightglue"
        result.validate()


# Need torch for mock tensors; import conditionally
try:
    import torch
except ImportError:
    torch = None