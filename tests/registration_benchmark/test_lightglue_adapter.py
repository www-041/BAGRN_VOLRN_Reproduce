"""Tests for :mod:`src.registration_benchmark.matchers.lightglue`.

Unit tests use mocks so they pass without a GPU or the lightglue package.
"""

from __future__ import annotations
from unittest import mock

import numpy as np
import pytest

from src.registration_benchmark.models import MatchSet, MatchView

# Need torch for mock tensors; import conditionally
try:
    import torch
except ImportError:
    torch = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_view(with_invalid: bool = False) -> MatchView:
    valid = np.ones((128, 128), dtype=bool)
    if with_invalid:
        valid[0:10, :] = False  # top rows invalid
    return MatchView(
        ref=np.random.default_rng(1).uniform(0, 1, (128, 128)).astype(np.float32),
        tgt=np.random.default_rng(2).uniform(0, 1, (128, 128)).astype(np.float32),
        ref_valid=valid.copy(),
        tgt_valid=valid.copy(),
        origin_x=50.0,
        origin_y=30.0,
        scale_x=0.5,
        scale_y=0.5,
    )


# ---------------------------------------------------------------------------
# Unit tests: match extraction helpers
# ---------------------------------------------------------------------------


class TestExtractLightGlueMatches:
    """Tests for :func:`_extract_lightglue_matches`."""

    def test_extracts_m_by_2_matches(self):
        """Official format: matches shaped (M, 2) with index pairs."""
        from src.registration_benchmark.matchers.lightglue import _extract_lightglue_matches

        kpts0 = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])
        kpts1 = torch.tensor([[15.0, 25.0], [35.0, 45.0], [55.0, 65.0]])

        # pairs: [ref_idx, tgt_idx]  — match (0↔2) and (2↔1)
        matches01 = {
            "matches": torch.tensor([[0, 2], [2, 1]]),
            "scores": torch.tensor([0.9, 0.7]),
        }

        ref_xy, tgt_xy, conf, src = _extract_lightglue_matches(
            {"keypoints": kpts0}, {"keypoints": kpts1}, matches01, 0.0
        )

        assert len(ref_xy) == 2
        # ref point 0 → (10, 20)
        np.testing.assert_allclose(ref_xy[0], [10.0, 20.0])
        # ref point 2 → (50, 60)
        np.testing.assert_allclose(ref_xy[1], [50.0, 60.0])
        # tgt point 2 → (55, 65)
        np.testing.assert_allclose(tgt_xy[0], [55.0, 65.0])
        # tgt point 1 → (35, 45)
        np.testing.assert_allclose(tgt_xy[1], [35.0, 45.0])
        np.testing.assert_allclose(conf, [0.9, 0.7])
        assert src == "scores"

    def test_extracts_matches0_format(self):
        """Legacy format: matches0 with -1 = unmatched."""
        from src.registration_benchmark.matchers.lightglue import _extract_lightglue_matches

        kpts0 = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        kpts1 = torch.tensor([[5.0, 5.0], [6.0, 6.0]])

        # kp0[0]↔kp1[1], kp0[1] unmatched, kp0[2]↔kp1[0]
        matches01 = {
            "matches0": torch.tensor([1, -1, 0]),
            "matching_scores0": torch.tensor([0.8, 0.0, 0.6]),
        }

        ref_xy, tgt_xy, conf, src = _extract_lightglue_matches(
            {"keypoints": kpts0}, {"keypoints": kpts1}, matches01, 0.0
        )

        assert len(ref_xy) == 2
        np.testing.assert_allclose(ref_xy[0], [0.0, 0.0])
        np.testing.assert_allclose(ref_xy[1], [2.0, 2.0])
        np.testing.assert_allclose(tgt_xy[0], [6.0, 6.0])
        np.testing.assert_allclose(tgt_xy[1], [5.0, 5.0])
        assert src == "matching_scores0"

    def test_confidence_filtering(self):
        """Matches below min_confidence are excluded."""
        from src.registration_benchmark.matchers.lightglue import _extract_lightglue_matches

        kpts0 = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        kpts1 = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        matches01 = {
            "matches": torch.tensor([[0, 0], [1, 1], [2, 2]]),
            "scores": torch.tensor([0.9, 0.3, 0.8]),
        }

        _, _, conf, _ = _extract_lightglue_matches(
            {"keypoints": kpts0}, {"keypoints": kpts1}, matches01, min_confidence=0.5
        )

        assert len(conf) == 2  # only 0.9 and 0.8 survive
        assert conf[0] == pytest.approx(0.9)
        assert conf[1] == pytest.approx(0.8)

    def test_fallback_ones_when_no_score(self):
        """When no score field exists, all confidences are 1.0."""
        from src.registration_benchmark.matchers.lightglue import _extract_lightglue_matches

        kpts0 = torch.tensor([[1.0, 2.0]])
        kpts1 = torch.tensor([[3.0, 4.0]])
        matches01 = {"matches": torch.tensor([[0, 0]])}

        _, _, conf, src = _extract_lightglue_matches(
            {"keypoints": kpts0}, {"keypoints": kpts1}, matches01, 0.0
        )

        assert conf[0] == 1.0
        assert src == "fallback_ones"


# ---------------------------------------------------------------------------
# Unit tests: valid-mask filter
# ---------------------------------------------------------------------------


class TestValidMaskFilter:
    """Tests for :func:`_filter_by_valid_mask`."""

    def test_invalid_mask_points_removed(self):
        from src.registration_benchmark.matchers.lightglue import _filter_by_valid_mask

        view = _make_view(with_invalid=True)
        # Points in (col, row) = (x, y)
        ref_xy = np.array([[60.0, 20.0], [60.0, 5.0], [80.0, 40.0]], dtype=np.float64)
        tgt_xy = np.array([[60.0, 20.0], [60.0, 5.0], [80.0, 40.0]], dtype=np.float64)
        conf = np.array([0.9, 0.8, 0.7])

        r, t, c = _filter_by_valid_mask(ref_xy, tgt_xy, conf, view)

        # Point at y=5 is in row 5 which is in top-10 invalid zone → removed
        assert len(r) == 2
        np.testing.assert_allclose(c, [0.9, 0.7])

    def test_all_valid_kept(self):
        from src.registration_benchmark.matchers.lightglue import _filter_by_valid_mask

        view = _make_view(with_invalid=False)
        ref_xy = np.array([[60.0, 60.0], [70.0, 70.0]])
        tgt_xy = np.array([[60.0, 60.0], [70.0, 70.0]])
        conf = np.array([0.5, 0.6])

        r, t, c = _filter_by_valid_mask(ref_xy, tgt_xy, conf, view)
        assert len(r) == 2


# ---------------------------------------------------------------------------
# Unit tests: coordinate mapping (mock full adapter)
# ---------------------------------------------------------------------------


class TestLightGlueAdapterUnit:
    """Mock-based unit tests for the full match_lightglue adapter."""

    @pytest.fixture
    def view(self):
        return _make_view(with_invalid=False)

    def test_coordinate_mapping_to_canvas(self, view):
        """After to_canvas, coordinates should be in common-grid space."""
        import src.registration_benchmark.matchers.lightglue as lg_mod

        kpts0 = torch.tensor([[10.0, 20.0]])
        kpts1 = torch.tensor([[15.0, 22.0]])

        with (
            mock.patch.object(lg_mod, "_LIGHTGLUE_AVAILABLE", True),
            mock.patch.object(lg_mod, "SuperPoint", create=True) as mock_sp,
            mock.patch.object(lg_mod, "LightGlue", create=True) as mock_lg,
            mock.patch.object(lg_mod, "rbd", create=True) as mock_rbd,
        ):
            # Mock SuperPoint
            mock_sp_inst = mock.MagicMock()
            mock_sp.return_value.eval.return_value.to.return_value = mock_sp_inst
            mock_sp_inst.extract.side_effect = [
                {"keypoints": kpts0.unsqueeze(0), "descriptors": torch.zeros(1, 1, 256)},
                {"keypoints": kpts1.unsqueeze(0), "descriptors": torch.zeros(1, 1, 256)},
            ]

            # Mock LightGlue matcher
            mock_lg_inst = mock.MagicMock()
            mock_lg.return_value.eval.return_value.to.return_value = mock_lg_inst
            mock_lg_inst.return_value = {
                "matches": torch.tensor([[[0, 0]]]),
                "scores": torch.tensor([[0.9]]),
            }

            # rbd strips batch dim
            def batchman(d):
                out = {}
                for k, v in d.items():
                    if isinstance(v, torch.Tensor) and v.ndim >= 1:
                        out[k] = v[0]
                    else:
                        out[k] = v
                return out
            mock_rbd.side_effect = batchman

            result = lg_mod.match_lightglue(view, device="cpu")

            assert result.method == "lightglue"
            result.validate()
            # Canvas: x = 50 + 10/0.5 = 70, y = 30 + 20/0.5 = 70
            assert result.ref_xy[0, 0] == pytest.approx(70.0)
            assert result.ref_xy[0, 1] == pytest.approx(70.0)

    def test_registers_in_runner(self):
        from src.registration_benchmark.matchers.lightglue import is_lightglue_available
        avail = is_lightglue_available()
        assert isinstance(avail, bool)

    def test_unavailable_raises_runtime_error(self, view):
        from src.registration_benchmark.matchers.lightglue import (
            _LIGHTGLUE_AVAILABLE,
            match_lightglue,
        )
        if not _LIGHTGLUE_AVAILABLE:
            with pytest.raises(RuntimeError, match="LightGlue is not available"):
                match_lightglue(view)
        else:
            pytest.skip("LightGlue is installed — run integration test instead")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestLightGlueAdapterIntegration:
    """Integration tests (only run when lightglue is actually available)."""

    @pytest.fixture
    def view(self):
        return _make_view(with_invalid=False)

    def test_smoke_on_small_image(self, view):
        """Run LightGlue on a 128×128 image — smoke test only."""
        from src.registration_benchmark.matchers.lightglue import (
            _LIGHTGLUE_AVAILABLE,
            match_lightglue,
        )
        if not _LIGHTGLUE_AVAILABLE:
            pytest.skip("LightGlue not available")

        try:
            result = match_lightglue(view, device="cpu", max_num_keypoints=512)
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("ssl", "certificate", "download",
                                       "connection", "network",
                                       "bad gateway", "502", "timeout")):
                pytest.skip(f"LightGlue real inference blocked (network): {e}")
            raise

        assert isinstance(result, MatchSet)
        assert result.method == "lightglue"
        result.validate()
        assert "confidence_source" in result.metadata