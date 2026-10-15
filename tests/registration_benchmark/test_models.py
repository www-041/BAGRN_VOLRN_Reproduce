"""Tests for :mod:`src.registration_benchmark.models`."""

import numpy as np
import pytest

from src.registration_benchmark.models import CommonGridPair, MatchSet, MatchView


class TestMatchSet:
    """Unit tests for :class:`MatchSet`."""

    @staticmethod
    def _make_valid(n: int = 10) -> MatchSet:
        rng = np.random.default_rng(42)
        return MatchSet(
            method="test",
            ref_xy=rng.uniform(0, 512, size=(n, 2)),
            tgt_xy=rng.uniform(0, 512, size=(n, 2)),
            confidence=rng.uniform(0, 1, size=n),
            runtime_sec=1.23,
        )

    def test_accepts_valid_arrays(self) -> None:
        ms = self._make_valid()
        ms.validate()  # must not raise

    def test_rejects_shape_mismatch(self) -> None:
        ms = self._make_valid()
        ms.tgt_xy = np.zeros((5, 2))  # different N
        with pytest.raises(ValueError, match="tgt_xy must match"):
            ms.validate()

    def test_rejects_non_2d_ref(self) -> None:
        ms = self._make_valid()
        ms.ref_xy = np.zeros(10)  # 1-D
        with pytest.raises(ValueError, match="ref_xy must have shape"):
            ms.validate()

    def test_rejects_non_2d_tgt(self) -> None:
        ms = self._make_valid()
        ms.tgt_xy = np.zeros(10)  # 1-D
        with pytest.raises(ValueError, match="tgt_xy must match"):
            ms.validate()

    def test_rejects_wrong_confidence_shape(self) -> None:
        ms = self._make_valid()
        ms.confidence = np.ones(5)
        with pytest.raises(ValueError, match="confidence must have shape"):
            ms.validate()

    def test_rejects_nan_in_ref(self) -> None:
        ms = self._make_valid()
        ms.ref_xy[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            ms.validate()

    def test_rejects_inf_in_tgt(self) -> None:
        ms = self._make_valid()
        ms.tgt_xy[2, 1] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            ms.validate()

    def test_zero_matches_valid(self) -> None:
        ms = MatchSet(
            method="empty",
            ref_xy=np.empty((0, 2)),
            tgt_xy=np.empty((0, 2)),
            confidence=np.empty(0),
            runtime_sec=0.0,
        )
        ms.validate()  # must not raise

    def test_metadata_default(self) -> None:
        ms = self._make_valid()
        assert ms.metadata == {}

    def test_metadata_persisted(self) -> None:
        ms = self._make_valid()
        ms.metadata["key"] = 42
        assert ms.metadata["key"] == 42


class TestMatchView:
    """Unit tests for :class:`MatchView` coordinate mapping."""

    @staticmethod
    def _make_view() -> MatchView:
        return MatchView(
            ref=np.zeros((256, 256), dtype=np.float32),
            tgt=np.zeros((256, 256), dtype=np.float32),
            ref_valid=np.ones((256, 256), dtype=bool),
            tgt_valid=np.ones((256, 256), dtype=bool),
            origin_x=100.0,
            origin_y=50.0,
            scale_x=0.5,
            scale_y=0.5,
        )

    def test_to_canvas_identity(self) -> None:
        """When origin=0 and scale=1, coordinates should be unchanged."""
        view = MatchView(
            ref=np.zeros((10, 10), dtype=np.float32),
            tgt=np.zeros((10, 10), dtype=np.float32),
            ref_valid=np.ones((10, 10), dtype=bool),
            tgt_valid=np.ones((10, 10), dtype=bool),
            origin_x=0.0,
            origin_y=0.0,
            scale_x=1.0,
            scale_y=1.0,
        )
        xy = np.array([[3.0, 7.0], [128.0, 256.0]])
        result = view.to_canvas(xy)
        np.testing.assert_allclose(result, xy, atol=1e-10)

    def test_to_canvas_with_offset_and_scale(self) -> None:
        view = self._make_view()
        xy = np.array([[0.0, 0.0], [128.0, 256.0]])
        result = view.to_canvas(xy)
        expected = np.array([
            [100.0 + 0.0 / 0.5, 50.0 + 0.0 / 0.5],
            [100.0 + 128.0 / 0.5, 50.0 + 256.0 / 0.5],
        ])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_to_canvas_rejects_1d(self) -> None:
        view = self._make_view()
        with pytest.raises(ValueError, match="must have shape"):
            view.to_canvas(np.array([1.0, 2.0]))

    def test_to_canvas_rejects_3col(self) -> None:
        view = self._make_view()
        with pytest.raises(ValueError, match="must have shape"):
            view.to_canvas(np.zeros((5, 3)))