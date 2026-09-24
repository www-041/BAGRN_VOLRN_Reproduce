"""Task 4 tests for the frozen LightGlue + DISK comparator."""

from __future__ import annotations

import numpy as np
import pytest

from src.registration_benchmark.models import MatchView
from src.registration_benchmark.matchers.lightglue_disk import (
    LIGHTGLUE_DISK_UNAVAILABLE,
    match_lightglue_disk,
)


def _view() -> MatchView:
    return MatchView(
        ref=np.zeros((40, 50), dtype=np.float32),
        tgt=np.zeros((40, 50), dtype=np.float32),
        ref_valid=np.ones((40, 50), dtype=bool),
        tgt_valid=np.ones((40, 50), dtype=bool),
        origin_x=100.0,
        origin_y=200.0,
        scale_x=0.5,
        scale_y=0.25,
    )


def test_disk_coordinates_are_native_extractor_input_coordinates(monkeypatch):
    import torch

    class FakeDisk:
        def __init__(self, **kwargs):
            pass

        def eval(self):
            return self

        def to(self, device):
            return self

        def extract(self, image, **kwargs):
            return {
                "keypoints": torch.tensor([[[10.0, 20.0]]]),
                "descriptors": torch.zeros((1, 1, 128)),
                "image_size": torch.tensor([[50.0, 40.0]]),
            }

    class FakeLightGlue:
        def __init__(self, **kwargs):
            pass

        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, data):
            return {
                "matches": torch.tensor([[[0, 0]]]),
                "scores": torch.tensor([[0.82]]),
            }

    monkeypatch.setattr("src.registration_benchmark.matchers.lightglue_disk._LIGHTGLUE_DISK_AVAILABLE", True)
    monkeypatch.setattr("src.registration_benchmark.matchers.lightglue_disk.DISK", FakeDisk)
    monkeypatch.setattr("src.registration_benchmark.matchers.lightglue_disk.LightGlue", FakeLightGlue)

    matches = match_lightglue_disk(_view(), device="cpu")

    matches.validate_for_geometry()
    np.testing.assert_allclose(matches.ref_xy, [[120.0, 280.0]])
    np.testing.assert_allclose(matches.tgt_xy, [[120.0, 280.0]])
    assert matches.metadata["output_coordinate_frame"] == "disk_extractor_input"
    assert matches.metadata["disk_internal_resize"] == 1024
    assert matches.metadata["confidence_semantics"] == "method_internal_only"
    assert matches.metadata["runtime_breakdown"]["feature_runtime_sec"] >= 0.0
    assert matches.metadata["runtime_breakdown"]["matcher_runtime_sec"] >= 0.0


def test_missing_disk_stack_is_explicit_and_never_falls_back(monkeypatch):
    monkeypatch.setattr(
        "src.registration_benchmark.matchers.lightglue_disk._LIGHTGLUE_DISK_AVAILABLE",
        False,
    )

    with pytest.raises(RuntimeError, match=LIGHTGLUE_DISK_UNAVAILABLE):
        match_lightglue_disk(_view(), device="cpu")
