import sys
from types import SimpleNamespace

import numpy as np

from src.multiscene_sift import pairwise
from src.multiscene_sift.models import OverlapEdge, PairwiseRegistration
from src.registration_benchmark.models import MatchView


def _view() -> MatchView:
    image = np.zeros((64, 64), dtype=np.float32)
    return MatchView(
        ref=image,
        tgt=image.copy(),
        ref_valid=np.ones((64, 64), dtype=bool),
        tgt_valid=np.ones((64, 64), dtype=bool),
        origin_x=0.0,
        origin_y=0.0,
        scale_x=1.0,
        scale_y=1.0,
    )


def test_loftr_session_constructs_model_once_and_closes_safely(monkeypatch):
    import torch
    from src.registration_benchmark.matchers import loftr

    calls = {"construct": 0, "infer": 0}

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, batch):
            calls["infer"] += 1
            return {
                "keypoints0": torch.tensor([[10.0, 10.0]]),
                "keypoints1": torch.tensor([[10.0, 10.0]]),
                "confidence": torch.tensor([0.9]),
            }

    def make_model(*args, **kwargs):
        calls["construct"] += 1
        return FakeModel()

    monkeypatch.setattr(loftr, "_LOFTR_AVAILABLE", True)
    monkeypatch.setattr(loftr.KF, "LoFTR", make_model)

    session = loftr.LoFTRMatcherSession(device="cpu")
    first = session.match(_view())
    second = session.match(_view())
    uncached = loftr.match_loftr(
        _view(), device="cpu", _matcher=FakeModel(), _model_init_runtime_sec=0.0
    )
    session.close()
    session.close()

    assert calls == {"construct": 1, "infer": 3}
    np.testing.assert_array_equal(first.ref_xy, uncached.ref_xy)
    np.testing.assert_array_equal(first.tgt_xy, uncached.tgt_xy)
    np.testing.assert_array_equal(first.confidence, uncached.confidence)
    assert first.metadata["runtime_breakdown"]["model_init_runtime_sec"] > 0.0
    assert second.metadata["runtime_breakdown"]["model_init_runtime_sec"] == 0.0
    assert first.metadata["runtime_breakdown"]["pair_inference_runtime_sec"] >= 0.0
    assert second.metadata["runtime_breakdown"]["pair_inference_runtime_sec"] >= 0.0


def test_run_all_pairs_reuses_one_session_and_closes_it(monkeypatch, tmp_path):
    seen = []
    closed = []

    class FakeSession:
        def close(self):
            closed.append(self)

    session = FakeSession()
    monkeypatch.setattr(pairwise, "_create_matcher_session", lambda *args, **kwargs: session, raising=False)

    def fake_register(*args, matcher_session=None, **kwargs):
        seen.append(matcher_session)
        return PairwiseRegistration(
            idx_i=0,
            idx_j=1,
            status="NO_OVERLAP",
            raw_matches=0,
            inliers=0,
            inlier_ratio=0.0,
            coverage=0.0,
            residual_median=float("nan"),
            residual_rmse=float("nan"),
            residual_p95=float("nan"),
            pair_pixel_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            pair_common_transform=None,
            runtime_sec=0.0,
            matcher="sift",
        )

    monkeypatch.setattr(pairwise, "register_pair", fake_register)
    monkeypatch.setattr(pairwise, "save_pairwise_summary", lambda *args, **kwargs: None)
    edges = [
        OverlapEdge(0, 1, 1.0, 0.5, 0.5),
        OverlapEdge(0, 2, 1.0, 0.5, 0.5),
    ]

    scenes = [type("SceneStub", (), {"name": f"scene-{i}"})() for i in range(3)]
    pairwise.run_all_pairs(scenes, edges, tmp_path, matcher="sift")

    assert seen == [session, session]
    assert closed == [session]


def test_pairwise_peak_reset_hook_exists_for_each_pair():
    assert callable(pairwise._reset_peak_gpu_memory_stats)


def test_pairwise_gpu_peak_helpers_reset_before_each_pair(monkeypatch):
    class FakeCuda:
        def __init__(self):
            self.reset_calls = 0
            self.peaks = iter((8.0 * 1024.0**2, 13.0 * 1024.0**2))

        def is_available(self):
            return True

        def reset_peak_memory_stats(self):
            self.reset_calls += 1

        def max_memory_allocated(self):
            return next(self.peaks)

    fake_cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    pairwise._reset_peak_gpu_memory_stats("cuda")
    first = pairwise._peak_gpu_memory_mb("cuda")
    pairwise._reset_peak_gpu_memory_stats("cuda")
    second = pairwise._peak_gpu_memory_mb("cuda")

    assert fake_cuda.reset_calls == 2
    assert first == 8.0
    assert second == 13.0
