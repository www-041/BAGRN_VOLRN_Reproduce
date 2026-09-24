"""Task 3 tests for the official EfficientLoFTR adapter contract."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from src.registration_benchmark.models import MatchView, PAIR_COMMON_GRID_FRAME
from src.registration_benchmark.matchers.efficient_loftr import (
    EFFICIENT_LOFTR_UNAVAILABLE,
    _model_to_view_coordinates,
    _install_kornia_grid_compat,
    _install_pytorch_lightning_compat,
    _load_checkpoint,
    _resolve_paths,
    is_efficient_loftr_available,
    match_efficient_loftr,
)


def _view() -> MatchView:
    return MatchView(
        ref=np.zeros((12, 16), dtype=np.float32),
        tgt=np.zeros((12, 16), dtype=np.float32),
        ref_valid=np.ones((12, 16), dtype=bool),
        tgt_valid=np.ones((12, 16), dtype=bool),
        origin_x=100.0,
        origin_y=200.0,
        scale_x=0.5,
        scale_y=0.25,
    )


def test_model_resize_coordinates_are_inverted_before_common_grid_mapping():
    model_ref = np.array([[4.0, 3.0], [0.0, 0.0]])
    model_tgt = np.array([[3.0, 2.0], [1.0, 1.0]])

    ref_view = _model_to_view_coordinates(model_ref, original_shape=(12, 16), model_shape=(6, 8))
    tgt_view = _model_to_view_coordinates(model_tgt, original_shape=(12, 16), model_shape=(6, 8))

    np.testing.assert_allclose(ref_view, [[8.0, 6.0], [0.0, 0.0]])
    np.testing.assert_allclose(tgt_view, [[6.0, 4.0], [2.0, 2.0]])

    view = _view()
    np.testing.assert_allclose(view.to_common_grid(ref_view), [[116.0, 224.0], [100.0, 200.0]])
    np.testing.assert_allclose(view.to_common_grid(tgt_view), [[112.0, 216.0], [104.0, 208.0]])


def test_efficient_loftr_unavailable_is_explicit_and_never_falls_back(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError(f"{EFFICIENT_LOFTR_UNAVAILABLE}: test dependency missing")

    monkeypatch.setattr(
        "src.registration_benchmark.matchers.efficient_loftr._load_model",
        unavailable,
    )

    with pytest.raises(RuntimeError, match=EFFICIENT_LOFTR_UNAVAILABLE):
        match_efficient_loftr(_view(), device="cpu")


def test_contract_metadata_declares_common_grid_and_internal_confidence():
    from src.registration_benchmark.models import make_matchset_from_view

    matches = make_matchset_from_view(
        method="efficient_loftr",
        view=_view(),
        ref_xy_view=np.array([[1.0, 2.0]]),
        tgt_xy_view=np.array([[1.5, 2.5]]),
        confidence=np.array([0.9]),
        runtime_sec=0.25,
    )

    matches.validate_for_geometry()
    assert matches.metadata["coordinate_frame"] == PAIR_COMMON_GRID_FRAME
    assert matches.metadata["confidence_semantics"] == "method_internal_only"


def test_mocked_inference_smoke_emits_common_grid_matchset(monkeypatch, tmp_path):
    import torch

    class FakeModel:
        def __call__(self, batch):
            batch["mkpts0_f"] = torch.tensor([[32.0, 32.0]])
            batch["mkpts1_f"] = torch.tensor([[31.0, 30.0]])
            batch["mconf"] = torch.tensor([0.8])
            return None

    repo = tmp_path / "EfficientLoFTR"
    (repo / "src" / "loftr").mkdir(parents=True)
    weights = repo / "weights" / "eloftr_outdoor.ckpt"
    weights.parent.mkdir()
    weights.write_bytes(b"test")
    view = MatchView(
        ref=np.zeros((70, 90), dtype=np.float32),
        tgt=np.zeros((70, 90), dtype=np.float32),
        ref_valid=np.ones((70, 90), dtype=bool),
        tgt_valid=np.ones((70, 90), dtype=bool),
        origin_x=100.0,
        origin_y=200.0,
        scale_x=1.0,
        scale_y=1.0,
    )
    monkeypatch.setattr(
        "src.registration_benchmark.matchers.efficient_loftr._load_model",
        lambda **kwargs: FakeModel(),
    )

    matches = match_efficient_loftr(view, device="cpu", repo_dir=repo)

    matches.validate_for_geometry()
    np.testing.assert_allclose(matches.ref_xy, [[145.0, 235.0]])
    np.testing.assert_allclose(matches.tgt_xy, [[143.59375, 232.8125]])
    assert matches.metadata["output_coordinate_frame"] == "resized_model_input"
    assert matches.metadata["resize_inverse_applied"] is True
    assert matches.metadata["runtime_breakdown"]["matcher_runtime_sec"] >= 0.0


def test_availability_probe_does_not_claim_missing_model_is_available():
    assert isinstance(is_efficient_loftr_available(), bool)


def test_efficient_loftr_kornia_grid_import_compatibility():
    _install_kornia_grid_compat()

    from kornia.utils.grid import create_meshgrid

    assert callable(create_meshgrid)


def test_efficient_loftr_lightning_utility_import_compatibility():
    _install_pytorch_lightning_compat()

    from pytorch_lightning.utilities import rank_zero_only

    assert rank_zero_only.rank == 0


def test_efficient_loftr_official_import_compatibility():
    repo_value = os.environ.get("EFFICIENT_LOFTR_REPO")
    if not repo_value:
        pytest.skip("EFFICIENT_LOFTR_REPO is not configured")

    try:
        repo, _ = _resolve_paths(repo_value, None)
    except RuntimeError:
        pytest.skip("configured EfficientLoFTR paths are unavailable")
    repo_src = repo / "src"
    if not (repo_src / "loftr").is_dir():
        pytest.skip("configured EfficientLoFTR source tree is unavailable")

    _install_kornia_grid_compat()
    import src as project_src

    if str(repo_src) not in project_src.__path__:
        project_src.__path__.insert(0, str(repo_src))
    sys.modules.pop("src.loftr", None)

    from src.loftr import LoFTR

    assert LoFTR is not None


def test_weights_directory_alias_resolves_to_official_repository_root(
    tmp_path, monkeypatch
):
    # The real-user environment may intentionally define a checkpoint path;
    # this unit test must exercise the weights-directory default in isolation.
    monkeypatch.delenv("EFFICIENT_LOFTR_WEIGHTS", raising=False)
    repo = tmp_path / "EfficientLoFTR"
    (repo / "src" / "loftr").mkdir(parents=True)
    weights_dir = repo / "weights"
    weights_dir.mkdir()
    checkpoint = weights_dir / "eloftr_outdoor.ckpt"
    checkpoint.write_bytes(b"test")

    resolved_repo, resolved_weights = _resolve_paths(str(weights_dir), None)

    assert resolved_repo == repo.resolve()
    assert resolved_weights == checkpoint.resolve()


def test_trusted_official_checkpoint_load_disables_weights_only(tmp_path):
    checkpoint = tmp_path / "eloftr_outdoor.ckpt"
    checkpoint.write_bytes(b"test")
    calls = []

    class FakeTorch:
        @staticmethod
        def load(path, *, map_location, weights_only):
            calls.append((path, map_location, weights_only))
            return {"state_dict": {}}

    loaded = _load_checkpoint(FakeTorch(), checkpoint)

    assert loaded == {"state_dict": {}}
    assert calls == [(str(checkpoint), "cpu", False)]
