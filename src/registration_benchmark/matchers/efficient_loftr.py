"""Official EfficientLoFTR adapter.

This adapter intentionally loads the official ``zju3dv/EfficientLoFTR``
repository from an explicit local checkout.  It does not silently substitute
Kornia LoFTR or any other matcher when the dependency or checkpoint is absent.

The official inference path accepts grayscale ``(N, 1, H, W)`` tensors in
``[0, 1]`` and resizes each image down to dimensions divisible by 32.  Its
``mkpts*_f`` outputs are therefore in the resized model-input frame.  This
module reverses that resize before applying the common-grid crop/scale mapping.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.registration_benchmark.models import MatchView, make_matchset_from_view

EFFICIENT_LOFTR_UNAVAILABLE = "EFFICIENT_LOFTR_UNAVAILABLE"
OFFICIAL_REPOSITORY = "https://github.com/zju3dv/EfficientLoFTR"
DEFAULT_WEIGHTS_NAME = "eloftr_outdoor.ckpt"
_MULTIPLE = 32


def is_efficient_loftr_available(
    repo_dir: str | os.PathLike[str] | None = None,
    weights_path: str | os.PathLike[str] | None = None,
) -> bool:
    """Return whether the explicitly configured official model is available."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False

    try:
        repo, weights = _resolve_paths(repo_dir, weights_path)
    except RuntimeError:
        return False

    return (repo / "src" / "loftr").is_dir() and weights.is_file()


def match_efficient_loftr(
    view: MatchView,
    device: str = "auto",
    confidence_threshold: float = 0.0,
    repo_dir: str | os.PathLike[str] | None = None,
    weights_path: str | os.PathLike[str] | None = None,
    model_type: str = "full",
    precision: str = "fp32",
):
    """Run official EfficientLoFTR and return common-grid correspondences.

    ``model_type`` is ``"full"`` by default, matching the official outdoor
    checkpoint.  ``precision`` is recorded for diagnostics; fp32 is the
    conservative default for this first integration.
    """
    if model_type not in {"full", "opt"}:
        raise ValueError("model_type must be 'full' or 'opt'")
    if precision not in {"fp32", "mp", "fp16"}:
        raise ValueError("precision must be 'fp32', 'mp', or 'fp16'")

    t0 = time.perf_counter()
    resolved_device = _resolve_device(device)
    model = _load_model(
        device=resolved_device,
        repo_dir=repo_dir,
        weights_path=weights_path,
        model_type=model_type,
        precision=precision,
    )

    import torch

    ref_model, ref_transform = _prepare_model_input(view.ref)
    tgt_model, tgt_transform = _prepare_model_input(view.tgt)
    ref_tensor = torch.from_numpy(ref_model)[None, None].to(resolved_device)
    tgt_tensor = torch.from_numpy(tgt_model)[None, None].to(resolved_device)
    if precision == "fp16":
        ref_tensor = ref_tensor.half()
        tgt_tensor = tgt_tensor.half()

    batch = {"image0": ref_tensor, "image1": tgt_tensor}
    with torch.no_grad():
        if precision == "mp" and resolved_device.startswith("cuda"):
            with torch.autocast(device_type="cuda"):
                returned = model(batch)
        else:
            returned = model(batch)

    output = returned if isinstance(returned, dict) else batch
    try:
        mkpts0_model = _to_numpy(output["mkpts0_f"])
        mkpts1_model = _to_numpy(output["mkpts1_f"])
        confidence = _to_numpy(output["mconf"]).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{EFFICIENT_LOFTR_UNAVAILABLE}: official output is missing "
            f"mkpts0_f/mkpts1_f/mconf ({exc})"
        ) from exc

    if mkpts0_model.shape != mkpts1_model.shape or mkpts0_model.ndim != 2 or mkpts0_model.shape[1] != 2:
        raise RuntimeError(
            f"{EFFICIENT_LOFTR_UNAVAILABLE}: invalid output coordinate shapes "
            f"{mkpts0_model.shape} and {mkpts1_model.shape}"
        )
    if confidence.shape != (len(mkpts0_model),):
        raise RuntimeError(
            f"{EFFICIENT_LOFTR_UNAVAILABLE}: mconf shape {confidence.shape} "
            f"does not match {len(mkpts0_model)} matches"
        )

    # Official mkpts*_f coordinates are in each resized model-input frame.
    ref_view = _model_to_view_coordinates(
        mkpts0_model, original_shape=view.ref.shape, model_shape=ref_transform["model_shape"]
    )
    tgt_view = _model_to_view_coordinates(
        mkpts1_model, original_shape=view.tgt.shape, model_shape=tgt_transform["model_shape"]
    )

    keep = confidence >= confidence_threshold
    keep &= _valid_match_mask(ref_view, tgt_view, view)
    ref_view = ref_view[keep]
    tgt_view = tgt_view[keep]
    confidence = confidence[keep]

    elapsed = time.perf_counter() - t0
    metadata = {
        "official_repository": OFFICIAL_REPOSITORY,
        "model_variant": model_type,
        "pretrained_weights": DEFAULT_WEIGHTS_NAME,
        "weights_path": str(_resolve_paths(repo_dir, weights_path)[1]),
        "precision": precision,
        "device": resolved_device,
        "confidence_threshold": confidence_threshold,
        "confidence_source": "official_mconf",
        "confidence_semantics": "method_internal_only",
        "output_coordinate_frame": "resized_model_input",
        "resize_policy": "floor_each_dimension_to_multiple_of_32",
        "ref_input_transform": ref_transform,
        "tgt_input_transform": tgt_transform,
        "resize_inverse_applied": True,
        "padding": {"ref": [0, 0, 0, 0], "tgt": [0, 0, 0, 0]},
    }
    return make_matchset_from_view(
        method="efficient_loftr",
        view=view,
        ref_xy_view=ref_view,
        tgt_xy_view=tgt_view,
        confidence=confidence,
        runtime_sec=elapsed,
        metadata=metadata,
        runtime_breakdown={
            "feature_runtime_sec": 0.0,
            "matcher_runtime_sec": elapsed,
        },
    )


def _model_to_view_coordinates(
    xy_model: np.ndarray,
    *,
    original_shape: tuple[int, int],
    model_shape: tuple[int, int],
) -> np.ndarray:
    """Undo official resize: model-frame ``[x,y]`` -> original view ``[x,y]``."""
    xy = np.asarray(xy_model, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy_model must have shape (N, 2)")
    original_h, original_w = original_shape
    model_h, model_w = model_shape
    if original_h <= 0 or original_w <= 0 or model_h <= 0 or model_w <= 0:
        raise ValueError("image shapes must be positive")
    out = xy.copy()
    out[:, 0] *= float(original_w) / float(model_w)
    out[:, 1] *= float(original_h) / float(model_h)
    return out


def _prepare_model_input(image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the official floor-to-32 resize and record its inverse mapping."""
    image = np.asarray(image, dtype=np.float32)
    original_h, original_w = image.shape
    model_h = (original_h // _MULTIPLE) * _MULTIPLE
    model_w = (original_w // _MULTIPLE) * _MULTIPLE
    if model_h < _MULTIPLE or model_w < _MULTIPLE:
        raise RuntimeError(
            f"{EFFICIENT_LOFTR_UNAVAILABLE}: overlap view must be at least 32x32"
        )
    if (model_h, model_w) == (original_h, original_w):
        resized = image
    else:
        resized = cv2.resize(image, (model_w, model_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32, copy=False), {
        "original_shape": [int(original_h), int(original_w)],
        "model_shape": [int(model_h), int(model_w)],
        "scale_x_model_per_original": float(model_w / original_w),
        "scale_y_model_per_original": float(model_h / original_h),
        "resize_inverse": "x_original=x_model/scale_x, y_original=y_model/scale_y",
    }


def _valid_match_mask(ref_xy: np.ndarray, tgt_xy: np.ndarray, view: MatchView) -> np.ndarray:
    keep = np.ones(len(ref_xy), dtype=bool)
    for i, (ref_point, tgt_point) in enumerate(zip(ref_xy, tgt_xy)):
        for point, valid in ((ref_point, view.ref_valid), (tgt_point, view.tgt_valid)):
            x, y = int(round(point[0])), int(round(point[1]))
            if not (0 <= y < valid.shape[0] and 0 <= x < valid.shape[1] and valid[y, x]):
                keep[i] = False
                break
    return keep


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _resolve_device(request: str) -> str:
    if request == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return request


def _resolve_paths(
    repo_dir: str | os.PathLike[str] | None,
    weights_path: str | os.PathLike[str] | None,
) -> tuple[Path, Path]:
    repo_value = repo_dir or os.environ.get("EFFICIENT_LOFTR_REPO")
    if not repo_value:
        raise RuntimeError(f"{EFFICIENT_LOFTR_UNAVAILABLE}: set EFFICIENT_LOFTR_REPO")
    repo_candidate = Path(repo_value).expanduser().resolve()
    # Accept the user's convenient ``...\EfficientLoFTR\weights`` setting
    # while keeping the official repository root for importing ``src\loftr``.
    if (
        repo_candidate.name.lower() == "weights"
        and (repo_candidate.parent / "src" / "loftr").is_dir()
    ):
        repo = repo_candidate.parent
        default_weights = repo_candidate / DEFAULT_WEIGHTS_NAME
    else:
        repo = repo_candidate
        default_weights = repo / "weights" / DEFAULT_WEIGHTS_NAME
    weight_value = weights_path or os.environ.get("EFFICIENT_LOFTR_WEIGHTS")
    weights = Path(weight_value).expanduser().resolve() if weight_value else default_weights
    return repo, weights


def _load_model(
    *,
    device: str,
    repo_dir: str | os.PathLike[str] | None,
    weights_path: str | os.PathLike[str] | None,
    model_type: str,
    precision: str,
):
    """Load official code and weights, wrapping all setup failures explicitly."""
    try:
        import torch

        repo, weights = _resolve_paths(repo_dir, weights_path)
        if not (repo / "src" / "loftr").is_dir():
            raise FileNotFoundError(f"official repository source not found: {repo}")
        if not weights.is_file():
            raise FileNotFoundError(f"pretrained weights not found: {weights}")

        import src as project_src
        repo_src = str(repo / "src")
        if repo_src not in project_src.__path__:
            project_src.__path__.insert(0, repo_src)
        _install_kornia_grid_compat()
        _install_pytorch_lightning_compat()
        from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

        config = copy.deepcopy(full_default_cfg if model_type == "full" else opt_default_cfg)
        matcher = LoFTR(config=config)
        checkpoint = _load_checkpoint(torch, weights)
        state_dict = checkpoint.get("state_dict", checkpoint)
        matcher.load_state_dict(state_dict)
        matcher = reparameter(matcher).eval().to(device)
        if precision == "fp16":
            matcher = matcher.half()
        return matcher
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith(EFFICIENT_LOFTR_UNAVAILABLE):
            raise
        raise RuntimeError(f"{EFFICIENT_LOFTR_UNAVAILABLE}: {exc}") from exc


def _load_checkpoint(torch_module, weights: Path):
    """Load the explicitly configured official checkpoint.

    The official EfficientLoFTR Lightning checkpoint contains non-tensor
    training metadata.  PyTorch 2.6 changed ``torch.load``'s default to
    ``weights_only=True``, which rejects that metadata before the model can
    extract its ``state_dict``.  ``weights_only=False`` is intentional here:
    this adapter only loads the user-selected official checkpoint path.  The
    fallback keeps the adapter compatible with older PyTorch releases that do
    not accept the keyword.
    """
    try:
        return torch_module.load(
            str(weights), map_location="cpu", weights_only=False
        )
    except TypeError:
        return torch_module.load(str(weights), map_location="cpu")


def _install_kornia_grid_compat() -> None:
    """Expose EfficientLoFTR's old Kornia grid import on newer Kornia.

    EfficientLoFTR imports ``create_meshgrid`` from ``kornia.utils.grid``;
    newer Kornia exposes the same function from ``kornia.geometry.grid``.
    Keep the compatibility shim local to this adapter so the shared LoFTR,
    LightGlue, and project-wide Kornia environment is not downgraded or
    globally modified outside an EfficientLoFTR load.
    """
    try:
        from kornia.utils.grid import create_meshgrid  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError):
        from kornia.geometry.grid import create_meshgrid

    import types

    compat_module = types.ModuleType("kornia.utils.grid")
    compat_module.create_meshgrid = create_meshgrid
    sys.modules["kornia.utils.grid"] = compat_module


def _install_pytorch_lightning_compat() -> None:
    """Provide EfficientLoFTR's tiny Lightning utility dependency if absent.

    The official inference source only needs ``rank_zero_only`` from the old
    Lightning utility module. Avoid installing the historical Lightning 1.x
    stack into the shared registration environment just for that symbol.
    """
    try:
        from pytorch_lightning.utilities import rank_zero_only  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import types

    def rank_zero_only(function=None):
        if function is None:
            return lambda wrapped: wrapped
        return function

    rank_zero_only.rank = 0
    utilities_module = types.ModuleType("pytorch_lightning.utilities")
    utilities_module.rank_zero_only = rank_zero_only
    lightning_module = types.ModuleType("pytorch_lightning")
    lightning_module.utilities = utilities_module
    sys.modules["pytorch_lightning"] = lightning_module
    sys.modules["pytorch_lightning.utilities"] = utilities_module
