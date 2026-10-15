"""Environment, input, and resource diagnostics for Task15 preflight."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

from src.multiscene_sift.b9_dataset import discover_b9_scenes


HARD_STOP_ENVIRONMENT_INVALID = "HARD_STOP_ENVIRONMENT_INVALID"
HARD_STOP_MATCHER_ASSET_MISSING = "HARD_STOP_MATCHER_ASSET_MISSING"
HARD_STOP_INPUT_INVALID = "HARD_STOP_INPUT_INVALID"


def resource_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import psutil

        snapshot["rss_mb"] = float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        snapshot["rss_mb"] = "NOT_MEASURED"
    try:
        import torch

        snapshot["torch_version"] = torch.__version__
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        snapshot["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception:
        snapshot["torch_version"] = "NOT_MEASURED"
        snapshot["cuda_available"] = False
        snapshot["cuda_device_count"] = 0
    return snapshot


def discover_selected_records(config) -> list[dict[str, Any]]:
    try:
        records = discover_b9_scenes(config.dataset.input_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(f"{HARD_STOP_INPUT_INVALID}: {exc}") from exc
    if config.dataset.scene_selection is not None:
        try:
            records = [records[index] for index in config.dataset.scene_selection]
        except IndexError as exc:
            raise RuntimeError(f"{HARD_STOP_INPUT_INVALID}: invalid scene_selection {exc}") from exc
    if len(records) != config.dataset.expected_scene_count:
        raise RuntimeError(
            f"{HARD_STOP_INPUT_INVALID}: expected {config.dataset.expected_scene_count} scenes, found {len(records)}"
        )
    if config.dataset.expected_scene_ids is not None:
        actual_ids = tuple(str(record.get("scene_id")) for record in records)
        if actual_ids != tuple(config.dataset.expected_scene_ids):
            raise RuntimeError(
                f"{HARD_STOP_INPUT_INVALID}: expected scene ids {tuple(config.dataset.expected_scene_ids)!r}, found {actual_ids!r}"
            )
    for record in records:
        if record.get("crs") != config.canonical_grid.crs:
            raise RuntimeError(f"{HARD_STOP_INPUT_INVALID}: {record['scene_id']} CRS mismatch")
        if abs(float(record.get("resolution_x_m", 0.0)) - config.canonical_grid.resolution_m) > 1e-6:
            raise RuntimeError(f"{HARD_STOP_INPUT_INVALID}: {record['scene_id']} x resolution mismatch")
        if abs(float(record.get("resolution_y_m", 0.0)) - config.canonical_grid.resolution_m) > 1e-6:
            raise RuntimeError(f"{HARD_STOP_INPUT_INVALID}: {record['scene_id']} y resolution mismatch")
        if not Path(record["b9_path"]).is_file() or not Path(record["mtl_path"]).is_file():
            raise RuntimeError(f"{HARD_STOP_INPUT_INVALID}: {record['scene_id']} source artifact missing")
    return records


def validate_environment(config) -> None:
    if sys.version_info[:2] < (3, 11):
        raise RuntimeError(f"{HARD_STOP_ENVIRONMENT_INVALID}: Python 3.11+ required")
    if not config.registration.checkpoint.is_file() or config.registration.matcher_repo is None or not config.registration.matcher_repo.is_dir():
        raise RuntimeError(f"{HARD_STOP_MATCHER_ASSET_MISSING}: official EfficientLoFTR checkout/checkpoint missing")
    try:
        import cv2  # noqa: F401
        import rasterio  # noqa: F401
        import scipy  # noqa: F401
        import shapely  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"{HARD_STOP_ENVIRONMENT_INVALID}: required dependency missing: {exc}") from exc
