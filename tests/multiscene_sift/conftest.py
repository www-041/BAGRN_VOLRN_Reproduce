"""Fixtures for multiscene_sift tests."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

# ---------------------------------------------------------------------------
# Synthetic test data helpers
# ---------------------------------------------------------------------------

RESOLUTION = 30.0  # metres per pixel
IMAGE_SIZE = 128   # pixels


def _make_band(
    path: Path,
    crs: str = "EPSG:32650",
    origin_x: float = 500000.0,
    origin_y: float = 4000000.0,
    width: int = IMAGE_SIZE,
    height: int = IMAGE_SIZE,
    dtype: str = "uint16",
) -> None:
    """Write a synthetic GeoTIFF with random texture."""
    rng = np.random.default_rng(abs(hash(str(path))) % (2**31))
    data = rng.integers(0, 65535, size=(height, width)).astype(dtype)
    transform = from_origin(origin_x, origin_y, RESOLUTION, RESOLUTION)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def make_five_scene_dataset(root: Path) -> list[str]:
    """Create a synthetic 5-scene dataset and return scene names.

    Scenes arranged in a 2-row layout so each overlaps neighbouring scenes:

        [0]──[1]──[2]
               │
        [3]──[4]

    Scene names follow DZ01V naming convention.
    """
    scene_names = [
        "DZ01V_L2_E113.0_N36.4_20260222031837_01_T1",
        "DZ01V_L2_E113.4_N36.4_20260222031838_01_T2",
        "DZ01V_L2_E113.8_N36.4_20260222031839_01_T3",
        "DZ01V_L2_E113.4_N36.2_20260222031840_01_T4",
        "DZ01V_L2_E113.8_N36.2_20260222031841_01_T5",
    ]

    # Offsets (in metres) for each scene origin
    # 128 px * 30 m = 3840 m per scene
    # Overlap ~ 1/3 of scene width
    half = IMAGE_SIZE * RESOLUTION  # 3840 m
    over = half * 2 / 3             # ~2560 m spacing for ~1/3 overlap
    base_x, base_y = 500000.0, 4000000.0

    offsets = [
        (0, 0),            # scene 0
        (over, 0),         # scene 1
        (over * 2, 0),     # scene 2
        (over, -over),     # scene 3
        (over * 2, -over), # scene 4
    ]

    for name, (ox, oy) in zip(scene_names, offsets):
        scene_dir = root / name
        scene_dir.mkdir(parents=True)
        for band in ("B14", "B8", "B5"):
            path = scene_dir / f"{name}_{band}.TIF"
            _make_band(path, origin_x=base_x + ox, origin_y=base_y + oy)

    return scene_names


def make_five_scene_path(tmp_path: Path) -> Path:
    """Create a synthetic five-scene dataset in a temp dir, return input root."""
    root = tmp_path / "input" / "flat"
    root.mkdir(parents=True)
    make_five_scene_dataset(root)
    return root


# ---------------------------------------------------------------------------
# autouse fixture for test isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _rasterio_env():
    """Ensure rasterio doesn't leak global state across tests."""
    yield
    # No explicit teardown needed for rasterio in test context