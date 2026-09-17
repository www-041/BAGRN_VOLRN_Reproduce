"""Tests for :mod:`src.multiscene_sift.radiometric`."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.multiscene_sift.models import Scene
from src.multiscene_sift.radiometric import (
    normalize_registered_band,
    BandRadiometricResult,
)
from src.multiscene_sift.band_geometry import apply_world_correction_to_transform
from src.multiscene_sift.mosaicking import raster_bounds_from_transform


def _make_scene_objects(tmp_path, n=3):
    """Create synthetic Scene-like objects for testing."""
    from src.multiscene_sift.models import Scene

    scenes = []
    for i in range(n):
        ox = 500000 + i * 2000
        oy = 4000000
        tf = from_origin(ox, oy, 30, 30)
        h, w = 128, 128

        # Write actual GeoTIFF with gradient + noise
        path = tmp_path / f"scene_{i}" / f"scene_{i}_B14.TIF"
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42 + i)
        y, x = np.mgrid[0:h, 0:w]
        data = (25000 + 5000 * np.sin(x * 0.1) + 3000 * np.cos(y * 0.08)
                + rng.uniform(0, 2000, (h, w))).astype("uint16")

        with rasterio.open(
            path, "w", driver="GTiff", height=h, width=w,
            count=1, dtype="uint16", crs="EPSG:32650", transform=tf,
        ) as dst:
            dst.write(data, 1)

        bounds = rasterio.coords.BoundingBox(
            ox, oy - h * 30, ox + w * 30, oy,
        )

        scenes.append(Scene(
            index=i, name=f"scene_{i}",
            directory=str(path.parent),
            band_paths={"B14": str(path)},
            crs=rasterio.crs.CRS.from_epsg(32650),
            transforms={"B14": tf},
            shapes={"B14": (h, w)},
            nodata={"B14": None},
            bounds={"B14": bounds},
        ))

    return scenes


class TestRadiometricAdapter:
    """Tests for :func:`normalize_registered_band`."""

    def test_returns_valid_result(self, tmp_path):
        """Basic call should return BandRadiometricResult with valid fields."""
        scenes = _make_scene_objects(tmp_path, n=3)
        G = [np.eye(3) for _ in scenes]

        result = normalize_registered_band(
            scenes, G, "B14", reference_idx=0,
        )

        assert isinstance(result, BandRadiometricResult)
        assert result.band == "B14"
        assert len(result.normalized_arrays) == 3
        assert len(result.overlaps) > 0
        assert result.bagrn_runtime_sec >= 0
        assert result.volrn_runtime_sec >= 0

    def test_output_shapes_unchanged(self, tmp_path):
        """Normalized arrays should have same shape as input."""
        scenes = _make_scene_objects(tmp_path, n=3)
        G = [np.eye(3) for _ in scenes]

        result = normalize_registered_band(scenes, G, "B14", reference_idx=0)

        for i in range(3):
            expected = (1, 128, 128)
            assert result.normalized_arrays[i].shape == expected

    def test_valid_outputs_finite(self, tmp_path):
        """Normalized output should have finite valid pixels."""
        scenes = _make_scene_objects(tmp_path, n=3)
        G = [np.eye(3) for _ in scenes]

        result = normalize_registered_band(scenes, G, "B14", reference_idx=0)

        for arr in result.normalized_arrays:
            assert np.isfinite(arr).any()

    def test_metric_keys_present(self, tmp_path):
        """All metric dicts should contain required keys."""
        scenes = _make_scene_objects(tmp_path, n=3)
        G = [np.eye(3) for _ in scenes]

        result = normalize_registered_band(scenes, G, "B14", reference_idx=0)

        required = {"adm", "adsd", "cd", "gl", "rdoa", "ave"}
        for metrics in [result.registered_metrics,
                        result.bagrn_metrics,
                        result.volrn_metrics]:
            for key in required:
                assert key in metrics, f"Missing {key}"

    def test_registered_gl_approx_zero(self, tmp_path):
        """Registered-stage GL should be ~0 (before==after)."""
        scenes = _make_scene_objects(tmp_path, n=3)
        G = [np.eye(3) for _ in scenes]

        result = normalize_registered_band(scenes, G, "B14", reference_idx=0)

        gl = result.registered_metrics.get("gl")
        assert gl is not None
        assert gl == pytest.approx(0.0, abs=1e-6)

    def test_reference_connectivity_required(self, tmp_path):
        """Disconnected radiometric graph should raise RuntimeError."""
        # Create scenes far apart so no overlap
        scenes = []
        for i in range(3):
            ox = 500000 + i * 100000  # far apart
            tf = from_origin(ox, 4000000, 30, 30)
            h, w = 64, 64
            path = tmp_path / f"isolated_{i}" / f"isolated_{i}_B14.TIF"
            path.parent.mkdir(parents=True, exist_ok=True)
            data = np.ones((h, w), dtype="uint16") * 30000
            with rasterio.open(
                path, "w", driver="GTiff", height=h, width=w,
                count=1, dtype="uint16", crs="EPSG:32650", transform=tf,
            ) as dst:
                dst.write(data, 1)

            bounds = rasterio.coords.BoundingBox(
                ox, 4000000 - h * 30, ox + w * 30, 4000000,
            )
            scenes.append(Scene(
                index=i, name=f"isolated_{i}",
                directory=str(path.parent),
                band_paths={"B14": str(path)},
                crs=rasterio.crs.CRS.from_epsg(32650),
                transforms={"B14": tf},
                shapes={"B14": (h, w)},
                nodata={"B14": None},
                bounds={"B14": bounds},
            ))

        G = [np.eye(3) for _ in scenes]

        with pytest.raises(RuntimeError, match="Radiometric overlap"):
            normalize_registered_band(scenes, G, "B14", reference_idx=0)