"""Dataset discovery and validation for five-scene SIFT registration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import rasterio
import rasterio.crs

from src.multiscene_sift.models import REQUIRED_BANDS, Scene

logger = logging.getLogger(__name__)


def discover_five_scenes(
    input_root: str,
    scene_names: list[str],
    bands: tuple[str, ...] = REQUIRED_BANDS,
) -> tuple[list[Scene], dict]:
    """Discover and validate five DZ01V scenes.

    Args:
        input_root: Path to the ``flat/`` input directory.
        scene_names: List of scene directory names (exactly 5).
        bands: Band suffixes to discover (default: B14, B8, B5).

    Returns:
        ``(scenes, manifest)`` where *manifest* is a JSON-serialisable dict.

    Raises:
        FileNotFoundError: If a required band file is missing.
        ValueError: If CRS or grid parameters are inconsistent within a scene.
    """
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    scenes: list[Scene] = []

    for idx, name in enumerate(scene_names):
        scene_dir = root / name
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

        band_paths: dict[str, str] = {}
        for band in bands:
            candidates = sorted(scene_dir.glob(f"*_{band}.TIF"))
            # Also try uppercase suffix
            if not candidates:
                candidates = sorted(scene_dir.glob(f"*_{band}.tif"))
            if not candidates:
                raise FileNotFoundError(
                    f"Band {band} not found in {scene_dir}"
                )
            if len(candidates) > 1:
                logger.warning(
                    "Multiple %s files found in %s, using %s",
                    band, name, candidates[0].name,
                )
            band_paths[band] = str(candidates[0])

        # --- Validate bands -------------------------------------------------
        _validate_scene_bands(scene_dir, name, band_paths, bands)

        # --- Open B14 for metadata ------------------------------------------
        with rasterio.open(band_paths[bands[0]]) as src:
            crs = src.crs
            shape = src.shape
            transform = src.transform
            nodata = src.nodata
            bounds = src.bounds

        # Read all band metadata
        transforms: dict[str, object] = {bands[0]: transform}
        shapes: dict[str, tuple[int, int]] = {bands[0]: shape}
        nodatas: dict[str, float | None] = {bands[0]: nodata}
        bounds_dict: dict[str, tuple[float, float, float, float]] = {bands[0]: bounds}

        for band in bands[1:]:
            with rasterio.open(band_paths[band]) as src:
                transforms[band] = src.transform
                shapes[band] = src.shape
                nodatas[band] = src.nodata
                bounds_dict[band] = src.bounds

        scenes.append(Scene(
            index=idx,
            name=name,
            directory=str(scene_dir),
            band_paths=band_paths,
            crs=crs,
            transforms=transforms,
            shapes=shapes,
            nodata=nodatas,
            bounds=bounds_dict,
        ))

    manifest = _build_manifest(scenes, bands)
    return scenes, manifest


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------


def _validate_scene_bands(
    scene_dir: Path,
    name: str,
    band_paths: dict[str, str],
    bands: tuple[str, ...],
) -> None:
    """Validate that within-scene bands share CRS, orientation, and pixel size."""
    ref_band = bands[0]
    with rasterio.open(band_paths[ref_band]) as ref_src:
        ref_crs = ref_src.crs
        ref_transform = ref_src.transform
        ref_res = (abs(ref_transform.a), abs(ref_transform.e))
        # Check orientation: a>0 (east), e<0 (north)
        if ref_transform.a <= 0:
            raise ValueError(
                f"{name}/{ref_band}: expected transform.a > 0 (east-going), "
                f"got {ref_transform.a}"
            )
        if ref_transform.e >= 0:
            raise ValueError(
                f"{name}/{ref_band}: expected transform.e < 0 (north-going), "
                f"got {ref_transform.e}"
            )
        ref_rot_b = ref_transform.b
        ref_rot_d = ref_transform.d

    for band in bands[1:]:
        with rasterio.open(band_paths[band]) as src:
            # CRS match
            if src.crs != ref_crs:
                raise ValueError(
                    f"{name}: CRS mismatch: {ref_band}={ref_crs}, "
                    f"{band}={src.crs}"
                )
            tf = src.transform
            res = (abs(tf.a), abs(tf.e))
            # Orientation match
            if tf.a <= 0 or tf.e >= 0:
                raise ValueError(
                    f"{name}/{band}: grid orientation mismatch "
                    f"(a={tf.a}, e={tf.e})"
                )
            # Pixel size match (within 1%)
            for dim in (0, 1):
                if ref_res[dim] > 0 and res[dim] > 0:
                    ratio = max(ref_res[dim], res[dim]) / min(ref_res[dim], res[dim])
                    if ratio > 1.01:
                        raise ValueError(
                            f"{name}: pixel size mismatch between "
                            f"{ref_band} ({ref_res[dim]:.2f}) and "
                            f"{band} ({res[dim]:.2f}) on dim {dim}"
                        )
            # Rotation terms consistent (b,d ~0 for north-up)
            if abs(tf.b - ref_rot_b) > 1e-9 or abs(tf.d - ref_rot_d) > 1e-9:
                raise ValueError(
                    f"{name}: rotation/skew mismatch between "
                    f"{ref_band} (b={ref_rot_b}, d={ref_rot_d}) and "
                    f"{band} (b={tf.b}, d={tf.d})"
                )


def _build_manifest(scenes: list[Scene], bands: tuple[str, ...]) -> dict:
    """Build a JSON-serialisable dataset manifest."""
    scene_list = []
    for s in scenes:
        scene_list.append({
            "index": s.index,
            "name": s.name,
            "directory": s.directory,
            "crs": str(s.crs),
            "bands": {
                b: {
                    "path": s.band_paths[b],
                    "shape": list(s.shapes[b]),
                    "resolution": (
                        abs(s.transforms[b].a),
                        abs(s.transforms[b].e),
                    ),
                    "bounds": [
                        s.bounds[b].left,
                        s.bounds[b].bottom,
                        s.bounds[b].right,
                        s.bounds[b].top,
                    ],
                    "nodata": s.nodata[b],
                }
                for b in bands
            },
        })
    return {"scenes": scene_list, "bands": list(bands), "n_scenes": len(scenes)}


def save_manifest(manifest: dict, path: str | Path) -> None:
    """Write the dataset manifest to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Dataset manifest saved: %s", path)