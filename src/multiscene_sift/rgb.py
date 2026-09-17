"""Three-band GeoTIFF stacking and 8-bit RGB preview."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

logger = logging.getLogger(__name__)


def stack_three_band_geotiff(
    band_paths: dict[str, str],
    output_path: str | Path,
) -> str:
    """Stack three single-band GeoTIFFs into one 3-band GeoTIFF.

    Verifies same CRS, transform, width, and height for all three bands.

    Args:
        band_paths: ``{"R": path, "G": path, "B": path}``.
        output_path: Output GeoTIFF path.

    Returns:
        Output file path.

    Raises:
        ValueError: If band grids are not aligned.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    band_order = ["R", "G", "B"]
    arrays = []
    ref_profile = None

    for key in band_order:
        path = band_paths[key]
        with rasterio.open(path) as src:
            if ref_profile is None:
                ref_profile = {
                    "crs": src.crs,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "dtype": src.dtypes[0],
                    "nodata": src.nodata,
                }
            else:
                if src.crs != ref_profile["crs"]:
                    raise ValueError(
                        f"CRS mismatch for band {key}: {src.crs} vs {ref_profile['crs']}"
                    )
                if src.transform != ref_profile["transform"]:
                    raise ValueError(
                        f"Transform mismatch for band {key}"
                    )
                if src.width != ref_profile["width"] or src.height != ref_profile["height"]:
                    raise ValueError(
                        f"Size mismatch for band {key}: "
                        f"{src.width}×{src.height} vs "
                        f"{ref_profile['width']}×{ref_profile['height']}"
                    )
            arrays.append(src.read(1))

    # Stack
    rgb = np.stack(arrays, axis=0).astype(ref_profile["dtype"])

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=ref_profile["height"],
        width=ref_profile["width"],
        count=3,
        dtype=ref_profile["dtype"],
        crs=ref_profile["crs"],
        transform=ref_profile["transform"],
        nodata=ref_profile["nodata"],
    ) as dst:
        dst.write(rgb)
        dst.descriptions = (
            "Band 1: B14 mapped to Red",
            "Band 2: B8 mapped to Green",
            "Band 3: B5 mapped to Blue",
        )

    logger.info("Three-band GeoTIFF saved: %s", output_path)
    return str(output_path)


def make_rgb_preview(
    tif_path: str | Path,
    output_path: str | Path,
    stretch_path: str | Path | None = None,
) -> str:
    """Create an 8-bit RGB preview PNG from a 3-band GeoTIFF.

    For each channel independently:
        1. Collect finite valid pixels
        2. Clip to [2nd, 98th] percentile
        3. Linear stretch to 0-255
        4. Convert to uint8

    Args:
        tif_path: Input 3-band GeoTIFF.
        output_path: Output PNG path.
        stretch_path: If given, save stretch parameters as JSON.

    Returns:
        Output PNG path.
    """
    from PIL import Image

    with rasterio.open(tif_path) as src:
        data = src.read()  # (3, H, W)

    stretched = np.zeros_like(data, dtype=np.uint8)
    stretch_params = {}

    for b in range(3):
        band = data[b].astype(np.float64)
        finite = band[np.isfinite(band)]
        if len(finite) == 0:
            stretched[b] = 0
            stretch_params[f"band_{b}"] = {"p2": 0, "p98": 0}
            continue

        p2 = np.percentile(finite, 2)
        p98 = np.percentile(finite, 98)
        stretch_params[f"band_{b}"] = {
            "percentile_2": float(p2),
            "percentile_98": float(p98),
        }

        if p98 > p2:
            clipped = np.clip(band, p2, p98)
            normalized = (clipped - p2) / (p98 - p2) * 255
            stretched[b] = normalized.astype(np.uint8)
        else:
            stretched[b] = 0

    # PIL expects (H, W, 3)
    img = Image.fromarray(np.transpose(stretched, (1, 2, 0)), mode="RGB")
    img.save(str(output_path))

    if stretch_path:
        with open(stretch_path, "w") as f:
            json.dump(stretch_params, f, indent=2)

    logger.info("RGB preview saved: %s", output_path)
    return str(output_path)