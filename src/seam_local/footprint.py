"""Shared valid-mask footprint construction for source-side geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
from rasterio.features import shapes as raster_shapes
from shapely.geometry import shape as shapely_shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def footprint_polygon_from_valid_mask(mask: np.ndarray, transform: Any) -> BaseGeometry:
    """Return the polygonized valid support using the frozen Task13A.1 semantics.

    The mask is rasterized into polygons with its holes and disconnected
    components intact, then combined exactly as the five-scene replay did.
    """

    valid = np.asarray(mask, dtype=bool)
    if valid.ndim != 2 or not np.any(valid):
        raise ValueError("frozen scene has no valid footprint")
    polygons = [
        shapely_shape(geometry)
        for geometry, value in raster_shapes(
            valid.astype(np.uint8), mask=valid, transform=transform
        )
        if int(value) == 1
    ]
    if not polygons:
        raise ValueError("frozen scene has no polygonized valid footprint")
    return unary_union(polygons)
