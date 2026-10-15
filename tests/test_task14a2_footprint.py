from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.transform import array_bounds
from shapely.geometry import Point, box

from src.seam_local.footprint import footprint_polygon_from_valid_mask


ROOT = Path(__file__).resolve().parents[1]
FIVE_MANIFEST = ROOT / "data/output/b9_five_scene_validation/seam_local_task13a/protocol.json"


def test_irregular_valid_mask_is_not_reduced_to_dataset_bbox():
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:5, 1:7] = True
    mask[2:4, 3:5] = False
    transform = Affine(2, 0, 100, 0, -2, 200)

    footprint = footprint_polygon_from_valid_mask(mask, transform)
    bbox = box(*array_bounds(mask.shape[0], mask.shape[1], transform))

    assert footprint.area < bbox.area
    assert not footprint.contains(Point(107, 193))
    assert footprint.bounds == (102.0, 190.0, 114.0, 198.0)


def test_empty_valid_mask_is_rejected():
    with pytest.raises(ValueError, match="valid footprint"):
        footprint_polygon_from_valid_mask(np.zeros((3, 4), dtype=bool), Affine.identity())


def test_old_and_shared_helper_are_geometry_equivalent_on_five_scene_masks():
    if not FIVE_MANIFEST.is_file():
        pytest.skip("frozen five-scene protocol is unavailable")
    from scripts.run_task13a1_source_side import _footprint_polygon

    import json

    protocol = json.loads(FIVE_MANIFEST.read_text(encoding="utf-8"))
    for scene in protocol["scenes"]:
        with rasterio.open(ROOT / scene["path"]) as src:
            mask = src.read_masks(1) > 0
            old = _footprint_polygon(mask, src.transform)
            new = footprint_polygon_from_valid_mask(mask, src.transform)
        assert old.equals(new)
        assert old.bounds == new.bounds
        assert old.area == new.area
        assert old.centroid.equals(new.centroid)
