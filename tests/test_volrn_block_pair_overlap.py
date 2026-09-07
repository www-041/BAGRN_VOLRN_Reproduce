"""Test VOLRN block pairs require real geographic overlap."""

import numpy as np
import pytest
from rasterio.transform import from_origin
from src.volrn import image_blocking


def test_volrn_block_pair_requires_geographic_overlap():
    """
    VOLRN should only create block pairs for images that actually overlap geographically.
    Two images in the same grid cell but with no geographic overlap should not form a pair.
    """
    # Create two 16x16 images in a large grid cell
    # Both fit within a 64x64 grid cell but don't overlap
    img0 = np.ones((1, 16, 16), dtype=np.float64)
    img1 = np.ones((1, 16, 16), dtype=np.float64)

    # Image 0: covers (0, 0) to (16, 16)
    transform0 = from_origin(0, 16, 1, 1)
    bounds0 = (0, 0, 16, 16)

    # Image 1: covers (20, 0) to (36, 16) - no overlap with image 0
    transform1 = from_origin(20, 16, 1, 1)
    bounds1 = (20, 0, 36, 16)

    arrays = [img0, img1]
    transforms = [transform0, transform1]
    bounds_list = [bounds0, bounds1]
    nodata_values = [None, None]

    # Use a large block size so both images fall in the same grid cell
    blocks, pairs = image_blocking(
        arrays, transforms, bounds_list, nodata_values,
        block_size=64, bands=[0]
    )

    # Both images should have blocks
    assert len(blocks) == 2

    # But there should be NO pairs since they don't overlap
    assert len(pairs) == 0, f"Expected 0 pairs for non-overlapping images, got {len(pairs)}"


def test_volrn_block_pair_with_real_overlap():
    """
    VOLRN should create block pairs for images that actually overlap geographically.
    """
    # Create two 16x16 images that DO overlap
    img0 = np.ones((1, 16, 16), dtype=np.float64)
    img1 = np.ones((1, 16, 16), dtype=np.float64)

    # Image 0: covers (0, 0) to (16, 16)
    transform0 = from_origin(0, 16, 1, 1)
    bounds0 = (0, 0, 16, 16)

    # Image 1: covers (8, 0) to (24, 16) - overlaps with image 0 in x=[8,16]
    transform1 = from_origin(8, 16, 1, 1)
    bounds1 = (8, 0, 24, 16)

    arrays = [img0, img1]
    transforms = [transform0, transform1]
    bounds_list = [bounds0, bounds1]
    nodata_values = [None, None]

    blocks, pairs = image_blocking(
        arrays, transforms, bounds_list, nodata_values,
        block_size=64, bands=[0]
    )

    # Both images should have blocks
    assert len(blocks) == 2

    # There should be 1 pair since they overlap
    assert len(pairs) == 1, f"Expected 1 pair for overlapping images, got {len(pairs)}"
