from __future__ import annotations

import numpy as np
import rasterio
from affine import Affine


def _write(path, values, valid, *, dtype="float32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
                       count=1, dtype=dtype, crs="EPSG:32650", transform=Affine(14, 0, 100, 0, -14, 200),
                       nodata=255 if dtype == "uint8" else -9999) as ds:
        ds.write(values.astype(dtype), 1)
        ds.write_mask(valid.astype(np.uint8) * 255)


def test_fixed_priority_mosaic_covers_union_and_labels_each_valid_source(tmp_path):
    from src.benchmark_v1_unified.raw_mosaic import build_fixed_source_mosaic

    shape = (7, 9)
    vals = [np.full(shape, 10 + i, dtype=np.float32) for i in range(3)]
    masks = [np.zeros(shape, dtype=bool) for _ in range(3)]
    masks[0][:, :5] = True
    masks[1][:, 3:8] = True
    masks[2][:, 7:] = True
    inputs, frozen_masks = [], []
    for i in range(3):
        ip = tmp_path / f"input_{i}.tif"
        mp = tmp_path / f"mask_{i}.tif"
        _write(ip, vals[i], masks[i])
        _write(mp, masks[i].astype(np.uint8), np.ones(shape, bool), dtype="uint8")
        inputs.append(ip)
        frozen_masks.append(mp)

    checks = []
    result = build_fixed_source_mosaic(
        inputs, frozen_masks,
        tmp_path / "mosaic.tif", tmp_path / "mosaic_mask.tif", tmp_path / "source_labels.tif",
        scene_ids=["S0", "S1", "S2"], expected_shape=shape,
        expected_transform=(14, 0, 100, 0, -14, 200), expected_crs="EPSG:32650",
        priority=[0, 1, 2], block_shape=(3, 4), check=lambda: checks.append("tile"),
    )
    with rasterio.open(tmp_path / "mosaic.tif") as ds:
        mosaic = ds.read(1)
        mosaic_valid = ds.dataset_mask() > 0
    with rasterio.open(tmp_path / "mosaic_mask.tif") as ds:
        explicit = ds.read(1) > 0
    with rasterio.open(tmp_path / "source_labels.tif") as ds:
        labels = ds.read(1)

    union = np.logical_or.reduce(masks)
    expected_labels = np.where(masks[0], 0, np.where(masks[1], 1, 2)).astype(np.uint8)
    expected_values = np.where(masks[0], 10, np.where(masks[1], 11, 12)).astype(np.float32)
    assert result["status"] == "COMPLETED"
    assert result["priority_scene_ids"] == ["S0", "S1", "S2"]
    assert np.array_equal(mosaic_valid, union)
    assert np.array_equal(explicit, union)
    assert np.array_equal(labels[union], expected_labels[union])
    assert np.array_equal(mosaic[union], expected_values[union])
    assert result["holes"] == result["extras"] == 0
    assert len(checks) == 9


def test_fixed_priority_mosaic_rejects_frozen_mask_outside_source_validity(tmp_path):
    import pytest
    from src.benchmark_v1_unified.raw_mosaic import build_fixed_source_mosaic

    image = np.ones((3, 4), dtype=np.float32)
    data_mask = np.ones((3, 4), dtype=bool)
    data_mask[1, 2] = False
    frozen = np.ones((3, 4), dtype=np.uint8)
    ip, mp = tmp_path / "input.tif", tmp_path / "mask.tif"
    _write(ip, image, data_mask)
    _write(mp, frozen, np.ones((3, 4), bool), dtype="uint8")
    with pytest.raises(ValueError, match="FROZEN_MASK_OUTSIDE_SOURCE_VALIDITY"):
        build_fixed_source_mosaic(
            [ip], [mp], tmp_path / "m.tif", tmp_path / "mm.tif", tmp_path / "labels.tif",
            scene_ids=["S0"], expected_shape=image.shape,
            expected_transform=(14, 0, 100, 0, -14, 200), expected_crs="EPSG:32650",
            priority=[0], block_shape=(2, 2),
        )


def test_fixed_priority_mosaic_cleans_partial_outputs_when_guard_stops_run(tmp_path):
    import pytest
    from src.benchmark_v1_unified.resource_guard import ResourceLimitExceeded
    from src.benchmark_v1_unified.raw_mosaic import build_fixed_source_mosaic

    values = np.ones((5, 5), dtype=np.float32)
    valid = np.ones((5, 5), dtype=bool)
    image, mask = tmp_path / "source.tif", tmp_path / "valid.tif"
    _write(image, values, valid)
    _write(mask, valid.astype(np.uint8), valid, dtype="uint8")
    count = 0

    def stop_after_first_tile():
        nonlocal count
        count += 1
        if count == 2:
            raise ResourceLimitExceeded("RSS_POLLING_LIMIT_TRIGGERED")

    with pytest.raises(ResourceLimitExceeded):
        build_fixed_source_mosaic(
            [image], [mask], tmp_path / "mosaic.tif", tmp_path / "mosaic_mask.tif", tmp_path / "labels.tif",
            scene_ids=["S0"], expected_shape=values.shape,
            expected_transform=(14, 0, 100, 0, -14, 200), expected_crs="EPSG:32650",
            priority=[0], block_shape=(2, 2), check=stop_after_first_tile,
        )
    assert not (tmp_path / "mosaic.tif").exists()
    assert not (tmp_path / "mosaic_mask.tif").exists()
    assert not (tmp_path / "labels.tif").exists()
    assert not list(tmp_path.glob("*.partial*"))
