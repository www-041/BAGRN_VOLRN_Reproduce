"""
镶嵌模块测试
"""

import tempfile
import os

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.mosaic import create_mosaic, _compute_weight_map, _combined_bounds


def test_compute_weight_map_uniform():
    """全部有效 -> 距离边界越远权重越大，最大值为 1。"""
    mask = np.ones((10, 10), dtype=bool)
    w = _compute_weight_map(mask)
    assert np.isclose(w.max(), 1.0)
    assert w.min() >= 0.0
    assert w[0, 0] < w[5, 5]  # 中心 > 角落


def test_compute_weight_map_center_higher():
    """中心像素权重应大于边缘像素。"""
    mask = np.ones((20, 20), dtype=bool)
    w = _compute_weight_map(mask)
    assert w[10, 10] > w[0, 0]


def test_compute_weight_map_with_nodata():
    """部分无效像素时，距离无效区越近权重越小。"""
    mask = np.ones((10, 10), dtype=bool)
    mask[:, :5] = False
    w = _compute_weight_map(mask)
    assert w[2, 7] > w[2, 5]


def test_combined_bounds_single():
    """单幅影像返回其自身包围盒。"""
    arr = np.ones((1, 10, 10))
    tr = from_origin(100, 200, 1, 1)
    l, b, r, t = _combined_bounds([arr], [tr])
    assert abs(l - 100) < 1e-6
    assert abs(r - 110) < 1e-6
    assert abs(b - 190) < 1e-6
    assert abs(t - 200) < 1e-6


def test_combined_bounds_two():
    """两幅影像，取并集。"""
    arr = np.ones((1, 10, 10))
    tr1 = from_origin(100, 200, 1, 1)
    tr2 = from_origin(108, 200, 1, 1)
    l, b, r, t = _combined_bounds([arr, arr], [tr1, tr2])
    assert abs(l - 100) < 1e-6
    assert abs(r - 118) < 1e-6


def test_create_mosaic_two_images():
    """两幅半重叠影像镶嵌 -> 输出覆盖联合范围。"""
    n_bands = 2
    shape = (10, 10)
    arr1 = np.ones((n_bands, *shape), dtype=np.float32) * 100
    arr2 = np.ones((n_bands, *shape), dtype=np.float32) * 200
    tr1 = from_origin(100, 200, 1, 1)
    tr2 = from_origin(105, 200, 1, 1)
    crs = "EPSG:4326"
    nodata = None

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        out_path = f.name

    try:
        result = create_mosaic(
            [arr1, arr2], [tr1, tr2], crs, [nodata, nodata],
            out_path,
        )
        assert os.path.exists(result)
        with rasterio.open(result) as ds:
            assert ds.count == n_bands
            assert ds.width > shape[1]
            # 左半部分（无重叠区）应接近 100
            left_data = ds.read(1)
            assert left_data[:, :5].mean() < 110
            # 右半部分（无重叠区）应接近 200
            right_data = ds.read(1)
            assert right_data[:, 10:].mean() > 190
            # 中间重叠列在 100~200 之间
            mid = ds.read(1)[:, 5:10].mean()
            assert 100 < mid < 200
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def test_create_mosaic_no_overlap():
    """无重叠的两幅影像 -> 直接并排放置。"""
    arr1 = np.ones((1, 10, 10), dtype=np.float32) * 100
    arr2 = np.ones((1, 10, 10), dtype=np.float32) * 200
    tr1 = from_origin(100, 200, 1, 1)
    tr2 = from_origin(120, 200, 1, 1)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        out_path = f.name

    try:
        result = create_mosaic(
            [arr1, arr2], [tr1, tr2], "EPSG:4326", [None, None],
            out_path,
        )
        with rasterio.open(result) as ds:
            assert ds.width == 30
            assert ds.height == 10
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def test_create_mosaic_nodata():
    """影像包含 nodata -> 镶嵌时 nodata 区域被跳过。"""
    n_bands = 1
    shape = (10, 10)
    arr1 = np.ones((n_bands, *shape), dtype=np.float32) * 100
    arr1[0, 0:5, :] = -9999  # 左半无效
    arr2 = np.ones((n_bands, *shape), dtype=np.float32) * 200
    tr1 = from_origin(100, 200, 1, 1)
    tr2 = from_origin(105, 200, 1, 1)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        out_path = f.name

    try:
        result = create_mosaic(
            [arr1, arr2], [tr1, tr2], "EPSG:4326",
            [-9999, None], out_path,
        )
        assert os.path.exists(result)
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
