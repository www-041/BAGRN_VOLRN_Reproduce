"""
重叠检测模块的单元测试

使用合成 GeoTIFF（通过 io_utils.create_synthetic_geotiff 生成）
测试 has_overlap、get_overlap_window、overlap_pixel_count 的正确性。
"""

import os
import tempfile
import numpy as np
import rasterio
from rasterio.windows import from_bounds

from src.io_utils import create_synthetic_geotiff
from src.overlap import has_overlap, get_overlap_window, overlap_pixel_count


# ---------------------------------------------------------------------------
# 辅助函数：在临时目录中创建测试影像
# ---------------------------------------------------------------------------

def _make_test_images(tmpdir):
    """
    创建三张合成 GeoTIFF，用于测试重叠检测：

    img_A:  4行×4列，范围 (0, 0) → (4, 4)，单位像素
    img_B:  4行×4列，范围 (2, 0) → (6, 4)，与 A 右半重叠
    img_C:  4行×4列，范围 (5, 0) → (9, 4)，与 B 右半重叠，与 A 不重叠
    """
    res = 1.0  # 每个像素 1 个地理单位
    rows = cols = 4
    crs = "EPSG:4326"

    # img_A: 左上角 (0, 0)
    path_a = os.path.join(tmpdir, "img_A.tif")
    create_synthetic_geotiff(
        path_a, rows, cols,
        transform=rasterio.Affine(res, 0, 0, 0, -res, 4),
        crs=crs, bands=1, fill_value=1.0,
    )

    # img_B: 左上角 (2, 0)
    path_b = os.path.join(tmpdir, "img_B.tif")
    create_synthetic_geotiff(
        path_b, rows, cols,
        transform=rasterio.Affine(res, 0, 2, 0, -res, 4),
        crs=crs, bands=1, fill_value=2.0,
    )

    # img_C: 左上角 (5, 0)
    path_c = os.path.join(tmpdir, "img_C.tif")
    create_synthetic_geotiff(
        path_c, rows, cols,
        transform=rasterio.Affine(res, 0, 5, 0, -res, 4),
        crs=crs, bands=1, fill_value=3.0,
    )

    return path_a, path_b, path_c


def _get_bounds_and_transform(path):
    """读取 GeoTIFF 的 bounds 和 transform。"""
    with rasterio.open(path) as src:
        return src.bounds, src.transform


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestHasOverlap:
    """测试 has_overlap 函数"""

    def test_overlapping_pair(self):
        """A 与 B 有重叠"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p_a, p_b, _ = _make_test_images(tmpdir)
            b1, _ = _get_bounds_and_transform(p_a)
            b2, _ = _get_bounds_and_transform(p_b)
            assert has_overlap(b1, b2) is True

    def test_non_overlapping_pair(self):
        """A 与 C 无重叠"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p_a, _, p_c = _make_test_images(tmpdir)
            b1, _ = _get_bounds_and_transform(p_a)
            b3, _ = _get_bounds_and_transform(p_c)
            assert has_overlap(b1, b3) is False

    def test_same_image(self):
        """同一张影像必然与自身重叠"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p_a, _, _ = _make_test_images(tmpdir)
            b1, _ = _get_bounds_and_transform(p_a)
            assert has_overlap(b1, b1) is True


class TestGetOverlapWindow:
    """测试 get_overlap_window 函数"""

    def test_overlap_window_shape(self):
        """A-B 重叠区域应为右半部分 4 列中的 2 列（像素坐标 x: 2~4）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p_a, p_b, _ = _make_test_images(tmpdir)
            b1, t1 = _get_bounds_and_transform(p_a)
            b2, t2 = _get_bounds_and_transform(p_b)
            result = get_overlap_window(b1, t1, b2, t2)
            assert result is not None

            (r1_s, r1_e, c1_s, c1_e), (r2_s, r2_e, c2_s, c2_e) = result

            # A: 重叠列范围应为 [2, 4)
            assert c1_s == 2
            assert c1_e == 4
            # A: 行范围为全部 [0, 4)
            assert r1_s == 0
            assert r1_e == 4

            # B: 重叠列范围应为 [0, 2)
            assert c2_s == 0
            assert c2_e == 2
            # B: 行范围为全部 [0, 4)
            assert r2_s == 0
            assert r2_e == 4

    def test_no_overlap_returns_none(self):
        """A-C 无重叠应返回 None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p_a, _, p_c = _make_test_images(tmpdir)
            b1, t1 = _get_bounds_and_transform(p_a)
            b3, t3 = _get_bounds_and_transform(p_c)
            result = get_overlap_window(b1, t1, b3, t3)
            assert result is None


class TestOverlapPixelCount:
    """测试 overlap_pixel_count 函数"""

    def test_count_correct(self):
        """A-B 重叠应为 2 列 × 4 行 = 8 像素"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p_a, p_b, _ = _make_test_images(tmpdir)
            b1, t1 = _get_bounds_and_transform(p_a)
            b2, t2 = _get_bounds_and_transform(p_b)
            count = overlap_pixel_count(b1, t1, b2, t2)
            assert count == 8

    def test_no_overlap_count_zero(self):
        """A-C 无重叠，像素数应为 0"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p_a, _, p_c = _make_test_images(tmpdir)
            b1, t1 = _get_bounds_and_transform(p_a)
            b3, t3 = _get_bounds_and_transform(p_c)
            count = overlap_pixel_count(b1, t1, b3, t3)
            assert count == 0
