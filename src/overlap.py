"""
重叠区域检测模块

提供函数用于判断两幅遥感影像的包围盒是否重叠，
以及计算重叠区域在各自像素坐标系下的窗口范围。
"""

import numpy as np
import rasterio
from shapely.geometry import box
from typing import Tuple, Optional


def has_overlap(bounds1: Tuple[float, float, float, float],
                bounds2: Tuple[float, float, float, float]) -> bool:
    """
    判断两幅影像的地理包围盒是否有重叠。

    参数
    ----------
    bounds1 : (left, bottom, right, top)
        影像 1 的地理范围（与 rasterio 的 bounds 格式一致）。
    bounds2 : (left, bottom, right, top)
        影像 2 的地理范围。

    返回
    -------
    bool
        有重叠返回 True，否则返回 False。
    """
    rect1 = box(*bounds1)
    rect2 = box(*bounds2)
    return rect1.intersects(rect2)


def intersection_bounds(bounds1: Tuple[float, float, float, float],
                        bounds2: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float]]:
    """
    计算两幅影像地理包围盒的交集范围。若无重叠返回 None。

    参数
    ----------
    bounds1, bounds2 : (left, bottom, right, top)
        两个影像的地理范围。

    返回
    -------
    (left, bottom, right, top) or None
        交集的地理范围。
    """
    rect1 = box(*bounds1)
    rect2 = box(*bounds2)
    if not rect1.intersects(rect2):
        return None
    inter = rect1.intersection(rect2)
    return inter.bounds


def get_overlap_window(
    bounds1: Tuple[float, float, float, float],
    transform1: rasterio.Affine,
    bounds2: Tuple[float, float, float, float],
    transform2: rasterio.Affine,
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    """
    计算两幅影像的重叠区域在各自像素坐标系中的窗口（行、列范围）。

    返回结果格式：
        ((row_start1, row_end1, col_start1, col_end1),
         (row_start2, row_end2, col_start2, col_end2))

    其中 row_end / col_end 遵循 Python 切片约定（不含终点）。
    若无重叠则返回 None。

    参数
    ----------
    bounds1 : (left, bottom, right, top)
        影像 1 的地理范围。
    transform1 : rasterio.Affine
        影像 1 的仿射变换（地理坐标 → 像素坐标）。
    bounds2 : (left, bottom, right, top)
        影像 2 的地理范围。
    transform2 : rasterio.Affine
        影像 2 的仿射变换。

    返回
    -------
    ((row_start1, row_end1, col_start1, col_end1),
     (row_start2, row_end2, col_start2, col_end2)) or None
    """
    # 计算地理交集包围盒
    inter = intersection_bounds(bounds1, bounds2)
    if inter is None:
        return None

    left, bottom, right, top = inter

    # 将交集角点投影到影像1的像素坐标
    col1_min, row1_min = ~transform1 * (left, top)       # 左上角
    col1_max, row1_max = ~transform1 * (right, bottom)   # 右下角

    # 将交集角点投影到影像2的像素坐标
    col2_min, row2_min = ~transform2 * (left, top)
    col2_max, row2_max = ~transform2 * (right, bottom)

    # 转换为整数像素范围（取整并裁剪有效范围）
    row_start1 = max(0, int(round(row1_min)))
    row_end1   = max(0, int(round(row1_max)))
    col_start1 = max(0, int(round(col1_min)))
    col_end1   = max(0, int(round(col1_max)))

    row_start2 = max(0, int(round(row2_min)))
    row_end2   = max(0, int(round(row2_max)))
    col_start2 = max(0, int(round(col2_min)))
    col_end2   = max(0, int(round(col2_max)))

    # 确保范围有效（行/列至少有 1 个像素）
    if row_end1 <= row_start1 or col_end1 <= col_start1:
        return None
    if row_end2 <= row_start2 or col_end2 <= col_start2:
        return None

    return ((row_start1, row_end1, col_start1, col_end1),
            (row_start2, row_end2, col_start2, col_end2))


def overlap_pixel_count(
    bounds1: Tuple[float, float, float, float],
    transform1: rasterio.Affine,
    bounds2: Tuple[float, float, float, float],
    transform2: rasterio.Affine,
) -> int:
    """
    计算两幅影像重叠区域的像素数量。

    参数
    ----------
    bounds1, bounds2 : (left, bottom, right, top)
        两幅影像的地理范围。
    transform1, transform2 : rasterio.Affine
        两幅影像的仿射变换。

    返回
    -------
    int
        重叠像素数。若无重叠返回 0。
    """
    windows = get_overlap_window(bounds1, transform1, bounds2, transform2)
    if windows is None:
        return 0
    (r1_s, r1_e, c1_s, c1_e), _ = windows
    return (r1_e - r1_s) * (c1_e - c1_s)


def detect_multi_overlap(bounds_list, transforms_list, min_pixels=100):
    """
    检测多幅影像之间所有重叠对，返回 overlap 列表（兼容 bagrn/volrn 格式）。

    参数
    ----------
    bounds_list : list of (left, bottom, right, top)
        各影像的地理范围。
    transforms_list : list of rasterio.Affine
        各影像的仿射变换。
    min_pixels : int
        最小重叠像素数，低于此值的对被忽略。

    返回
    -------
    list of dict
        每个元素包含 idx_i, idx_j, window_i, window_j, pixel_count。
    """
    n = len(bounds_list)
    overlaps = []
    for i in range(n):
        for j in range(i + 1, n):
            win = get_overlap_window(
                bounds_list[i], transforms_list[i],
                bounds_list[j], transforms_list[j])
            if win is not None:
                (ri_s, ri_e, ci_s, ci_e), (rj_s, rj_e, cj_s, cj_e) = win
                pix = (ri_e - ri_s) * (ci_e - ci_s)
                if pix >= min_pixels:
                    overlaps.append({
                        'idx_i': i,
                        'idx_j': j,
                        'window_i': (ri_s, ri_e, ci_s, ci_e),
                        'window_j': (rj_s, rj_e, cj_s, cj_e),
                        'pixel_count': pix,
                    })
    return overlaps
