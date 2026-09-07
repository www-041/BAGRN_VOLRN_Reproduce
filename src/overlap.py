"""
重叠区域检测模块

提供函数用于判断两幅遥感影像的包围盒是否重叠，
以及计算重叠区域在各自像素坐标系下的窗口范围。
"""

import math
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
        有正面积重叠返回 True，否则返回 False。
        仅边/角接触（零面积交集）返回 False。
    """
    rect1 = box(*bounds1)
    rect2 = box(*bounds2)
    if not rect1.intersects(rect2):
        return False
    inter = rect1.intersection(rect2)
    return inter.area > 0


def intersection_bounds(bounds1: Tuple[float, float, float, float],
                        bounds2: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float]]:
    """
    计算两幅影像地理包围盒的交集范围。若无正面积重叠返回 None。

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
    if inter.area <= 0:
        return None
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
    使用 floor/ceil 保守覆盖地理交集的分数像素边界。
    若无正面积重叠则返回 None。

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
    # 计算地理交集包围盒（已确保正面积）
    inter = intersection_bounds(bounds1, bounds2)
    if inter is None:
        return None

    left, bottom, right, top = inter
    inv_t1 = ~transform1
    inv_t2 = ~transform2

    # 将交集所有四个角投影到影像1的像素坐标，取保守范围
    geo_corners = [(left, top), (right, top), (left, bottom), (right, bottom)]
    pix1 = [inv_t1 * (gx, gy) for gx, gy in geo_corners]
    row1_vals = [p[1] for p in pix1]
    col1_vals = [p[0] for p in pix1]

    # floor 起点，ceil 终点 → 保守覆盖分数边界
    row_start1 = int(math.floor(min(row1_vals)))
    row_end1   = int(math.ceil(max(row1_vals)))
    col_start1 = int(math.floor(min(col1_vals)))
    col_end1   = int(math.ceil(max(col1_vals)))

    # 同样处理影像2
    pix2 = [inv_t2 * (gx, gy) for gx, gy in geo_corners]
    row2_vals = [p[1] for p in pix2]
    col2_vals = [p[0] for p in pix2]

    row_start2 = int(math.floor(min(row2_vals)))
    row_end2   = int(math.ceil(max(row2_vals)))
    col_start2 = int(math.floor(min(col2_vals)))
    col_end2   = int(math.ceil(max(col2_vals)))

    # 裁剪到非负范围
    row_start1 = max(0, row_start1)
    col_start1 = max(0, col_start1)
    row_start2 = max(0, row_start2)
    col_start2 = max(0, col_start2)

    # 确保范围有效（至少有 1 个像素）
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
