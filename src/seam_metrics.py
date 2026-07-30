"""
接缝评估与羽化宽度实验模块

实现BAGRN-VOLRN项目中的接缝质量评价指标和羽化宽度优化实验。
包括辐射跳变、梯度不连续、色差、清晰度、纹理损失等指标的计算。

参考论文:
"Joint block adjustment and variational optimization for global and local
radiometric normalization toward multiple remote sensing image mosaicking"
"""

from __future__ import annotations

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _safe_array(arr: NDArray, nodata: Optional[float] = None) -> NDArray:
    """将输入转为float64并处理NoData，返回掩码（True=有效像素）"""
    a = np.asarray(arr, dtype=np.float64)
    mask = np.ones(a.shape, dtype=bool)
    if nodata is not None:
        mask &= a != nodata
    return a, mask


def _valid_stats(a: NDArray, mask: NDArray) -> Tuple[NDArray, NDArray]:
    """返回有效像素值的展开一维数组"""
    return a[mask], mask


def compute_seam_jump(
    arr_a: NDArray,
    arr_b: NDArray,
    seam_cols: NDArray,
    band: int = 0,
    strip_width: int = 10,
    nodata_a: Optional[float] = None,
    nodata_b: Optional[float] = None,
) -> Dict[str, float]:
    """
    计算接缝两侧的辐射跳变

    Parameters
    ----------
    arr_a, arr_b : 2D数组
        重叠区域的单波段图像数据
    seam_cols : 1D数组
        每行接缝的列位置
    band : int
        波段索引（暂保留接口）
    strip_width : int
        接缝两侧用于比较的像素宽度
    nodata_a, nodata_b : float, optional
        NoData值

    Returns
    -------
    dict: mean_jump, p95_jump, max_jump
    """
    a, mask_a = _safe_array(arr_a, nodata_a)
    b, mask_b = _safe_array(arr_b, nodata_b)

    rows, cols = a.shape
    seam_cols = np.asarray(seam_cols, dtype=np.int64)

    jumps = []

    for r in range(rows):
        sc = int(seam_cols[r])
        if sc < strip_width or sc + strip_width >= cols:
            continue

        left_start = max(0, sc - strip_width)
        left_end = sc
        right_start = sc + 1
        right_end = min(cols, sc + 1 + strip_width)

        left_region = a[r, left_start:left_end]
        right_region = b[r, right_start:right_end]

        left_mask = mask_a[r, left_start:left_end]
        right_mask = mask_b[r, right_start:right_end]

        if not np.any(left_mask) or not np.any(right_mask):
            continue

        mean_left = np.mean(left_region[left_mask])
        mean_right = np.mean(right_region[right_mask])
        jumps.append(abs(mean_left - mean_right))

    if len(jumps) == 0:
        return {"mean_jump": np.nan, "p95_jump": np.nan, "max_jump": np.nan}

    jumps = np.array(jumps, dtype=np.float64)
    return {
        "mean_jump": float(np.mean(jumps)),
        "p95_jump": float(np.percentile(jumps, 95)),
        "max_jump": float(np.max(jumps)),
    }


def compute_seam_gradient_discontinuity(
    arr_a: NDArray,
    arr_b: NDArray,
    seam_cols: NDArray,
    nodata_a: Optional[float] = None,
    nodata_b: Optional[float] = None,
) -> Dict[str, float]:
    """
    计算接缝法线方向的梯度不连续性

    沿法线方向（水平）计算接缝两侧的梯度绝对值差异。
    梯度不连续性越大说明接缝越明显。

    Parameters
    ----------
    arr_a, arr_b : 2D数组
        重叠区域的单波段图像数据
    seam_cols : 1D数组
        每行接缝的列位置
    nodata_a, nodata_b : float, optional
        NoData值

    Returns
    -------
    dict: mean_discontinuity, p95_discontinuity
    """
    a, mask_a = _safe_array(arr_a, nodata_a)
    b, mask_b = _safe_array(arr_b, nodata_b)

    rows, cols = a.shape
    seam_cols = np.asarray(seam_cols, dtype=np.int64)

    discontinuities = []

    for r in range(rows):
        sc = int(seam_cols[r])
        if sc < 2 or sc + 2 >= cols:
            continue

        # 左侧梯度：在接缝左侧2像素处的水平梯度
        left_grad = abs(a[r, sc] - a[r, sc - 1])
        # 右侧梯度：在接缝右侧1像素处的水平梯度
        right_grad = abs(b[r, sc + 1] - b[r, sc]) if sc + 1 < cols else 0.0

        # 检查NoData
        if not mask_a[r, sc] or not mask_a[r, sc - 1]:
            continue
        if sc + 1 < cols and (not mask_b[r, sc + 1] or not mask_b[r, sc]):
            continue

        discontinuities.append(abs(left_grad - right_grad))

    if len(discontinuities) == 0:
        return {"mean_discontinuity": np.nan, "p95_discontinuity": np.nan}

    discontinuities = np.array(discontinuities, dtype=np.float64)
    return {
        "mean_discontinuity": float(np.mean(discontinuities)),
        "p95_discontinuity": float(np.percentile(discontinuities, 95)),
    }


def compute_seam_color_distance(
    arr_a: NDArray,
    arr_b: NDArray,
    seam_cols: NDArray,
    nodata_a: Optional[float] = None,
    nodata_b: Optional[float] = None,
) -> Dict[str, float]:
    """
    计算接缝多波段色差

    使用欧氏距离衡量接缝两侧的颜色差异。
    适用于多波段(RGB)图像。

    Parameters
    ----------
    arr_a, arr_b : 2D或3D数组
        重叠区域图像数据，shape=(H,W)或(H,W,C)
    seam_cols : 1D数组
        每行接缝的列位置
    nodata_a, nodata_b : float, optional
        NoData值（所有波段共用同一NoData）

    Returns
    -------
    dict: mean_distance, p95_distance
    """
    a, mask_a = _safe_array(arr_a, nodata_a)
    b, mask_b = _safe_array(arr_b, nodata_b)

    if a.ndim == 2:
        a = a[:, :, np.newaxis]
        b = b[:, :, np.newaxis]
        mask_a = mask_a[:, :, np.newaxis]
        mask_b = mask_b[:, :, np.newaxis]

    rows, cols, n_bands = a.shape
    seam_cols = np.asarray(seam_cols, dtype=np.int64)

    distances = []

    for r in range(rows):
        sc = int(seam_cols[r])
        if sc < 1 or sc + 1 >= cols:
            continue

        # 检查所有波段的NoData
        valid_left = mask_a[r, sc - 1, :].all()
        valid_right = mask_b[r, sc + 1, :].all()
        if not valid_left or not valid_right:
            continue

        vec_left = a[r, sc - 1, :]
        vec_right = b[r, sc + 1, :]
        dist = np.sqrt(np.sum((vec_left - vec_right) ** 2))
        distances.append(dist)

    if len(distances) == 0:
        return {"mean_distance": np.nan, "p95_distance": np.nan}

    distances = np.array(distances, dtype=np.float64)
    return {
        "mean_distance": float(np.mean(distances)),
        "p95_distance": float(np.percentile(distances, 95)),
    }


def compute_sharpness(
    arr: NDArray,
    method: str = "tenengrad",
    nodata: Optional[float] = None,
) -> float:
    """
    计算图像清晰度（锐度）

    支持两种方法:
    - tenengrad: 使用Sobel梯度能量，值越大越清晰
    - laplacian: 使用拉普拉斯算子方差

    Parameters
    ----------
    arr : 2D数组
        输入图像
    method : str
        'tenengrad' 或 'laplacian'
    nodata : float, optional
        NoData值

    Returns
    -------
    float: 清晰度值（越高越清晰）
    """
    a, mask = _safe_array(arr, nodata)

    if a.size == 0:
        return np.nan

    # 使用有效像素填充NoData区域为均值，避免边缘伪影
    if not mask.all():
        valid_mean = np.mean(a[mask]) if mask.any() else 0.0
        a[~mask] = valid_mean

    if method == "tenengrad":
        # Sobel算子计算梯度
        from scipy.ndimage import sobel

        gx = sobel(a, axis=1)
        gy = sobel(a, axis=0)
        # 梯度能量: |Gx|^2 + |Gy|^2
        gradient_energy = gx**2 + gy**2
        return float(np.mean(gradient_energy))

    elif method == "laplacian":
        from scipy.ndimage import laplace

        lap = laplace(a)
        # 拉普拉斯方差作为清晰度指标
        return float(np.var(lap))

    else:
        raise ValueError(f"未知方法: {method}, 请使用 'tenengrad' 或 'laplacian'")


def compute_sharpness_retention(
    arr_before: NDArray,
    arr_after: NDArray,
    nodata: Optional[float] = None,
) -> Dict[str, float]:
    """
    比较羽化前后的清晰度保留率

    Parameters
    ----------
    arr_before, arr_after : 2D数组
        羽化前后的图像
    nodata : float, optional
        NoData值

    Returns
    -------
    dict: sharpness_before, sharpness_after, retention_ratio
    """
    s_before = compute_sharpness(arr_before, method="tenengrad", nodata=nodata)
    s_after = compute_sharpness(arr_after, method="tenengrad", nodata=nodata)

    if np.isnan(s_before) or np.isnan(s_after) or s_before == 0:
        retention = np.nan
    else:
        retention = s_after / s_before

    return {
        "sharpness_before": s_before,
        "sharpness_after": s_after,
        "retention_ratio": retention,
    }


def compute_texture_loss_in_feather_zone(
    arr_before: NDArray,
    arr_after: NDArray,
    feather_mask: NDArray,
    nodata: Optional[float] = None,
) -> Dict[str, float]:
    """
    测量羽化区域的纹理损失

    使用局部标准差作为纹理复杂度指标。
    纹理损失比 = 1 - (羽化后纹理 / 羽化前纹理)

    Parameters
    ----------
    arr_before, arr_after : 2D数组
        羽化前后的图像
    feather_mask : 2D布尔数组
        羽化区域掩码（True=羽化区域）
    nodata : float, optional
        NoData值

    Returns
    -------
    dict: texture_before, texture_after, texture_loss_ratio
    """
    from scipy.ndimage import uniform_filter

    a_before, mask_before = _safe_array(arr_before, nodata)
    a_after, mask_after = _safe_array(arr_after, nodata)

    combined_mask = feather_mask & mask_before & mask_after

    if not combined_mask.any():
        return {
            "texture_before": np.nan,
            "texture_after": np.nan,
            "texture_loss_ratio": np.nan,
        }

    # 使用局部标准差衡量纹理复杂度（窗口大小7x7）
    window_size = 7
    mean_before = uniform_filter(a_before, size=window_size)
    mean_sq_before = uniform_filter(a_before**2, size=window_size)
    local_var_before = np.maximum(mean_sq_before - mean_before**2, 0)
    texture_before = np.mean(np.sqrt(local_var_before[combined_mask]))

    mean_after = uniform_filter(a_after, size=window_size)
    mean_sq_after = uniform_filter(a_after**2, size=window_size)
    local_var_after = np.maximum(mean_sq_after - mean_after**2, 0)
    texture_after = np.mean(np.sqrt(local_var_after[combined_mask]))

    if texture_before == 0 or np.isnan(texture_before):
        loss_ratio = np.nan
    else:
        loss_ratio = 1.0 - (texture_after / texture_before)

    return {
        "texture_before": float(texture_before),
        "texture_after": float(texture_after),
        "texture_loss_ratio": float(loss_ratio),
    }


def compute_all_seam_metrics(
    arr_a: NDArray,
    arr_b: NDArray,
    seam_cols: NDArray,
    band: int = 0,
    strip_width: int = 10,
    nodata_a: Optional[float] = None,
    nodata_b: Optional[float] = None,
) -> Dict[str, float]:
    """
    一次性计算接缝对的所有指标

    Parameters
    ----------
    arr_a, arr_b : 2D或3D数组
        重叠区域图像数据
    seam_cols : 1D数组
        每行接缝的列位置
    band : int
        用于单波段指标的波段索引
    strip_width : int
        接缝两侧比较宽度
    nodata_a, nodata_b : float, optional
        NoData值

    Returns
    -------
    dict: 包含所有接缝指标的字典
    """
    # 提取单波段用于单波段指标
    if arr_a.ndim == 3:
        a_band = arr_a[:, :, band]
        b_band = arr_b[:, :, band]
    else:
        a_band = arr_a
        b_band = arr_b

    metrics = {}

    # 辐射跳变
    jump = compute_seam_jump(a_band, b_band, seam_cols, strip_width=strip_width,
                             nodata_a=nodata_a, nodata_b=nodata_b)
    metrics.update(jump)

    # 梯度不连续
    grad = compute_seam_gradient_discontinuity(a_band, b_band, seam_cols,
                                               nodata_a=nodata_a, nodata_b=nodata_b)
    metrics.update(grad)

    # 多波段色差
    color = compute_seam_color_distance(arr_a, arr_b, seam_cols,
                                        nodata_a=nodata_a, nodata_b=nodata_b)
    metrics.update(color)

    # 接缝处清晰度
    metrics["sharpness_a"] = compute_sharpness(a_band, nodata=nodata_a)
    metrics["sharpness_b"] = compute_sharpness(b_band, nodata=nodata_b)

    return metrics


# ---------------------------------------------------------------------------
# 接缝质量综合指标
# ---------------------------------------------------------------------------

def compute_seam_quality(
    arr_a: NDArray,
    arr_b: NDArray,
    seam_cols: NDArray,
    strip_width: int = 10,
    nodata_a: Optional[float] = None,
    nodata_b: Optional[float] = None,
) -> Dict[str, float]:
    """
    综合接缝质量指标：辐射跳变 + 梯度不连续 + 色差。

    对于多波段数组 (3D)，逐波段计算指标后取均值。
    对于单波段数组 (2D)，直接计算。

    Returns
    -------
    dict: mean_jump, p95_jump, mean_discontinuity, p95_discontinuity,
          mean_distance, p95_distance
    """
    if arr_a.size == 0 or arr_b.size == 0:
        return {
            "mean_jump": np.nan, "p95_jump": np.nan,
            "mean_discontinuity": np.nan, "p95_discontinuity": np.nan,
            "mean_distance": np.nan, "p95_distance": np.nan,
        }

    if arr_a.ndim == 3 and arr_b.ndim == 3:
        n_bands = arr_a.shape[2]
        all_jumps, all_grads, all_dists = [], [], []
        for b in range(n_bands):
            a_band = arr_a[:, :, b]
            b_band = arr_b[:, :, b]
            # 排除 NoData 一致的波段
            valid_a = np.isfinite(a_band)
            valid_b = np.isfinite(b_band)
            if not (valid_a.any() and valid_b.any()):
                continue
            jump = compute_seam_jump(a_band, b_band, seam_cols,
                                     strip_width=strip_width,
                                     nodata_a=nodata_a, nodata_b=nodata_b)
            grad = compute_seam_gradient_discontinuity(a_band, b_band, seam_cols,
                                                       nodata_a=nodata_a, nodata_b=nodata_b)
            color = compute_seam_color_distance(a_band, b_band, seam_cols,
                                                nodata_a=nodata_a, nodata_b=nodata_b)
            all_jumps.append(jump)
            all_grads.append(grad)
            all_dists.append(color)

        if not all_jumps:
            return {
                "mean_jump": np.nan, "p95_jump": np.nan,
                "mean_discontinuity": np.nan, "p95_discontinuity": np.nan,
                "mean_distance": np.nan, "p95_distance": np.nan,
            }

        # 均值跨波段
        def _avg(key):
            vals = [d[key] for d in all_jumps + all_grads + all_dists if key in d and np.isfinite(d[key])]
            return float(np.mean(vals)) if vals else np.nan

        jump_mean = float(np.nanmean([j["mean_jump"] for j in all_jumps]))
        jump_p95 = float(np.nanmean([j["p95_jump"] for j in all_jumps]))
        grad_mean = float(np.nanmean([g["mean_discontinuity"] for g in all_grads]))
        grad_p95 = float(np.nanmean([g["p95_discontinuity"] for g in all_grads]))
        dist_mean = float(np.nanmean([d["mean_distance"] for d in all_dists]))
        dist_p95 = float(np.nanmean([d["p95_distance"] for d in all_dists]))
        return {
            "mean_jump": jump_mean, "p95_jump": jump_p95,
            "mean_discontinuity": grad_mean, "p95_discontinuity": grad_p95,
            "mean_distance": dist_mean, "p95_distance": dist_p95,
        }

    # 2D 单波段
    jump = compute_seam_jump(arr_a, arr_b, seam_cols,
                             strip_width=strip_width,
                             nodata_a=nodata_a, nodata_b=nodata_b)
    grad = compute_seam_gradient_discontinuity(arr_a, arr_b, seam_cols,
                                               nodata_a=nodata_a, nodata_b=nodata_b)
    color = compute_seam_color_distance(arr_a, arr_b, seam_cols,
                                        nodata_a=nodata_a, nodata_b=nodata_b)
    return {**jump, **grad, **color}


def compute_seam_length(
    seam_mask: NDArray,
    pixel_size: float = 1.0,
) -> float:
    """
    计算接缝线长度（像素数 × 像素尺寸）。

    直接使用 mask（True=接缝像素），不要求特定波段数。
    空 mask 返回 0。

    Parameters
    ----------
    seam_mask : NDArray
        2D 布尔/数值数组，True/非零表示接缝位置。
    pixel_size : float
        像素尺寸（地面分辨率），用于将像素数转换为实际长度。

    Returns
    -------
    float : 接缝长度（像素单位 × pixel_size）
    """
    if seam_mask is None or seam_mask.size == 0:
        return 0.0
    mask = np.asarray(seam_mask, dtype=bool)
    return float(np.sum(mask)) * pixel_size


def compute_seam_transition_width(
    arr_a: NDArray,
    arr_b: NDArray,
    seam_cols: NDArray,
    nodata_a: Optional[float] = None,
    nodata_b: Optional[float] = None,
) -> Dict[str, float]:
    """
    计算接缝过渡宽度：梯度变化最大的两侧像素数。

    从接缝线向两侧扩展，找到梯度下降到峰值 10% 处的距离。
    支持 2D 和 3D（多波段，逐波段计算后均值）。

    Returns
    -------
    dict: mean_transition_width, p95_transition_width
    """
    if arr_a.size == 0 or arr_b.size == 0:
        return {"mean_transition_width": np.nan, "p95_transition_width": np.nan}

    if arr_a.ndim == 3:
        n_bands = arr_a.shape[2]
        widths = []
        for b in range(n_bands):
            a_band = arr_a[:, :, b]
            b_band = arr_b[:, :, b]
            result = compute_seam_transition_width(a_band, b_band, seam_cols,
                                                   nodata_a=nodata_a, nodata_b=nodata_b)
            w = result["mean_transition_width"]
            if np.isfinite(w):
                widths.append(w)
        mean_w = float(np.mean(widths)) if widths else np.nan
        p95_w = float(np.percentile(widths, 95)) if len(widths) >= 2 else mean_w
        return {"mean_transition_width": mean_w, "p95_transition_width": p95_w}

    # 2D case
    a, mask_a = _safe_array(arr_a, nodata_a)
    b, mask_b = _safe_array(arr_b, nodata_b)
    rows, cols = a.shape
    seam_cols = np.asarray(seam_cols, dtype=np.int64)

    transition_widths: List[float] = []
    for r in range(rows):
        sc = int(seam_cols[r])
        if sc < 2 or sc + 2 >= cols:
            continue

        # Compute horizontal gradient magnitude around the seam
        left_side = []
        for offset in range(1, min(sc + 1, cols // 2)):
            c = sc - offset
            if mask_a[r, c] and mask_a[r, c - 1]:
                left_side.append(abs(float(a[r, c]) - float(a[r, c - 1])))
            else:
                break

        right_side = []
        for offset in range(1, min(cols - sc, cols // 2)):
            c = sc + offset
            if mask_b[r, c] and mask_b[r, c - 1]:
                right_side.append(abs(float(b[r, c]) - float(b[r, c - 1])))
            else:
                break

        if not left_side or not right_side:
            continue

        # Peak at seam (average of last left and first right)
        peak = (left_side[-1] + right_side[0]) / 2.0 if left_side and right_side else 0.0
        if peak < 1e-10:
            continue

        threshold = peak * 0.1

        # Left width: how far before gradient drops below threshold
        lw = 0
        for i, val in enumerate(reversed(left_side)):
            if val >= threshold:
                lw = i + 1
                break

        # Right width
        rw = 0
        for i, val in enumerate(right_side):
            if val >= threshold:
                rw = i + 1
                break

        transition_widths.append(float(lw + rw))

    if not transition_widths:
        return {"mean_transition_width": np.nan, "p95_transition_width": np.nan}

    return {
        "mean_transition_width": float(np.mean(transition_widths)),
        "p95_transition_width": float(np.percentile(transition_widths, 95)),
    }


def _build_seam_line(
    overlaps: List[Tuple[int, int]],
    transforms: List,
    shape: Tuple[int, int],
) -> NDArray:
    """
    根据重叠区域构建简化接缝线（垂直中线）

    Parameters
    ----------
    overlaps : list of (row, col)
        重叠区域坐标列表
    transforms : list
        各瓦片的仿射变换
    shape : (H, W)
        输出网格尺寸

    Returns
    -------
    1D数组，每行的接缝列位置
    """
    rows, cols = shape
    seam_cols = np.full(rows, cols // 2, dtype=np.int64)
    return seam_cols


def evaluate_mosaic_seams(
    arrays: List[NDArray],
    transforms: List,
    nodata_values: List[Optional[float]],
    output_grid_info: Dict,
    feather_widths: List[int] = [2, 4, 8, 16, 32, 64],
) -> List[Dict]:
    """
    评估不同羽化宽度下的马赛克接缝质量

    Parameters
    ----------
    arrays : list of NDArray
        输入图像列表（2D或3D）
    transforms : list
        各图像的仿射变换矩阵
    nodata_values : list of float
        各图像的NoData值
    output_grid_info : dict
        输出网格信息，包含shape, bounds等
    feather_widths : list of int
        待测试的羽化宽度列表（像素）

    Returns
    -------
    list of dict: 每个羽化宽度对应的指标字典
    """
    from scipy.ndimage import uniform_filter

    results = []

    # 获取网格形状
    if "shape" in output_grid_info:
        grid_rows, grid_cols = output_grid_info["shape"]
    else:
        grid_rows = output_grid_info.get("height", 1024)
        grid_cols = output_grid_info.get("width", 1024)

    # 生成简化的接缝线（垂直中线）
    seam_cols = np.full(grid_rows, grid_cols // 2, dtype=np.int64)

    for fw in feather_widths:
        result = {"feather_width": fw}

        # 对每对相邻瓦片计算指标
        for i in range(len(arrays) - 1):
            arr_a = arrays[i]
            arr_b = arrays[i + 1]

            # 确保2D
            if arr_a.ndim == 3:
                a_2d = arr_a[:, :, 0]
                b_2d = arr_b[:, :, 0]
            else:
                a_2d = arr_a
                b_2d = arr_b

            # 确保尺寸一致（取最小）
            min_rows = min(a_2d.shape[0], b_2d.shape[0])
            min_cols = min(a_2d.shape[1], b_2d.shape[1])

            if min_rows == 0 or min_cols == 0:
                continue

            a_crop = a_2d[:min_rows, :min_cols]
            b_crop = b_2d[:min_rows, :min_cols]

            nd_a = nodata_values[i] if i < len(nodata_values) else None
            nd_b = nodata_values[i + 1] if i + 1 < len(nodata_values) else None

            # 使用该瓦片对的接缝线
            pair_seam = seam_cols[:min_rows]

            # 计算接缝指标
            pair_metrics = compute_all_seam_metrics(
                a_crop, b_crop, pair_seam,
                strip_width=fw,
                nodata_a=nd_a, nodata_b=nd_b,
            )

            # 添加前缀避免键冲突
            prefix = f"pair_{i}_{i+1}_"
            for k, v in pair_metrics.items():
                result[prefix + k] = v

            # 计算羽化后的指标
            mask_a, mask_b = np.ones_like(a_crop, dtype=bool), np.ones_like(b_crop, dtype=bool)
            if nd_a is not None:
                mask_a &= a_crop != nd_a
            if nd_b is not None:
                mask_b &= b_crop != nd_b

            # 简单线性羽化
            feather_mask = np.zeros((min_rows, min_cols), dtype=bool)
            for r in range(min_rows):
                sc = int(pair_seam[r])
                left = max(0, sc - fw)
                right = min(min_cols, sc + fw)
                feather_mask[r, left:right] = True

            if feather_mask.any():
                # 羽化混合
                a_f = a_crop.copy().astype(np.float64)
                b_f = b_crop.copy().astype(np.float64)

                for r in range(min_rows):
                    sc = int(pair_seam[r])
                    left = max(0, sc - fw)
                    right = min(min_cols, sc + fw)
                    span = right - left
                    if span > 0:
                        alpha = np.linspace(0, 1, span)
                        for c_idx in range(left, right):
                            a_f[r, c_idx] = a_crop[r, c_idx] * (1 - alpha[c_idx - left]) + b_crop[r, c_idx] * alpha[c_idx - left]

                # 清晰度保留
                sharp_ret = compute_sharpness_retention(a_crop, a_f, nodata=nd_a)
                result[prefix + "sharpness_retention"] = sharp_ret["retention_ratio"]

                # 纹理损失
                tex_loss = compute_texture_loss_in_feather_zone(a_crop, a_f, feather_mask, nodata=nd_a)
                result[prefix + "texture_loss_ratio"] = tex_loss["texture_loss_ratio"]

        # 汇总所有瓦片对的平均指标
        jump_keys = [k for k in result if "mean_jump" in k and k.startswith("pair_")]
        if jump_keys:
            result["avg_mean_jump"] = float(np.nanmean([result[k] for k in jump_keys]))

        grad_keys = [k for k in result if "mean_discontinuity" in k and k.startswith("pair_")]
        if grad_keys:
            result["avg_mean_discontinuity"] = float(np.nanmean([result[k] for k in grad_keys]))

        color_keys = [k for k in result if "mean_distance" in k and k.startswith("pair_")]
        if color_keys:
            result["avg_mean_distance"] = float(np.nanmean([result[k] for k in color_keys]))

        sharp_keys = [k for k in result if "sharpness_retention" in k and k.startswith("pair_")]
        if sharp_keys:
            result["avg_sharpness_retention"] = float(np.nanmean([result[k] for k in sharp_keys]))

        tex_keys = [k for k in result if "texture_loss_ratio" in k and k.startswith("pair_")]
        if tex_keys:
            result["avg_texture_loss_ratio"] = float(np.nanmean([result[k] for k in tex_keys]))

        results.append(result)

    return results
