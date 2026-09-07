"""
VOLRN 系数诊断与 RBF 位移场诊断模块

提供局部辐射归一化后处理的诊断工具：
  1. analyze_volrn_coefficients — 分析逐块 a/b 系数分布、异常值与空间梯度
  2. check_local_overenhancement — 检测局部过增强现象
  3. analyze_rbf_displacement — 分析 RBF 位移场的质量指标
  4. save_diagnostics_outputs — 将诊断结果保存到 CSV / JSON / GeoJSON
"""

import json
import csv
import os
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy.ndimage import sobel
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to Python builtins for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj

def _safe_stats(arr: np.ndarray) -> Dict[str, float]:
    """对一维数组计算常用统计量，安全处理 NaN / Inf。"""
    arr64 = arr.astype(np.float64)
    valid = np.isfinite(arr64)
    vals = arr64[valid]
    if len(vals) == 0:
        return {
            'min': np.nan, 'max': np.nan, 'mean': np.nan, 'std': np.nan,
            'p01': np.nan, 'p05': np.nan, 'p50': np.nan,
            'p95': np.nan, 'p99': np.nan,
        }
    return {
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'p01': float(np.percentile(vals, 1)),
        'p05': float(np.percentile(vals, 5)),
        'p50': float(np.percentile(vals, 50)),
        'p95': float(np.percentile(vals, 95)),
        'p99': float(np.percentile(vals, 99)),
    }


def _mad(arr: np.ndarray) -> float:
    """计算中位绝对偏差 (MAD)，用于稳健离群检测。"""
    arr64 = arr.astype(np.float64)
    valid = np.isfinite(arr64)
    vals = arr64[valid]
    if len(vals) < 2:
        return 0.0
    median = np.median(vals)
    return float(np.median(np.abs(vals - median)))


def _robust_zscore(arr: np.ndarray) -> np.ndarray:
    """基于 MAD 的稳健 Z-score。"""
    arr64 = arr.astype(np.float64)
    valid = np.isfinite(arr64)
    vals = arr64[valid]
    if len(vals) < 2:
        return np.zeros_like(arr64)
    median = np.median(vals)
    mad = np.median(np.abs(vals - median))
    if mad < 1e-12:
        return np.zeros_like(arr64)
    # 使用 1.4826 * MAD 作为尺度估计（假设近似正态）
    scale = 1.4826 * mad
    z = np.full_like(arr64, np.nan)
    z[valid] = (arr64[valid] - median) / scale
    return z


def _gradient_energy(arr: np.ndarray) -> float:
    """计算 2D 数组的梯度能量（梯度幅值的平方和）。"""
    arr64 = arr.astype(np.float64)
    gx = sobel(arr64, axis=1, mode='constant')
    gy = sobel(arr64, axis=0, mode='constant')
    return float(np.sum(gx**2 + gy**2))


# ---------------------------------------------------------------------------
# 1. VOLRN 系数分析
# ---------------------------------------------------------------------------

def analyze_volrn_coefficients(
    block_coefficients: np.ndarray,
    blocks_info: List[Any],
    nodata_values: Optional[List[Optional[float]]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    分析 VOLRN 逐块 a/b 系数分布、异常值和空间梯度。

    参数
    ----------
    block_coefficients : np.ndarray, shape (n_bands, n_blocks, 2)
        逐块系数，axis=2 第 0 列为 a，第 1 列为 b。
    blocks_info : list of BlockInfo or dict
        每个块的信息，需包含 block_id, image_idx, grid_m, grid_n,
        center_x, center_y, mu, sigma。
    nodata_values : list of float or None, optional
        每幅影像的 nodata 值（当前用于过滤无效统计）。
    output_dir : str or None
        若指定则在该目录写入 CSV。

    返回
    -------
    dict 包含：
        per_band_stats : list of dict — 逐波段统计
        outlier_blocks : list of dict — 被标记为异常的块
        summary : dict — 全局摘要
    """
    if block_coefficients is None or block_coefficients.size == 0:
        logger.warning("block_coefficients 为空或 None，跳过 VOLRN 系数分析")
        return {
            'per_band_stats': [],
            'outlier_blocks': [],
            'summary': {'n_bands': 0, 'n_blocks': 0, 'skipped': True},
        }

    n_bands, n_blocks, _ = block_coefficients.shape

    per_band_stats: List[Dict[str, Any]] = []
    outlier_blocks: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        'n_bands': n_bands,
        'n_blocks': n_blocks,
    }

    total_a_neg_count = 0

    for b_idx in range(n_bands):
        a_vals = block_coefficients[b_idx, :, 0].copy()
        b_vals = block_coefficients[b_idx, :, 1].copy()

        # a 系数统计
        a_stats = _safe_stats(a_vals)
        a_mad = _mad(a_vals)
        a_z = _robust_zscore(a_vals)
        # a <= 0 计数
        a_neg_mask = np.isfinite(a_vals) & (a_vals <= 0)
        a_neg_count = int(np.sum(a_neg_mask))
        total_a_neg_count += a_neg_count

        # b 系数统计
        b_stats = _safe_stats(b_vals)
        b_mad = _mad(b_vals)

        # b / band_dynamic_range
        band_dr = 1.0  # Reference dynamic range for normalization (data-dependent in practice)
        b_over_dr = b_stats['mean'] / band_dr if band_dr > 1e-12 else np.nan

        # 异常块检测（|z| > 3 的块）
        band_outliers: List[Dict[str, Any]] = []
        for k in range(n_blocks):
            zk = a_z[k]
            if np.isfinite(zk) and abs(zk) > 3.0:
                blk = blocks_info[k]
                outlier_record = {
                    'band': b_idx,
                    'block_id': blk['block_id'] if isinstance(blk, dict) else blk.block_id,
                    'image_idx': blk['image_idx'] if isinstance(blk, dict) else blk.image_idx,
                    'grid_m': blk['grid_m'] if isinstance(blk, dict) else blk.grid_m,
                    'grid_n': blk['grid_n'] if isinstance(blk, dict) else blk.grid_n,
                    'a': float(a_vals[k]),
                    'b': float(b_vals[k]),
                    'a_zscore': float(zk),
                }
                band_outliers.append(outlier_record)
                outlier_blocks.append(outlier_record)

        # 空间梯度：相邻块系数差值
        spatial_grad_a_list: List[float] = []
        spatial_grad_b_list: List[float] = []
        for k1 in range(n_blocks):
            for k2 in range(k1 + 1, n_blocks):
                bi = blocks_info[k1]
                bj = blocks_info[k2]
                gm_i = bi['grid_m'] if isinstance(bi, dict) else bi.grid_m
                gn_i = bi['grid_n'] if isinstance(bi, dict) else bi.grid_n
                gm_j = bj['grid_m'] if isinstance(bj, dict) else bj.grid_m
                gn_j = bj['grid_n'] if isinstance(bj, dict) else bj.grid_n
                # 仅考虑相邻块（Chebyshev 距离 = 1）
                if max(abs(gm_i - gm_j), abs(gn_i - gn_j)) == 1:
                    da = abs(a_vals[k1] - a_vals[k2])
                    db = abs(b_vals[k1] - b_vals[k2])
                    if np.isfinite(da):
                        spatial_grad_a_list.append(da)
                    if np.isfinite(db):
                        spatial_grad_b_list.append(db)

        grad_a_stats = _safe_stats(np.array(spatial_grad_a_list)) if spatial_grad_a_list else {}
        grad_b_stats = _safe_stats(np.array(spatial_grad_b_list)) if spatial_grad_b_list else {}

        # RMS of a/b residuals: sqrt(mean(a^2)), sqrt(mean(b^2))
        # Measures deviation from identity transform (a=1, b=0)
        finite_a = a_vals[np.isfinite(a_vals)]
        finite_b = b_vals[np.isfinite(b_vals)]
        a_rms = float(np.sqrt(np.mean((finite_a - 1) ** 2))) if len(finite_a) > 0 else np.nan  # Deviation from identity (a=1)
        b_rms = float(np.sqrt(np.mean(finite_b ** 2))) if len(finite_b) > 0 else np.nan

        band_stat = {
            'band': b_idx,
            'a': a_stats,
            'a_mad': a_mad,
            'a_rms': a_rms,
            'b': b_stats,
            'b_mad': b_mad,
            'b_rms': b_rms,
            'a_neg_count': a_neg_count,
            'b_over_dynamic_range': b_over_dr,
            'spatial_gradient_a': grad_a_stats,
            'spatial_gradient_b': grad_b_stats,
            'n_outliers': len(band_outliers),
        }
        per_band_stats.append(band_stat)

    summary['total_a_neg_count'] = total_a_neg_count
    summary['total_outlier_blocks'] = len(outlier_blocks)
    summary['per_band'] = per_band_stats

    # Global RMS of all a/b coefficients across bands
    all_a = block_coefficients[:, :, 0].ravel()
    all_b = block_coefficients[:, :, 1].ravel()
    all_a_finite = all_a[np.isfinite(all_a)]
    all_b_finite = all_b[np.isfinite(all_b)]
    summary['global_a_rms'] = float(np.sqrt(np.mean((all_a_finite - 1) ** 2))) if len(all_a_finite) > 0 else np.nan  # Deviation from identity
    summary['global_b_rms'] = float(np.sqrt(np.mean(all_b_finite ** 2))) if len(all_b_finite) > 0 else np.nan

    # Convert everything to JSON-safe types
    summary = _json_safe(summary)
    outlier_blocks = _json_safe(outlier_blocks)

    # 可选：写 CSV
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'volrn_coefficients.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'band', 'block_id', 'image_idx', 'grid_m', 'grid_n',
                'a', 'b', 'a_zscore', 'is_outlier',
            ])
            for b_idx in range(n_bands):
                a_vals = block_coefficients[b_idx, :, 0]
                b_vals = block_coefficients[b_idx, :, 1]
                a_z = _robust_zscore(a_vals)
                for k in range(n_blocks):
                    blk = blocks_info[k]
                    blk_id = blk['block_id'] if isinstance(blk, dict) else blk.block_id
                    img_idx = blk['image_idx'] if isinstance(blk, dict) else blk.image_idx
                    gm = blk['grid_m'] if isinstance(blk, dict) else blk.grid_m
                    gn = blk['grid_n'] if isinstance(blk, dict) else blk.grid_n
                    zk = a_z[k]
                    is_out = bool(np.isfinite(zk) and abs(zk) > 3.0)
                    writer.writerow([
                        b_idx, blk_id, img_idx, gm, gn,
                        float(a_vals[k]), float(b_vals[k]),
                        float(zk) if np.isfinite(zk) else '',
                        is_out,
                    ])

    return {
        'per_band_stats': per_band_stats,
        'outlier_blocks': outlier_blocks,
        'summary': summary,
    }


# ---------------------------------------------------------------------------
# 2. 检测局部过增强
# ---------------------------------------------------------------------------

def check_local_overenhancement(
    arrays_before: List[np.ndarray],
    arrays_after: List[np.ndarray],
    nodata_values: List[Optional[float]],
    block_size: int = 200,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    检测 VOLRN 是否导致局部过增强。

    评估指标：
      - 局部标准差比值（after/before）：> 1.5 视为过增强
      - 局部梯度能量比值（after/before）：> 2.0 视为过增强
      - 饱和像素比率变化

    参数
    ----------
    arrays_before, arrays_after : list of np.ndarray
        归一化前后的影像数组 (bands, rows, cols)。
    nodata_values : list of float or None
    block_size : int
        分块大小（像素），用于逐块比较。
    output_dir : str or None

    返回
    -------
    dict 包含逐块和聚合统计。
    """
    assert len(arrays_before) == len(arrays_after)
    n_images = len(arrays_before)
    n_bands = arrays_before[0].shape[0]

    per_image_results: List[Dict[str, Any]] = []
    all_std_ratios: List[float] = []
    all_grad_ratios: List[float] = []

    for img_idx in range(n_images):
        arr_b = arrays_before[img_idx].astype(np.float64)
        arr_a = arrays_after[img_idx].astype(np.float64)
        nd = nodata_values[img_idx]
        rows, cols = arr_b.shape[1], arr_b.shape[2]

        # 构建有效掩膜
        if nd is not None:
            valid_mask = (arr_b != nd) & np.isfinite(arr_b) & (arr_a != nd) & np.isfinite(arr_a)
        else:
            valid_mask = np.isfinite(arr_b) & np.isfinite(arr_a)

        block_std_ratios: List[float] = []
        block_grad_ratios: List[float] = []
        block_sat_change: List[float] = []

        for r0 in range(0, rows, block_size):
            for c0 in range(0, cols, block_size):
                r1 = min(r0 + block_size, rows)
                c1 = min(c0 + block_size, cols)

                # 提取块
                patch_b = arr_b[:, r0:r1, c0:c1]
                patch_a = arr_a[:, r0:r1, c0:c1]
                patch_valid = valid_mask[:, r0:r1, c0:c1]

                # 逐波段统计
                for b_idx in range(n_bands):
                    pv = patch_valid[b_idx]
                    if pv.sum() < 10:
                        continue

                    vb = patch_b[b_idx][pv]
                    va = patch_a[b_idx][pv]

                    # 局部标准差比值
                    std_b = float(np.std(vb))
                    std_a = float(np.std(va))
                    if std_b > 1e-10:
                        ratio = std_a / std_b
                        block_std_ratios.append(ratio)
                        all_std_ratios.append(ratio)

                    # 局部梯度能量比值
                    grad_b = _gradient_energy(patch_b[b_idx])
                    grad_a = _gradient_energy(patch_a[b_idx])
                    if grad_b > 1e-10:
                        gr = grad_a / grad_b
                        block_grad_ratios.append(gr)
                        all_grad_ratios.append(gr)

                    # 饱和像素：接近 0 或 1（归一化后）
                    # 这里使用原始数据范围检查极端值
                    sat_b = float(np.sum((vb < 0.01) | (vb > 0.99))) / len(vb)
                    sat_a = float(np.sum((va < 0.01) | (va > 0.99))) / len(va)
                    block_sat_change.append(sat_a - sat_b)

        img_result = {
            'image_idx': img_idx,
            'mean_std_ratio': float(np.mean(block_std_ratios)) if block_std_ratios else np.nan,
            'p95_std_ratio': float(np.percentile(block_std_ratios, 95)) if block_std_ratios else np.nan,
            'mean_grad_ratio': float(np.mean(block_grad_ratios)) if block_grad_ratios else np.nan,
            'p95_grad_ratio': float(np.percentile(block_grad_ratios, 95)) if block_grad_ratios else np.nan,
            'mean_sat_change': float(np.mean(block_sat_change)) if block_sat_change else np.nan,
            'n_blocks': len(block_std_ratios),
            'n_overenhanced_std': sum(1 for r in block_std_ratios if r > 1.5),
            'n_overenhanced_grad': sum(1 for r in block_grad_ratios if r > 2.0),
        }
        per_image_results.append(img_result)

    summary = {
        'n_images': n_images,
        'global_mean_std_ratio': float(np.mean(all_std_ratios)) if all_std_ratios else np.nan,
        'global_mean_grad_ratio': float(np.mean(all_grad_ratios)) if all_grad_ratios else np.nan,
        'total_overenhanced_std': sum(ir['n_overenhanced_std'] for ir in per_image_results),
        'total_overenhanced_grad': sum(ir['n_overenhanced_grad'] for ir in per_image_results),
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, 'overenhancement.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(_json_safe({
                'per_image': per_image_results,
                'summary': summary,
            }), f, indent=2, ensure_ascii=False)

    return {
        'per_image': per_image_results,
        'summary': summary,
    }


# ---------------------------------------------------------------------------
# 3. RBF 位移场诊断
# ---------------------------------------------------------------------------

def analyze_rbf_displacement(
    local_dx: np.ndarray,
    local_dy: np.ndarray,
    control_points_xy: np.ndarray,
    valid_mask: np.ndarray,
    clip_limit: Optional[float] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    分析 RBF 位移场的质量指标。

    参数
    ----------
    local_dx, local_dy : np.ndarray, shape (rows, cols)
        RBF 插值后的位移场。
    control_points_xy : np.ndarray, shape (n_pts, 2)
        控制点坐标 (x, y)。
    valid_mask : np.ndarray, shape (rows, cols), dtype=bool
        有效区域掩膜。
    clip_limit : float or None
        位移裁剪阈值（像素），用于统计超限像素数。
    output_dir : str or None

    返回
    -------
    dict 包含位移统计、梯度统计、控制点密度等。
    """
    dx64 = local_dx.astype(np.float64)
    dy64 = local_dy.astype(np.float64)
    vm = valid_mask.astype(bool)

    # 有效像素位移
    dx_valid = dx64[vm]
    dy_valid = dy64[vm]
    mag_valid = np.sqrt(dx_valid**2 + dy_valid**2)

    # 位移统计
    dx_stats = _safe_stats(dx_valid)
    dy_stats = _safe_stats(dy_valid)
    mag_stats = _safe_stats(mag_valid)

    # 补充 P90
    if len(mag_valid) > 0:
        mag_stats['p90'] = float(np.percentile(mag_valid, 90))
    else:
        mag_stats['p90'] = np.nan

    # NaN / Inf 计数
    total_pixels = int(dx64.size)
    nan_count = int(np.sum(~np.isfinite(dx64) | ~np.isfinite(dy64)))
    inf_count = int(np.sum(np.isinf(dx64) | np.isinf(dy64)))

    # 梯度统计
    gx_dx = sobel(dx64, axis=1, mode='constant')
    gy_dx = sobel(dx64, axis=0, mode='constant')
    gx_dy = sobel(dy64, axis=1, mode='constant')
    gy_dy = sobel(dy64, axis=0, mode='constant')

    grad_mag = np.sqrt(gx_dx**2 + gy_dx**2 + gx_dy**2 + gy_dy**2)
    grad_valid = grad_mag[vm]
    grad_stats = _safe_stats(grad_valid)

    # 裁剪限制
    clip_count = 0
    clip_ratio = 0.0
    if clip_limit is not None and clip_limit > 0:
        clip_mask = np.abs(dx_valid) > clip_limit | np.abs(dy_valid) > clip_limit
        clip_count = int(np.sum(clip_mask))
        clip_ratio = clip_count / len(dx_valid) if len(dx_valid) > 0 else 0.0

    # 控制点密度与最近邻距离
    n_pts = len(control_points_xy)
    nn_dists: List[float] = []
    density = 0.0
    hull_coverage = 0.0

    if n_pts > 2:
        # 最近邻距离
        from scipy.spatial import KDTree
        tree = KDTree(control_points_xy)
        dists, _ = tree.query(control_points_xy, k=2)
        nn_dists = dists[:, 1].tolist()  # 排除自身

        # 控制点密度 = 有效区域内每平方像素的控制点数
        valid_area = float(np.sum(vm))
        density = n_pts / valid_area if valid_area > 0 else 0.0

        # 凸包覆盖率
        try:
            hull = ConvexHull(control_points_xy)
            hull_area = hull.volume if control_points_xy.shape[1] == 2 else hull.area
            # 凸包面积占有效面积比例
            hull_coverage = hull_area / valid_area if valid_area > 0 else 0.0
        except Exception:
            hull_coverage = 0.0

    nn_stats = _safe_stats(np.array(nn_dists)) if nn_dists else {}

    # 外推区域比率（无控制点邻域的区域）
    # 简化：距离最近控制点超过 block_size 的像素视为外推区域
    extrapolation_ratio = 0.0
    if n_pts > 0:
        # 用下采样避免内存问题
        step = max(1, min(local_dx.shape[0], local_dx.shape[1]) // 200)
        ds_valid = vm[::step, ::step]
        ds_points_y, ds_points_x = np.where(ds_valid)
        if len(ds_points_y) > 0:
            # 像素坐标
            ds_xy = np.column_stack([ds_points_x.astype(np.float64),
                                     ds_points_y.astype(np.float64)])
            # 控制点坐标（像素）
            ctrl_xy_px = control_points_xy.astype(np.float64)
            tree_ds = KDTree(ctrl_xy_px)
            dists_ds, _ = tree_ds.query(ds_xy)
            # 超过 block_size 的视为外推
            ext_step = step  # 近似面积比
            ext_count = int(np.sum(dists_ds > step * 10))  # 10 倍步长作为外推阈值
            extrapolation_ratio = ext_count / len(ds_points_y) if len(ds_points_y) > 0 else 0.0

    result = {
        'dx_statistics': dx_stats,
        'dy_statistics': dy_stats,
        'magnitude_statistics': mag_stats,
        'gradient_statistics': grad_stats,
        'nan_pixel_count': nan_count,
        'inf_pixel_count': inf_count,
        'total_valid_pixels': int(np.sum(vm)),
        'total_pixels': total_pixels,
        'clip_limit': clip_limit,
        'clip_pixel_count': clip_count,
        'clip_ratio': clip_ratio,
        'control_point_count': n_pts,
        'control_point_density': density,
        'nearest_neighbor_statistics': nn_stats,
        'hull_coverage_ratio': hull_coverage,
        'extrapolation_ratio': extrapolation_ratio,
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, 'rbf_displacement.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(_json_safe(result), f, indent=2, ensure_ascii=False)

    return result


# ---------------------------------------------------------------------------
# 4. 保存诊断输出
# ---------------------------------------------------------------------------

def save_diagnostics_outputs(
    diagnostics: Dict[str, Any],
    output_dir: str,
    prefix: str = '',
) -> None:
    """
    将诊断结果保存为 CSV、JSON 和可选 GeoJSON。

    参数
    ----------
    diagnostics : dict
        由上述诊断函数返回的结果字典。
    output_dir : str
        输出目录。
    prefix : str
        文件名前缀（可选）。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存 VOLRN 系数 CSV
    if 'per_band_stats' in diagnostics and 'summary' in diagnostics:
        # 已由 analyze_volrn_coefficients 保存 CSV，此处保存汇总 JSON
        json_path = os.path.join(output_dir, f'{prefix}volrn_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(_json_safe(diagnostics), f, indent=2, ensure_ascii=False)

    # 保存过增强结果
    if 'per_image' in diagnostics and 'summary' in diagnostics:
        # 检查是否为 overenhancement 结果
        if any('mean_std_ratio' in img for img in diagnostics.get('per_image', [])):
            json_path = os.path.join(output_dir, f'{prefix}overenhancement.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(_json_safe(diagnostics), f, indent=2, ensure_ascii=False)

    # 保存 RBF 位移场结果
    if 'dx_statistics' in diagnostics:
        json_path = os.path.join(output_dir, f'{prefix}rbf_displacement.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(_json_safe(diagnostics), f, indent=2, ensure_ascii=False)

    # 保存异常块 GeoJSON
    outlier_blocks = diagnostics.get('outlier_blocks', [])
    if outlier_blocks and 'center_x' in outlier_blocks[0]:
        features = []
        for ob in outlier_blocks:
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [ob['center_x'], ob['center_y']],
                },
                'properties': {
                    'band': ob.get('band', -1),
                    'block_id': ob.get('block_id', -1),
                    'image_idx': ob.get('image_idx', -1),
                    'grid_m': ob.get('grid_m', -1),
                    'grid_n': ob.get('grid_n', -1),
                    'a': ob.get('a', 0.0),
                    'b': ob.get('b', 0.0),
                    'a_zscore': ob.get('a_zscore', 0.0),
                },
            }
            features.append(feature)

        geojson = {
            'type': 'FeatureCollection',
            'features': features,
        }
        geojson_path = os.path.join(output_dir, f'{prefix}outlier_blocks.geojson')
        with open(geojson_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)
