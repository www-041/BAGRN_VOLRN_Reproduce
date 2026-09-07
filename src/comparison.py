"""
对比方法模块

实现论文常用的辐射归一化对比方法：
  1. 直方图匹配 (Histogram Matching)
  2. 矩匹配 (Moment Matching)
  3. Wallis 滤波 (Wallis Filter)
"""

import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# 辅助：提取重叠区有效像素
# ---------------------------------------------------------------------------

def _overlap_valid_pixels(array_i, array_j, window_i, window_j,
                          nodata_i, nodata_j, band):
    """提取重叠区有效像素，独立过滤（支持不同分辨率）。"""
    r1s, r1e, c1s, c1e = window_i
    r2s, r2e, c2s, c2e = window_j
    pi = array_i[band, r1s:r1e, c1s:c1e].ravel()
    pj = array_j[band, r2s:r2e, c2s:c2e].ravel()
    mi = np.isfinite(pi)
    mj = np.isfinite(pj)
    if nodata_i is not None:
        mi &= (pi != nodata_i)
    if nodata_j is not None:
        mj &= (pj != nodata_j)
    # 独立过滤，不要求相同长度
    return pi[mi], pj[mj]


# ---------------------------------------------------------------------------
# 1. 直方图匹配 (Histogram Matching)
# ---------------------------------------------------------------------------

def histogram_matching_normalize(
    arrays: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    control_idx: int = 0,
    n_bins: int = 65536,
    spanning_tree: Optional[List[tuple]] = None,
) -> List[np.ndarray]:
    """
    直方图匹配归一化。

    支持 spanning_tree 传播：不与 control 直接重叠的影像，
    沿生成树由父影像向子影像传播。
    """
    n_images = len(arrays)
    n_bands = arrays[0].shape[0]
    result = [arr.astype(np.float64, copy=True) for arr in arrays]

    # 构建父子关系：如果提供了 spanning_tree，用它；否则用重叠关系传播
    if spanning_tree is not None:
        parent_map = {}
        for i, j in spanning_tree:
            parent_map[j] = i
            parent_map.setdefault(i, None)
    else:
        # 自动检测：找与 control 直接重叠的影像
        parent_map = {control_idx: None}
        for ov in overlaps:
            pi, pj = ov['idx_i'], ov['idx_j']
            if pi == control_idx and pj not in parent_map:
                parent_map[pj] = pi
            elif pj == control_idx and pi not in parent_map:
                parent_map[pi] = pj

    # 按层级顺序处理（先处理父，再处理子）
    processed = {control_idx}
    order = [control_idx]
    while len(order) < n_images:
        added = False
        for idx in range(n_images):
            if idx in processed:
                continue
            parent = parent_map.get(idx)
            if parent is not None and parent in processed:
                order.append(idx)
                processed.add(idx)
                added = True
        if not added:
            # 剩余无法传播的影像，跳过
            for idx in range(n_images):
                if idx not in processed:
                    order.append(idx)
                    processed.add(idx)
            break

    for band in range(n_bands):
        for idx in order:
            if idx == control_idx:
                continue

            # 确定参考影像（父影像）
            ref_idx = parent_map.get(idx, control_idx)
            if ref_idx is None:
                ref_idx = control_idx

            # 找这对重叠
            win_pair = None
            for ov in overlaps:
                if (ov['idx_i'] == ref_idx and ov['idx_j'] == idx) or \
                   (ov['idx_i'] == idx and ov['idx_j'] == ref_idx):
                    win_pair = ov
                    break
            if win_pair is None:
                continue

            if win_pair['idx_i'] == ref_idx:
                p_ref, p_src = _overlap_valid_pixels(
                    result[ref_idx], arrays[idx],
                    win_pair['window_i'], win_pair['window_j'],
                    nodata_values[ref_idx], nodata_values[idx], band,
                )
            else:
                p_src, p_ref = _overlap_valid_pixels(
                    arrays[idx], result[ref_idx],
                    win_pair['window_i'], win_pair['window_j'],
                    nodata_values[idx], nodata_values[ref_idx], band,
                )

            if len(p_ref) < 10 or len(p_src) < 10:
                continue

            # 计算直方图累积分布
            vmin = min(p_ref.min(), p_src.min())
            vmax = max(p_ref.max(), p_src.max())
            if vmax - vmin < 1e-12:
                continue

            hist_ref, edges = np.histogram(p_ref, bins=n_bins, range=(vmin, vmax), density=True)
            hist_src, _ = np.histogram(p_src, bins=n_bins, range=(vmin, vmax), density=True)

            cdf_ref = hist_ref.cumsum()
            cdf_ref = cdf_ref / cdf_ref[-1]
            cdf_src = hist_src.cumsum()
            cdf_src = cdf_src / cdf_src[-1]

            # 映射表：src 值 → 与 ref CDF 最接近的 bin
            bin_centers = (edges[:-1] + edges[1:]) / 2
            mapping = np.interp(cdf_src, cdf_ref, bin_centers)

            # 对整幅影像做映射
            nd = nodata_values[idx]
            img_band = result[idx][band]
            valid = np.isfinite(img_band)
            if nd is not None:
                valid &= (img_band != nd)

            # 离散化像素值到 bin 中心再查表
            pixel_indices = np.digitize(img_band, edges) - 1
            pixel_indices = np.clip(pixel_indices, 0, len(mapping) - 1)
            corrected = mapping[pixel_indices]
            img_band[valid] = corrected[valid]

    return result


# ---------------------------------------------------------------------------
# 2. 矩匹配 (Moment Matching)
# ---------------------------------------------------------------------------

def moment_matching_normalize(
    arrays: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    control_idx: int = 0,
    spanning_tree: Optional[List[tuple]] = None,
) -> List[np.ndarray]:
    """
    矩匹配归一化。

    支持 spanning_tree 传播：不与 control 直接重叠的影像，
    沿生成树由父影像向子影像传播。
    """
    n_images = len(arrays)
    n_bands = arrays[0].shape[0]
    result = [arr.astype(np.float64, copy=True) for arr in arrays]

    # 构建父子关系
    if spanning_tree is not None:
        parent_map = {}
        for i, j in spanning_tree:
            parent_map[j] = i
            parent_map.setdefault(i, None)
    else:
        parent_map = {control_idx: None}
        for ov in overlaps:
            pi, pj = ov['idx_i'], ov['idx_j']
            if pi == control_idx and pj not in parent_map:
                parent_map[pj] = pi
            elif pj == control_idx and pi not in parent_map:
                parent_map[pi] = pj

    # 按层级顺序处理
    processed = {control_idx}
    order = [control_idx]
    while len(order) < n_images:
        added = False
        for idx in range(n_images):
            if idx in processed:
                continue
            parent = parent_map.get(idx)
            if parent is not None and parent in processed:
                order.append(idx)
                processed.add(idx)
                added = True
        if not added:
            for idx in range(n_images):
                if idx not in processed:
                    order.append(idx)
                    processed.add(idx)
            break

    for band in range(n_bands):
        for idx in order:
            if idx == control_idx:
                continue

            ref_idx = parent_map.get(idx, control_idx)
            if ref_idx is None:
                ref_idx = control_idx

            # 提取与父影像的重叠区统计量
            mu_ref_list, mu_src_list = [], []
            std_ref_list, std_src_list = [], []

            for ov in overlaps:
                if (ov['idx_i'] == ref_idx and ov['idx_j'] == idx) or \
                   (ov['idx_i'] == idx and ov['idx_j'] == ref_idx):
                    if ov['idx_i'] == ref_idx:
                        p_ref, p_src = _overlap_valid_pixels(
                            result[ref_idx], arrays[idx],
                            ov['window_i'], ov['window_j'],
                            nodata_values[ref_idx], nodata_values[idx], band,
                        )
                    else:
                        p_src, p_ref = _overlap_valid_pixels(
                            arrays[idx], result[ref_idx],
                            ov['window_i'], ov['window_j'],
                            nodata_values[idx], nodata_values[ref_idx], band,
                        )

                    if len(p_ref) < 5 or len(p_src) < 5:
                        continue
                    mu_ref_list.append(p_ref.mean())
                    mu_src_list.append(p_src.mean())
                    std_ref_list.append(p_ref.std())
                    std_src_list.append(p_src.std())

            if not mu_ref_list:
                continue

            mu_ref = np.mean(mu_ref_list)
            mu_src = np.mean(mu_src_list)
            std_ref = np.mean(std_ref_list)
            std_src = np.mean(std_src_list)

            if std_src < 1e-12:
                continue

            # 矩匹配：corrected = (src - μ_src) * (σ_ref / σ_src) + μ_ref
            nd = nodata_values[idx]
            img_band = result[idx][band]
            valid = np.isfinite(img_band)
            if nd is not None:
                valid &= (img_band != nd)

            img_band[valid] = (img_band[valid] - mu_src) * (std_ref / std_src) + mu_ref

    return result


# ---------------------------------------------------------------------------
# 3. Wallis 滤波
# ---------------------------------------------------------------------------

def _wallis_kernel_1d(size: int) -> np.ndarray:
    """生成一维 Wallis 高斯核（截断高斯）。"""
    sigma = size / 6.0
    x = np.arange(-(size // 2), size // 2 + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def wallis_normalize(
    arrays: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    control_idx: int = 0,
    window_size: int = 51,
    target_mean: Optional[List[float]] = None,
    target_std: Optional[List[float]] = None,
) -> List[np.ndarray]:
    """
    Wallis 滤波归一化（局部自适应矩匹配）。

    控制影像默认保持不变。
    对每个局部窗口，计算局部均值和标准差，并调整到目标值。
    target_mean 和 target_std 默认从控制影像与所有影像的重叠区按波段统计。
    """
    n_images = len(arrays)
    n_bands = arrays[0].shape[0]

    # Per-band target values from control vs other images overlap stats
    if target_mean is None or target_std is None:
        all_mu = [[] for _ in range(n_bands)]
        all_std = [[] for _ in range(n_bands)]
        for band in range(n_bands):
            for idx in range(n_images):
                if idx == control_idx:
                    continue
                for ov in overlaps:
                    if (ov["idx_i"] == control_idx and ov["idx_j"] == idx) or \
                       (ov["idx_i"] == idx and ov["idx_j"] == control_idx):
                        if ov["idx_i"] == control_idx:
                            p_ref, _ = _overlap_valid_pixels(
                                arrays[control_idx], arrays[idx],
                                ov["window_i"], ov["window_j"],
                                nodata_values[control_idx], nodata_values[idx], band,
                            )
                        else:
                            p_ref, _ = _overlap_valid_pixels(
                                arrays[idx], arrays[control_idx],
                                ov["window_i"], ov["window_j"],
                                nodata_values[idx], nodata_values[control_idx], band,
                            )
                        if len(p_ref) > 5:
                            all_mu[band].append(p_ref.mean())
                            all_std[band].append(p_ref.std())

        # Per-band targets
        target_mean = [np.mean(m) if m else 0.0 for m in all_mu]
        target_std = [np.mean(s) if s else 1.0 for s in all_std]

    # 生成一维高斯核
    if window_size % 2 == 0:
        window_size += 1
    kernel_1d = _wallis_kernel_1d(window_size)

    result = [arr.astype(np.float64, copy=True) for arr in arrays]

    for band in range(n_bands):
        for idx in range(n_images):
            if idx == control_idx:
                continue  # 控制影像保持不变
            nd = nodata_values[idx]
            img = arrays[idx][band].astype(np.float64)

            # Valid-weighted convolution to handle NoData/NaN/Inf
            valid_mask = np.isfinite(img)
            if nd is not None:
                valid_mask &= (img != nd)
            valid_f = valid_mask.astype(np.float64)
            weighted_img = np.where(valid_mask, img, 0.0)

            local_sum = _separable_convolve(weighted_img, kernel_1d)
            local_valid = _separable_convolve(valid_f, kernel_1d)
            local_valid = np.maximum(local_valid, 1e-10)
            local_mean = local_sum / local_valid

            local_sq_sum = _separable_convolve(np.where(valid_mask, img ** 2, 0.0), kernel_1d)
            local_sq_mean = local_sq_sum / local_valid
            local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
            local_std = np.sqrt(local_var)

            # Wallis 校正
            corrected = (img - local_mean) * (target_std[band] / (local_std + 1e-12)) + target_mean[band]

            result[idx][band][valid_mask] = corrected[valid_mask]

    return result


def _separable_convolve(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    """用两个一维卷积实现二维高斯滤波（快于直接二维卷积）。"""
    from scipy.ndimage import convolve1d
    temp = convolve1d(img, kernel_1d, axis=0, mode='reflect')
    result = convolve1d(temp, kernel_1d, axis=1, mode='reflect')
    return result


# ---------------------------------------------------------------------------
# 统一接口
# ---------------------------------------------------------------------------

METHODS = {
    "histogram_matching": histogram_matching_normalize,
    "moment_matching": moment_matching_normalize,
    "wallis": wallis_normalize,
}


def run_comparison(
    method: str,
    arrays: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    control_idx: int = 0,
    spanning_tree: Optional[List[tuple]] = None,
    **kwargs,
) -> List[np.ndarray]:
    """
    运行指定的对比方法。

    参数
    ----------
    method : str
        "histogram_matching" / "moment_matching" / "wallis"
    arrays : list of np.ndarray
    nodata_values : list
    overlaps : list
    control_idx : int
    spanning_tree : list of (i, j) tuples or None
        可靠的生成树边，用于传播。None 则自动检测直接重叠。
    **kwargs : 传递给具体方法的额外参数。

    返回
    -------
    list of np.ndarray
    """
    if method == "histogram_matching":
        return histogram_matching_normalize(
            arrays, nodata_values, overlaps, control_idx,
            spanning_tree=spanning_tree, **kwargs)
    elif method == "moment_matching":
        return moment_matching_normalize(
            arrays, nodata_values, overlaps, control_idx,
            spanning_tree=spanning_tree, **kwargs)
    elif method == "wallis":
        return wallis_normalize(
            arrays, nodata_values, overlaps, control_idx,
            **kwargs)
    else:
        raise ValueError(f"未知对比方法: {method}，可用: {list(METHODS.keys())}")
