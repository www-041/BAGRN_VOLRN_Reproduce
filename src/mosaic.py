"""
镶嵌模块

将多幅归一化后的 GeoTIFF 影像拼接为一幅全景镶嵌图。
重叠区采用距离加权羽化（distance-weighted feathering）融合。
"""

import numpy as np
import rasterio
import logging
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt, sobel
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _find_min_cost_seam(arr_a, arr_b, mask_a, mask_b,
                         diff_weight=1.0, grad_weight=0.5):
    """在两景重叠区中寻找最小代价接缝线（自上而下动态规划）。

    代价函数：两影像像素差 + 梯度差 + 两侧 nodata 惩罚。
    返回：seam_cols（长度=overlap_height），接缝在重叠区中的列偏移。

    Parameters
    ----------
    arr_a, arr_b : 2D ndarray, 重叠区内的影像值
    mask_a, mask_b : 2D bool, 有效像素掩膜
    diff_weight : float, 像素差代价权重
    grad_weight : float, 梯度差代价权重

    Returns
    -------
    seam_cols : 1D ndarray of int, 每行接缝列索引
    """
    ph, pw = arr_a.shape

    # 联合有效掩膜（接缝必须在两景都有效的位置）
    valid = mask_a & mask_b
    if valid.sum() < ph:
        # 退化：找有效像素最多的列
        col_valid = valid.sum(axis=0)
        best_col = int(np.argmax(col_valid))
        return np.full(ph, best_col, dtype=np.int32)

    # --- 逐行计算最小代价接缝 ---
    # 代价 = diff_weight * |A-B| + grad_weight * |gradA-gradB|
    a = arr_a.astype(np.float64)
    b = arr_b.astype(np.float64)

    diff_cost = np.abs(a - b)

    # 梯度代价
    gx_a = np.abs(sobel(a, axis=1))
    gy_a = np.abs(sobel(a, axis=0))
    gx_b = np.abs(sobel(b, axis=1))
    gy_b = np.abs(sobel(b, axis=0))
    grad_cost = np.sqrt((gx_a - gx_b)**2 + (gy_a - gy_b)**2)

    cost = diff_weight * diff_cost + grad_weight * grad_cost

    # nodata 惩罚：只有一景有效时高代价
    cost[~valid] = 1e10

    # DP 累积代价
    INF = 1e18
    dp = np.full((ph, pw), INF, dtype=np.float64)
    parent = np.full((ph, pw), -1, dtype=np.int32)

    # 第一行：直接代价
    dp[0, :] = cost[0, :]

    # 逐行扫描，允许左右各移动 1 列（保持接缝连续平滑）
    for r in range(1, ph):
        for c in range(pw):
            best_prev = -1
            best_val = INF
            for dc in [-1, 0, 1]:
                pc = c + dc
                if 0 <= pc < pw and dp[r-1, pc] < best_val:
                    best_val = dp[r-1, pc]
                    best_prev = pc
            if best_prev >= 0:
                dp[r, c] = cost[r, c] + best_val
                parent[r, c] = best_prev

    # 从最后一行最小代价处回溯
    seam_cols = np.zeros(ph, dtype=np.int32)
    seam_cols[ph-1] = int(np.argmin(dp[ph-1, :]))
    for r in range(ph-2, -1, -1):
        seam_cols[r] = parent[r+1, seam_cols[r+1]]

    return seam_cols


def _compute_weight_map(mask: np.ndarray) -> np.ndarray:
    """
    计算羽化权重图。

    对每个有效像素，权重 = 到最近无效/边缘像素的欧氏距离。
    中心区域权重大、边缘权重小，保证重叠区平滑过渡。
    """
    dist = distance_transform_edt(mask)
    if dist.max() > 0:
        dist = dist / dist.max()
    return dist


def _combined_bounds(arrays, transforms):
    """计算所有影像的联合地理包围盒。"""
    left = float('inf')
    bottom = float('inf')
    right = float('-inf')
    top = float('-inf')
    for arr, tr in zip(arrays, transforms):
        rows, cols = arr.shape[1], arr.shape[2]
        l = tr.c
        b = tr.f + tr.e * rows
        r = tr.c + tr.a * cols
        t = tr.f
        left = min(left, l)
        bottom = min(bottom, b)
        right = max(right, r)
        top = max(top, t)
    return left, bottom, right, top


def create_mosaic(
    arrays: List[np.ndarray],
    transforms: List[rasterio.Affine],
    crs: str,
    nodata_values: List[Optional[float]],
    output_path: str,
    resolution: Optional[float] = None,
    mode: str = "weighted",
    feather_width: int = 64,
    return_diagnostics: bool = False,
) -> str:
    """
    创建镶嵌图。

    流程：
      1. 计算联合包围盒与输出网格
      2. 将每幅影像投影到输出网格
      3. 根据 mode 选择融合策略：
         - "weighted": 全局距离加权平均（原始方法）
         - "source_selection": 每个像素仅选取距离边缘最远的单幅影像
         - "narrow_feather": 仅在接缝两侧 feather_width 像素内羽化，
           其余区域使用 source_selection
      4. 写出 GeoTIFF

    参数
    ----------
    arrays : list of np.ndarray
        形状均为 (bands, rows, cols)。
    transforms : list of rasterio.Affine
    crs : str
        输出 CRS（所有影像需一致）。
    nodata_values : list of float or None
    output_path : str
    resolution : float or None
        输出分辨率（像素地理尺寸）。None 则用所有影像中的最精细分辨率。
    mode : str
        "weighted" | "source_selection" | "narrow_feather"
    feather_width : int
        narrow_feather 模式下接缝两侧羽化半宽（像素）。

    返回
    -------
    output_path : str
    """
    n_images = len(arrays)
    if n_images == 0:
        raise ValueError("没有影像可供镶嵌")
    n_bands = arrays[0].shape[0]

    from src.experiment_config import VALID_MOSAIC_MODES
    if mode not in VALID_MOSAIC_MODES:
        raise ValueError(f"未知的镶嵌模式: '{mode}'，可选值: {VALID_MOSAIC_MODES}")

    # ---- 1. 联合包围盒 ----
    left, bottom, right, top = _combined_bounds(arrays, transforms)
    out_width_geo = right - left
    out_height_geo = top - bottom

    if resolution is None:
        resolution = min(abs(tr.a) for tr in transforms)

    if resolution <= 0:
        raise ValueError(f"无效分辨率: {resolution}")

    import math
    width = max(1, int(math.ceil(out_width_geo / resolution)))
    height = max(1, int(math.ceil(out_height_geo / resolution)))

    out_transform = rasterio.Affine(resolution, 0, left, 0, -resolution, top)

    print(f"  镶嵌画布: {width}×{height}, 分辨率={resolution:.2f}")
    print(f"  地理范围: ({left:.2f}, {bottom:.2f}, {right:.2f}, {top:.2f})")

    # ---- 2. 投影所有影像到输出网格 ----
    projected = []
    for idx in range(n_images):
        arr = arrays[idx]
        tr = transforms[idx]
        nd = nodata_values[idx]

        # NoData 感知重投影：用 NaN 作为中间值，避免 0 值污染
        proj_bands = np.full((n_bands, height, width), np.nan, dtype=np.float64)
        for b in range(n_bands):
            reproject(
                source=arr[b].astype(np.float64),
                destination=proj_bands[b],
                src_transform=tr,
                src_crs=crs,
                dst_transform=out_transform,
                dst_crs=crs,
                src_nodata=nd,
                dst_nodata=np.nan,
                init_dest_nodata=True,
                resampling=Resampling.bilinear,
            )

        # 有效像素掩膜（非 NaN）
        proj_mask = np.all(np.isfinite(proj_bands), axis=0)

        if not proj_mask.any():
            projected.append(None)
            continue

        projected.append((proj_bands, proj_mask, nd))
        print(f"  已投影 [{idx+1}/{n_images}] {width}×{height}")

    # ---- 3. 融合 ----
    n_bands = arrays[0].shape[0]
    result = np.full((n_bands, height, width), 0.0, dtype=np.float64)
    final_mask = np.zeros((height, width), dtype=bool)

    if mode == "source_selection":
        # 每个像素只选距离边缘最远的单幅影像
        best_idx = np.full((height, width), -1, dtype=np.int32)
        best_weight = np.zeros((height, width), dtype=np.float64)

        for k, proj in enumerate(projected):
            if proj is None:
                continue
            bands, mask, nd = proj
            weight = distance_transform_edt(mask).astype(np.float64)
            weight[~mask] = 0.0
            better = weight > best_weight
            best_idx[better] = k
            best_weight[better] = weight[better]

        final_mask = best_idx >= 0
        for k, proj in enumerate(projected):
            if proj is None:
                continue
            bands, mask, nd = proj
            pix = best_idx == k
            if pix.any():
                result[:, pix] = bands[:, pix]

    elif mode == "narrow_feather":
        # ---- 逐接缝最小代价羽化 ----
        # 1. source selection 求每像素最优来源
        # 2. 从 best_idx 4邻域检测真实接缝来源对
        # 3. 逐对处理：求最小代价接缝 → 有符号距离 → 对称羽化
        # 4. processed_feather_mask 防止多接缝重叠覆盖

        n_proj = len(projected)

        # --- 阶段 1：source selection ---
        all_weights = []
        for proj in projected:
            if proj is None:
                all_weights.append(None)
                continue
            w = distance_transform_edt(proj[1]).astype(np.float64)
            w[~proj[1]] = 0.0
            all_weights.append(w)

        best_idx = np.full((height, width), -1, dtype=np.int32)
        best_w = np.zeros((height, width), dtype=np.float64)
        for k, w in enumerate(all_weights):
            if w is None:
                continue
            better = w > best_w
            best_idx[better] = k
            best_w[better] = w[better]

        # 初始化 source selection 结果
        proj_data = {}
        for k, proj in enumerate(projected):
            if proj is None:
                continue
            proj_data[k] = proj[0]

        for k in proj_data:
            pix = best_idx == k
            if pix.any():
                result[:, pix] = proj_data[k][:, pix]

        # --- 阶段 2：从 best_idx 检测真实接缝来源对 ---
        actual_seam_pairs = set()
        for dy, dx in [(0, 1), (1, 0)]:
            src_r = slice(0, height - dy if dy else height)
            src_c = slice(0, width - dx if dx else width)
            dst_r = slice(dy, height)
            dst_c = slice(dx, width)

            label_a = best_idx[src_r, src_c]
            label_b = best_idx[dst_r, dst_c]

            valid = (label_a >= 0) & (label_b >= 0) & (label_a != label_b)
            if valid.any():
                pairs_a = label_a[valid]
                pairs_b = label_b[valid]
                for la, lb in zip(pairs_a, pairs_b):
                    actual_seam_pairs.add(tuple(sorted((int(la), int(lb)))))

        actual_seam_pairs = sorted(actual_seam_pairs)
        print(f"  Actual seam pairs detected: {len(actual_seam_pairs)}")
        for i, j in actual_seam_pairs:
            print(f"    [{i}]-[{j}]")

        if not actual_seam_pairs:
            final_mask = best_idx >= 0
            print(f"  No seams, using source selection only")
        else:
            # --- 阶段 3：逐对处理实际接缝 ---
            processed_feather_mask = np.zeros((height, width), dtype=bool)
            processed_seam_pairs = []

            for seam_pair in actual_seam_pairs:
                si, sj = seam_pair
                if si not in proj_data or sj not in proj_data:
                    continue

                mask_i = projected[si][1]
                mask_j = projected[sj][1]
                overlap = mask_i & mask_j

                if overlap.sum() < 10:
                    processed_seam_pairs.append(seam_pair)
                    continue

                # 该来源对的真实边界掩膜
                boundary = np.zeros((height, width), dtype=bool)
                for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    src_r = slice(max(0, -dy), height + min(0, -dy))
                    src_c = slice(max(0, -dx), width + min(0, -dx))
                    dst_r = slice(max(0, dy), height + min(0, dy))
                    dst_c = slice(max(0, dx), width + min(0, dx))

                    la = best_idx[src_r, src_c]
                    lb = best_idx[dst_r, dst_c]
                    is_pair_boundary = ((la == si) & (lb == sj)) | ((la == sj) & (lb == si))
                    boundary[dst_r, dst_c] |= is_pair_boundary

                if not boundary.any():
                    processed_seam_pairs.append(seam_pair)
                    continue

                # 重叠区范围
                rows = np.any(overlap, axis=1)
                cols = np.any(overlap, axis=0)
                r0, r1 = np.where(rows)[0][[0, -1]]
                c0, c1 = np.where(cols)[0][[0, -1]]
                ov_h = r1 - r0 + 1
                ov_w = c1 - c0 + 1

                # 提取重叠区影像
                ov_i = proj_data[si][:, r0:r1+1, c0:c1+1]
                ov_j = proj_data[sj][:, r0:r1+1, c0:c1+1]
                ov_mi = mask_i[r0:r1+1, c0:c1+1]
                ov_mj = mask_j[r0:r1+1, c0:c1+1]

                # 自动选择接缝方向
                if ov_w >= ov_h:
                    # 重叠区较宽 → 自上而下接缝（列方向变化）
                    seam_local = _find_min_cost_seam(
                        ov_i[0], ov_j[0], ov_mi, ov_mj,
                        diff_weight=1.0, grad_weight=0.5)
                    seam_global = seam_local + c0
                    orientation = 'vertical'

                    signed_dist = np.full((height, width), 0.0, dtype=np.float64)
                    for r in range(r0, r0 + len(seam_global)):
                        if r >= height:
                            break
                        sc = seam_global[r - r0]
                        signed_dist[r, :] = np.arange(width, dtype=np.float64) - sc
                else:
                    # 重叠区较高 → 自左向右接缝（行方向变化）
                    seam_local = _find_min_cost_seam(
                        ov_i[0].T, ov_j[0].T, ov_mi.T, ov_mj.T,
                        diff_weight=1.0, grad_weight=0.5)
                    seam_global = seam_local + r0
                    orientation = 'horizontal'

                    signed_dist = np.full((height, width), 0.0, dtype=np.float64)
                    for c in range(c0, c0 + len(seam_global)):
                        if c >= width:
                            break
                        sr = seam_global[c - c0]
                        signed_dist[:, c] = np.arange(height, dtype=np.float64) - sr

                # 羽化区宽度保护：最大允许宽度 = 重叠区最小宽度的一半
                ov_min_dim = min(ov_h, ov_w)
                max_feather_width = max(1, ov_min_dim // 2)
                requested_width = feather_width
                if feather_width > max_feather_width:
                    effective_width = max_feather_width
                    feather_status = "clamped"
                    feather_reason = f"requested={feather_width} > max={max_feather_width}"
                    logger.info("  [%d]-[%d] feather_width clamped: %d → %d (%s)",
                                si, sj, feather_width, effective_width, feather_reason)
                else:
                    effective_width = feather_width
                    feather_status = "ok"
                    feather_reason = ""

                # 羽化区：当前 pair 的重叠区 + effective_width + 未处理
                feather_zone = ((np.abs(signed_dist) <= effective_width) &
                                overlap & ~processed_feather_mask)

                if not feather_zone.any():
                    continue

                # 对称权重
                w_sj = np.clip(0.5 + signed_dist / (2.0 * max(effective_width, 1)),
                               0.0, 1.0)
                w_si = 1.0 - w_sj

                # 羽化区内逐像元混合
                feather_pixels = np.argwhere(feather_zone)
                n_blended = 0
                for py, px in feather_pixels:
                    si_valid = mask_i[py, px]
                    sj_valid = mask_j[py, px]

                    if si_valid and sj_valid:
                        ws = float(w_si[py, px])
                        result[:, py, px] = (proj_data[si][:, py, px] * ws +
                                              proj_data[sj][:, py, px] * (1.0 - ws))
                    elif si_valid:
                        result[:, py, px] = proj_data[si][:, py, px]
                    elif sj_valid:
                        result[:, py, px] = proj_data[sj][:, py, px]
                    else:
                        continue
                    n_blended += 1

                processed_feather_mask |= feather_zone
                processed_seam_pairs.append(seam_pair)

                print(f"  Feather seam [{si}]-[{sj}]: "
                      f"seam_pixels={boundary.sum()}, "
                      f"blended_pixels={n_blended}, "
                      f"requested_width={requested_width}, "
                      f"effective_width={effective_width}, "
                      f"status={feather_status}, "
                      f"orientation={orientation}")

            # --- 阶段 4：断言所有接缝已处理 ---
            actual_set = set(actual_seam_pairs)
            processed_set = set(processed_seam_pairs)
            if actual_set != processed_set:
                missing = actual_set - processed_set
                raise RuntimeError(
                    f"Seam processing incomplete: missing {missing}. "
                    f"actual={actual_set}, processed={processed_set}")

            final_mask = best_idx >= 0
            print(f"\n  Processed seam pairs: {processed_seam_pairs}")
            print(f"  Total feather pixels: {int(processed_feather_mask.sum())}")

    else:
        # mode == "weighted": 全局距离加权平均（原始方法）
        sum_values = np.zeros((n_bands, height, width), dtype=np.float64)
        sum_weights = np.zeros((height, width), dtype=np.float64)

        for proj in projected:
            if proj is None:
                continue
            bands, mask, nd = proj
            weight = distance_transform_edt(mask).astype(np.float64)
            weight[~mask] = 0.0
            weight[mask] += 1e-6
            safe_bands = np.where(mask[np.newaxis, :, :], bands, 0.0)
            sum_values += safe_bands * weight[np.newaxis, :, :]
            sum_weights += weight

        final_mask = sum_weights > 1e-12
        denom = sum_weights[np.newaxis, :, :]
        np.divide(sum_values, denom, out=result, where=denom > 1e-12)

    # 验证 Union vs Final
    union_mask = np.zeros((height, width), dtype=bool)
    for proj in projected:
        if proj is not None:
            union_mask |= proj[1]
    print(f"  Union pixels: {union_mask.sum()}")
    print(f"  Final valid pixels: {final_mask.sum()}")

    # NaN 检查
    nan_in_valid = np.isnan(result[:, final_mask]).sum()
    inf_in_valid = np.isinf(result[:, final_mask]).sum()
    print(f"  NaN inside final mask: {nan_in_valid}")
    print(f"  Inf inside final mask: {inf_in_valid}")

    # 确定输出 nodata 值
    # 如果所有 nodata 都是 None，使用 NaN（浮点输出）；否则使用声明的 nodata
    any_nodata = any(nd is not None for nd in nodata_values)
    if any_nodata:
        nd_val = next(nd for nd in nodata_values if nd is not None)
    else:
        nd_val = np.nan  # 无声明 nodata 时，用 NaN 表示未覆盖区

    # 强制清理：有效区内残留 NaN/Inf → nodata（检查所有波段）
    bad_mask = ~np.all(np.isfinite(result), axis=0)
    result[:, bad_mask] = nd_val
    result[:, ~final_mask] = nd_val

    # 最终断言
    nan_check = np.isnan(result[:, final_mask & ~bad_mask]).sum()
    assert nan_check == 0, f"Assertion failed: {nan_check} NaN in final valid pixels"

    # ---- 4. 写出 ----
    # 如果 nodata 是 None 且结果含 NaN，使用 float32 保留 NaN
    out_dtype = arrays[0].dtype.name
    has_nan_nodata = (np.isnan(nd_val) if np.isscalar(nd_val) and isinstance(nd_val, float) else False)
    if has_nan_nodata:
        out_dtype = "float32"

    profile = {
        "driver": "GTiff",
        "dtype": out_dtype,
        "count": n_bands,
        "height": height,
        "width": width,
        "transform": out_transform,
        "crs": crs,
        "compress": "lzw",
        "nodata": None if has_nan_nodata else nd_val,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(result.astype(out_dtype))

    if return_diagnostics:
        diag = {
            'output_path': output_path,
            'mode': mode,
            'feather_width': feather_width,
            'canvas_size': (width, height),
            'resolution': resolution,
            'n_images': n_images,
            'n_bands': n_bands,
            'valid_pixels': int(final_mask.sum()),
            'total_pixels': width * height,
            'nan_in_valid': int(nan_in_valid),
            'inf_in_valid': int(inf_in_valid),
            'source_index_map': best_idx if 'best_idx' in dir() else None,
        }
        return output_path, diag

    return output_path
