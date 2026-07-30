"""
光谱保真度验证模块

验证辐射归一化前后的多波段光谱形状保持度。
注意：输入可能为 DN 值，因此使用"多波段相对光谱形状保持度"术语，
而非"反射率光谱保真度"（除非输入已确认为反射率）。

主要函数
--------
compute_sam                  光谱角映射（SAM）逐像素
compute_spectral_rmse        光谱 RMSE 逐像素
compute_relative_spectral_rmse  相对光谱 RMSE（归一化能量）
compute_band_ratio_error     稳定波段比值保持度
compute_spectral_correlation 波段间相关系数变化
sample_spectral_points       分层采样索引
compute_all_spectral         一站式计算所有光谱指标
"""

import numpy as np
from scipy.ndimage import sobel
from typing import List, Optional, Tuple, Dict, Any
import csv
import os


# ===========================================================================
# 辅助函数
# ===========================================================================

def _build_valid_mask(
    array: np.ndarray,
    nodata: Optional[float],
) -> np.ndarray:
    """
    构建有效像素掩码：排除 nodata 和非有限值。

    参数
    ----------
    array : np.ndarray, shape (bands, rows, cols)
    nodata : float or None

    返回
    -------
    np.ndarray, shape (rows, cols), dtype=bool
        所有波段均为有效值的像素掩码。
    """
    # 先排除非有限值
    valid = np.isfinite(array).all(axis=0)
    # 再排除 nodata
    if nodata is not None:
        valid &= ~np.any(array == nodata, axis=0)
    return valid


def _validate_inputs(
    original: np.ndarray,
    normalized: np.ndarray,
    nodata: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    确保输入形状一致并返回 float64 副本。

    参数
    ----------
    original : np.ndarray, shape (bands, rows, cols)
    normalized : np.ndarray, shape (bands, rows, cols)
    nodata : float or None

    返回
    -------
    orig_f : np.ndarray, dtype=float64
    norm_f : np.ndarray, dtype=float64

    异常
    ------
    ValueError 如果形状不匹配或波段数 < 2。
    """
    if original.shape != normalized.shape:
        raise ValueError(
            f"形状不匹配: original {original.shape} vs normalized {normalized.shape}"
        )
    if original.ndim != 3 or original.shape[0] < 2:
        raise ValueError(
            f"需要 (bands, rows, cols) 且 bands >= 2, 当前 {original.shape}"
        )
    # 额外安全检查：确保数组不全为 NaN
    if not np.any(np.isfinite(original)):
        raise ValueError("original 数组全部为 NaN/Inf，无法计算")
    if not np.any(np.isfinite(normalized)):
        raise ValueError("normalized 数组全部为 NaN/Inf，无法计算")
    return original.astype(np.float64, copy=True), normalized.astype(np.float64, copy=True)


def _percentiles(data: np.ndarray) -> Dict[str, float]:
    """计算常用分位数统计量。"""
    if data.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "p90": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.nanmean(data)),
        "median": float(np.nanmedian(data)),
        "p90": float(np.nanpercentile(data, 90)),
        "p95": float(np.nanpercentile(data, 95)),
    }


# ===========================================================================
# 1. SAM — Spectral Angle Mapper（逐像素光谱角）
# ===========================================================================

def compute_sam(
    original: np.ndarray,
    normalized: np.ndarray,
    nodata: Optional[float],
    valid_mask: Optional[np.ndarray] = None,
    bands: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    scene_id: str = "",
) -> Dict[str, Any]:
    """
    逐像素计算光谱角映射（SAM）。

    数学定义（单位：度）
    ----------------------
    对于每个像素 p，设 b 维光谱向量为 s_o(p) 和 s_n(p)：

        SAM(p) = arccos( <s_o, s_n> / (||s_o|| * ||s_n||) )

    SAM = 0° 表示光谱方向完全一致；值越小越好。

    本函数同时过滤：
      - 零向量（||s_o|| == 0 或 ||s_n|| == 0）
      - 低能量向量（||s_o|| / mean < 1e-6 或 ||s_n|| / mean < 1e-6）

    参数
    ----------
    original : np.ndarray, shape (bands, rows, cols)
        归一化前的影像（DN 或反射率）。
    normalized : np.ndarray, shape (bands, rows, cols)
        归一化后的影像。
    nodata : float or None
        NoData 值；为 None 时仅排除非有限值。
    valid_mask : np.ndarray of bool, optional
        额外的有效掩码（例如重叠区标记）。与 nodata 掩码取交集。

    返回
    -------
    dict
        mean   : float — SAM 均值（度）
        median : float — SAM 中位数（度）
        p90    : float — SAM 90 分位数（度）
        p95    : float — SAM 95 分位数（度）
        per_pixel : np.ndarray, shape (rows, cols), dtype=float64
            逐像素 SAM 值；无效像素为 NaN。
    """
    orig_f, norm_f = _validate_inputs(original, normalized, nodata)

    # 有效像素掩码：所有波段均有效
    all_valid = _build_valid_mask(orig_f, nodata)
    all_valid &= _build_valid_mask(norm_f, nodata)
    if valid_mask is not None:
        all_valid &= valid_mask

    # 计算光谱向量范数
    norm_orig = np.sqrt(np.sum(orig_f ** 2, axis=0))   # (rows, cols)
    norm_norm = np.sqrt(np.sum(norm_f ** 2, axis=0))

    # 过滤零向量和极低能量向量
    # 以全局平均能量为参考阈值
    mean_energy = max(norm_orig.mean(), norm_norm.mean(), 1e-12)
    energy_thresh = mean_energy * 1e-6

    nonzero_mask = (norm_orig > energy_thresh) & (norm_norm > energy_thresh)
    all_valid &= nonzero_mask

    # 初始化逐像素 SAM 为 NaN
    sam_map = np.full(orig_f.shape[1:], np.nan, dtype=np.float64)

    if not all_valid.any():
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "per_pixel": sam_map,
        }

    # 逐像素内积和余弦值
    # s_o · s_n = Σ_b orig[b] * norm[b]
    dot_product = np.sum(orig_f * norm_f, axis=0)
    denom = norm_orig * norm_norm

    # 余弦值裁剪到 [-1, 1] 以避免浮点误差导致 arccos 域外
    cos_angle = np.clip(dot_product / np.maximum(denom, 1e-30), -1.0, 1.0)

    # SAM（弧度 → 度）
    sam_rad = np.arccos(cos_angle)
    sam_deg = np.degrees(sam_rad)

    sam_map[all_valid] = sam_deg[all_valid]

    # 汇总统计
    stats = _percentiles(sam_deg[all_valid])
    stats["per_pixel"] = sam_map

    # 可选：输出 per-scene CSV（带真实波段名）
    if output_dir is not None and bands is not None:
        os.makedirs(output_dir, exist_ok=True)
        n_bands = original.shape[0]
        band_labels = [f"B{b+1}" for b in bands] if bands else [f"B{b+1}" for b in range(n_bands)]
        csv_path = os.path.join(output_dir, f"sam_{scene_id}.csv" if scene_id else "sam.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["row", "col", "sam_deg"] + band_labels)
            valid_rows, valid_cols = np.where(all_valid)
            for idx in range(len(valid_rows)):
                r, c = valid_rows[idx], valid_cols[idx]
                row_data = [int(r), int(c), float(sam_deg[r, c])]
                row_data += [float(original[b, r, c]) for b in range(n_bands)]
                writer.writerow(row_data)

    return stats


# ===========================================================================
# 2. 光谱 RMSE（逐像素）
# ===========================================================================

def compute_spectral_rmse(
    original: np.ndarray,
    normalized: np.ndarray,
    nodata: Optional[float],
) -> Dict[str, float]:
    """
    逐像素计算光谱 RMSE。

    数学定义
    --------
    对于每个像素 p，设 b 波段向量 s_o(p) 和 s_n(p)：

        spectral_RMSE(p) = sqrt( (1/B) * Σ_b (s_o_b(p) - s_n_b(p))^2 )

    该指标衡量归一化前后各波段数值的绝对偏差，
    对于 DN 值影像同样适用。

    返回
    -------
    dict
        mean, median, p90, p95 — 光谱 RMSE 的统计量。
    """
    orig_f, norm_f = _validate_inputs(original, normalized, nodata)
    all_valid = _build_valid_mask(orig_f, nodata) & _build_valid_mask(norm_f, nodata)

    if not all_valid.any():
        return {"mean": float("nan"), "median": float("nan"),
                "p90": float("nan"), "p95": float("nan")}

    n_bands = orig_f.shape[0]
    # 逐波段平方差
    sq_diff = np.sum((orig_f - norm_f) ** 2, axis=0)  # (rows, cols)
    # RMSE = sqrt(mean of squared differences)
    rmse_map = np.sqrt(sq_diff / n_bands)

    return _percentiles(rmse_map[all_valid])


# ===========================================================================
# 3. 相对光谱 RMSE（归一化能量）
# ===========================================================================

def compute_relative_spectral_rmse(
    original: np.ndarray,
    normalized: np.ndarray,
    nodata: Optional[float],
) -> Dict[str, float]:
    """
    逐像素计算相对光谱 RMSE = spectral_RMSE / mean_energy。

    数学定义
    --------
        mean_energy(p) = (1/B) * Σ_b (|s_o_b(p)| + |s_n_b(p)|) / 2

        relative_spectral_RMSE(p) = spectral_RMSE(p) / mean_energy(p)

    当 mean_energy(p) < 阈值（全局均值的 1e-6）时视为退化情形，
    对应像素设为 NaN。

    返回
    -------
    dict
        mean, median, p90, p95 — 相对光谱 RMSE 的统计量。
    """
    orig_f, norm_f = _validate_inputs(original, normalized, nodata)
    all_valid = _build_valid_mask(orig_f, nodata) & _build_valid_mask(norm_f, nodata)

    if not all_valid.any():
        return {"mean": float("nan"), "median": float("nan"),
                "p90": float("nan"), "p95": float("nan")}

    n_bands = orig_f.shape[0]
    sq_diff = np.sum((orig_f - norm_f) ** 2, axis=0)
    rmse_map = np.sqrt(sq_diff / n_bands)

    # 平均能量 = 各波段绝对值的均值
    mean_energy = (np.mean(np.abs(orig_f), axis=0) + np.mean(np.abs(norm_f), axis=0)) / 2.0

    # 阈值过滤
    global_energy = mean_energy[all_valid].mean()
    thresh = global_energy * 1e-6
    reliable = all_valid & (mean_energy > thresh)

    rel_rmse = np.full(orig_f.shape[1:], np.nan, dtype=np.float64)
    rel_rmse[reliable] = rmse_map[reliable] / mean_energy[reliable]

    return _percentiles(rel_rmse[reliable])


# ===========================================================================
# 4. 波段比值保持度
# ===========================================================================

def compute_band_ratio_error(
    original: np.ndarray,
    normalized: np.ndarray,
    nodata: Optional[float],
    band_pairs: List[Tuple[int, int]],
    band_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    scene_id: str = "",
) -> Dict[int, Dict[str, float]]:
    """
    对指定波段对计算比值保持度（归一化前后比值的相对误差）。

    数学定义
    --------
    对波段对 (i, j)，定义：

        R_orig(p)  = s_o_i(p) / s_o_j(p)
        R_norm(p)  = s_n_i(p) / s_n_j(p)

        ratio_error(p) = |R_norm(p) - R_orig(p)| / max(|R_orig(p)|, ε)

    其中 ε = 1e-10 防止除零。

    仅使用 s_o_j(p) 和 s_n_j(p) 均大于阈值的像素。

    返回
    -------
    dict
        键为 (i, j) 元组，值为 {"mean", "median", "p90", "p95"}。
    """
    orig_f, norm_f = _validate_inputs(original, normalized, nodata)
    all_valid = _build_valid_mask(orig_f, nodata) & _build_valid_mask(norm_f, nodata)

    n_bands = orig_f.shape[0]
    results = {}

    for (bi, bj) in band_pairs:
        if bi < 0 or bi >= n_bands or bj < 0 or bj >= n_bands:
            raise ValueError(f"波段索引超界: ({bi}, {bj}), 最大合法值 {n_bands - 1}")

        # 分母波段有效条件：原始和归一化均有效且绝对值大于阈值
        denom_orig = orig_f[bj]
        denom_norm = norm_f[bj]
        energy_thresh = max(abs(denom_orig[all_valid]).mean(),
                           abs(denom_norm[all_valid]).mean(), 1e-12) * 1e-6

        ratio_valid = (all_valid
                       & (np.abs(denom_orig) > energy_thresh)
                       & (np.abs(denom_norm) > energy_thresh))

        if not ratio_valid.any():
            results[(bi, bj)] = {"mean": float("nan"), "median": float("nan"),
                                 "p90": float("nan"), "p95": float("nan")}
            continue

        r_orig = orig_f[bi, ratio_valid] / denom_orig[ratio_valid]
        r_norm = norm_f[bi, ratio_valid] / denom_norm[ratio_valid]

        # 相对误差
        abs_diff = np.abs(r_norm - r_orig)
        denom_ref = np.maximum(np.abs(r_orig), 1e-10)
        rel_err = abs_diff / denom_ref

        results[(bi, bj)] = _percentiles(rel_err)

    # 可选：输出 per-scene CSV（带真实波段名）
    if output_dir is not None and band_pairs:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"band_ratio_{scene_id}.csv" if scene_id else "band_ratio.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["band_pair", "band_i_name", "band_j_name", "mean", "median", "p90", "p95"]
            writer.writerow(header)
            for (bi, bj), stats in results.items():
                name_i = band_names[bi] if band_names and bi < len(band_names) else f"B{bi+1}"
                name_j = band_names[bj] if band_names and bj < len(band_names) else f"B{bj+1}"
                writer.writerow([
                    f"({bi},{bj})", name_i, name_j,
                    stats.get("mean", float("nan")),
                    stats.get("median", float("nan")),
                    stats.get("p90", float("nan")),
                    stats.get("p95", float("nan")),
                ])

    return results


# ===========================================================================
# 5. 波段间相关系数变化
# ===========================================================================

def compute_spectral_correlation(
    original: np.ndarray,
    normalized: np.ndarray,
    nodata: Optional[float],
) -> Dict[str, Any]:
    """
    计算归一化前后波段间相关系数矩阵的变化。

    数学定义
    --------
    对每波段 b，计算：

        corr_b = PearsonCorr(orig_b, norm_b)

    同时输出相关系数矩阵的 Frobenius 范数变化量（衡量波段间
    线性关系的保持程度）。

    返回
    -------
    dict
        per_band : list of float — 各波段 Pearson 相关系数
        corr_orig : np.ndarray — 原始波段间相关矩阵
        corr_norm : np.ndarray — 归一化后波段间相关矩阵
        frobenius_diff : float — ‖C_orig - C_norm‖_F（Frobenius 范数差）
    """
    orig_f, norm_f = _validate_inputs(original, normalized, nodata)
    all_valid = _build_valid_mask(orig_f, nodata) & _build_valid_mask(norm_f, nodata)

    n_bands = orig_f.shape[0]

    if not all_valid.any():
        return {
            "per_band": [float("nan")] * n_bands,
            "corr_orig": [[float("nan")] * n_bands for _ in range(n_bands)],
            "corr_norm": [[float("nan")] * n_bands for _ in range(n_bands)],
            "frobenius_diff": float("nan"),
        }

    # 提取有效像素的 2D 矩阵 (pixels, bands)
    orig_2d = orig_f[:, all_valid].T
    norm_2d = norm_f[:, all_valid].T

    # 逐波段 Pearson 相关系数
    per_band_corr = []
    for b in range(n_bands):
        c = np.corrcoef(orig_2d[:, b], norm_2d[:, b])[0, 1]
        per_band_corr.append(float(c))

    # 波段间相关矩阵
    corr_orig = np.corrcoef(orig_2d, rowvar=False)
    corr_norm = np.corrcoef(norm_2d, rowvar=False)

    # Frobenius 范数差
    frob_diff = float(np.linalg.norm(corr_orig - corr_norm, ord="fro"))

    return {
        "per_band": per_band_corr,
        "corr_orig": corr_orig.tolist() if isinstance(corr_orig, np.ndarray) else corr_orig,
        "corr_norm": corr_norm.tolist() if isinstance(corr_norm, np.ndarray) else corr_norm,
        "frobenius_diff": frob_diff,
    }


# ===========================================================================
# 6. 分层采样
# ===========================================================================

def sample_spectral_points(
    original: np.ndarray,
    normalized: np.ndarray,
    nodata: Optional[float],
    overlaps: List[dict],
    n_samples: int = 1000,
    seed: int = 42,
    include_overlap: bool = True,
    include_nonoverlap: bool = True,
    include_near_seam: bool = False,
    seam_buffer: int = 5,
) -> Dict[str, Any]:
    """
    分层采样光谱分析点。

    将像素分为以下区域：
      1. 重叠区（overlap）
      2. 非重叠区（non-overlap）
      3. 接缝线附近（near-seam）——重叠区边缘 5 像素以内

    参数
    ----------
    original, normalized : np.ndarray, shape (bands, rows, cols)
    nodata : float or None
    overlaps : list of dict
        每项含 idx_i, idx_j, window_i, window_j。
    n_samples : int
        总采样点数。
    seed : int
        随机种子。
    include_overlap : bool
        是否采样重叠区。
    include_nonoverlap : bool
        是否采样非重叠区。
    include_near_seam : bool
        是否采样接缝线附近区域。
    seam_buffer : int
        接缝线缓冲区像素数。

    返回
    -------
    dict
        indices : list of (row, col) — 采样点坐标
        region_labels : list of str — 各采样点所属区域
        overlap_mask : np.ndarray, dtype=bool — 重叠区掩码
        near_seam_mask : np.ndarray, dtype=bool — 接缝线附近掩码
    """
    orig_f, norm_f = _validate_inputs(original, normalized, nodata)
    rows, cols = orig_f.shape[1], orig_f.shape[2]

    # 构建有效像素掩码
    all_valid = _build_valid_mask(orig_f, nodata) & _build_valid_mask(norm_f, nodata)

    # 构建重叠区掩码
    overlap_mask = np.zeros((rows, cols), dtype=bool)
    for ov in overlaps:
        r1s, r1e, c1s, c1e = ov["window_i"]
        r2s, r2e, c2s, c2e = ov["window_j"]
        overlap_mask[r1s:r1e, c1s:c1e] = True
        overlap_mask[r2s:r2e, c2s:c2e] = True

    # 接缝线掩码：重叠区边缘区域
    near_seam_mask = np.zeros((rows, cols), dtype=bool)
    if include_near_seam and seam_buffer > 0:
        from scipy.ndimage import binary_dilation, binary_erosion
        dilated = binary_dilation(overlap_mask, iterations=seam_buffer)
        eroded = binary_erosion(overlap_mask, iterations=seam_buffer)
        near_seam_mask = dilated & ~eroded

    # 非重叠区掩码
    nonoverlap_mask = all_valid & ~overlap_mask

    # 可用区域及权重
    regions = []
    if include_overlap:
        regions.append(("overlap", overlap_mask & all_valid))
    if include_near_seam:
        regions.append(("near_seam", near_seam_mask & all_valid))
    if include_nonoverlap:
        regions.append(("non_overlap", nonoverlap_mask & all_valid))

    # 若无可用区域
    total_valid = sum(m.sum() for _, m in regions)
    if total_valid == 0:
        return {
            "indices": [],
            "region_labels": [],
            "overlap_mask": overlap_mask,
            "near_seam_mask": near_seam_mask,
        }

    # 按比例分配各区域采样数
    rng = np.random.RandomState(seed)
    all_indices = []
    all_labels = []

    for region_name, mask in regions:
        n_in_region = mask.sum()
        if n_in_region == 0:
            continue
        # 按有效像素比例分配采样数
        proportion = n_in_region / total_valid
        n_draw = max(1, int(round(n_samples * proportion)))
        n_draw = min(n_draw, n_in_region)

        ys, xs = np.where(mask)
        chosen = rng.choice(len(ys), size=n_draw, replace=False)
        for idx in chosen:
            all_indices.append((int(ys[idx]), int(xs[idx])))
            all_labels.append(region_name)

    return {
        "indices": all_indices,
        "region_labels": all_labels,
        "overlap_mask": overlap_mask,
        "near_seam_mask": near_seam_mask,
    }


# ===========================================================================
# 7. 一站式计算所有光谱指标
# ===========================================================================

def compute_all_spectral(
    arrays_before: List[np.ndarray],
    arrays_after: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    bands: Optional[List[int]] = None,
    band_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    n_bins: int = 256,
) -> Dict[str, Any]:
    """
    一次性计算所有光谱保真度指标。

    对每幅影像分别计算 SAM、光谱 RMSE、相对光谱 RMSE，
    并汇总波段比值和相关系数信息。

    参数
    ----------
    arrays_before : list of np.ndarray
        归一化前的影像数组，每个形状为 (bands, rows, cols)。
    arrays_after : list of np.ndarray
        归一化后的影像数组。
    nodata_values : list of float or None
        各影像的 NoData 值。
    overlaps : list of dict
        重叠信息。
    bands : list of int, optional
        要处理的波段索引（0-based）。None 表示全部波段。
    n_bins : int
        直方图 bin 数（暂未使用，保留接口）。

    返回
    -------
    dict
        per_image : list of dict — 各影像的逐像素 SAM 和 RMSE 统计
        aggregate : dict — 所有影像的聚合统计
        correlation : dict — 各影像的波段相关性信息
    """
    n_images = len(arrays_before)
    if n_images == 0:
        return {"per_image": [], "aggregate": {}, "correlation": []}

    n_bands = arrays_before[0].shape[0]
    if bands is None:
        bands = list(range(n_bands))

    per_image = []
    corr_results = []

    for idx in range(n_images):
        orig = arrays_before[idx]
        norm = arrays_after[idx]
        nd = nodata_values[idx]
        scene_id = f"scene_{idx}"

        try:
            sam = compute_sam(orig, norm, nd, bands=bands,
                              output_dir=output_dir, scene_id=scene_id)
            rmse = compute_spectral_rmse(orig, norm, nd)
            rel_rmse = compute_relative_spectral_rmse(orig, norm, nd)
            corr = compute_spectral_correlation(orig, norm, nd)
        except (ValueError, Exception) as exc:
            import logging as _log
            _log.getLogger(__name__).warning("Scene %d 光谱计算失败: %s", idx, exc)
            nan_stats = {"mean": float("nan"), "median": float("nan"),
                         "p90": float("nan"), "p95": float("nan")}
            sam = nan_stats
            rmse = nan_stats
            rel_rmse = nan_stats
            corr = {"per_band": [], "corr_orig": [], "corr_norm": [],
                    "frobenius_diff": float("nan")}

        per_image.append({
            "image_index": idx,
            "sam": {k: v for k, v in sam.items() if k != "per_pixel"},
            "spectral_rmse": rmse,
            "relative_spectral_rmse": rel_rmse,
            "n_valid_pixels": int(_build_valid_mask(
                orig.astype(np.float64), nd).sum()),
        })
        corr_results.append(corr)

    # 聚合统计（对所有影像取均值）
    agg_sam = {
        k: np.nanmean([pi["sam"][k] for pi in per_image])
        for k in ["mean", "median", "p90", "p95"]
    }
    agg_rmse = {
        k: np.nanmean([pi["spectral_rmse"][k] for pi in per_image])
        for k in ["mean", "median", "p90", "p95"]
    }
    agg_rel_rmse = {
        k: np.nanmean([pi["relative_spectral_rmse"][k] for pi in per_image])
        for k in ["mean", "median", "p90", "p95"]
    }

    # 波段比值：使用第一对稳定波段 (0, 1) 作为示例
    # 实际使用时可通过接口扩展
    default_pairs = [(0, min(1, n_bands - 1))]
    ratio_errors = {}
    for idx in range(n_images):
        try:
            re = compute_band_ratio_error(
                arrays_before[idx], arrays_after[idx],
                nodata_values[idx], default_pairs,
                band_names=band_names, output_dir=output_dir,
                scene_id=f"scene_{idx}")
            ratio_errors[idx] = {str(k): v for k, v in re.items()}
        except Exception:
            ratio_errors[idx] = {}

    return {
        "per_image": per_image,
        "aggregate": {
            "sam": agg_sam,
            "spectral_rmse": agg_rmse,
            "relative_spectral_rmse": agg_rel_rmse,
        },
        "correlation": corr_results,
        "band_ratio_errors": ratio_errors,
    }
