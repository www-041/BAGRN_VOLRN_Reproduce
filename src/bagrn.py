"""
BAGRN — Block Adjustment based Global Radiometric Normalization

实现论文中的全局辐射归一化第一阶段（Stage 1）。
关键公式对应关系：
  - Eq.(1)  : θ_i - θ_j = μ_j - μ_i
  - Eq.(3)  : D_α X_μ = L_α
  - Eq.(4-6): 控制影像约束 θ_r = 0
  - Eq.(7)  : D X_μ = L
  - Eq.(8)  : 权重矩阵 P（重叠面积占比）
  - Eq.(10-11): Moment matching f' = ω f + υ
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# 辅助：从重叠窗口提取有效像素（排除 nodata）
# ---------------------------------------------------------------------------

def _overlap_means_stds(
    arrays: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    bands: List[int],
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    对每对重叠影像、每个指定波段，计算重叠区（排除 nodata）的 mean 和 std。

    参数
    ----------
    arrays : list of np.ndarray
        每幅影像的数组，形状均为 (bands, rows, cols)。
    nodata_values : list of float or None
        每幅影像的 nodata 值。
    overlaps : list of dict
        重叠信息，每项含 idx_i, idx_j, window_i, window_j 四个 key。
    bands : list of int
        要处理的波段索引（0-based）。

    返回
    -------
    pair_means : list of np.ndarray
        pair_means[k] 形状 (len(bands), 2)，两列分别为影像 i、j 的均值。
    pair_stds  : list of np.ndarray
        与 pair_means 形状相同的标准差。
    pair_pixels : np.ndarray
        每对重叠的有效像素数（1D）。
    """
    n_pairs = len(overlaps)
    n_bands = len(bands)
    pair_means = [np.zeros((n_bands, 2)) for _ in range(n_pairs)]
    pair_stds  = [np.zeros((n_bands, 2)) for _ in range(n_pairs)]
    pair_pixels = np.zeros(n_pairs, dtype=np.int64)

    for k, ov in enumerate(overlaps):
        i, j = ov["idx_i"], ov["idx_j"]
        r1_s, r1_e, c1_s, c1_e = ov["window_i"]
        r2_s, r2_e, c2_s, c2_e = ov["window_j"]

        arr_i = arrays[i]
        arr_j = arrays[j]
        nd_i = nodata_values[i]
        nd_j = nodata_values[j]

        pair_pixels[k] = (r1_e - r1_s) * (c1_e - c1_s)

        for b_idx, band in enumerate(bands):
            patch_i = arr_i[band, r1_s:r1_e, c1_s:c1_e]
            patch_j = arr_j[band, r2_s:r2_e, c2_s:c2_e]

            # 排除 nodata（两幅影像分辨率可能不同，分开计算有效像素）
            mask_i = np.isfinite(patch_i) if nd_i is None else (patch_i != nd_i)
            mask_j = np.isfinite(patch_j) if nd_j is None else (patch_j != nd_j)

            if mask_i.sum() == 0 or mask_j.sum() == 0:
                pair_means[k][b_idx] = [0.0, 0.0]
                pair_stds[k][b_idx]  = [0.0, 0.0]
            else:
                pair_means[k][b_idx] = [float(patch_i[mask_i].mean()), float(patch_j[mask_j].mean())]
                pair_stds[k][b_idx]  = [float(patch_i[mask_i].std()),  float(patch_j[mask_j].std())]

    return pair_means, pair_stds, pair_pixels


# ---------------------------------------------------------------------------
# 构建并求解 BAGRN 稀疏线性系统
# ---------------------------------------------------------------------------

def _solve_compensation(
    n_images: int,
    overlaps: List[dict],
    pair_values: List[np.ndarray],
    pair_pixels: np.ndarray,
    control_idx: int,
) -> np.ndarray:
    """
    构建并求解 BAGRN 的加权最小二乘系统。

    对均值 μ 和标准差 σ 求解形式一致，因此复用此函数。

    Eq.(1)  : θ_i - θ_j = val_j - val_i  (val 是 μ 或 σ)
    Eq.(3)  : D_α X = L_α
    Eq.(5)  : D_β X = 0  (控制影像)
    Eq.(7)  : D X = L
    Eq.(8)  : 加权 W — 重叠像素数占比

    参数
    ----------
    n_images : int
        影像总数 N。
    overlaps : list of dict
        每项含 idx_i, idx_j。
    pair_values : list of np.ndarray
        pair_values[k] 形状 (n_bands, 2)，两列分别为 i、j 的均值/标准差。
    pair_pixels : np.ndarray
        每对重叠的有效像素数。
    control_idx : int
        控制影像的索引，其补偿值设为 0。

    返回
    -------
    comp : np.ndarray
        形状 (n_images,)，每幅影像的补偿值。
    """
    n_pairs = len(overlaps)
    n_bands = pair_values[0].shape[0]

    # --- 权重 W（Eq.8）---
    total_pixels = pair_pixels.sum()
    if total_pixels == 0:
        weights = np.ones(n_pairs) / n_pairs
    else:
        weights = pair_pixels.astype(np.float64) / total_pixels

    # --- 对所有波段联合求解（每个波段独立建系统，但此处按波段循环）---
    # 论文中每个波段独立处理
    comp = np.zeros((n_bands, n_images), dtype=np.float64)

    for b_idx in range(n_bands):
        # ---- D_α 与 L_α ----
        # D_α: 稀疏矩阵 (n_pairs × n_images), 每行 +1 在 i, -1 在 j
        rows_da, cols_da, data_da = [], [], []
        L_alpha = np.zeros(n_pairs, dtype=np.float64)

        for k, ov in enumerate(overlaps):
            i, j = ov["idx_i"], ov["idx_j"]
            mu_i = pair_values[k][b_idx, 0]
            mu_j = pair_values[k][b_idx, 1]

            # 方程: θ_i - θ_j = μ_j - μ_i  (Eq.1)
            rows_da.extend([k, k])
            cols_da.extend([i, j])
            data_da.extend([1.0, -1.0])
            L_alpha[k] = mu_j - mu_i

        D_alpha = sparse.csr_matrix(
            (data_da, (rows_da, cols_da)),
            shape=(n_pairs, n_images),
        )

        # ---- D_β 与 L_β: 控制影像约束 (Eq.4-5) ----
        D_beta = sparse.csr_matrix(
            ([1.0], ([0], [control_idx])),
            shape=(1, n_images),
        )
        L_beta = np.zeros(1)

        # ---- 组装 D 和 L (Eq.7) ----
        D = sparse.vstack([D_alpha, D_beta], format="csr")
        L = np.concatenate([L_alpha, L_beta])

        # ---- 权重矩阵 P (Eq.8) ----
        sqrt_w = np.sqrt(weights)
        p_diag = np.concatenate([sqrt_w, [1.0]])
        P_sqrt = sparse.diags(p_diag, format="csr")

        # ---- 加权最小二乘: min || sqrt(P) (D X - L) ||² ----
        A_w = P_sqrt @ D
        b_w = P_sqrt @ L

        result = lsqr(A_w, b_w, atol=1e-10, btol=1e-10, show=False)
        comp[b_idx, :] = result[0]

    return comp


# ---------------------------------------------------------------------------
# Moment matching 应用到整幅影像
# ---------------------------------------------------------------------------

def _apply_moment_matching(
    array: np.ndarray,
    nodata: Optional[float],
    mu_orig: np.ndarray,
    sigma_orig: np.ndarray,
    theta_mu: np.ndarray,
    theta_sigma: np.ndarray,
    bands: List[int],
) -> np.ndarray:
    """
    对整幅影像逐波段应用 moment matching（Eq.10-11）。

    对每个波段：
      μ' = μ_orig + θ_μ
      σ' = σ_orig + θ_σ
      ω  = σ' / σ_orig
      υ  = μ' - ω · μ_orig
      f' = ω · f + υ

    参数
    ----------
    array : np.ndarray
        原始影像数组 (bands, rows, cols)。
    nodata : float or None
        NoData 值。
    mu_orig : np.ndarray
        原始均值，形状 (n_bands,)。
    sigma_orig : np.ndarray
        原始标准差，形状 (n_bands,)。
    theta_mu : np.ndarray
        均值补偿，形状 (n_bands,)。
    theta_sigma : np.ndarray
        标准差补偿，形状 (n_bands,)。
    bands : list of int
        要处理的波段索引。

    返回
    -------
    result : np.ndarray
        归一化后的数组，形状与 array 相同。
    """
    result = array.astype(np.float64, copy=True)

    for b_idx, band in enumerate(bands):
        mu_t = mu_orig[b_idx] + theta_mu[b_idx]    # 补偿后均值 μ'
        sg_t = sigma_orig[b_idx] + theta_sigma[b_idx]  # 补偿后标准差 σ'

        # 避免除零
        sg_orig = sigma_orig[b_idx]
        if sg_orig < 1e-12 or sg_t < 1e-12:
            omega = 1.0
        else:
            omega = sg_t / sg_orig   # ω (Eq.11)

        upsilon = mu_t - omega * mu_orig[b_idx]  # υ (Eq.11)

        band_data = result[band]
        if nodata is None:
            band_data[:] = omega * band_data + upsilon
        else:
            mask = band_data != nodata
            band_data[mask] = omega * band_data[mask] + upsilon

    return result


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def bagrn_normalize(
    arrays: List[np.ndarray],
    nodata_values: List[Optional[float]],
    overlaps: List[dict],
    control_idx: int = 0,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    BAGRN 全局辐射归一化主函数。

    流程：
      1. 对每对重叠区域，计算每个波段的 μ、σ（排除 nodata）
      2. 构建加权最小二乘系统，求解 θ_μ、θ_σ
      3. 对每幅影像做 moment matching，得到归一化结果

    参数
    ----------
    arrays : list of np.ndarray
        每幅影像的像素数组，形状均为 (bands, rows, cols)。
    nodata_values : list of float or None
        各影像的 nodata 值。
    overlaps : list of dict
        重叠信息，每项包含：
            idx_i, idx_j : int         — 重叠的两幅影像索引
            window_i     : (rS, rE, cS, cE)  — 在影像 i 中的窗口
            window_j     : (rS, rE, cS, cE)  — 在影像 j 中的窗口
    control_idx : int
        控制影像索引（其补偿值为 0）。默认为 0。

    返回
    -------
    normalized : list of np.ndarray
        归一化后的影像数组，与输入形状相同。
    theta_mu : np.ndarray
        每幅影像每个波段的均值补偿，形状 (n_bands, n_images)。
    theta_sigma : np.ndarray
        每幅影像每个波段的标注差补偿，形状 (n_bands, n_images)。
    """
    if len(arrays) == 0:
        return [], np.array([]), np.array([])

    n_images = len(arrays)
    n_bands = arrays[0].shape[0]
    bands = list(range(n_bands))

    # ---- 步骤 1: 计算重叠区的 μ 和 σ ----
    pair_means, pair_stds, pair_pixels = _overlap_means_stds(
        arrays, nodata_values, overlaps, bands,
    )

    # ---- 步骤 2: 求解补偿系数 ----
    theta_mu = _solve_compensation(
        n_images, overlaps, pair_means, pair_pixels, control_idx,
    )
    theta_sigma = _solve_compensation(
        n_images, overlaps, pair_stds, pair_pixels, control_idx,
    )

    # ---- 步骤 3: 对每幅影像计算全局 μ、σ 然后 moment matching ----
    normalized = []
    for img_idx in range(n_images):
        arr = arrays[img_idx]
        nd = nodata_values[img_idx]

        mu_i = np.zeros(n_bands)
        sg_i = np.zeros(n_bands)
        for b_idx, band in enumerate(bands):
            band_data = arr[band]
            if nd is None:
                valid = np.isfinite(band_data)
            else:
                valid = band_data != nd
            mu_i[b_idx] = band_data[valid].mean() if valid.sum() > 0 else 0.0
            sg_i[b_idx] = band_data[valid].std()  if valid.sum() > 0 else 0.0

        result = _apply_moment_matching(
            arr, nd,
            mu_i, sg_i,
            theta_mu[:, img_idx],
            theta_sigma[:, img_idx],
            bands,
        )
        normalized.append(result)

    return normalized, theta_mu, theta_sigma
