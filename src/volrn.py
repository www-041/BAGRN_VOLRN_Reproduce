"""
VOLRN — Variational Optimization based Local Radiometric Normalization

实现论文中的局部辐射归一化第二阶段（Stage 2）。
关键公式对应关系：
  - Eq.(12)   : f'_k = a · f_k + b
  - Eq.(13)   : μ' = aμ + b,  σ' = aσ
  - Eq.(15)   : ε_μ = a_i μ_i + b_i − (a_j μ_j + b_j),  ε_σ = a_i σ_i − a_j σ_j
  - Eq.(17)   : ε = Bx
  - Eq.(22)   : τ_μ = a_k μ_k + b_k − μ_k,  τ_σ = a_k σ_k − σ_k
  - Eq.(24)   : τ = Ax − b
  - Eq.(29)   : E(x) = ½||Bx||₂² + λ||Ax − b||₁
  - Eq.(30-33): ADMM 求解
  - Eq.(34-35): 反距离加权插值
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator
from typing import List, Tuple, Optional, Dict
from shapely.geometry import box
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------

@dataclass
class BlockInfo:
    """单个 image block 的信息。"""
    block_id: int          # 全局唯一块编号 (0 ~ T-1)
    image_idx: int         # 所属影像索引
    grid_m: int            # 网格行索引 (第几行网格)
    grid_n: int            # 网格列索引 (第几列网格)
    window: Tuple[int, int, int, int]  # (r_s, r_e, c_s, c_e) 在源影像中的窗口
    center_x: float        # 块中心地理 x 坐标（用于 IDW）
    center_y: float        # 块中心地理 y 坐标（用于 IDW）
    mu: np.ndarray         # 各波段均值，形状 (n_bands,)
    sigma: np.ndarray      # 各波段标准差，形状 (n_bands,)
    valid_mask: np.ndarray # 各波段有效掩膜，形状 (n_bands, rows, cols) —— 仅保存，不占大内存


@dataclass
class BlockPairInfo:
    """一对有重叠关系的 image block 的信息。"""
    block_id_i: int
    block_id_j: int
    grid_m: int
    grid_n: int


# ---------------------------------------------------------------------------
# Step 2.3.1: 图像分块
# ---------------------------------------------------------------------------

def _compute_combined_bounds(
    bounds_list: List[Tuple[float, float, float, float]],
) -> Tuple[float, float, float, float]:
    """计算所有影像的联合最小外接矩形（地理坐标）。"""
    left   = min(b[0] for b in bounds_list)
    bottom = min(b[1] for b in bounds_list)
    right  = max(b[2] for b in bounds_list)
    top    = max(b[3] for b in bounds_list)
    return left, bottom, right, top


def _pixel_window_from_bounds(
    geo_bounds: Tuple[float, float, float, float],
    transform,
    img_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """根据地理包围盒，计算在影像中的像素窗口。"""
    rows, cols = img_shape
    left, bottom, right, top = geo_bounds

    c_min, r_min = ~transform * (left, top)
    c_max, r_max = ~transform * (right, bottom)

    r_s = max(0, int(np.floor(min(r_min, r_max))))
    r_e = min(rows, int(np.ceil(max(r_min, r_max))))
    c_s = max(0, int(np.floor(min(c_min, c_max))))
    c_e = min(cols, int(np.ceil(max(c_min, c_max))))

    if r_e <= r_s or c_e <= c_s:
        return None
    return (r_s, r_e, c_s, c_e)


def image_blocking(
    arrays: List[np.ndarray],
    transforms: List,
    bounds_list: List[Tuple[float, float, float, float]],
    nodata_values: List[Optional[float]],
    block_size: int,
    bands: List[int],
) -> Tuple[List[BlockInfo], List[BlockPairInfo]]:
    """
    对所有影像进行规则网格分块（Section 2.3.1）。
    """
    left, bottom, right, top = _compute_combined_bounds(bounds_list)
    width_geo = right - left
    height_geo = top - bottom

    res_x = abs(transforms[0].a)
    res_y = abs(transforms[0].e)
    block_geo_x = block_size * res_x
    block_geo_y = block_size * res_y

    if block_geo_x <= 0 or block_geo_y <= 0:
        raise ValueError("block_geo 尺寸必须 > 0")

    n_cols_grid = max(1, int(np.ceil(width_geo / block_geo_x)))
    n_rows_grid = max(1, int(np.ceil(height_geo / block_geo_y)))

    n_bands = len(bands)
    blocks: List[BlockInfo] = []
    pairs: List[BlockPairInfo] = []
    block_id_counter = 0

    for gm in range(n_rows_grid):
        for gn in range(n_cols_grid):
            cell_left   = left + gn * block_geo_x
            cell_right  = min(left + (gn + 1) * block_geo_x, right)
            cell_top    = top  - gm * block_geo_y
            cell_bottom = max(top - (gm + 1) * block_geo_y, bottom)

            if cell_right <= cell_left or cell_top <= cell_bottom:
                continue

            cell_bounds = (cell_left, cell_bottom, cell_right, cell_top)
            cell_box = box(*cell_bounds)

            cell_blocks: Dict[int, BlockInfo] = {}

            for img_idx, (arr, tr, bnds, nd) in enumerate(
                zip(arrays, transforms, bounds_list, nodata_values)
            ):
                img_box = box(*bnds)
                if not cell_box.intersects(img_box):
                    continue

                window = _pixel_window_from_bounds(
                    cell_bounds, tr, (arr.shape[1], arr.shape[2])
                )
                if window is None:
                    continue

                r_s, r_e, c_s, c_e = window

                mu_block = np.zeros(n_bands)
                sg_block = np.zeros(n_bands)
                block_valid = True
                block_mask = np.zeros((n_bands, r_e - r_s, c_e - c_s), dtype=bool)

                for b_idx, band in enumerate(bands):
                    patch = arr[band, r_s:r_e, c_s:c_e]
                    if nd is not None:
                        valid = (patch != nd) & np.isfinite(patch)
                    else:
                        valid = np.isfinite(patch)

                    block_mask[b_idx] = valid
                    if valid.sum() > 0:
                        mu_block[b_idx] = float(patch[valid].mean())
                        sg_block[b_idx] = float(patch[valid].std())
                    else:
                        block_valid = False
                        break

                if not block_valid:
                    continue

                center_x = (cell_left + cell_right) / 2.0
                center_y = (cell_top + cell_bottom) / 2.0

                blk = BlockInfo(
                    block_id=block_id_counter,
                    image_idx=img_idx,
                    grid_m=gm,
                    grid_n=gn,
                    window=window,
                    center_x=center_x,
                    center_y=center_y,
                    mu=mu_block.copy(),
                    sigma=sg_block.copy(),
                    valid_mask=block_mask,
                )
                cell_blocks[img_idx] = blk
                blocks.append(blk)
                block_id_counter += 1

            img_idxs = list(cell_blocks.keys())
            for ii in range(len(img_idxs)):
                for jj in range(ii + 1, len(img_idxs)):
                    pairs.append(BlockPairInfo(
                        block_id_i=cell_blocks[img_idxs[ii]].block_id,
                        block_id_j=cell_blocks[img_idxs[jj]].block_id,
                        grid_m=gm,
                        grid_n=gn,
                    ))

    return blocks, pairs


# ---------------------------------------------------------------------------
# Step 2.3.2: 构建变分模型稀疏矩阵
# ---------------------------------------------------------------------------

def _build_volrn_system(
    blocks: List[BlockInfo],
    pairs: List[BlockPairInfo],
    band_idx: int,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, float]:
    """对指定波段构建 VOLRN 稀疏矩阵 B、A 和向量 b（Eq.15-28）。"""
    T = len(blocks)
    M = len(pairs)
    n_eqs_B = 2 * M
    n_eqs_A = 2 * T
    n_vars = 2 * T

    b_rows, b_cols, b_data = [], [], []

    for p_idx, pair in enumerate(pairs):
        bi = pair.block_id_i
        bj = pair.block_id_j
        mu_i = blocks[bi].mu[band_idx]
        mu_j = blocks[bj].mu[band_idx]
        sg_i = blocks[bi].sigma[band_idx]
        sg_j = blocks[bj].sigma[band_idx]

        row_mu = 2 * p_idx
        b_rows.extend([row_mu, row_mu, row_mu, row_mu])
        b_cols.extend([2*bi, 2*bi+1, 2*bj, 2*bj+1])
        b_data.extend([mu_i, 1.0, -mu_j, -1.0])

        row_sg = 2 * p_idx + 1
        b_rows.extend([row_sg, row_sg])
        b_cols.extend([2*bi, 2*bj])
        b_data.extend([sg_i, -sg_j])

    B = sparse.csr_matrix(
        (b_data, (b_rows, b_cols)),
        shape=(n_eqs_B, n_vars),
    )

    a_rows, a_cols, a_data = [], [], []
    b_vec = np.zeros(n_eqs_A)

    for k in range(T):
        mu_k = blocks[k].mu[band_idx]
        sg_k = blocks[k].sigma[band_idx]

        row_mu = 2 * k
        a_rows.extend([row_mu, row_mu])
        a_cols.extend([2*k, 2*k+1])
        a_data.extend([mu_k, 1.0])
        b_vec[row_mu] = mu_k

        row_sg = 2 * k + 1
        a_rows.extend([row_sg])
        a_cols.extend([2*k])
        a_data.extend([sg_k])
        b_vec[row_sg] = sg_k

    A = sparse.csr_matrix(
        (a_data, (a_rows, a_cols)),
        shape=(n_eqs_A, n_vars),
    )

    mu_vals = np.array([blocks[k].mu[band_idx] for k in range(T)])
    mu_scale = float(np.mean(np.abs(mu_vals)))

    return B, A, b_vec, mu_scale


# ---------------------------------------------------------------------------
# Step 2.3.3: ADMM 优化求解
# ---------------------------------------------------------------------------

def _soft_threshold(v: np.ndarray, kappa: float) -> np.ndarray:
    """软阈值函数 S_κ(v) = sign(v) · max(|v| − κ, 0)"""
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)


def _admm_solver(
    B: sparse.csr_matrix,
    A: sparse.csr_matrix,
    b: np.ndarray,
    lambda_param: float,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-4,
    verbose: bool = False,
) -> Tuple[np.ndarray, bool, int]:
    """
    用 ADMM 求解变分模型（Eq.30-33）。
    """
    n_vars = B.shape[1]
    n_eqs_A = A.shape[0]

    x = np.ones(n_vars)
    x[1::2] = 0.0
    z = np.zeros(n_eqs_A)
    u = np.zeros(n_eqs_A)

    BtB = B.T @ B
    AtA = A.T @ A
    At = A.T
    M = BtB + rho * AtA
    M = M.tocsr()

    converged = False
    for it in range(max_iter):
        rhs = rho * (At @ (z - u + b))
        x_new, cg_info = cg(M, rhs, x0=x, rtol=1e-6, maxiter=500, atol=1e-10)
        if cg_info != 0 and verbose:
            print(f"  [警告] CG 未收敛 (info={cg_info})，迭代 {it+1}")

        Az_minus_b = A @ x_new - b
        z_new = _soft_threshold(Az_minus_b + u, lambda_param / rho)
        u_new = u + Az_minus_b - z_new

        x_diff = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-12)
        if x_diff < tol:
            if verbose:
                print(f"ADMM 收敛于迭代 {it + 1}, rel_diff={x_diff:.2e}")
            x = x_new
            z = z_new
            u = u_new
            converged = True
            n_iters = it + 1
            break

        x = x_new
        z = z_new
        u = u_new
    else:
        if verbose:
            print(f"ADMM 达到最大迭代 {max_iter}")
        n_iters = max_iter

    return x, converged, n_iters


# ---------------------------------------------------------------------------
# Step 2.3.4: 九邻域 IDW 插值
# ---------------------------------------------------------------------------

def _idw_interpolate(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    has_block: np.ndarray,
    all_gm: List[int],
    all_gn: List[int],
    pixel_gm: np.ndarray,
    pixel_gn: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    九邻域反距离加权插值（论文 Eq.34-35）— 向量化实现。

    对每个像素，在其周围的 3×3 网格块中，按距离倒数加权插值 a, b 系数。
    无有效邻域时回退到最近邻块。
    """
    n_gm = len(all_gm)
    n_gn = len(all_gn)

    gm_arr = np.array(all_gm, dtype=np.float64)
    gn_arr = np.array(all_gn, dtype=np.float64)

    rows_img = len(pixel_gm)
    cols_img = len(pixel_gn)
    a_out = np.zeros((rows_img, cols_img), dtype=np.float64)
    b_out = np.zeros((rows_img, cols_img), dtype=np.float64)

    # 有效块的位置和值
    valid_mask = has_block
    valid_gm_idx, valid_gn_idx = np.where(valid_mask)
    if len(valid_gm_idx) == 0:
        return a_out, b_out

    valid_a = grid_a[valid_gm_idx, valid_gn_idx]
    valid_b = grid_b[valid_gm_idx, valid_gn_idx]
    valid_gm = gm_arr[valid_gm_idx]
    valid_gn = gn_arr[valid_gn_idx]

    # 像素网格
    PG, PN = np.meshgrid(pixel_gm, pixel_gn, indexing='ij')
    # PG, PN shape: (rows_img, cols_img)
    # 重塑为 (N_pixels,)
    pg_flat = PG.ravel()
    pn_flat = PN.ravel()
    n_pixels = len(pg_flat)

    # 计算所有像素到所有有效块的距离: (N_pixels, N_valid_blocks)
    # 使用分块计算避免内存爆炸
    CHUNK = 5000
    a_flat = np.zeros(n_pixels, dtype=np.float64)
    b_flat = np.zeros(n_pixels, dtype=np.float64)

    for start in range(0, n_pixels, CHUNK):
        end = min(start + CHUNK, n_pixels)
        pg_chunk = pg_flat[start:end]
        pn_chunk = pn_flat[start:end]
        n_chunk = end - start

        # 距离矩阵: (n_chunk, N_valid_blocks)
        dg = pg_chunk[:, None] - valid_gm[None, :]  # (n_chunk, N_blocks)
        dn = pn_chunk[:, None] - valid_gn[None, :]
        dist = np.sqrt(dg**2 + dn**2)  # (n_chunk, N_blocks)

        # 对每个像素取最近的9个块
        k = min(9, dist.shape[1])
        if k <= 1:
            nearest_idx = np.zeros((n_chunk, 1), dtype=int)
        else:
            nearest_idx = np.argpartition(dist, k - 1, axis=1)[:, :k]  # (n_chunk, k)

        # 提取最近邻的距离和值
        row_idx = np.arange(n_chunk)[:, None]
        near_dist = dist[row_idx, nearest_idx]  # (n_chunk, k)
        near_a = valid_a[nearest_idx]  # (n_chunk, k)
        near_b = valid_b[nearest_idx]  # (n_chunk, k)

        # 距离为0的直接使用
        exact = near_dist < 1e-10
        has_exact = exact.any(axis=1)

        # 非精确匹配的用IDW
        inv_dist = np.zeros_like(near_dist)
        mask = near_dist > 1e-10
        np.divide(1.0, near_dist, out=inv_dist, where=mask)
        weights = np.where(mask, inv_dist, 0.0)
        w_sum = weights.sum(axis=1, keepdims=True)
        w_sum = np.where(w_sum > 0, w_sum, 1.0)
        idw_a = (weights * near_a).sum(axis=1) / w_sum.ravel()
        idw_b = (weights * near_b).sum(axis=1) / w_sum.ravel()

        # 精确匹配的直接使用
        if has_exact.any():
            exact_rows = np.where(has_exact)[0]
            for pi in exact_rows:
                col = np.argmax(exact[pi])
                idw_a[pi] = near_a[pi, col]
                idw_b[pi] = near_b[pi, col]

        a_flat[start:end] = idw_a
        b_flat[start:end] = idw_b

    a_out = a_flat.reshape(rows_img, cols_img)
    b_out = b_flat.reshape(rows_img, cols_img)

    return a_out, b_out


def _interpolate_and_apply(
    arrays: List[np.ndarray],
    nodata_values: List[Optional[float]],
    blocks: List[BlockInfo],
    all_x_per_band: np.ndarray,
    block_size: int,
    transforms: List,
    bounds_list: List[Tuple[float, float, float, float]],
    bands: List[int],
) -> List[np.ndarray]:
    """
    逐波段 IDW 插值并应用 f' = a*f + b。

    all_x_per_band : shape (n_bands, 2*T)，每个波段独立的系数。
    """
    n_blocks = len(blocks)
    n_bands_total = arrays[0].shape[0]

    grid_map = {}
    for blk in blocks:
        key = (blk.grid_m, blk.grid_n)
        if key not in grid_map:
            grid_map[key] = {}
        grid_map[key][blk.image_idx] = blk.block_id

    all_gm = sorted(set(blk.grid_m for blk in blocks))
    all_gn = sorted(set(blk.grid_n for blk in blocks))

    left_all, bottom_all, right_all, top_all = _compute_combined_bounds(bounds_list)
    res_x = abs(transforms[0].a)
    res_y = abs(transforms[0].e)
    block_geo_x = block_size * res_x
    block_geo_y = block_size * res_y

    results = []

    for img_idx, arr in enumerate(arrays):
        nd = nodata_values[img_idx]
        result = arr.astype(np.float64, copy=True)
        rows_img, cols_img = arr.shape[1], arr.shape[2]

        img_blocks = [blk for blk in blocks if blk.image_idx == img_idx]
        if not img_blocks:
            results.append(result)
            continue

        # 像素中心 → 坐标
        pixel_gm = np.zeros(rows_img, dtype=np.float64)
        pixel_gn = np.zeros(cols_img, dtype=np.float64)
        for row in range(rows_img):
            _, geo_y = transforms[img_idx] * (0, row)
            pixel_gm[row] = (top_all - geo_y) / block_geo_y - 0.5
        for col in range(cols_img):
            geo_x, _ = transforms[img_idx] * (col, 0)
            pixel_gn[col] = (geo_x - left_all) / block_geo_x - 0.5

        # 逐波段独立插值和应用
        for b_idx, band in enumerate(bands):
            # 提取该波段的系数
            block_a = np.zeros(n_blocks)
            block_b = np.zeros(n_blocks)
            for k in range(n_blocks):
                block_a[k] = all_x_per_band[b_idx, 2 * k]
                block_b[k] = all_x_per_band[b_idx, 2 * k + 1]

            a_grid = np.zeros((len(all_gm), len(all_gn)))
            b_grid = np.zeros((len(all_gm), len(all_gn)))
            has_block = np.zeros((len(all_gm), len(all_gn)), dtype=bool)

            for blk in img_blocks:
                gi = all_gm.index(blk.grid_m)
                gj = all_gn.index(blk.grid_n)
                a_grid[gi, gj] = block_a[blk.block_id]
                b_grid[gi, gj] = block_b[blk.block_id]
                has_block[gi, gj] = True

            # 九邻域 IDW 插值
            a_vals, b_vals = _idw_interpolate(
                a_grid, b_grid, has_block,
                all_gm, all_gn, pixel_gm, pixel_gn,
            )

            # 应用 f' = a*f + b，仅对有效像素
            val = result[band]
            if nd is not None:
                valid = (val != nd) & np.isfinite(val)
            else:
                valid = np.isfinite(val)
            result[band, valid] = a_vals[valid] * val[valid] + b_vals[valid]
            # 将 nodata 像素设为 np.nan
            result[band, ~valid] = np.nan

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def volrn_normalize(
    arrays: List[np.ndarray],
    transforms: List,
    bounds_list: List[Tuple[float, float, float, float]],
    nodata_values: List[Optional[float]],
    block_size_pixels: int = 200,
    lambda_param: float = 0.5,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-4,
    verbose: bool = False,
    return_diagnostics: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    VOLRN 局部辐射归一化主函数。

    参数
    ----------
    arrays : list of np.ndarray
    transforms : list of rasterio.Affine
    bounds_list : list of (left, bottom, right, top)
    nodata_values : list of float or None
    block_size_pixels : int
    lambda_param : float
    rho : float
    max_iter : int
    tol : float
    verbose : bool
    return_diagnostics : bool
        若 True，额外返回 diagnostics dict。

    返回
    -------
    results : list of np.ndarray
    block_coefficients : np.ndarray, shape (n_bands, n_blocks, 2)
    diagnostics : dict (仅 return_diagnostics=True 时)
    """
    if len(arrays) == 0:
        if return_diagnostics:
            return [], np.array([]), {}
        return [], np.array([])

    n_bands = arrays[0].shape[0]
    bands = list(range(n_bands))

    # ---- Step 0: 同波段共同缩放归一化 ----
    # 关键修复：所有影像的同一波段使用相同的 vmin 和 scale，
    # 不能逐影像各自计算，否则会破坏 BAGRN 已建立的共同辐射尺度。
    common_ranges = []  # 每个波段的 (vmin, scale)
    for b_idx in range(n_bands):
        all_valid_vals = []
        for img_idx, arr in enumerate(arrays):
            nd = nodata_values[img_idx]
            patch = arr[b_idx]
            if nd is not None:
                valid = (patch != nd) & np.isfinite(patch)
            else:
                valid = np.isfinite(patch)
            if valid.any():
                all_valid_vals.append(patch[valid])
        if all_valid_vals:
            all_vals = np.concatenate(all_valid_vals)
            vmin = float(np.percentile(all_vals, 1))
            vmax = float(np.percentile(all_vals, 99))
            scale = vmax - vmin
            if scale < 1e-10:
                scale = 1.0
        else:
            vmin, scale = 0.0, 1.0
        common_ranges.append((vmin, scale))

    if verbose:
        for b_idx in range(n_bands):
            vmin, scale = common_ranges[b_idx]
            print(f"  波段 {b_idx}: 公共范围 [{vmin:.1f}, {vmin+scale:.1f}], scale={scale:.1f}")

    # 归一化所有影像，同时保护 NoData
    norm_arrays = []
    nodata_masks = []  # 保存每幅影像每个波段的有效掩膜
    for img_idx, arr in enumerate(arrays):
        nd = nodata_values[img_idx]
        arr_norm = np.zeros_like(arr, dtype=np.float64)
        img_masks = np.zeros((n_bands, arr.shape[1], arr.shape[2]), dtype=bool)
        for b_idx in range(n_bands):
            vmin, scale = common_ranges[b_idx]
            patch = arr[b_idx].astype(np.float64)
            if nd is not None:
                valid = (patch != nd) & np.isfinite(patch)
            else:
                valid = np.isfinite(patch)
            img_masks[b_idx] = valid
            # 有效像素归一化，无效像素设为 0（后续不参与计算）
            arr_norm[b_idx, valid] = (patch[valid] - vmin) / scale
            arr_norm[b_idx, ~valid] = 0.0
        norm_arrays.append(arr_norm)
        nodata_masks.append(img_masks)

    # ---- Step 1: 图像分块 ----
    blocks, pairs = image_blocking(
        norm_arrays, transforms, bounds_list, nodata_values,
        block_size_pixels, bands,
    )

    if len(blocks) == 0 or len(pairs) == 0:
        if return_diagnostics:
            return [arr.copy() for arr in arrays], np.array([]), {
                'n_blocks': 0,
                'n_pairs': 0,
                'n_bands': n_bands,
                'block_size': block_size_pixels,
                'lambda': lambda_param,
                'rho': rho,
                'all_converged': True,
                'block_details': [],
            }
        return [arr.copy() for arr in arrays], np.array([])

    T = len(blocks)
    n_coeffs = 2 * T

    if verbose:
        print(f"VOLRN: {len(blocks)} blocks, {len(pairs)} pairs")

    # ---- Step 2-3: 逐波段构建系统并求解（各波段独立） ----
    all_x = np.zeros((n_bands, n_coeffs))
    band_converged = np.zeros(n_bands, dtype=bool)
    band_iterations = np.zeros(n_bands, dtype=int)

    for b_idx, band in enumerate(bands):
        B, A, b_vec, mu_scale = _build_volrn_system(blocks, pairs, b_idx)

        if verbose:
            print(f"  波段 {band}: μ_scale={mu_scale:.3f} (归一化后), λ={lambda_param}, ρ={rho}")

        x, converged, n_iters = _admm_solver(B, A, b_vec, lambda_param, rho, max_iter, tol, verbose)
        band_converged[b_idx] = converged
        band_iterations[b_idx] = n_iters
        if not converged:
            if verbose:
                print(f"  波段 {band}: ADMM 未收敛，使用单位变换")
            x = np.zeros(n_coeffs)
            x[0::2] = 1.0
            x[1::2] = 0.0
        all_x[b_idx, :] = x

    all_converged = bool(band_converged.all())
    if not all_converged and verbose:
        print("  警告: 部分波段 ADMM 未收敛")

    # ---- Step 4-5: 逐波段 IDW 插值并应用 ----
    results = _interpolate_and_apply(
        norm_arrays, nodata_values, blocks, all_x,
        block_size_pixels, transforms, bounds_list, bands,
    )

    # ---- Step 6: 反归一化回原始数据范围 ----
    for img_idx in range(len(results)):
        for b_idx in range(n_bands):
            vmin, scale = common_ranges[b_idx]
            results[img_idx][b_idx] = results[img_idx][b_idx] * scale + vmin
            # 恢复 NoData
            nd = nodata_values[img_idx]
            if nd is not None:
                results[img_idx][b_idx, ~nodata_masks[img_idx][b_idx]] = nd

    # 组装 block 系数输出: shape (n_bands, n_blocks, 2)
    block_coeffs = np.zeros((n_bands, T, 2))
    for b_idx in range(n_bands):
        block_coeffs[b_idx, :, 0] = all_x[b_idx, 0::2]  # a
        block_coeffs[b_idx, :, 1] = all_x[b_idx, 1::2]  # b

    if return_diagnostics:
        block_list = []
        for blk in blocks:
            block_list.append({
                'block_id': blk.block_id,
                'image_idx': blk.image_idx,
                'grid_m': blk.grid_m,
                'grid_n': blk.grid_n,
                'center_x': blk.center_x,
                'center_y': blk.center_y,
            })

        diag_blocks = []
        for b_idx in range(n_bands):
            for k in range(T):
                diag_blocks.append({
                    'band': b_idx,
                    'block_id': blocks[k].block_id,
                    'image_idx': blocks[k].image_idx,
                    'grid_m': blocks[k].grid_m,
                    'grid_n': blocks[k].grid_n,
                    'center_x': blocks[k].center_x,
                    'center_y': blocks[k].center_y,
                    'mu': float(blocks[k].mu[b_idx]),
                    'sigma': float(blocks[k].sigma[b_idx]),
                    'a': float(all_x[b_idx, 2*k]),
                    'b': float(all_x[b_idx, 2*k+1]),
                    'converged': bool(band_converged[b_idx]),
                    'iterations': int(band_iterations[b_idx]),
                })

        diagnostics = {
            'n_blocks': T,
            'n_pairs': len(pairs),
            'n_bands': n_bands,
            'block_size': block_size_pixels,
            'lambda': lambda_param,
            'rho': rho,
            'all_converged': all_converged,
            'band_converged': band_converged.tolist(),
            'band_iterations': band_iterations.tolist(),
            'block_details': diag_blocks,
        }
        return results, block_coeffs, diagnostics

    return results, block_coeffs
