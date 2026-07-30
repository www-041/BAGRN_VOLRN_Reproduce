"""
VOLRN 局部辐射归一化的单元测试

使用合成小型数组验证：
  - 分块逻辑正确
  - 变分模型能收敛
  - 局部接缝差异在 VOLRN 后变小
  - 与 BAGRN 联合使用时效果提升
"""

import numpy as np
import rasterio
import pytest
from src.volrn import volrn_normalize, image_blocking, _build_volrn_system
from src.bagrn import bagrn_normalize


# ===================================================================
# 辅助：生成合成测试影像
# ===================================================================

def _make_test_pair(seed: int = 42):
    """
    生成一对 8×8 单波段合成影像，水平方向有 4 列重叠：

    img_0: 地理范围 (0,0)~(8,8), 均值 ~100, 标准差 ~10
    img_1: 地理范围 (4,0)~(12,8), 均值 ~130, 标准差 ~18
    两图在 x=4~8 范围重叠（4 列）。
    """
    np.random.seed(seed)
    rows = cols = 8
    res = 1.0

    arr0 = (100 + 10 * np.random.randn(1, rows, cols)).astype(np.float32)
    arr1 = (130 + 18 * np.random.randn(1, rows, cols)).astype(np.float32)

    tr0 = rasterio.Affine(res, 0, 0, 0, -res, rows)
    tr1 = rasterio.Affine(res, 0, 4, 0, -res, rows)

    bnds0 = (0, 0, cols, rows)
    bnds1 = (4, 0, cols + 4, rows)

    imgs = [arr0, arr1]
    trs = [tr0, tr1]
    bnds = [bnds0, bnds1]
    nds = [None, None]

    return imgs, trs, bnds, nds


def _make_test_triplet(seed: int = 7):
    """
    生成三张 6×6 单波段影像，模拟条带排列：

    img_0: (0,0)~(6,6), 均值 100, 标准差 10
    img_1: (3,0)~(9,6), 均值 130, 标准差 18
    img_2: (6,0)~(12,6), 均值 85, 标准差 12
    重叠: (0,1) 在 x=3~6, (1,2) 在 x=6~9 → 实际上 0 和 2 不重叠
    """
    np.random.seed(seed)
    rows = cols = 6
    res = 1.0

    arr0 = (100 + 10 * np.random.randn(1, rows, cols)).astype(np.float32)
    arr1 = (130 + 18 * np.random.randn(1, rows, cols)).astype(np.float32)
    arr2 = (85 + 12 * np.random.randn(1, rows, cols)).astype(np.float32)

    tr0 = rasterio.Affine(res, 0, 0, 0, -res, rows)
    tr1 = rasterio.Affine(res, 0, 3, 0, -res, rows)
    tr2 = rasterio.Affine(res, 0, 6, 0, -res, rows)

    bnds0 = (0, 0, cols, rows)
    bnds1 = (3, 0, cols + 3, rows)
    bnds2 = (6, 0, cols + 6, rows)

    imgs = [arr0, arr1, arr2]
    trs = [tr0, tr1, tr2]
    bnds = [bnds0, bnds1, bnds2]
    nds = [None, None, None]

    return imgs, trs, bnds, nds


# ===================================================================
# 测试用例
# ===================================================================

class TestImageBlocking:
    """测试图像分块模块"""

    def test_block_count(self):
        """block 数量应 > 0"""
        imgs, trs, bnds, nds = _make_test_pair()
        blocks, pairs = image_blocking(
            imgs, trs, bnds, nds, block_size=4, bands=[0],
        )
        assert len(blocks) > 0
        assert len(pairs) > 0

    def test_each_block_has_valid_stats(self):
        """每个 block 的 μ、σ 应为有效数值"""
        imgs, trs, bnds, nds = _make_test_pair()
        blocks, pairs = image_blocking(
            imgs, trs, bnds, nds, block_size=4, bands=[0],
        )
        for blk in blocks:
            assert np.isfinite(blk.mu[0])
            assert np.isfinite(blk.sigma[0])
            assert blk.sigma[0] > 0


class TestBuildSystem:
    """测试变分模型系统构建"""

    def test_matrix_shapes(self):
        """B, A 矩阵形状应正确"""
        imgs, trs, bnds, nds = _make_test_pair()
        blocks, pairs = image_blocking(
            imgs, trs, bnds, nds, block_size=4, bands=[0],
        )
        T = len(blocks)
        M = len(pairs)
        B, A, b_vec, mu_scale = _build_volrn_system(blocks, pairs, band_idx=0)

        assert B.shape == (2 * M, 2 * T), f"B shape {B.shape}"
        assert A.shape == (2 * T, 2 * T), f"A shape {A.shape}"
        assert b_vec.shape == (2 * T,), f"b shape {b_vec.shape}"
        assert mu_scale > 0, f"mu_scale should be positive, got {mu_scale}"


class TestVolrnBasic:
    """VOLRN 基本功能测试"""

    def test_output_shape(self):
        """输出数组形状与输入一致"""
        imgs, trs, bnds, nds = _make_test_pair()
        results, coeffs = volrn_normalize(
            imgs, trs, bnds, nds,
            block_size_pixels=4,
            lambda_param=0.5,
            max_iter=50,
        )
        assert len(results) == len(imgs)
        for r, orig in zip(results, imgs):
            assert r.shape == orig.shape

    def test_coefficient_shape(self):
        """block_coefficients 形状应为 (n_bands, n_blocks, 2)"""
        imgs, trs, bnds, nds = _make_test_pair()
        blocks, _ = image_blocking(
            imgs, trs, bnds, nds, block_size=4, bands=[0],
        )
        results, coeffs = volrn_normalize(
            imgs, trs, bnds, nds,
            block_size_pixels=4,
            lambda_param=0.5,
            max_iter=50,
        )
        n_bands = imgs[0].shape[0]
        assert coeffs.shape == (n_bands, len(blocks), 2)

    def test_energy_decreases_after_admm(self):
        """
        ADMM 求解后，目标函数 E(x) = ½||Bx||₂² + λ||Ax−b||₁
        应比初始值 (x=identity) 降低。
        """
        imgs, trs, bnds, nds = _make_test_pair()
        blocks, pairs = image_blocking(
            imgs, trs, bnds, nds, block_size=4, bands=[0],
        )
        B, A, b_vec, mu_scale = _build_volrn_system(blocks, pairs, band_idx=0)
        T = len(blocks)
        n_vars = 2 * T

        from src.volrn import _admm_solver
        scale = mu_scale if mu_scale > 0 else 1.0
        lambda_scaled = 0.5 * scale
        rho_scaled = 1.0 * scale
        x_sol, converged, n_iters = _admm_solver(B, A, b_vec, lambda_param=lambda_scaled, rho=rho_scaled,
                             max_iter=100, tol=1e-4)

        def energy(x):
            return 0.5 * np.linalg.norm(B @ x) ** 2 + \
                   lambda_scaled * np.linalg.norm(A @ x - b_vec, ord=1)

        # 初始值为单位变换 (a=1, b=0)
        x_init = np.zeros(n_vars)
        x_init[0::2] = 1.0
        e0 = energy(x_init)
        e_sol = energy(x_sol)
        assert e_sol <= e0 * 1.01 + 1e-3, f"E(x_init)={e0:.4f}, E(x_sol)={e_sol:.4f}"

    def test_volrn_does_not_degrade_constant_images(self):
        """
        常量影像 + BAGRN 完美校正后，VOLRN 的 L1 正则应使大部分 block
        保持 a≈1, b≈0，不会引入新的人工痕迹。
        """
        rows = cols = 12
        res = 1.0
        # img_0: 均值 100 的常量影像
        arr0 = np.full((1, rows, cols), 100.0, dtype=np.float32)
        # img_1: 均值 150 的常量影像
        arr1 = np.full((1, rows, cols), 150.0, dtype=np.float32)

        tr0 = rasterio.Affine(res, 0, 0, 0, -res, rows)
        tr1 = rasterio.Affine(res, 0, 4, 0, -res, rows)
        bnds0 = (0, 0, cols, rows)
        bnds1 = (4, 0, cols + 4, rows)
        imgs = [arr0, arr1]
        trs = [tr0, tr1]
        bnds = [bnds0, bnds1]
        nds = [None, None]

        from src.overlap import get_overlap_window
        ov = [{
            "idx_i": 0, "idx_j": 1,
            "window_i": get_overlap_window(bnds[0], trs[0], bnds[1], trs[1])[0],
            "window_j": get_overlap_window(bnds[0], trs[0], bnds[1], trs[1])[1],
        }]
        bagrn_results, _, _ = bagrn_normalize(imgs, nds, ov, control_idx=0)
        volrn_results, _ = volrn_normalize(
            bagrn_results, trs, bnds, nds,
            block_size_pixels=4, lambda_param=0.5, max_iter=100,
        )

        # 重叠区均值的绝对差不应超过 1.0
        r0s, r0e, c0s, c0e = ov[0]["window_i"]
        r1s, r1e, c1s, c1e = ov[0]["window_j"]
        diff = np.abs(
            volrn_results[0][0, r0s:r0e, c0s:c0e].mean() -
            volrn_results[1][0, r1s:r1e, c1s:c1e].mean()
        )
        assert diff < 1.0, f"VOLRN 后常量图重叠区均值差 {diff:.4f}"


class TestVolrnTriplet:
    """三张影像的 VOLRN 测试"""

    def test_triplet_pipeline(self):
        """
        对三张影像依次经过 BAGRN + VOLRN，验证重叠区辐射差异缩小。
        """
        imgs, trs, bnds, nds = _make_test_triplet()

        # BAGRN 重叠
        from src.overlap import get_overlap_window
        ov01 = get_overlap_window(bnds[0], trs[0], bnds[1], trs[1])
        ov12 = get_overlap_window(bnds[1], trs[1], bnds[2], trs[2])
        overlaps = [
            {"idx_i": 0, "idx_j": 1, "window_i": ov01[0], "window_j": ov01[1]},
            {"idx_i": 1, "idx_j": 2, "window_i": ov12[0], "window_j": ov12[1]},
        ]
        bagrn_results, _, _ = bagrn_normalize(imgs, nds, overlaps, control_idx=0)

        # VOLRN
        volrn_results, _ = volrn_normalize(
            bagrn_results, trs, bnds, nds,
            block_size_pixels=3,
            lambda_param=0.5,
            max_iter=50,
        )

        # 计算每对重叠的 MAD
        for ov_idx, ov in enumerate(overlaps):
            r0s, r0e, c0s, c0e = ov["window_i"]
            r1s, r1e, c1s, c1e = ov["window_j"]

            mad_bagrn = np.abs(
                bagrn_results[ov["idx_i"]][0, r0s:r0e, c0s:c0e] -
                bagrn_results[ov["idx_j"]][0, r1s:r1e, c1s:c1e]
            ).mean()
            mad_volrn = np.abs(
                volrn_results[ov["idx_i"]][0, r0s:r0e, c0s:c0e] -
                volrn_results[ov["idx_j"]][0, r1s:r1e, c1s:c1e]
            ).mean()

            assert mad_volrn <= mad_bagrn + 0.1, \
                f"重叠 {ov_idx}: BAGRN MAD={mad_bagrn:.4f}, VOLRN MAD={mad_volrn:.4f}"
