"""
BAGRN 全局辐射归一化的单元测试

使用合成小尺寸数组验证：
  - 重叠区 μ、σ 计算正确性
  - 加权最小二乘求解的补偿值正确
  - Moment matching 后重叠区辐射差异应趋近于 0
  - nodata 像元不被修改
"""

import numpy as np
import pytest
from src.bagrn import bagrn_normalize


def _make_simple_arrays():
    """
    创建 3 张合成影像（单波段 int16）：
      img_0 (控制): 范围 (0,0)~(4,4), 均值 ~100, 标准差 ~10
      img_1         : 范围 (2,0)~(6,4), 均值 ~120, 标准差 ~15
      img_2         : 范围 (5,0)~(9,4), 均值 ~90,  标准差 ~12

    重叠关系：
      (0,1): 列 2~4 宽 2 列 × 4 行 = 8 像素
      (1,2): 列 0~1 宽 1 列 × 4 行 = 4 像素（img_1 中列 0~1 相当于地理 5~6）
      注：img_2 左上角在地理 x=5，所以 img_1 的列 0~1 与 img_2 的列 0~1 重叠
    """
    np.random.seed(42)
    rows = cols = 4

    def _make(mean, std):
        arr = mean + std * np.random.randn(1, rows, cols)
        return arr.astype(np.float32)

    imgs = [_make(100, 10), _make(120, 15), _make(90, 12)]
    nodata = [None, None, None]

    overlaps = [
        # (0,1): img_0 中列 [2,4), img_1 中列 [0,2)
        {"idx_i": 0, "idx_j": 1,
         "window_i": (0, 4, 2, 4),
         "window_j": (0, 4, 0, 2)},
        # (1,2): img_1 中列 [3,4), img_2 中列 [0,1)
        {"idx_i": 1, "idx_j": 2,
         "window_i": (0, 4, 3, 4),
         "window_j": (0, 4, 0, 1)},
    ]

    return imgs, nodata, overlaps


def _make_arrays_with_nodata():
    """
    创建带有 nodata 的影像，验证 nodata 不被修改且不参与统计。
    img_0 在第 0 行第 0 列有一个 nodata 像素。
    """
    np.random.seed(123)
    rows = cols = 4

    arr0 = 100 + 10 * np.random.randn(1, rows, cols)
    arr0[0, 0, 0] = -9999  # nodata
    arr1 = 120 + 15 * np.random.randn(1, rows, cols)
    arr2 = 90 + 12 * np.random.randn(1, rows, cols)

    imgs = [arr0.astype(np.float32), arr1.astype(np.float32), arr2.astype(np.float32)]
    nodata = [-9999, None, None]

    overlaps = [
        {"idx_i": 0, "idx_j": 1,
         "window_i": (0, 4, 2, 4),
         "window_j": (0, 4, 0, 2)},
        {"idx_i": 1, "idx_j": 2,
         "window_i": (0, 4, 3, 4),
         "window_j": (0, 4, 0, 1)},
    ]
    return imgs, nodata, overlaps


# ===================================================================
# 测试用例
# ===================================================================

class TestBagrnBasic:
    """基础功能测试"""

    def test_output_shape(self):
        """归一化后形状不变"""
        imgs, nd, ov = _make_simple_arrays()
        result, _, _ = bagrn_normalize(imgs, nd, ov, control_idx=0)
        assert len(result) == len(imgs)
        for r, orig in zip(result, imgs):
            assert r.shape == orig.shape

    def test_overlap_mean_consistency(self):
        """归一化后重叠区的均值应趋于一致（控制影像为参考）"""
        imgs, nd, ov = _make_simple_arrays()
        result, tm, ts = bagrn_normalize(imgs, nd, ov, control_idx=0)

        # 取 img_0 和 img_1 的重叠区均值
        r0s, r0e, c0s, c0e = ov[0]["window_i"]
        r1s, r1e, c1s, c1e = ov[0]["window_j"]

        patch0 = result[0][0, r0s:r0e, c0s:c0e]
        patch1 = result[1][0, r1s:r1e, c1s:c1e]

        mean0 = patch0.mean()
        mean1 = patch1.mean()
        # 控制图与 img_1 重叠区均值差应很小
        assert abs(mean0 - mean1) < 1.0, f"均值差 {abs(mean0-mean1):.4f} > 1.0"

    def test_overlap_std_consistency(self):
        """归一化后重叠区的标准差应趋于一致"""
        imgs, nd, ov = _make_simple_arrays()
        result, _, _ = bagrn_normalize(imgs, nd, ov, control_idx=0)

        r0s, r0e, c0s, c0e = ov[0]["window_i"]
        r1s, r1e, c1s, c1e = ov[0]["window_j"]

        patch0 = result[0][0, r0s:r0e, c0s:c0e]
        patch1 = result[1][0, r1s:r1e, c1s:c1e]

        std0 = patch0.std()
        std1 = patch1.std()
        assert abs(std0 - std1) < 1.5, f"标准差差 {abs(std0-std1):.4f} > 1.5"

    def test_control_image_unchanged(self):
        """控制影像（补偿为 0）不应发生显著变化"""
        imgs, nd, ov = _make_simple_arrays()
        result, _, _ = bagrn_normalize(imgs, nd, ov, control_idx=0)

        diff = np.abs(result[0] - imgs[0])
        # 控制影像均值补偿为 0，但 moment matching 中若 ω≠1 仍可能有变化
        # 控制影像补偿为 0 意味着 μ' = μ，σ' = σ，所以 ω=1, υ=0
        # 则应完全不变
        assert diff.max() < 1e-5, f"控制影像变化 max={diff.max():.6f}"

    def test_compensation_shapes(self):
        """theta_mu 和 theta_sigma 形状正确"""
        imgs, nd, ov = _make_simple_arrays()
        _, tm, ts = bagrn_normalize(imgs, nd, ov, control_idx=0)
        n_bands = imgs[0].shape[0]
        n_imgs = len(imgs)
        assert tm.shape == (n_bands, n_imgs)
        assert ts.shape == (n_bands, n_imgs)


class TestBagrnNodata:
    """Nodata 处理测试"""

    def test_nodata_preserved(self):
        """Nodata 像素在归一化后值不变"""
        imgs, nd, ov = _make_arrays_with_nodata()
        result, _, _ = bagrn_normalize(imgs, nd, ov, control_idx=0)

        # img_0 的 (0,0) 是 nodata
        assert result[0][0, 0, 0] == -9999, "nodata 被修改"

    def test_nodata_excluded_from_stats(self):
        """含有 nodata 时仍能得到合理结果（阈值放宽，小样本随机波动）"""
        imgs, nd, ov = _make_arrays_with_nodata()
        result, _, _ = bagrn_normalize(imgs, nd, ov, control_idx=0)

        r0s, r0e, c0s, c0e = ov[0]["window_i"]
        r1s, r1e, c1s, c1e = ov[0]["window_j"]

        patch0 = result[0][0, r0s:r0e, c0s:c0e]
        patch1 = result[1][0, r1s:r1e, c1s:c1e]

        mean0 = patch0.mean()
        mean1 = patch1.mean()
        assert abs(mean0 - mean1) < 3.5, f"nodata 场景均值差 {abs(mean0-mean1):.4f}"


class TestBagrnSingleBand:
    """单波段影像测试（退化情况）"""

    def test_two_images_one_overlap(self):
        """只有两张图、一组重叠的最简情况"""
        np.random.seed(7)
        arr0 = np.array([[[50, 51, 52, 53],
                          [50, 51, 52, 53],
                          [50, 51, 52, 53],
                          [50, 51, 52, 53]]], dtype=np.float32)
        arr1 = np.array([[[70, 71, 72, 73],
                          [70, 71, 72, 73],
                          [70, 71, 72, 73],
                          [70, 71, 72, 73]]], dtype=np.float32)

        imgs = [arr0, arr1]
        nd = [None, None]
        # 两图完全重叠
        ov = [{"idx_i": 0, "idx_j": 1,
               "window_i": (0, 4, 0, 4),
               "window_j": (0, 4, 0, 4)}]

        result, tm, ts = bagrn_normalize(imgs, nd, ov, control_idx=0)

        # 控制图补偿为 0，不应变化
        assert np.allclose(result[0], arr0)

        # img_1 应与 img_0 在重叠区一致
        diff = np.abs(result[1] - arr0)
        assert diff.mean() < 2.0


class TestBagrnMultiBand:
    """多波段影像测试"""

    def test_three_band_rgb(self):
        """3 波段 RGB 影像"""
        np.random.seed(1)
        rows = cols = 4

        arr0 = 100 + 10 * np.random.randn(3, rows, cols)
        arr1 = 130 + 18 * np.random.randn(3, rows, cols)

        imgs = [arr0.astype(np.float32), arr1.astype(np.float32)]
        nd = [None, None]
        ov = [{"idx_i": 0, "idx_j": 1,
               "window_i": (0, 4, 0, 4),
               "window_j": (0, 4, 0, 4)}]

        result, tm, ts = bagrn_normalize(imgs, nd, ov, control_idx=0)

        assert tm.shape == (3, 2)
        assert ts.shape == (3, 2)

        # 每个波段的重叠区均值一致
        for b in range(3):
            diff_m = abs(result[0][b].mean() - result[1][b].mean())
            assert diff_m < 1.5, f"波段 {b} 均值差 {diff_m:.4f}"
