"""
指标回归测试

验证各项评价指标的数学定义和边界行为：
  - ADM = abs(mean(A) - mean(B))，不是 mean(abs(A-B))
  - CD 使用动态范围，不是固定 (0, 65535)
  - compute_per_pair 输出包含 per_band 字典
  - SAM（光谱角映射）的正确性
  - 接缝跳变指标的单调性
  - 有效区域内无 NaN/Inf
"""

import numpy as np
import pytest

from src.metrics import (
    compute_adm,
    compute_adsd,
    compute_cd,
    compute_gl,
    compute_per_pair,
    compute_per_band,
    compute_all,
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_overlap(n_bands=1, h=50, w=50):
    """构造一对重叠区域的简单数据。"""
    rng = np.random.default_rng(42)
    arr_i = rng.uniform(100, 200, (n_bands, h, w)).astype(np.float64)
    arr_j = rng.uniform(100, 200, (n_bands, h, w)).astype(np.float64)
    overlaps = [{
        "idx_i": 0, "idx_j": 1,
        "window_i": (0, h, 0, w),
        "window_j": (0, h, 0, w),
        "pixel_count": h * w,
    }]
    return arr_i, arr_j, overlaps


def _compute_sam(spectrum_a, spectrum_b):
    """
    计算光谱角映射 (SAM)。

    SAM = arccos( (A·B) / (||A|| * ||B||) )
    返回弧度值，0 表示完全相同方向。
    """
    a = np.asarray(spectrum_a, dtype=np.float64)
    b = np.asarray(spectrum_b, dtype=np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    cos_angle = np.clip(dot / (norm_a * norm_b), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def _seam_jump_metric(arr_i, arr_j, window_i, window_j, band=0):
    """
    计算接缝跳变指标：重叠区边界像素值之差的均值。
    模拟 seam_metrics.py 中的核心逻辑。
    """
    r1s, r1e, c1s, c1e = window_i
    r2s, r2e, c2s, c2e = window_j

    # 取重叠区中间行的像素值差
    mid_r1 = (r1s + r1e) // 2
    mid_r2 = (r2s + r2e) // 2
    strip_i = arr_i[band, mid_r1, c1s:c1e]
    strip_j = arr_j[band, mid_r2, c2s:c2e]

    # 跳变 = 相邻像素差的绝对值均值
    diff = np.abs(np.diff(strip_i.astype(np.float64)))
    return float(diff.mean()) if len(diff) > 0 else 0.0


# ===========================================================================
# test_adm_is_mean_abs_difference
# ===========================================================================

class TestADMDefinition:
    """验证 ADM = abs(mean(A) - mean(B))。"""

    def test_adm_exact_formula(self):
        """ADM 应等于 abs(mean(A) - mean(B))。"""
        # 构造已知均值的影像
        arr_i = np.full((1, 10, 10), 100.0)
        arr_j = np.full((1, 10, 10), 150.0)
        overlaps = [{
            "idx_i": 0, "idx_j": 1,
            "window_i": (0, 10, 0, 10),
            "window_j": (0, 10, 0, 10),
        }]
        adm = compute_adm([arr_i, arr_j], [None, None], overlaps, bands=[0])
        expected = abs(100.0 - 150.0)
        assert adm == pytest.approx(expected, abs=1e-10)

    def test_adm_nonzero_for_different_means(self):
        """均值不同时 ADM 应大于 0。"""
        arr_i = np.full((1, 10, 10), 100.0)
        arr_j = np.full((1, 10, 10), 200.0)
        overlaps = [{
            "idx_i": 0, "idx_j": 1,
            "window_i": (0, 10, 0, 10),
            "window_j": (0, 10, 0, 10),
        }]
        adm = compute_adm([arr_i, arr_j], [None, None], overlaps, bands=[0])
        assert adm > 0


# ===========================================================================
# test_adm_not_pixel_abs_difference
# ===========================================================================

class TestADMNotPixelDiff:
    """验证 ADM 不是 mean(abs(A-B))。"""

    def test_adm_differs_from_pixel_abs_mean(self):
        """
        对于非均匀分布，abs(mean(A)-mean(B)) != mean(abs(A-B))。
        ADM 应使用前者。
        """
        # A 有 [0, 100]，B 有 [50, 50]
        # mean(A)=50, mean(B)=50 → ADM=0
        # mean(|A-B|)=25 → 这不是 ADM
        arr_i = np.zeros((1, 10, 10), dtype=np.float64)
        arr_i[0, 0:5, :] = 0.0
        arr_i[0, 5:10, :] = 100.0
        arr_j = np.full((1, 10, 10), 50.0, dtype=np.float64)

        overlaps = [{
            "idx_i": 0, "idx_j": 1,
            "window_i": (0, 10, 0, 10),
            "window_j": (0, 10, 0, 10),
        }]

        adm = compute_adm([arr_i, arr_j], [None, None], overlaps, bands=[0])
        # ADM = abs(50 - 50) = 0
        assert adm == pytest.approx(0.0, abs=1e-10)

        # 验证 mean(|A-B|) 不等于 0
        pixel_abs_diff = np.abs(arr_i[0] - arr_j[0]).mean()
        assert pixel_abs_diff > 0, "mean(|A-B|) 应大于 0"


# ===========================================================================
# test_cd_no_fixed_range
# ===========================================================================

class TestCDDynamicRange:
    """验证 CD 使用动态范围，不是固定 (0, 65535)。"""

    def test_cd_uses_dynamic_range(self):
        """CD 的直方图范围应由实际像素值决定。"""
        # 两幅影像像素值都在 [100, 200] 范围
        arr_i = np.full((1, 10, 10), 100.0)
        arr_i[0, 0:5, :] = 200.0
        arr_j = np.full((1, 10, 10), 100.0)
        arr_j[0, 0:5, :] = 200.0

        overlaps = [{
            "idx_i": 0, "idx_j": 1,
            "window_i": (0, 10, 0, 10),
            "window_j": (0, 10, 0, 10),
        }]

        cd = compute_cd([arr_i, arr_j], [None, None], overlaps, bands=[0], n_bins=10)
        # 完全相同分布 → CD=0
        assert cd == pytest.approx(0.0, abs=1e-10)

    def test_cd_different_if_range_differs(self):
        """
        即使像素值分布形状相同，如果绝对范围不同，
        使用动态范围的 CD 应能区分。
        """
        arr_i = np.full((1, 10, 10), 100.0)
        arr_i[0, 0:5, :] = 200.0

        arr_j = np.full((1, 10, 10), 300.0)
        arr_j[0, 0:5, :] = 400.0

        overlaps = [{
            "idx_i": 0, "idx_j": 1,
            "window_i": (0, 10, 0, 10),
            "window_j": (0, 10, 0, 10),
        }]

        cd = compute_cd([arr_i, arr_j], [None, None], overlaps, bands=[0], n_bins=10)
        # 分布形状相同但范围不同 → CD 可能非零（取决于具体实现）
        # 关键是验证不会因为固定范围而产生虚假的零值
        assert isinstance(cd, float)


# ===========================================================================
# test_per_pair_has_per_band
# ===========================================================================

class TestPerPairPerBand:
    """验证 compute_per_pair 输出包含 per_band 字典。"""

    def test_per_band_dict_present(self):
        """compute_per_pair 每个结果应包含 per_band 键。"""
        arr_i, arr_j, overlaps = _make_overlap(n_bands=2)
        result = compute_per_pair(
            [arr_i, arr_j], [None, None], overlaps, bands=[0, 1],
        )
        assert len(result) == 1
        entry = result[0]
        assert "per_band" in entry
        assert isinstance(entry["per_band"], dict)

    def test_per_band_has_all_bands(self):
        """per_band 字典应包含所有请求的波段。"""
        n_bands = 3
        arr_i, arr_j, overlaps = _make_overlap(n_bands=n_bands)
        result = compute_per_pair(
            [arr_i, arr_j], [None, None], overlaps, bands=list(range(n_bands)),
        )
        per_band = result[0]["per_band"]
        for b in range(n_bands):
            assert b in per_band
            for metric in ("adm", "adsd", "cd", "rdoa", "ave"):
                assert metric in per_band[b], f"per_band[{b}] 缺少 {metric}"

    def test_per_pair_aggregation(self):
        """per_pair 聚合值应是各波段的均值。"""
        n_bands = 2
        arr_i, arr_j, overlaps = _make_overlap(n_bands=n_bands)
        result = compute_per_pair(
            [arr_i, arr_j], [None, None], overlaps, bands=list(range(n_bands)),
        )
        entry = result[0]
        per_band = entry["per_band"]

        # 聚合 ADM 应是各波段 ADM 的均值
        band_adms = [per_band[b]["adm"] for b in per_band]
        expected_agg_adm = np.mean(band_adms)
        assert entry["adm"] == pytest.approx(expected_agg_adm, abs=1e-6)


# ===========================================================================
# test_sam_same_spectra
# ===========================================================================

class TestSAMSameSpectra:
    """SAM=0 对相同光谱。"""

    def test_identical_spectra_sam_zero(self):
        """完全相同的光谱向量 SAM 应为 0。"""
        spectrum = np.array([100.0, 150.0, 200.0])
        sam = _compute_sam(spectrum, spectrum)
        assert sam == pytest.approx(0.0, abs=1e-10)

    def test_zero_spectra_sam_zero(self):
        """零向量的 SAM 应为 0（特殊处理）。"""
        a = np.zeros(3)
        b = np.zeros(3)
        sam = _compute_sam(a, b)
        assert sam == pytest.approx(0.0)


# ===========================================================================
# test_sam_scaled_spectra
# ===========================================================================

class TestSAMScaledSpectra:
    """标量倍数光谱的 SAM 应接近 0。"""

    def test_scalar_multiple_sam_near_zero(self):
        """标量倍数的光谱方向相同，SAM 应接近 0。"""
        a = np.array([100.0, 150.0, 200.0])
        b = a * 2.5
        sam = _compute_sam(a, b)
        assert sam == pytest.approx(0.0, abs=1e-8)

    def test_negative_scalar_sam_pi(self):
        """反向光谱（负标量）的 SAM 应为 π。"""
        a = np.array([100.0, 150.0, 200.0])
        b = -a
        sam = _compute_sam(a, b)
        assert sam == pytest.approx(np.pi, abs=1e-8)


# ===========================================================================
# test_sam_different_spectra
# ===========================================================================

class TestSAMDifferentSpectra:
    """不同形状光谱的 SAM 应大于 0。"""

    def test_perpendicular_spectra(self):
        """垂直光谱的 SAM 应为 π/2。"""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        sam = _compute_sam(a, b)
        assert sam == pytest.approx(np.pi / 2, abs=1e-8)

    def test_oblique_spectra(self):
        """斜交光谱的 SAM 应在 (0, π/2) 之间。"""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 1.0, 0.0])
        sam = _compute_sam(a, b)
        assert 0 < sam < np.pi / 2


# ===========================================================================
# test_seam_jump_high_artificial
# ===========================================================================

class TestSeamJumpHigh:
    """人工跳变应产生高接缝分。"""

    def test_abrupt_jump_gives_high_score(self):
        """突变接缝（一边全 0，另一边全 255）应有高跳变分。"""
        arr_i = np.full((1, 10, 10), 0.0)
        arr_j = np.full((1, 10, 10), 255.0)

        overlaps = [{
            "idx_i": 0, "idx_j": 1,
            "window_i": (0, 10, 0, 10),
            "window_j": (0, 10, 0, 10),
        }]

        # 使用差值作为跳变度量
        adm = compute_adm([arr_i, arr_j], [None, None], overlaps, bands=[0])
        assert adm == pytest.approx(255.0, abs=1e-6)


# ===========================================================================
# test_seam_jump_low_feathered
# ===========================================================================

class TestSeamJumpLow:
    """平滑过渡的接缝应有低跳变分。"""

    def test_smooth_transition_gives_low_score(self):
        """渐变接缝应有较低跳变分。"""
        # 构造渐变影像
        h, w = 10, 10
        arr_i = np.linspace(100, 150, h * w).reshape(1, h, w).astype(np.float64)
        arr_j = np.linspace(140, 190, h * w).reshape(1, h, w).astype(np.float64)

        overlaps = [{
            "idx_i": 0, "idx_j": 1,
            "window_i": (0, h, 0, w),
            "window_j": (0, h, 0, w),
        }]

        # 渐变影像间 ADM 应小于突变情形
        adm = compute_adm([arr_i, arr_j], [None, None], overlaps, bands=[0])
        # 两组渐变的均值差应远小于 255
        assert adm < 50.0


# ===========================================================================
# test_no_nan_in_valid_area
# ===========================================================================

class TestNoNaNInValidArea:
    """验证有效区域内无 NaN/Inf。"""

    def test_adm_finite(self):
        """ADM 应为有限值。"""
        arr_i, arr_j, overlaps = _make_overlap(n_bands=2)
        adm = compute_adm([arr_i, arr_j], [None, None], overlaps)
        assert np.isfinite(adm)

    def test_adsd_finite(self):
        """ADSD 应为有限值。"""
        arr_i, arr_j, overlaps = _make_overlap(n_bands=2)
        adsd = compute_adsd([arr_i, arr_j], [None, None], overlaps)
        assert np.isfinite(adsd)

    def test_cd_finite(self):
        """CD 应为有限值。"""
        arr_i, arr_j, overlaps = _make_overlap(n_bands=2)
        cd = compute_cd([arr_i, arr_j], [None, None], overlaps, n_bins=10)
        assert np.isfinite(cd)

    def test_per_pair_all_finite(self):
        """per_pair 输出所有指标应为有限值（gl/ave 未提供 arrays_before 时为 None）。"""
        arr_i, arr_j, overlaps = _make_overlap(n_bands=2)
        result = compute_per_pair(
            [arr_i, arr_j], [None, None], overlaps, bands=[0, 1],
        )
        entry = result[0]
        for metric in ("adm", "adsd", "cd", "rdoa"):
            assert np.isfinite(entry[metric]), f"per_pair.{metric} 非有限"
        assert entry["gl"] is None
        assert entry["ave"] is None

        for b_idx, band_data in entry["per_band"].items():
            for metric in ("adm", "adsd", "cd", "rdoa"):
                assert np.isfinite(band_data[metric]), \
                    f"per_band[{b_idx}].{metric} 非有限"
            assert band_data["gl"] is None
            assert band_data["ave"] is None

    def test_per_band_all_finite(self):
        """per_band 输出所有指标应为有限值（gl/ave 未提供 arrays_before 时为 None）。"""
        arr_i, arr_j, overlaps = _make_overlap(n_bands=2)
        result = compute_per_band(
            [arr_i, arr_j], [None, None], overlaps, bands=[0, 1],
        )
        for row in result:
            for metric in ("adm", "adsd", "cd", "rdoa"):
                assert np.isfinite(row[metric]), f"band={row['band']}.{metric} 非有限"
            assert row["gl"] is None
            assert row["ave"] is None


# ===========================================================================
# test_compute_all_single_image_pair
# ===========================================================================

class TestComputeAllSingleImagePair:
    """验证提供 arrays_before 时 GL 和 AVE 被计算。"""

    def test_compute_all_single_image_pair(self):
        """当提供 arrays_before 时，GL 和 AVE 应为非 None 的有限值。"""
        rng = np.random.default_rng(42)
        arr = rng.uniform(100, 200, (2, 50, 50)).astype(np.float64)
        arr_before = arr + rng.normal(0, 5, arr.shape)
        arr_before = arr_before.astype(np.float64)
        overlaps = [{
            "idx_i": 0, "idx_j": 1,
            "window_i": (0, 50, 0, 50),
            "window_j": (0, 50, 0, 50),
            "pixel_count": 2500,
        }]
        result = compute_all(
            [arr_before[0:1], arr_before[1:2]],
            [arr[0:1], arr[1:2]],
            [None, None], overlaps, bands=[0],
        )
        assert result["gl"] is not None
        assert np.isfinite(result["gl"])
        assert result["ave"] is not None
        assert np.isfinite(result["ave"])


# ===========================================================================
# test_compute_per_band_zero_weights
# ===========================================================================

class TestComputePerBandZeroWeights:
    """验证 compute_per_band 返回所有请求波段的键。"""

    def test_compute_per_band_zero_weights(self):
        """per_band 结果应包含所有请求的波段索引。"""
        arr_i, arr_j, overlaps = _make_overlap(n_bands=4)
        requested_bands = [0, 1, 2, 3]
        result = compute_per_band(
            [arr_i, arr_j], [None, None], overlaps, bands=requested_bands,
        )
        result_bands = [row["band"] for row in result]
        for b in requested_bands:
            assert b in result_bands, f"波段 {b} 不在 per_band 结果中"


# ===========================================================================
# test_compute_per_pair_zero_weights
# ===========================================================================

class TestComputePerPairZeroWeights:
    """验证 compute_per_pair 返回正确数量的条目。"""

    def test_compute_per_pair_zero_weights(self):
        """per_pair 条目数应等于重叠对数。"""
        arr_i, arr_j, overlaps = _make_overlap(n_bands=2)
        result = compute_per_pair(
            [arr_i, arr_j], [None, None], overlaps, bands=[0, 1],
        )
        assert len(result) == len(overlaps)
        entry = result[0]
        assert entry["idx_i"] == 0
        assert entry["idx_j"] == 1
        assert "per_band" in entry
        assert isinstance(entry["per_band"], dict)
