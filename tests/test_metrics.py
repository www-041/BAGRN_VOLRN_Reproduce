"""
测试 src/metrics.py 中的各项评价指标。
"""

import numpy as np
import pytest
from src.metrics import (
    compute_adm,
    compute_adsd,
    compute_cd,
    compute_gl,
    compute_rdoa,
    compute_ave,
    compute_all,
)


# ---------------------------------------------------------------------------
# 辅助：生成测试用的虚拟影像和 overlap 信息
# ---------------------------------------------------------------------------

def _make_test_images():
    """
    创建 3 幅影像（2 波段 × 20×20）及虚拟 overlap。
    """
    rng = np.random.default_rng(42)
    arrays = [rng.normal(100, 10, (2, 20, 20)).astype(np.float32) for _ in range(3)]
    nodata = [None, None, None]

    # 设定重叠区使均值/标准差可预测
    # Image 0 vs Image 1：左上角重叠
    arrays[0][0, 0:10, 0:10] = 110.0  # band 0 重叠区
    arrays[0][1, 0:10, 0:10] = 50.0   # band 1 重叠区
    arrays[1][0, 0:10, 0:10] = 120.0
    arrays[1][1, 0:10, 0:10] = 60.0

    # Image 1 vs Image 2：右下角重叠
    arrays[1][0, 10:20, 10:20] = 90.0
    arrays[1][1, 10:20, 10:20] = 40.0
    arrays[2][0, 10:20, 10:20] = 100.0
    arrays[2][1, 10:20, 10:20] = 30.0

    overlaps = [
        {"idx_i": 0, "idx_j": 1, "window_i": (0, 10, 0, 10), "window_j": (0, 10, 0, 10)},
        {"idx_i": 1, "idx_j": 2, "window_i": (10, 20, 10, 20), "window_j": (10, 20, 10, 20)},
    ]

    # 被归一化后的影像（模拟：简单调亮）
    arrays_after = [a * 1.05 for a in arrays]

    return arrays, arrays_after, nodata, overlaps


# ---------------------------------------------------------------------------
# TestADM
# ---------------------------------------------------------------------------

class TestADM:
    def test_adm_identical_images(self):
        arr = np.ones((1, 10, 10))
        overlaps = [{"idx_i": 0, "idx_j": 1, "window_i": (0, 5, 0, 5), "window_j": (0, 5, 0, 5)}]
        val = compute_adm([arr, arr], [None, None], overlaps, bands=[0])
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_adm_known_difference(self):
        a = np.zeros((1, 10, 10))
        b = np.ones((1, 10, 10)) * 5.0
        overlaps = [{"idx_i": 0, "idx_j": 1, "window_i": (0, 5, 0, 5), "window_j": (0, 5, 0, 5)}]
        val = compute_adm([a, b], [None, None], overlaps, bands=[0])
        assert val == pytest.approx(5.0, abs=1e-6)

    def test_adm_multiband_average(self):
        a = np.zeros((2, 10, 10))
        b = np.zeros((2, 10, 10))
        a[0, 0:5, 0:5] = 10.0
        b[0, 0:5, 0:5] = 20.0
        a[1, 0:5, 0:5] = 30.0
        b[1, 0:5, 0:5] = 50.0
        overlaps = [{"idx_i": 0, "idx_j": 1, "window_i": (0, 5, 0, 5), "window_j": (0, 5, 0, 5)}]
        val = compute_adm([a, b], [None, None], overlaps)
        # band0: |10-20|=10, band1: |30-50|=20 → avg=15
        assert val == pytest.approx(15.0, abs=1e-6)

    def test_adm_no_overlaps_returns_zero(self):
        a = np.ones((1, 10, 10))
        val = compute_adm([a, a], [None, None], [])
        assert val == pytest.approx(0.0)

    def test_adm_nodata_excluded(self):
        a = np.ones((1, 10, 10))
        b = np.ones((1, 10, 10))
        a[0, 0:5, 0:5] = 999.0  # nodata
        b[0, 0:5, 0:5] = 999.0
        overlaps = [{"idx_i": 0, "idx_j": 1, "window_i": (0, 5, 0, 5), "window_j": (0, 5, 0, 5)}]
        val = compute_adm([a, b], [None, None], overlaps, bands=[0])
        # 排除 nodata 后重叠区剩下 0 像素 → total_diff/n_pairs 中该对贡献 0
        assert val == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# TestADSD
# ---------------------------------------------------------------------------

class TestADSD:
    def test_adsd_identical_images(self):
        arr = np.ones((1, 10, 10))
        overlaps = [{"idx_i": 0, "idx_j": 1, "window_i": (0, 5, 0, 5), "window_j": (0, 5, 0, 5)}]
        val = compute_adsd([arr, arr], [None, None], overlaps, bands=[0])
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_adsd_known_std_diff(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 5, (1, 20, 20)).astype(np.float32)
        b = rng.normal(100, 15, (1, 20, 20)).astype(np.float32)
        overlaps = [{"idx_i": 0, "idx_j": 1, "window_i": (0, 20, 0, 20), "window_j": (0, 20, 0, 20)}]

        val = compute_adsd([a, b], [None, None], overlaps, bands=[0])
        diff = abs(a[0].std() - b[0].std())
        assert val == pytest.approx(diff, abs=1e-4)


# ---------------------------------------------------------------------------
# TestCD
# ---------------------------------------------------------------------------

class TestCD:
    def test_cd_identical_images(self):
        arr = np.ones((1, 10, 10)) * 50.0
        overlaps = [{"idx_i": 0, "idx_j": 1, "window_i": (0, 5, 0, 5), "window_j": (0, 5, 0, 5)}]
        val = compute_cd([arr, arr], [None, None], overlaps, bands=[0], n_bins=10)
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_cd_different_histograms(self):
        a = np.zeros((1, 10, 10), dtype=np.float32)
        b = np.ones((1, 10, 10), dtype=np.float32)
        a[0, 0:5, 0:5] = 0.0
        b[0, 0:5, 0:5] = 255.0
        overlaps = [{"idx_i": 0, "idx_j": 1, "window_i": (0, 5, 0, 5), "window_j": (0, 5, 0, 5)}]
        val = compute_cd([a, b], [None, None], overlaps, bands=[0], n_bins=5)
        # 两个分布完全不重叠 → ΔH 较大且 > 0
        assert val > 0.0

    def test_cd_no_overlaps_returns_zero(self):
        a = np.ones((1, 5, 5))
        val = compute_cd([a, a], [None, None], [])
        assert val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestGL
# ---------------------------------------------------------------------------

class TestGL:
    def test_gl_identical_images(self):
        arr = np.ones((1, 10, 10))
        val = compute_gl([arr], [arr], [None], bands=[0])
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_gl_slightly_modified(self):
        rng = np.random.default_rng(1)
        before = rng.normal(100, 20, (1, 30, 30)).astype(np.float32)
        after = before + rng.normal(0, 1, (1, 30, 30)).astype(np.float32)  # 小噪声
        val = compute_gl([before], [after], [None], bands=[0])
        # 梯度方向应接近，GL 值应较小
        assert 0.0 <= val < 0.5

    def test_gl_nodata_excluded(self):
        before = np.ones((1, 10, 10))
        after = np.ones((1, 10, 10)) * 2.0
        before[0, 0:5, 0:5] = -999.0
        after[0, 0:5, 0:5] = -999.0
        val = compute_gl([before], [after], [-999.0], bands=[0])
        # 非 nodata 区域完全一致
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_gl_multiband(self):
        a = np.ones((2, 10, 10))
        val = compute_gl([a], [a], [None], bands=[0, 1])
        assert val == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Test RDOA / Ave
# ---------------------------------------------------------------------------

class TestRDOA:
    def test_rdoa_formula(self):
        assert compute_rdoa(1, 2, 3) == pytest.approx(2.0)


class TestAve:
    def test_ave_formula(self):
        assert compute_ave(1, 2, 3, 4) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Test compute_all 集成
# ---------------------------------------------------------------------------

class TestComputeAll:
    def test_all_metrics_return_same_shapes(self):
        arrays, arrays_after, nodata, overlaps = _make_test_images()
        result = compute_all(arrays, arrays_after, nodata, overlaps)
        for key in ("adm", "adsd", "cd", "gl", "rdoa", "ave"):
            assert key in result, f"缺少指标 {key}"
            assert isinstance(result[key], float), f"{key} 类型为 {type(result[key])}"

    def test_identical_before_after(self):
        a = [np.ones((1, 10, 10))]
        ov = [{"idx_i": 0, "idx_j": 1 if len(a) > 1 else 0, "window_i": (0, 5, 0, 5), "window_j": (0, 5, 0, 5)}]
        result = compute_all(a + a, a + a, [None, None], ov)
        assert result["adm"] == pytest.approx(0.0, abs=1e-10)
        assert result["adsd"] == pytest.approx(0.0, abs=1e-10)
        assert result["cd"] == pytest.approx(0.0, abs=1e-10)
        assert result["gl"] == pytest.approx(0.0, abs=1e-10)
        assert result["rdoa"] == pytest.approx(0.0, abs=1e-10)
        assert result["ave"] == pytest.approx(0.0, abs=1e-10)
