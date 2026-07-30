"""
对比方法测试
"""

import numpy as np
from src.comparison import (
    histogram_matching_normalize,
    moment_matching_normalize,
    wallis_normalize,
    run_comparison,
)


def _make_test_data():
    """生成两幅简单重叠影像用于测试。"""
    arrays = []
    arr1 = np.ones((1, 20, 20), dtype=np.float32) * 100
    arr2 = np.ones((1, 20, 20), dtype=np.float32) * 200
    arrays = [arr1, arr2]
    nodata = [None, None]
    overlaps = [{
        "idx_i": 0, "idx_j": 1,
        "window_i": (0, 20, 0, 20),
        "window_j": (0, 20, 0, 20),
    }]
    return arrays, nodata, overlaps


def test_histogram_matching_output_shape():
    """直方图匹配输出形状应与输入一致。"""
    arrays, nodata, overlaps = _make_test_data()
    result = histogram_matching_normalize(arrays, nodata, overlaps)
    assert len(result) == len(arrays)
    assert result[0].shape == arrays[0].shape


def test_histogram_matching_control_unchanged():
    """控制影像应保持不变（或变化极小）。"""
    arrays, nodata, overlaps = _make_test_data()
    result = histogram_matching_normalize(arrays, nodata, overlaps, control_idx=0)
    assert np.allclose(result[0], arrays[0], atol=1e-5)


def test_moment_matching_output_shape():
    """矩匹配输出形状应与输入一致。"""
    arrays, nodata, overlaps = _make_test_data()
    result = moment_matching_normalize(arrays, nodata, overlaps)
    assert len(result) == len(arrays)
    assert result[0].shape == arrays[0].shape


def test_moment_matching_identical_images():
    """若影像原本一致，矩匹配不应改变它们。"""
    arr = np.ones((1, 20, 20), dtype=np.float32) * 100
    arrays = [arr.copy(), arr.copy()]
    nodata = [None, None]
    overlaps = [{
        "idx_i": 0, "idx_j": 1,
        "window_i": (0, 20, 0, 20),
        "window_j": (0, 20, 0, 20),
    }]
    result = moment_matching_normalize(arrays, nodata, overlaps)
    assert np.allclose(result[0], result[1], atol=1e-5)


def test_moment_matching_control_unchanged():
    """矩匹配后控制影像应不变。"""
    arrays, nodata, overlaps = _make_test_data()
    result = moment_matching_normalize(arrays, nodata, overlaps, control_idx=0)
    assert np.allclose(result[0], arrays[0], atol=1e-5)


def test_wallis_output_shape():
    """Wallis 滤波输出形状应与输入一致。"""
    arrays, nodata, overlaps = _make_test_data()
    result = wallis_normalize(arrays, nodata, overlaps, window_size=11)
    assert len(result) == len(arrays)
    assert result[0].shape == arrays[0].shape


def test_run_comparison_histogram():
    """统一接口应能调用直方图匹配。"""
    arrays, nodata, overlaps = _make_test_data()
    result = run_comparison("histogram_matching", arrays, nodata, overlaps)
    assert len(result) == len(arrays)


def test_run_comparison_moment():
    """统一接口应能调用矩匹配。"""
    arrays, nodata, overlaps = _make_test_data()
    result = run_comparison("moment_matching", arrays, nodata, overlaps)
    assert len(result) == len(arrays)


def test_run_comparison_invalid():
    """统一接口应能处理未知方法名。"""
    arrays, nodata, overlaps = _make_test_data()
    try:
        run_comparison("invalid_method", arrays, nodata, overlaps)
        assert False, "应抛出 ValueError"
    except ValueError:
        assert True
