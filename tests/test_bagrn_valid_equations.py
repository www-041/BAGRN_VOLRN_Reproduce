"""Test BAGRN skips equations with no valid observations."""

import numpy as np
import pytest
from src.bagrn import bagrn_normalize


def test_bagrn_skips_no_valid_overlap():
    """
    当某对影像在重叠区无有效像素时，该方程应被跳过，
    而不是创建假的 θ_i = θ_j 约束。
    """
    # 创建 3 景影像，每景 1 波段，8x8 像素
    # 影像 0 和 1 有有效重叠
    # 影像 2 在重叠区全为 nodata
    img0 = np.ones((1, 8, 8), dtype=np.float64) * 100.0
    img1 = np.ones((1, 8, 8), dtype=np.float64) * 110.0  # 比 img0 高 10
    img2 = np.ones((1, 8, 8), dtype=np.float64) * 120.0
    
    # 影像 2 在右半部分设为 nodata
    img2[:, :, 4:] = -9999.0
    
    arrays = [img0, img1, img2]
    nodata_values = [None, None, -9999.0]
    
    # 定义重叠对
    # 对 0-1: 完全重叠
    # 对 1-2: 右半部分重叠，但 img2 在该区域全为 nodata
    overlaps = [
        {"idx_i": 0, "idx_j": 1, "window_i": (0, 8, 0, 8), "window_j": (0, 8, 0, 8)},
        {"idx_i": 1, "idx_j": 2, "window_i": (0, 8, 4, 8), "window_j": (0, 8, 4, 8)},
    ]
    
    # 运行 BAGRN
    result, theta_mu, theta_sigma = bagrn_normalize(
        arrays, nodata_values, overlaps, control_idx=0
    )
    
    # 验证结果
    # 影像 0 是控制影像，补偿应为 0
    assert theta_mu[0, 0] == pytest.approx(0.0, abs=1e-10)
    
    # 影像 1 应该被校正（与影像 0 有有效重叠）
    # 由于 img1 比 img0 高 10，补偿应约为 -10
    assert theta_mu[0, 1] < 0
    
    # 影像 2 与影像 1 的重叠区无有效像素
    # 该方程应被跳过，影像 2 的补偿应为 0（或接近 0）
    # 因为没有任何有效约束来校正它
    assert theta_mu[0, 2] == pytest.approx(0.0, abs=1e-6)


def test_bagrn_all_valid_overlaps():
    """
    当所有重叠对都有有效像素时，应正常求解。
    """
    img0 = np.ones((1, 8, 8), dtype=np.float64) * 100.0
    img1 = np.ones((1, 8, 8), dtype=np.float64) * 110.0
    img2 = np.ones((1, 8, 8), dtype=np.float64) * 120.0
    
    arrays = [img0, img1, img2]
    nodata_values = [None, None, None]
    
    overlaps = [
        {"idx_i": 0, "idx_j": 1, "window_i": (0, 8, 0, 8), "window_j": (0, 8, 0, 8)},
        {"idx_i": 1, "idx_j": 2, "window_i": (0, 8, 0, 8), "window_j": (0, 8, 0, 8)},
    ]
    
    result, theta_mu, theta_sigma = bagrn_normalize(
        arrays, nodata_values, overlaps, control_idx=0
    )
    
    # 所有影像都应有非零补偿（除了控制影像）
    assert theta_mu[0, 0] == pytest.approx(0.0, abs=1e-10)
    assert theta_mu[0, 1] != 0.0
    assert theta_mu[0, 2] != 0.0
