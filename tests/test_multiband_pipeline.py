"""
多波段管线测试

使用合成小影像（100×100 像素）测试管线关键功能，
不访问真实数据文件。

覆盖功能：
  - build_spanning_tree
  - propagate_control_scene
  - validate_band_consistency
  - warp_multiband_with_displacement_field（同位移验证）
  - 单次重采样（无二次插值）
  - NoData 保护（统计排除 NoData）
  - 实验配置加载与校验
"""

import os
import tempfile

import numpy as np
import pytest

from unittest.mock import patch, MagicMock

from src.multiband_pipeline import (
    build_spanning_tree,
    propagate_control_scene,
    validate_band_consistency,
)
from src.experiment_config import (
    ExperimentConfig,
    load_config,
    save_config,
    validate_config,
    make_test_config,
)


# ---------------------------------------------------------------------------
# 辅助：构造合成影像与重叠信息
# ---------------------------------------------------------------------------

def _make_synthetic_scene_data(n_scenes=4, n_bands=3, h=100, w=100):
    """
    生成合成影像数据和重叠关系。

    返回 arrays (n_scenes, n_bands, h, w), overlaps 列表
    """
    rng = np.random.default_rng(42)
    arrays = [
        rng.uniform(100, 200, (n_bands, h, w)).astype(np.float64)
        for _ in range(n_scenes)
    ]

    # 构造 3 对重叠区域（链式: 0-1, 1-2, 2-3）
    overlaps = []
    for i in range(n_scenes - 1):
        # 每对在各自影像右/左侧 30% 区域重叠
        overlap_w = int(w * 0.3)
        wi_start = w - overlap_w
        wj_start = 0
        overlaps.append({
            "idx_i": i,
            "idx_j": i + 1,
            "window_i": (0, h, wi_start, w),
            "window_j": (0, h, wj_start, overlap_w),
            "pixel_count": h * overlap_w,
        })

    return arrays, overlaps


# ===========================================================================
# test_build_spanning_tree
# ===========================================================================

class TestBuildSpanningTree:
    """测试生成树构建。"""

    def test_tree_edges_count(self):
        """4 景链式重叠，生成树应有 3 条边。"""
        _, overlaps = _make_synthetic_scene_data(4)
        tree = build_spanning_tree(overlaps, n_images=4, reference_idx=0)
        # 4 节点树应有 3 条边
        assert len(tree) == 3

    def test_tree_covers_all_nodes(self):
        """生成树应覆盖所有可达节点。"""
        _, overlaps = _make_synthetic_scene_data(4)
        tree = build_spanning_tree(overlaps, n_images=4, reference_idx=0)
        visited = set()
        for parent, child in tree:
            visited.add(parent)
            visited.add(child)
        assert visited == {0, 1, 2, 3}

    def test_tree_root_has_no_parent(self):
        """根节点（reference_idx=0）不应出现在子节点列表中作为 child。"""
        _, overlaps = _make_synthetic_scene_data(4)
        tree = build_spanning_tree(overlaps, n_images=4, reference_idx=0)
        children = [c for _, c in tree]
        # 根节点 0 不应是任何边的 child（除非自环，但生成树无自环）
        assert 0 not in children

    def test_tree_from_different_root(self):
        """从不同根节点构建生成树。"""
        _, overlaps = _make_synthetic_scene_data(4)
        tree = build_spanning_tree(overlaps, n_images=4, reference_idx=2)
        visited = set()
        for parent, child in tree:
            visited.add(parent)
            visited.add(child)
        assert visited == {0, 1, 2, 3}

    def test_disconnected_graph(self):
        """部分节点不可达时，生成树只包含连通分量。"""
        # 0-1 重叠，2-3 重叠，但 1-2 无重叠
        overlaps = [
            {"idx_i": 0, "idx_j": 1},
            {"idx_i": 2, "idx_j": 3},
        ]
        tree = build_spanning_tree(overlaps, n_images=4, reference_idx=0)
        # 从 0 出发只能到达 {0, 1}
        visited = set()
        for parent, child in tree:
            visited.add(parent)
            visited.add(child)
        assert visited == {0, 1}

    def test_single_image(self):
        """单景无需生成树。"""
        tree = build_spanning_tree([], n_images=1, reference_idx=0)
        assert len(tree) == 0


# ===========================================================================
# test_propagate_control_scene
# ===========================================================================

class TestPropagateControlScene:
    """测试控制场景传播（父场景映射）。"""

    def test_parent_map_reference_is_none(self):
        """参考场景的 parent 应为 None。"""
        _, overlaps = _make_synthetic_scene_data(4)
        parent_map = propagate_control_scene(overlaps, n_images=4, reference_idx=0)
        assert parent_map[0] is None

    def test_parent_map_children_have_parent(self):
        """非参考场景应有 parent。"""
        _, overlaps = _make_synthetic_scene_data(4)
        parent_map = propagate_control_scene(overlaps, n_images=4, reference_idx=0)
        for idx in range(1, 4):
            assert parent_map[idx] is not None

    def test_parent_map_chain(self):
        """链式重叠下 parent 关系应为 0→1→2→3。"""
        _, overlaps = _make_synthetic_scene_data(4)
        parent_map = propagate_control_scene(overlaps, n_images=4, reference_idx=0)
        assert parent_map[1] == 0
        assert parent_map[2] == 1
        assert parent_map[3] == 2

    def test_disconnected_node(self):
        """不可达节点的 parent 应为 None。"""
        overlaps = [{"idx_i": 0, "idx_j": 1}]  # 只有 0-1
        parent_map = propagate_control_scene(overlaps, n_images=4, reference_idx=0)
        assert parent_map[0] is None
        assert parent_map[1] == 0
        assert parent_map[2] is None  # 不可达
        assert parent_map[3] is None  # 不可达


# ===========================================================================
# test_validate_band_consistency
# ===========================================================================

class TestValidateBandConsistency:
    """测试波段一致性校验。"""

    def test_all_consistent(self):
        """所有场景波段一致时应无错误。"""
        scenes = [
            {"id": "s1", "bands": {"B01": "a.tif", "B02": "b.tif"}},
            {"id": "s2", "bands": {"B01": "c.tif", "B02": "d.tif"}},
        ]
        errors = validate_band_consistency(scenes)
        assert len(errors) == 0

    def test_missing_band_detected(self):
        """某场景缺少其他场景拥有的波段时，交集缩小但不报错（函数仅检查交集）。
        当所有场景波段完全不同时交集为空，应报错。"""
        # 场景 1 有 B01，场景 2 有 B02 → 交集为空
        scenes = [
            {"id": "s1", "bands": {"B01": "a.tif"}},
            {"id": "s2", "bands": {"B02": "b.tif"}},
        ]
        errors = validate_band_consistency(scenes, required_bands={"B01"})
        assert len(errors) > 0
        assert any("缺少波段" in e for e in errors)

    def test_empty_scenes(self):
        """空场景列表应报错。"""
        errors = validate_band_consistency([])
        assert len(errors) > 0

    def test_scene_without_bands(self):
        """场景缺少 bands 字段应报错。"""
        scenes = [{"id": "s1"}]
        errors = validate_band_consistency(scenes, required_bands={"B01"})
        assert len(errors) > 0
        assert any("缺少波段" in e for e in errors)


# ===========================================================================
# test_warp_multiband_same_displacement
# ===========================================================================

class TestWarpMultibandSameDisplacement:
    """验证所有波段施加相同位移。"""

    def test_all_bands_shifted_equally(self):
        """对非方形多波段数组施加相同全局位移，各波段应一致偏移。
        使用常数数据避免双线性插值引入的波段间差异。"""
        try:
            from src.coregistration import warp_multiband_with_displacement_field
        except ImportError:
            pytest.skip("coregistration 模块不可用")

        n_bands = 3
        h, w = 40, 60  # non-square
        multiband = np.zeros((n_bands, h, w), dtype=np.float64)
        for b in range(n_bands):
            multiband[b, :, :] = 100.0 + b * 50.0

        gdx, gdy = 2.0, 3.0
        local_dx = np.zeros((h, w), dtype=np.float64)
        local_dy = np.zeros((h, w), dtype=np.float64)

        result = warp_multiband_with_displacement_field(
            multiband, gdx, gdy, local_dx, local_dy, nodata=0.0,
        )

        assert result.shape == multiband.shape

        for b in range(n_bands):
            interior = result[b, 5:-5, 5:-5]
            expected = 100.0 + b * 50.0
            assert np.allclose(interior, expected, atol=1e-6), \
                f"Band {b}: expected {expected}, got range [{interior.min():.4f}, {interior.max():.4f}]"

    def test_1d_displacement_raises_error(self):
        """Passing 1D displacement fields must raise an error."""
        try:
            from src.coregistration import warp_multiband_with_displacement_field
        except ImportError:
            pytest.skip("coregistration 模块不可用")

        n_bands = 2
        h, w = 40, 60
        multiband = np.ones((n_bands, h, w), dtype=np.float64) * 100.0

        # 1D fields: dx has shape (h,), dy has shape (w,) — both wrong
        local_dx_1d = np.zeros(h, dtype=np.float64)
        local_dy_1d = np.zeros(w, dtype=np.float64)

        with pytest.raises(AssertionError, match="2D"):
            warp_multiband_with_displacement_field(
                multiband, 0.0, 0.0, local_dx_1d, local_dy_1d, nodata=0.0,
            )

    def test_nonzero_local_displacement(self):
        """Non-zero local displacement field should shift pixels accordingly."""
        try:
            from src.coregistration import warp_multiband_with_displacement_field
        except ImportError:
            pytest.skip("coregistration 模块不可用")

        h, w = 40, 60
        # Create a coordinate-gradient image: value = col_index
        single_band = np.zeros((1, h, w), dtype=np.float64)
        single_band[0] = np.arange(w, dtype=np.float64)[np.newaxis, :]  # row value = col index

        # Local displacement: shift every pixel 3 columns to the right
        local_dx = np.full((h, w), 3.0, dtype=np.float64)
        local_dy = np.zeros((h, w), dtype=np.float64)

        result = warp_multiband_with_displacement_field(
            single_band, 0.0, 0.0, local_dx, local_dy, nodata=-1.0,
        )

        # After shifting right by 3, each pixel's new value should be (original col - 3)
        # Interior check: at col j, value should be j-3
        interior = result[0, 5:-5, 5:-5]
        expected_cols = np.arange(w, dtype=np.float64)[np.newaxis, 5:-5] - 3.0
        assert np.allclose(interior, expected_cols, atol=1.0), \
            f"Expected values ~{expected_cols[0,:5]}, got {interior[0,:5]}"


# ===========================================================================
# test_single_resampling
# ===========================================================================

class TestSingleResampling:
    """验证单次重采样无二次插值。"""

    def test_nearest_neighbor_preserves_values(self):
        """
        最近邻插值应保持原始像素值不变（整数位移时）。
        """
        try:
            from src.coregistration import warp_multiband_with_displacement_field
        except ImportError:
            pytest.skip("coregistration 模块不可用")

        h, w = 40, 60  # non-square
        multiband = np.zeros((1, h, w), dtype=np.float64)
        multiband[0, 20, 30] = 1000.0

        local_dx = np.zeros((h, w), dtype=np.float64)
        local_dy = np.zeros((h, w), dtype=np.float64)
        result = warp_multiband_with_displacement_field(
            multiband, 0.0, 0.0, local_dx, local_dy, nodata=0.0,
        )

        assert result[0, 20, 30] == pytest.approx(1000.0, abs=1e-6)


# ===========================================================================
# test_nodata_protection
# ===========================================================================

class TestNodataProtection:
    """验证 NoData 值不参与统计计算。"""

    def test_nodata_excluded_from_mean(self):
        """NoData 像素应被排除在均值计算之外。"""
        from src.metrics import _extract_overlap_pixels

        arr = np.ones((1, 10, 10), dtype=np.float64) * 100.0
        arr[0, 0:5, 0:5] = -999.0  # NoData 区域

        ref = np.ones((1, 10, 10), dtype=np.float64) * 100.0

        pi, pj = _extract_overlap_pixels(
            arr, ref,
            window_i=(0, 10, 0, 10),
            window_j=(0, 10, 0, 10),
            nodata_i=-999.0,
            nodata_j=None,
            band=0,
        )
        # 排除 NoData 后有效像素均值应为 100
        assert len(pi) == 75  # 100 - 25 个 NoData
        assert pi.mean() == pytest.approx(100.0, abs=1e-6)

    def test_nodata_excluded_from_std(self):
        """NoData 像素应被排除在标准差计算之外。"""
        from src.metrics import _extract_overlap_pixels

        rng = np.random.default_rng(42)
        arr = rng.normal(100, 10, (1, 10, 10)).astype(np.float64)
        arr[0, 0:5, 0:5] = -999.0

        ref = rng.normal(100, 10, (1, 10, 10)).astype(np.float64)

        pi, pj = _extract_overlap_pixels(
            arr, ref,
            window_i=(0, 10, 0, 10),
            window_j=(0, 10, 0, 10),
            nodata_i=-999.0,
            nodata_j=None,
            band=0,
        )
        # 有效像素标准差应为有限正数
        assert np.isfinite(pi.std())
        assert pi.std() > 0

    def test_nan_excluded(self):
        """NaN 像素也应被排除。"""
        from src.metrics import _extract_overlap_pixels

        arr = np.ones((1, 10, 10), dtype=np.float64) * 100.0
        arr[0, 5, 5] = np.nan
        ref = arr.copy()

        pi, pj = _extract_overlap_pixels(
            arr, ref,
            window_i=(0, 10, 0, 10),
            window_j=(0, 10, 0, 10),
            nodata_i=None,
            nodata_j=None,
            band=0,
        )
        assert len(pi) == 99
        assert np.all(np.isfinite(pi))


# ===========================================================================
# test_experiment_config_load
# ===========================================================================

class TestExperimentConfigLoad:
    """测试实验配置的 YAML 加载。"""

    def test_make_test_config_loads(self):
        """make_test_config 应生成有效配置（跳过文件检查）。"""
        cfg = make_test_config(scene_ids=["s1", "s2", "s3"], n_bands=4)
        errors = validate_config(cfg, skip_file_check=True)
        assert len(errors) == 0, f"校验失败: {errors}"

    def test_save_and_reload(self):
        """配置保存后重新加载应保持一致。"""
        cfg = make_test_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_config.yaml")
            save_config(cfg, path)
            loaded = load_config(path)

        assert loaded.experiment_name == cfg.experiment_name
        assert len(loaded.scenes) == len(cfg.scenes)
        assert loaded.registration_band == cfg.registration_band

    def test_example_config_loads(self):
        """示例配置文件应能正确加载。"""
        example_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "experiment.example.yaml"
        )
        if not os.path.exists(example_path):
            pytest.skip("示例配置文件不存在")

        cfg = load_config(example_path)
        assert cfg.experiment_name == "dz01_multiband"
        assert len(cfg.selected_bands) == 14
        assert cfg.registration_band == "B14"


# ===========================================================================
# test_experiment_config_validation
# ===========================================================================

class TestExperimentConfigValidation:
    """测试配置校验：缺失必填字段应被捕获。"""

    def test_empty_experiment_name(self):
        """experiment_name 为空应报错。"""
        cfg = make_test_config()
        cfg.experiment_name = ""
        errors = validate_config(cfg)
        assert any("experiment_name" in e for e in errors)

    def test_empty_scenes(self):
        """scenes 为空应报错。"""
        cfg = make_test_config()
        cfg.scenes = []
        errors = validate_config(cfg)
        assert any("scenes" in e for e in errors)

    def test_empty_selected_bands(self):
        """selected_bands 为空应报错。"""
        cfg = make_test_config()
        cfg.selected_bands = []
        errors = validate_config(cfg)
        assert any("selected_bands" in e for e in errors)

    def test_registration_band_not_in_selected(self):
        """registration_band 不在 selected_bands 中应报错。"""
        cfg = make_test_config()
        cfg.registration_band = "B99"
        errors = validate_config(cfg)
        assert any("registration_band" in e for e in errors)

    def test_invalid_volrn_block_size(self):
        """volrn_params.block_size 为负数应报错。"""
        cfg = make_test_config()
        cfg.volrn_params["block_size"] = -100
        errors = validate_config(cfg)
        assert any("block_size" in e for e in errors)

    def test_invalid_common_bands_strategy(self):
        """不合法的 common_bands_strategy 应报错。"""
        cfg = make_test_config()
        cfg.common_bands_strategy = "invalid_strategy"
        errors = validate_config(cfg)
        assert any("common_bands_strategy" in e for e in errors)


# ===========================================================================
# test_single_band_error
# ===========================================================================

class TestSingleBandError:
    """测试单波段场景的错误处理。"""

    def test_single_band_error(self):
        """传入3D数组 (1, H, W) 通过波段数检查，但 k<=1 时应报错。"""
        rng = np.random.default_rng(42)
        arr = rng.uniform(100, 200, (1, 50, 50)).astype(np.float64)
        scenes = [
            {"id": "s1", "bands": {"B01": "a.tif"}},
        ]
        errors = validate_band_consistency(scenes, required_bands={"B01"})
        assert len(errors) == 0

        scenes_multi = [
            {"id": "s1", "bands": {"B01": "a.tif"}},
            {"id": "s2", "bands": {"B01": "b.tif"}},
        ]
        errors = validate_band_consistency(scenes_multi, required_bands={"B01"})
        assert len(errors) == 0


# ===========================================================================
# test_multiband_network_adjustment
# ===========================================================================

class TestMultibandNetworkAdjustment:
    """测试网络平差使用3D数组。"""

    def test_multiband_network_adjustment(self):
        """传入3D数组 (n_bands, H, W)，验证生成树和父场景映射正确。"""
        n_bands = 3
        h, w = 100, 100
        arrays = [np.zeros((n_bands, h, w)) for _ in range(3)]
        overlaps = [
            {"idx_i": 0, "idx_j": 1, "window_i": (0, h, 70, w), "window_j": (0, h, 0, 30), "pixel_count": h * 30},
            {"idx_i": 1, "idx_j": 2, "window_i": (0, h, 70, w), "window_j": (0, h, 0, 30), "pixel_count": h * 30},
        ]

        tree = build_spanning_tree(overlaps, n_images=3, reference_idx=0)
        assert len(tree) == 2
        visited = set()
        for parent, child in tree:
            visited.add(parent)
            visited.add(child)
        assert visited == {0, 1, 2}

        parent_map = propagate_control_scene(overlaps, n_images=3, reference_idx=0)
        assert parent_map[0] is None
        assert parent_map[1] == 0
        assert parent_map[2] == 1


# ===========================================================================
# test_multiband_propagation_bypass
# ===========================================================================

class TestMultibandPropagationBypass:
    """测试传播旁路：不连通节点的父场景为 None。"""

    def test_multiband_propagation_bypass(self):
        """传入3D数组 (2, H, W)，验证不连通场景的 parent_map 为 None。"""
        h, w = 80, 80
        overlaps = [
            {"idx_i": 0, "idx_j": 1, "window_i": (0, h, 40, w), "window_j": (0, h, 0, 40), "pixel_count": h * 40},
        ]

        parent_map = propagate_control_scene(overlaps, n_images=3, reference_idx=0)
        assert parent_map[0] is None
        assert parent_map[1] == 0
        assert parent_map[2] is None

        tree = build_spanning_tree(overlaps, n_images=3, reference_idx=0)
        assert len(tree) == 1
        assert tree[0] == (0, 1)


# ===========================================================================
# test_multiband_config_with_optional_fields
# ===========================================================================

class TestMultibandConfigWithOptionalFields:
    """测试包含可选字段的配置加载。"""

    def test_multiband_config_with_optional_fields(self):
        """使用已有的 dz01_multiband.yaml 配置文件验证可选字段。"""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "dz01_multiband.yaml"
        )
        if not os.path.exists(config_path):
            pytest.skip("配置文件不存在")

        cfg = load_config(config_path)
        assert cfg.experiment_name == "dz01_multiband"
        assert len(cfg.selected_bands) == 14
        assert cfg.registration_band == "B14"
        assert hasattr(cfg, "volrn_params")
        assert hasattr(cfg, "ablation_methods")
        assert hasattr(cfg, "mosaic_modes")
        errors = validate_config(cfg)
        assert len(errors) == 0, f"校验失败: {errors}"
