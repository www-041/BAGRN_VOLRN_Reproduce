"""
通用多景多波段处理管线模块

取代四景硬编码管线，提供可配置的通用管线接口：
  - 多景加载与波段验证
  - 重叠区域自动检测
  - 基于生成树的多波段配准
  - 多种辐射归一化方法并行对比
  - 镶嵌、评价指标一站式输出

关键设计原则：
  - 不修改已有的 four_image_pipeline.py
  - 仅依赖 src/ 中已有模块的公开接口
  - 支持 dry-run / smoke-test / caching
"""

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from collections import deque

import numpy as np

from src.experiment_config import ExperimentConfig, get_common_bands, validate_config
from src.io_utils import read_geotiff, write_geotiff, read_geotiff_band
from src.overlap import get_overlap_window, overlap_pixel_count, has_overlap
from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.metrics import compute_all, compute_per_pair, compute_per_band, save_metrics_csv
from src.comparison import run_comparison
from src.mosaic import create_mosaic

# ---------------------------------------------------------------------------
# 日志设置
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# 辅助函数
# ===========================================================================

def validate_band_consistency(scenes_config: List[Dict[str, Any]], required_bands=None) -> List[str]:
    """
    校验所有场景的波段一致性。

    检查：
      1. 每个场景都拥有所需的波段名集合。
      2. 分辨率是否一致（严格模式下必须完全一致）。

    参数
    ----------
    scenes_config : list of dict
        实验配置中的 scenes 列表。
    required_bands : set or None
        必须存在的波段集合（来自 selected_bands）。None 则使用交集。

    返回
    -------
    list of str
        错误列表，空列表表示全部通过。
    """
    errors: List[str] = []
    if not scenes_config:
        errors.append("scenes 列表为空")
        return errors

    # Use required_bands if provided (from selected_bands), else use intersection
    if required_bands:
        required = set(required_bands)
    else:
        # Fall back to intersection
        band_sets = []
        for scene in scenes_config:
            bands = scene.get("bands", {})
            band_sets.append(set(bands.keys()))
        if not band_sets:
            return errors
        required = band_sets[0]
        for s in band_sets[1:]:
            required = required & s
        if not required:
            errors.append("所有场景的公共波段集合为空（交集为空）")
            return errors

    for idx, scene in enumerate(scenes_config):
        if not isinstance(scene, dict):
            continue
        sid = scene.get("id", f"scene_{idx}")
        scene_bands = set(scene.get("bands", {}).keys())
        missing = required - scene_bands
        if missing:
            errors.append(
                f"scenes[{idx}] (id={sid}) 缺少波段: {sorted(missing)}"
            )
    return errors


def build_spanning_tree(
    overlaps: List[dict],
    n_images: int,
    reference_idx: int = 0,
) -> List[Tuple[int, int]]:
    """
    从重叠关系构建生成树（BFS）。

    以 reference_idx 为根，使用 BFS 找到一棵覆盖所有可达节点的生成树。
    未连接的节点被忽略。

    参数
    ----------
    overlaps : list of dict
        重叠信息列表，每项包含 idx_i, idx_j。
    n_images : int
        影像总数。
    reference_idx : int
        参考影像索引（生成树的根）。

    返回
    -------
    list of (int, int)
        生成树的边列表，(parent, child)。
    """
    adj: Dict[int, List[int]] = defaultdict(list)
    for ov in overlaps:
        i, j = ov["idx_i"], ov["idx_j"]
        adj[i].append(j)
        adj[j].append(i)

    visited = set([reference_idx])
    queue = deque([reference_idx])
    tree_edges: List[Tuple[int, int]] = []

    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                tree_edges.append((node, nb))
                queue.append(nb)

    return tree_edges


def propagate_control_scene(
    overlaps: List[dict],
    n_images: int,
    reference_idx: int = 0,
) -> Dict[int, Optional[int]]:
    """
    确定每个场景的父场景（基于 BFS 连通性）。

    返回
    -------
    dict
        scene_idx -> parent_idx，参考场景的 parent 为 None。
    """
    adj: Dict[int, List[int]] = defaultdict(list)
    for ov in overlaps:
        i, j = ov["idx_i"], ov["idx_j"]
        adj[i].append(j)
        adj[j].append(i)

    parent_map: Dict[int, Optional[int]] = {reference_idx: None}
    queue = deque([reference_idx])

    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in parent_map:
                parent_map[nb] = node
                queue.append(nb)

    # 未连接的节点也给一个默认值
    for idx in range(n_images):
        if idx not in parent_map:
            parent_map[idx] = None

    return parent_map


# ===========================================================================
# 主管线类
# ===========================================================================

class MultibandPipeline:
    """
    通用多景多波段辐射归一化管线。

    用法
    ----
    >>> from src.experiment_config import load_config
    >>> from src.multiband_pipeline import MultibandPipeline
    >>> cfg = load_config("config.yaml")
    >>> pipe = MultibandPipeline(cfg)
    >>> results = pipe.run()
    """

    def __init__(self, config: ExperimentConfig):
        """
        初始化管线。

        参数
        ----------
        config : ExperimentConfig
            实验配置对象。
        """
        self.config = config
        self.output_root = config.output_root
        self.smoke = config.smoke
        self.dry_run = config.dry_run
        self.smoke_crop_size = config.smoke_crop_size

        # 解析公共波段列表
        self.common_bands = get_common_bands(config)
        self.n_bands = len(self.common_bands)

        # 确定配准波段索引
        if config.registration_band in self.common_bands:
            self.registration_band_idx = self.common_bands.index(config.registration_band)
        else:
            self.registration_band_idx = self.n_bands - 1  # 回退到最后一波段

        # 控制场景索引
        self.control_idx = 0
        if config.control_scene and config.control_scene != "auto":
            for idx, scene in enumerate(config.scenes):
                if scene.get("id") == config.control_scene:
                    self.control_idx = idx
                    break

        # 缓存路径
        self._cache_dir = os.path.join(self.output_root, ".cache")

        # 冒烟模式：预收集各景元数据以确定裁剪中心
        self._smoke_meta: Optional[List[Dict[str, Any]]] = None
        if self.smoke:
            self._smoke_meta = self._collect_smoke_metadata()

    def _requested_normalization_methods(self) -> List[str]:
        """
        Get the list of requested normalization methods in order.
        
        Returns base methods (original, bagrn, bagrn_volrn) plus any
        ablation methods, with duplicates removed while preserving order.
        """
        base_methods = ["original", "bagrn", "bagrn_volrn"]
        all_methods = base_methods + list(self.config.ablation_methods)
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for method in all_methods:
            if method not in seen:
                seen.add(method)
                result.append(method)
        
        return result

    # -----------------------------------------------------------------------
    # 冒烟模式辅助
    # -----------------------------------------------------------------------

    def _collect_smoke_metadata(self) -> List[Dict[str, Any]]:
        """冒烟模式：预读各景元数据，确定裁剪中心。

        策略：不做全局交集中心，而是记录各景 bounds，
        后续在 detect_overlaps 之后为每条生成树边选择真实重叠中心裁剪。
        """
        import rasterio
        meta_list = []
        for scene_cfg in self.config.scenes:
            bands_dict = scene_cfg.get("bands", {})
            band_path = bands_dict.get(self.common_bands[0])
            if band_path is None:
                meta_list.append({})
                continue
            with rasterio.open(band_path) as src:
                tr = src.transform
                h, w = src.height, src.width
                left = tr.c
                top = tr.f
                right = left + tr.a * w
                bottom = top + tr.e * h
            meta_list.append({
                "bounds": (left, bottom, right, top),
                "height": h, "width": w, "transform": tr,
            })

        if meta_list:
            # 限制场景数：保留距所有景几何中心最近的 smoke_scene_count 景
            all_lefts = [m["bounds"][0] for m in meta_list if m]
            all_rights = [m["bounds"][2] for m in meta_list if m]
            all_tops = [m["bounds"][3] for m in meta_list if m]
            all_bottoms = [m["bounds"][1] for m in meta_list if m]
            center_x = (min(all_lefts) + max(all_rights)) / 2
            center_y = (min(all_bottoms) + max(all_tops)) / 2
            self._smoke_geo_center = (center_x, center_y)

            if len(meta_list) > self.config.smoke_scene_count:
                import math
                def _dist_to_center(m):
                    if not m or "bounds" not in m:
                        return float("inf")
                    lb, bot, rb, tp = m["bounds"]
                    mx = (lb + rb) / 2
                    my = (bot + tp) / 2
                    return math.hypot(mx - center_x, my - center_y)
                indexed = list(enumerate(meta_list))
                indexed.sort(key=lambda t: _dist_to_center(t[1]))
                keep = {i for i, _ in indexed[: self.config.smoke_scene_count]}
                self.config.scenes = [
                    s for i, s in enumerate(self.config.scenes) if i in keep
                ]
                meta_list = [m for i, m in enumerate(meta_list) if i in keep]
                logger.info(
                    "冒烟模式: 保留最近 %d 景（共 %d 景）",
                    len(meta_list), len(self.config.scenes),
                )
        else:
            self._smoke_geo_center = None

        return meta_list

    def _compute_overlap_center(self, bounds_i, bounds_j):
        """计算两景重叠区的地理中心坐标。"""
        left = max(bounds_i[0], bounds_j[0])
        right = min(bounds_i[2], bounds_j[2])
        bottom = max(bounds_i[1], bounds_j[1])
        top = min(bounds_i[3], bounds_j[3])
        if left >= right or bottom >= top:
            return None
        return ((left + right) / 2, (bottom + top) / 2)

    def _smoke_crop_around_center(self, arr: np.ndarray, tr, center_xy=None):
        """冒烟模式：围绕指定地理中心裁剪 smoke_crop_size x smoke_crop_size 区域。"""
        from rasterio.transform import Affine
        crop_size = self.smoke_crop_size
        if arr.ndim == 2:
            rows, cols = arr.shape
        else:
            _, rows, cols = arr.shape

        if center_xy is not None:
            cx, cy = center_xy
            inv_tr = ~tr
            col_center, row_center = inv_tr * (cx, cy)
            col_center, row_center = int(round(col_center)), int(round(row_center))
        elif self._smoke_geo_center is not None:
            cx, cy = self._smoke_geo_center
            inv_tr = ~tr
            col_center, row_center = inv_tr * (cx, cy)
            col_center, row_center = int(round(col_center)), int(round(row_center))
        else:
            row_center = rows // 2
            col_center = cols // 2

        r0 = max(0, row_center - crop_size // 2)
        c0 = max(0, col_center - crop_size // 2)
        r1 = min(rows, r0 + crop_size)
        c1 = min(cols, c0 + crop_size)
        if r1 - r0 < crop_size and rows >= crop_size:
            r0 = max(0, rows - crop_size)
            r1 = rows
        if c1 - c0 < crop_size and cols >= crop_size:
            c0 = max(0, cols - crop_size)
            c1 = cols

        if arr.ndim == 2:
            cropped = arr[r0:r1, c0:c1]
        else:
            cropped = arr[:, r0:r1, c0:c1]
        new_tr = tr * Affine.translation(c0, r0)
        return cropped, new_tr

    def _smoke_recrop_by_spanning_tree(
        self, scene_data: Dict[str, Any], tree_edges: List[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """冒烟模式：按生成树边的真实重叠中心重新裁剪各景。

        对每条树边 (parent, child)，取两景重叠区中心作为裁剪中心。
        每景的裁剪中心 = 其所有树边重叠中心的均值（保证该景与所有邻居重叠）。
        """
        import rasterio
        from rasterio.transform import Affine

        if not self.smoke or not tree_edges or not self._smoke_meta:
            return scene_data

        n_scenes = len(scene_data["arrays"])
        crop_size = self.smoke_crop_size

        # 为每条树边计算重叠中心
        edge_centers: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for parent, child in tree_edges:
            bounds_p = self._smoke_meta[parent].get("bounds")
            bounds_c = self._smoke_meta[child].get("bounds")
            if bounds_p and bounds_c:
                center = self._compute_overlap_center(bounds_p, bounds_c)
                if center:
                    edge_centers[(parent, child)] = center

        if not edge_centers:
            logger.warning("冒烟模式: 无法计算任何树边重叠中心，保持原始裁剪")
            return scene_data

        # 为每景计算裁剪中心 = 其所有关联树边重叠中心的均值
        scene_centers: Dict[int, Tuple[float, float]] = {}
        for idx in range(n_scenes):
            related_centers = []
            for (p, c), center in edge_centers.items():
                if p == idx or c == idx:
                    related_centers.append(center)
            if related_centers:
                avg_x = np.mean([c[0] for c in related_centers])
                avg_y = np.mean([c[1] for c in related_centers])
                scene_centers[idx] = (avg_x, avg_y)

        # 重新读取并裁剪各景
        new_arrays = []
        new_transforms = []
        new_bounds = []
        scenes = self.config.scenes

        for idx in range(n_scenes):
            center = scene_centers.get(idx)
            if center is None:
                new_arrays.append(scene_data["arrays"][idx])
                new_transforms.append(scene_data["transforms"][idx])
                new_bounds.append(scene_data["bounds"][idx])
                continue

            scene_cfg = scenes[idx]
            bands_dict = scene_cfg.get("bands", {})
            band_arrays = []
            tr = None
            for band_name in self.common_bands:
                band_path = bands_dict.get(band_name)
                if band_path is None:
                    continue
                arr, tr, _, _ = read_geotiff(band_path)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                arr, tr = self._smoke_crop_around_center(arr, tr, center_xy=center)
                band_arrays.append(arr)

            if band_arrays:
                multiband = np.stack(band_arrays, axis=0)
                new_arrays.append(multiband)
                new_transforms.append(tr)
                h, w = multiband.shape[1], multiband.shape[2]
                left = tr.c
                top = tr.f
                right = left + tr.a * w
                bottom = top + tr.e * h
                new_bounds.append((left, bottom, right, top))
            else:
                new_arrays.append(scene_data["arrays"][idx])
                new_transforms.append(scene_data["transforms"][idx])
                new_bounds.append(scene_data["bounds"][idx])

        scene_data["arrays"] = new_arrays
        scene_data["transforms"] = new_transforms
        scene_data["bounds"] = new_bounds

        logger.info(
            "冒烟模式: 按 %d 条生成树边重叠中心重新裁剪 %d 景",
            len(edge_centers), n_scenes,
        )
        return scene_data

    # -----------------------------------------------------------------------
    # a. 加载场景
    # -----------------------------------------------------------------------

    def load_scenes(self) -> Dict[str, Any]:
        """
        加载所有场景，验证波段可用性，处理多分辨率。

        返回
        -------
        dict
            {
                'arrays': list of np.ndarray,         # (bands, rows, cols)
                'transforms': list of Affine,
                'crs': str,
                'nodata_values': list of float|None,
                'bounds': list of (left, bottom, right, top),
                'scene_ids': list of str,
                'band_names': list of str,
                'resolution': float,                   # 公共分辨率
            }
        """
        logger.info("加载场景数据...")
        t0 = time.time()

        scenes = self.config.scenes
        n_scenes = len(scenes)

        arrays: List[np.ndarray] = []
        transforms = []
        crs_list: List[str] = []
        nodata_values: List[Optional[float]] = []
        bounds_list: List[Tuple[float, float, float, float]] = []
        scene_ids: List[str] = []

        for idx, scene_cfg in enumerate(scenes):
            sid = scene_cfg.get("id", f"scene_{idx}")
            scene_ids.append(sid)
            bands_dict = scene_cfg.get("bands", {})

            # 加载公共波段
            band_arrays = []
            for band_name in self.common_bands:
                band_path = bands_dict.get(band_name)
                if band_path is None:
                    raise ValueError(
                        f"场景 {sid} 缺少波段 {band_name}，"
                        f"可用波段: {list(bands_dict.keys())}"
                    )
                if self.smoke:
                    # 冒烟模式：先读取完整影像（用于准确重叠检测）
                    arr, tr, crs, nd = read_geotiff(band_path)
                    if arr.ndim == 3 and arr.shape[0] == 1:
                        arr = arr[0]  # 单波段文件 → (rows, cols)
                    elif arr.ndim == 3 and arr.shape[0] > 1:
                        raise ValueError(
                            f"场景 {sid} 波段 {band_name} 文件包含 {arr.shape[0]} 个波段，"
                            f"期望单波段文件"
                        )
                    band_arrays.append(arr)
                    if not band_arrays or len(band_arrays) == 1:
                        transforms.append(tr)
                        crs_list.append(crs)
                        nodata_values.append(nd)
                else:
                    arr, tr, crs, nd = read_geotiff(band_path)
                    if arr.ndim == 3 and arr.shape[0] == 1:
                        arr = arr[0]  # 单波段文件 → (rows, cols)
                    elif arr.ndim == 3 and arr.shape[0] > 1:
                        raise ValueError(
                            f"场景 {sid} 波段 {band_name} 文件包含 {arr.shape[0]} 个波段，"
                            f"期望单波段文件"
                        )
                    band_arrays.append(arr)
                    if not band_arrays or len(band_arrays) == 1:
                        transforms.append(tr)
                        crs_list.append(crs)
                        nodata_values.append(nd)

            # 组装多波段数组
            multiband = np.stack(band_arrays, axis=0)

            # After loading all bands for a scene, check intra-scene consistency
            if len(band_arrays) > 1:
                ref_shape = band_arrays[0].shape
                for bi, ba in enumerate(band_arrays[1:], 1):
                    if ba.shape != ref_shape:
                        raise ValueError(
                            f"场景 {sid} 波段形状不一致: 波段0={ref_shape}, 波段{bi}={ba.shape}"
                        )

            arrays.append(multiband)

            # 计算地理范围
            if self.smoke:
                h, w = multiband.shape[1], multiband.shape[2]
            else:
                with open(bands_dict[self.common_bands[0]], "rb") as f:
                    import rasterio
                    with rasterio.open(bands_dict[self.common_bands[0]]) as src:
                        h, w = src.height, src.width
            tr = transforms[idx]
            left = tr.c
            top = tr.f
            right = left + tr.a * w
            bottom = top + tr.e * h
            bounds_list.append((left, bottom, right, top))

        # 冒烟模式：验证全幅影像至少存在一对重叠区域
        if self.smoke and len(bounds_list) >= 2:
            has_overlap_any = False
            for i in range(len(bounds_list)):
                for j in range(i + 1, len(bounds_list)):
                    if has_overlap(bounds_list[i], bounds_list[j]):
                        has_overlap_any = True
                        break
                if has_overlap_any:
                    break
            if not has_overlap_any:
                logger.warning(
                    "冒烟模式全幅影像无重叠区域！scene_count=%d",
                    len(bounds_list),
                )

        # 统一分辨率
        # Strict mode validation
        if self.config.common_bands_strategy == "strict":
            from src.scene_preflight import validate_strict_scene_grids
            validate_strict_scene_grids(scene_ids, transforms, crs_list)
            # Use first scene's resolution (all should be same in strict mode)
            resolution = abs(transforms[0].a)
        else:
            # Non-strict mode: use minimum resolution
            resolutions = [abs(tr.a) for tr in transforms]
            resolution = min(resolutions)
            
            # Still validate CRS consistency
            if len(set(crs_list)) > 1:
                raise ValueError(
                    f"CRS 不一致: {dict(zip(scene_ids, crs_list))}。"
                    f"所有场景必须使用相同 CRS"
                )
        
        crs = crs_list[0]

        elapsed = time.time() - t0
        logger.info(
            "加载完成: %d 景, %d 波段, 分辨率=%.4f, 耗时 %.1fs",
            n_scenes, self.n_bands, resolution, elapsed,
        )

        return {
            "arrays": arrays,
            "transforms": transforms,
            "crs": crs,
            "nodata_values": nodata_values,
            "bounds": bounds_list,
            "scene_ids": scene_ids,
            "band_names": self.common_bands,
            "resolution": resolution,
        }

    # -----------------------------------------------------------------------
    # b. 检测重叠
    # -----------------------------------------------------------------------

    def detect_overlaps(self, scene_data: Dict[str, Any]) -> List[dict]:
        """
        检测所有场景对之间的重叠区域。

        参数
        ----------
        scene_data : dict
            由 load_scenes() 返回的数据字典。

        返回
        -------
        list of dict
            每个元素包含:
                idx_i, idx_j, window_i, window_j, pixel_count,
                per_band_mean, per_band_std
        """
        logger.info("检测重叠区域...")
        t0 = time.time()

        transforms = scene_data["transforms"]
        bounds_list = scene_data["bounds"]
        arrays = scene_data["arrays"]
        nodata_values = scene_data["nodata_values"]
        n_images = len(arrays)
        min_pixels = 1000 if not self.smoke else 100

        overlaps: List[dict] = []

        for i in range(n_images):
            for j in range(i + 1, n_images):
                win = get_overlap_window(
                    bounds_list[i], transforms[i],
                    bounds_list[j], transforms[j],
                )
                if win is None:
                    continue

                (ri_s, ri_e, ci_s, ci_e), (rj_s, rj_e, cj_s, cj_e) = win
                pix = (ri_e - ri_s) * (ci_e - ci_s)

                if pix < min_pixels:
                    continue

                # 计算重叠区逐波段统计
                per_band_stats = {}
                for b_idx in range(self.n_bands):
                    pi = arrays[i][b_idx, ri_s:ri_e, ci_s:ci_e]
                    pj = arrays[j][b_idx, rj_s:rj_e, cj_s:cj_e]

                    nd_i = nodata_values[i]
                    nd_j = nodata_values[j]

                    mi = np.isfinite(pi)
                    if nd_i is not None:
                        mi &= (pi != nd_i)
                    mj = np.isfinite(pj)
                    if nd_j is not None:
                        mj &= (pj != nd_j)
                    
                    # Use independent valid masks to support different-sized windows
                    valid_i = pi[mi]
                    valid_j = pj[mj]

                    if valid_i.size > 0 and valid_j.size > 0:
                        per_band_stats[b_idx] = {
                            "mean_i": float(valid_i.mean()),
                            "mean_j": float(valid_j.mean()),
                            "std_i": float(valid_i.std()),
                            "std_j": float(valid_j.std()),
                        }

                overlaps.append({
                    "idx_i": i,
                    "idx_j": j,
                    "window_i": (ri_s, ri_e, ci_s, ci_e),
                    "window_j": (rj_s, rj_e, cj_s, cj_e),
                    "pixel_count": pix,
                    "per_band_stats": per_band_stats,
                })

        elapsed = time.time() - t0
        logger.info(
            "检测到 %d 对重叠区域, 耗时 %.1fs", len(overlaps), elapsed,
        )

        return overlaps

    # -----------------------------------------------------------------------
    # c. 配准
    # -----------------------------------------------------------------------

    def register_scenes(
        self,
        scene_data: Dict[str, Any],
        overlaps: List[dict],
        registration_band_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        对所有场景执行多波段配准。

        流程：
          1. 用配准波段执行逐对匹配
          2. 构建生成树
          3. 网络平差求解全局位移
          4. 用 warp_multiband_with_displacement_field 对所有波段施加位移
          5. 可选 RBF 局部精化

        参数
        ----------
        scene_data : dict
            由 load_scenes() 返回的数据字典。
        overlaps : list of dict
            由 detect_overlaps() 返回的重叠列表。
        registration_band_idx : int or None
            配准使用的波段索引。None 则使用 self.registration_band_idx。

        返回
        -------
        dict
            {
                'registered_arrays': list of np.ndarray,
                'global_shifts': np.ndarray (n_images, 2),
                'pair_matches': list of dict,
                'diagnostics': dict,
            }
        """
        from src.coregistration import (
            collect_block_matches,
            multi_image_network_adjustment,
            warp_multiband_with_displacement_field,
            rematch_pair_on_registered,
        )

        if registration_band_idx is None:
            registration_band_idx = self.registration_band_idx

        logger.info(
            "开始配准 (波段索引=%d, 波段名=%s)...",
            registration_band_idx,
            self.common_bands[registration_band_idx] if registration_band_idx < len(self.common_bands) else "?",
        )
        t0 = time.time()

        arrays = scene_data["arrays"]
        transforms = scene_data["transforms"]
        nodata_values = scene_data["nodata_values"]
        n_images = len(arrays)

        if n_images <= 1:
            return {
                "registered_arrays": [a.copy() for a in arrays],
                "global_shifts": np.zeros((n_images, 2)),
                "pair_matches": [],
                "diagnostics": {"skipped": True, "reason": "单景无需配准"},
            }

        # ---- Step 1: 逐对匹配（用配准波段） ----
        from src.coregistration import (
    phase_correlation_from_overlap,
    robust_shift_estimate,
    classify_registration_quality,
)

        pair_measurements: List[dict] = []
        rejected_edges: List[dict] = []

        for ov in overlaps:
            i, j = ov["idx_i"], ov["idx_j"]
            arr_i_reg = arrays[i][registration_band_idx]
            arr_j_reg = arrays[j][registration_band_idx]
            nd_i = nodata_values[i]  # Keep None as None, don't convert to 0
            nd_j = nodata_values[j]  # Keep None as None, don't convert to 0

            # 尝试1: 块匹配
            matches, screening = collect_block_matches(
                arr_i_reg, transforms[i],
                arr_j_reg, transforms[j],
                nd_i, nd_j,
                block_size=512,
                max_global_shift=40,
                confidence_threshold=0.5,
            )

            if matches:
                confs = np.array([m["confidence"] for m in matches])
                dxs = np.array([m["shift_dx"] for m in matches])
                dys = np.array([m["shift_dy"] for m in matches])
                shift_dx = float(np.average(dxs, weights=confs))
                shift_dy = float(np.average(dys, weights=confs))
                res = np.hypot(dxs - shift_dx, dys - shift_dy)
                rmse = float(np.sqrt(np.mean(res**2)))
                p95 = float(np.percentile(res, 95)) if len(res) > 0 else 0.0

                pair_measurements.append({
                    "idx_i": i, "idx_j": j,
                    "shift_dx": shift_dx, "shift_dy": shift_dy,
                    "confidence": float(confs.mean()),
                    "n_blocks": len(matches),
                    "rmse": rmse, "p95": p95,
                    "matches": matches, "screening": screening,
                    "method": "block_match",
                })
                logger.info(
                    "  [%d]-[%d] block_match: dx=%.4f, dy=%.4f, conf=%.3f, "
                    "blocks=%d, rmse=%.3f, p95=%.3f",
                    i, j, shift_dx, shift_dy, float(confs.mean()),
                    len(matches), rmse, p95,
                )
                continue

            # 尝试2: overlap phase correlation fallback（地理重叠区）
            logger.info("  [%d]-[%d] 块匹配失败，尝试 phase correlation fallback...", i, j)
            valid_i = np.isfinite(arr_i_reg) & (arr_i_reg != nd_i)
            valid_j = np.isfinite(arr_j_reg) & (arr_j_reg != nd_j)
            try:
                sy, sx, conf_pc = phase_correlation_from_overlap(
                    arr_i_reg, transforms[i],
                    arr_j_reg, transforms[j],
                    nodata_ref=nd_i, nodata_tgt=nd_j,
                )
            except Exception as exc:
                logger.warning("  [%d]-[%d] phase correlation 异常: %s", i, j, exc)
                sy, sx, conf_pc = 0.0, 0.0, 0.0

            if conf_pc > 0.3 and (abs(sy) < 40 and abs(sx) < 40):
                pair_measurements.append({
                    "idx_i": i, "idx_j": j,
                    "shift_dx": float(sx), "shift_dy": float(sy),
                    "confidence": float(conf_pc),
                    "n_blocks": 1,
                    "rmse": 0.0, "p95": 0.0,
                    "matches": [], "screening": screening,
                    "method": "overlap_phase_correlation",
                })
                logger.info(
                    "  [%d]-[%d] overlap_phase_correlation: dx=%.4f, dy=%.4f, conf=%.3f",
                    i, j, float(sx), float(sy), float(conf_pc),
                )
                continue

            # 尝试3: 拒绝该边
            reason = "block_match无匹配且phase_correlation失败"
            if screening.get("low_valid", 0) > 0:
                reason += f" (low_valid={screening['low_valid']})"
            if screening.get("low_texture", 0) > 0:
                reason += f" (low_texture={screening['low_texture']})"
            rejected_edges.append({
                "idx_i": i, "idx_j": j,
                "reason": reason, "screening": screening,
            })
            logger.warning("  [%d]-[%d] 拒绝: %s", i, j, reason)

        if not pair_measurements:
            raise ValueError("没有可用的匹配对，无法进行配准")

        # ---- Step 1.5: 连通性分析 ----
        # 几何重叠边（所有 detect_overlaps 发现的对）
        geometric_edges = [(ov["idx_i"], ov["idx_j"]) for ov in overlaps]
        # 有效匹配边
        matching_edges = [(p["idx_i"], p["idx_j"]) for p in pair_measurements]
        # 拒绝边
        rejected_list = [(r["idx_i"], r["idx_j"], r["reason"]) for r in rejected_edges]

        logger.info("几何重叠边: %d 条 %s", len(geometric_edges), geometric_edges)
        logger.info("有效匹配边: %d 条 %s", len(matching_edges), matching_edges)
        if rejected_list:
            logger.info("拒绝边: %d 条", len(rejected_list))
            for ri, rj, reason in rejected_list:
                logger.info("  [%d]-[%d] 原因: %s", ri, rj, reason)

        # 计算连通分量（BFS）
        adj_all = defaultdict(set)
        for i, j in geometric_edges:
            adj_all[i].add(j)
            adj_all[j].add(i)
        visited_global = set()
        components: List[set] = []
        for node in range(n_images):
            if node in visited_global:
                continue
            comp = set()
            queue = deque([node])
            while queue:
                n = queue.popleft()
                if n in comp:
                    continue
                comp.add(n)
                visited_global.add(n)
                for nb in adj_all[n]:
                    if nb not in comp:
                        queue.append(nb)
            components.append(comp)
        logger.info("连通分量: %d 个 %s", len(components),
                     [sorted(c) for c in components])

        # 从有效匹配边构建生成树（BFS from control_idx）
        adj_match = defaultdict(set)
        for p in pair_measurements:
            adj_match[p["idx_i"]].add(p["idx_j"])
            adj_match[p["idx_j"]].add(p["idx_i"])

        visited = set([self.control_idx])
        queue = deque([self.control_idx])
        spanning_tree_edges: List[Tuple[int, int]] = []
        while queue:
            node = queue.popleft()
            for nb in adj_match[node]:
                if nb not in visited:
                    visited.add(nb)
                    spanning_tree_edges.append((node, nb))
                    queue.append(nb)

        unreachable = set(range(n_images)) - visited
        scene_ids = scene_data.get("scene_ids", [str(i) for i in range(n_images)])
        unreachable_ids = [scene_ids[i] for i in sorted(unreachable)]

        logger.info("生成树边: %d 条 %s", len(spanning_tree_edges), spanning_tree_edges)
        logger.info("可达场景: %d/%d %s", len(visited), n_images, sorted(visited))
        if unreachable:
            logger.warning("不可达场景: %s", unreachable_ids)

        # smoke 模式：不可达则报错（不允许继续）
        if unreachable:
            if self.smoke:
                raise ValueError(
                    f"smoke 五景验收失败: 场景 {unreachable_ids} 不可达，"
                    f"要求所有 {n_images} 景连通"
                )
            else:
                raise ValueError(
                    f"网络不连通: 场景 {unreachable_ids} 无法从控制景到达"
                )

        # ---- Step 2: 网络平差 ----
        # 用所有实测边做网络平差
        adj_result = multi_image_network_adjustment(
            pair_measurements, n_images, self.control_idx,
        )
        global_shifts = adj_result["global_shifts"]
        logger.info("网络平差完成, 闭环误差数: %d", len(adj_result.get("loop_errors", [])))

        # ---- Step 2.5: Registration Quality Gate ----
        reg_params = self.config.registration_params
        # Collect all pairwise shifts for quality assessment
        all_shifts_x = []
        all_shifts_y = []
        all_confs = []
        for p in pair_measurements:
            all_shifts_x.append(p["shift_dx"])
            all_shifts_y.append(p["shift_dy"])
            all_confs.append(p["confidence"])
        
        if all_shifts_x:
            robust_result = robust_shift_estimate(
                np.array(all_shifts_x), np.array(all_shifts_y), np.array(all_confs),
                reg_params,
            )
            quality = classify_registration_quality(robust_result, reg_params)
            required_quality = reg_params.get("required_quality", "pass")
            
            logger.info("配准质量: %s (要求: %s)", quality, required_quality)
            logger.info("  置信度=%.3f, 中值残差=%.3f, RMSE=%.3f, P95=%.3f, 内点数=%d",
                       robust_result["confidence"], robust_result["residual_median"],
                       robust_result["residual_rmse"], robust_result["residual_p95"],
                       robust_result["n_inliers"])
            
            quality_order = {"pass": 0, "warn": 1, "fail": 2}
            if quality_order.get(quality, 2) > quality_order.get(required_quality, 0):
                raise ValueError(
                    f"配准质量 {quality} 不满足要求 {required_quality}。"
                    f"RMSE={robust_result['residual_rmse']:.3f}, "
                    f"P95={robust_result['residual_p95']:.3f}"
                )

        # ---- Step 3: 对所有波段施加全局位移 ----
        registered_arrays: List[np.ndarray] = []
        for idx in range(n_images):
            gdx = global_shifts[idx, 0]
            gdy = global_shifts[idx, 1]

            if abs(gdx) < 1e-6 and abs(gdy) < 1e-6:
                registered_arrays.append(arrays[idx].astype(np.float64))
            else:
                # 构建全零局部场（先做全局配准，局部精化在后面）
                h, w = arrays[idx].shape[1], arrays[idx].shape[2]
                local_dx = np.zeros((h, w), dtype=np.float64)
                local_dy = np.zeros((h, w), dtype=np.float64)
                nd_val = nodata_values[idx]  # Keep None as None, don't convert to 0.0

                warped = warp_multiband_with_displacement_field(
                    arrays[idx], gdx, gdy,
                    local_dx, local_dy, nd_val,
                )
                registered_arrays.append(warped)

            logger.info("  [%d] 全局位移: dx=%.4f, dy=%.4f", idx, gdx, gdy)

        elapsed = time.time() - t0
        logger.info("配准完成, 耗时 %.1fs", elapsed)

        return {
            "registered_arrays": registered_arrays,
            "global_shifts": global_shifts,
            "pair_matches": pair_measurements,
            "spanning_tree": spanning_tree_edges,
            "geometric_edges": geometric_edges,
            "matching_edges": matching_edges,
            "rejected_edges": rejected_list,
            "connected_components": [sorted(c) for c in components],
            "unreachable_scenes": unreachable_ids,
            "diagnostics": {
                "n_pairs_matched": len(pair_measurements),
                "n_rejected": len(rejected_edges),
                "n_geometric": len(geometric_edges),
                "n_components": len(components),
                "global_shifts": global_shifts.tolist(),
                "loop_errors": adj_result.get("loop_errors", []),
                "elapsed_sec": elapsed,
            },
        }

    # -----------------------------------------------------------------------
    # d. 辐射归一化
    # -----------------------------------------------------------------------

    def apply_radiometric_normalization(
        self,
        registered_data: Dict[str, Any],
        overlaps: List[dict],
        methods: Optional[List[str]] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """
        对配准后的数据应用多种辐射归一化方法。

        支持的方法：
          - 'original': 不做归一化（原始数据）
          - 'bagrn': 全局归一化（BAGRN）
          - 'volrn_only': 仅局部归一化（无 BAGRN 前处理）
          - 'bagrn_volrn': BAGRN + VOLRN（论文主方法）
          - 'histogram_matching': 直方图匹配（基线）
          - 'moment_matching': 矩匹配（基线）
          - 'wallis': Wallis 滤波（基线）

        参数
        ----------
        registered_data : dict
            由 register_scenes() 返回的数据字典。
        overlaps : list of dict
            重叠区域信息。
        methods : list of str or None
            要执行的方法列表。None 则使用 config 中的 ablation_methods。

        返回
        -------
        dict
            method_name -> list of np.ndarray (归一化后的数组)。
        """
        if methods is None:
            methods = ["original", "bagrn", "bagrn_volrn"] + list(self.config.ablation_methods)

        # 去重保持顺序
        seen = set()
        unique_methods = []
        for m in methods:
            if m not in seen:
                seen.add(m)
                unique_methods.append(m)
        methods = unique_methods

        logger.info("执行辐射归一化, 方法: %s", methods)
        t0 = time.time()

        arrays = registered_data["registered_arrays"]
        transforms = registered_data.get("transforms", [])
        bounds_list = registered_data.get("bounds", [])
        nodata_values_list = registered_data.get("nodata_values", [None] * len(arrays))

        control_idx = self.control_idx
        volrn_params = self.config.volrn_params

        # 构建生成树用于传播型方法
        tree_edges = build_spanning_tree(overlaps, len(arrays), control_idx)
        spanning_tree = tree_edges if tree_edges else None

        results: Dict[str, List[np.ndarray]] = {}
        bagrn_cache = None

        for method in methods:
            logger.info("  方法: %s", method)
            t_method = time.time()

            try:
                if method == "original":
                    results[method] = [a.copy() for a in arrays]

                elif method == "bagrn":
                    normalized, _, _ = bagrn_normalize(
                        arrays, nodata_values_list, overlaps, control_idx,
                    )
                    bagrn_cache = normalized
                    results[method] = normalized

                elif method == "volrn_only":
                    # 无 BAGRN 前处理的 VOLRN
                    normalized, _ = volrn_normalize(
                        arrays, transforms, bounds_list, nodata_values_list,
                        block_size_pixels=volrn_params.get("block_size", 400),
                        lambda_param=volrn_params.get("lambda", 0.5),
                        rho=volrn_params.get("rho", 1.0),
                        max_iter=volrn_params.get("max_iter", 200),
                        tol=volrn_params.get("tol", 1e-4),
                        verbose=False,
                    )
                    results[method] = normalized

                elif method == "bagrn_volrn":
                    if bagrn_cache is not None:
                        bagrn_normalized = bagrn_cache
                    else:
                        bagrn_normalized, _, _ = bagrn_normalize(
                            arrays, nodata_values_list, overlaps, control_idx,
                        )
                    volrn_normalized, volrn_block_coeffs, volrn_diagnostics = volrn_normalize(
                        bagrn_normalized, transforms, bounds_list, nodata_values_list,
                        block_size_pixels=volrn_params.get("block_size", 400),
                        lambda_param=volrn_params.get("lambda", 0.5),
                        rho=volrn_params.get("rho", 1.0),
                        max_iter=volrn_params.get("max_iter", 200),
                        tol=volrn_params.get("tol", 1e-4),
                        verbose=False,
                        return_diagnostics=True,
                    )
                    results[method] = volrn_normalized
                    results[f"{method}_block_coefficients"] = volrn_block_coeffs

                elif method in ("histogram_matching", "moment_matching", "wallis"):
                    normalized = run_comparison(
                        method, arrays, nodata_values_list, overlaps,
                        control_idx=control_idx,
                        spanning_tree=spanning_tree,
                    )
                    results[method] = normalized

                else:
                    logger.warning("未知方法 '%s'，跳过", method)
                    continue

            except Exception as exc:
                logger.error("  方法 '%s' 失败: %s", method, exc)
                continue

            elapsed_m = time.time() - t_method
            logger.info("    %s 完成, 耗时 %.1fs", method, elapsed_m)

        elapsed = time.time() - t0
        logger.info("所有归一化方法完成, 总耗时 %.1fs", elapsed)

        return results

    # -----------------------------------------------------------------------
    # e. 生成镶嵌
    # -----------------------------------------------------------------------

    def compute_mosaics(
        self,
        normalized_dict: Dict[str, List[np.ndarray]],
        transforms: list,
        crs: str,
        nodata_values: list,
        output_dir: str,
    ) -> Dict[str, List[str]]:
        """
        对每种归一化方法生成镶嵌图。

        参数
        ----------
        normalized_dict : dict
            method_name -> list of np.ndarray。
        transforms : list
            仿射变换列表。
        crs : str
            坐标参考系。
        nodata_values : list
            各景 NoData 值。
        output_dir : str
            输出目录。

        返回
        -------
        dict
            method_name -> list of mosaic file paths。
        """
        logger.info("生成镶嵌图...")
        t0 = time.time()

        os.makedirs(output_dir, exist_ok=True)
        mosaic_results: Dict[str, List[str]] = {}

        for method, arrays in normalized_dict.items():
            method_dir = os.path.join(output_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            paths: List[str] = []

            for mode in self.config.mosaic_modes:
                mode = mode.strip()  # 清理可能的前导空格
                if mode == "narrow_feather":
                    for fw in self.config.feather_widths:
                        out_name = f"mosaic_{method}_{mode}_fw{fw}.tif"
                        out_path = os.path.join(method_dir, out_name)

                        if self.dry_run:
                            logger.info("  [DRY-RUN] 将生成: %s", out_path)
                            paths.append(out_path)
                            continue

                        try:
                            create_mosaic(
                                arrays, transforms, crs, nodata_values,
                                out_path,
                                mode=mode,
                                feather_width=fw,
                            )
                            paths.append(out_path)
                            logger.info("  已生成: %s", out_path)
                        except Exception as exc:
                            logger.error("  镶嵌失败 [%s, %s, fw=%d]: %s", method, mode, fw, exc)
                else:
                    out_name = f"mosaic_{method}_{mode}.tif"
                    out_path = os.path.join(method_dir, out_name)

                    if self.dry_run:
                        logger.info("  [DRY-RUN] 将生成: %s", out_path)
                        paths.append(out_path)
                        continue

                    try:
                        create_mosaic(
                            arrays, transforms, crs, nodata_values,
                            out_path,
                            mode=mode,
                        )
                        paths.append(out_path)
                        logger.info("  已生成: %s", out_path)
                    except Exception as exc:
                        logger.error("  镶嵌失败 [%s, %s]: %s", method, mode, exc)

            mosaic_results[method] = paths

        elapsed = time.time() - t0
        logger.info("镶嵌完成, 耗时 %.1fs", elapsed)

        return mosaic_results

    # -----------------------------------------------------------------------
    # f. 评价指标
    # -----------------------------------------------------------------------

    def evaluate_metrics(
        self,
        normalized_dict: Dict[str, List[np.ndarray]],
        overlaps: List[dict],
        nodata_values: list,
        bands: Optional[List[int]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, dict]:
        """
        对每种归一化方法计算评价指标。

        指标包括 ADM, ADSD, CD, GL, RDOA, Ave（论文 Section 3.1）。

        参数
        ----------
        normalized_dict : dict
            method_name -> list of np.ndarray。
        overlaps : list of dict
            重叠区域信息。
        nodata_values : list
            各景 NoData 值。
        bands : list of int or None
            波段索引列表。None 则处理所有波段。
        output_dir : str or None
            CSV 输出目录。None 则不保存。

        返回
        -------
        dict
            method_name -> {adm, adsd, cd, gl, rdoa, ave, per_pair, per_band}。
        """
        logger.info("计算评价指标...")
        t0 = time.time()

        if bands is None:
            bands = list(range(self.n_bands))

        # 获取原始数组用于 GL 计算
        original_arrays = normalized_dict.get("original", None)

        metrics_results: Dict[str, dict] = {}

        for method, arrays in normalized_dict.items():
            if not isinstance(arrays, list) or not arrays or not isinstance(arrays[0], np.ndarray):
                continue
            t_method = time.time()
            logger.info("  评估方法: %s", method)

            # 全局指标
            metrics = compute_all(
                original_arrays if original_arrays is not None else arrays,
                arrays, nodata_values, overlaps, bands,
            )

            # 逐对指标
            per_pair = compute_per_pair(arrays, nodata_values, overlaps, bands)

            # 逐波段指标
            per_band = compute_per_band(arrays, nodata_values, overlaps, bands)

            metrics_results[method] = {
                **metrics,
                "per_pair": per_pair,
                "per_band": per_band,
            }

            # 保存 CSV
            if output_dir:
                method_dir = os.path.join(output_dir, method)
                os.makedirs(method_dir, exist_ok=True)
                save_metrics_csv(
                    metrics, method_dir,
                    bands=bands, overlaps=overlaps,
                    arrays_before=original_arrays if original_arrays is not None else arrays,
                    arrays_after=arrays,
                    nodata_values=nodata_values,
                )

            elapsed_m = time.time() - t_method
            logger.info("    %s 评估完成, 耗时 %.1fs", method, elapsed_m)

        elapsed = time.time() - t0
        logger.info("指标计算完成, 耗时 %.1fs", elapsed)

        return metrics_results

    # -----------------------------------------------------------------------
    # f2. 光谱保持指标
    # -----------------------------------------------------------------------

    def evaluate_spectral(
        self,
        normalized_dict: Dict[str, List[np.ndarray]],
        original_arrays: List[np.ndarray],
        nodata_values: list,
        band_names: List[str],
        overlaps: List[dict],
        output_dir: Optional[str] = None,
    ) -> Dict[str, dict]:
        """
        计算光谱保持指标：SAM, spectral RMSE, correlation。

        对每种归一化方法（除 original 外），逐景比较归一化前后光谱。
        """
        from src.spectral_metrics import compute_all_spectral

        logger.info("计算光谱保持指标...")
        t0 = time.time()

        spectral_results: Dict[str, dict] = {}

        for method, arrays in normalized_dict.items():
            if method == "original":
                continue
            if not isinstance(arrays, list) or not arrays or not isinstance(arrays[0], np.ndarray):
                continue

            logger.info("  光谱指标: %s", method)
            try:
                method_dir = os.path.join(output_dir, method) if output_dir else None
                spectral = compute_all_spectral(
                    original_arrays, arrays, nodata_values, overlaps,
                    band_names=band_names, output_dir=method_dir,
                )
                spectral_results[method] = spectral
            except Exception as exc:
                logger.error("  光谱指标计算失败 [%s]: %s", method, exc)
                spectral_results[method] = {"error": str(exc)}

        elapsed = time.time() - t0
        logger.info("光谱指标完成, 耗时 %.1fs", elapsed)
        return spectral_results

    # -----------------------------------------------------------------------
    # f3. 数据质量检查
    # -----------------------------------------------------------------------

    def check_data_quality(
        self,
        normalized_dict: Dict[str, List[np.ndarray]],
        scene_data: Dict[str, Any],
    ) -> Dict[str, dict]:
        """
        检查每种方法输出的数据质量：NaN/Inf/valid pixels/shape/transform/CRS。

        如果有效数据区出现 NaN 或 Inf，标记该方法为失败。
        使用 normalized_dict["original"] 作为 reference valid mask。
        """
        logger.info("执行数据质量检查...")
        quality_results: Dict[str, dict] = {}

        nodata_values = scene_data["nodata_values"]
        transforms = scene_data["transforms"]
        crs = scene_data["crs"]
        resolution = scene_data["resolution"]
        band_names = scene_data["band_names"]
        
        # 获取 reference (original) 作为 valid mask 的基准
        reference_arrays = normalized_dict.get("original", None)

        for method, arrays in normalized_dict.items():
            if not isinstance(arrays, list) or not arrays or not isinstance(arrays[0], np.ndarray):
                continue

            method_quality = {"status": "pass", "issues": []}
            n_scenes = len(arrays)

            for i, arr in enumerate(arrays):
                nd = nodata_values[i] if i < len(nodata_values) else None
                bands_shape = arr.shape
                n_bands_out = bands_shape[0]

                # 使用 reference 构建 valid mask
                if reference_arrays and i < len(reference_arrays):
                    ref = reference_arrays[i]
                    ref_valid = np.isfinite(ref).all(axis=0)
                    if nd is not None:
                        ref_valid &= ~np.any(ref == nd, axis=0)
                else:
                    # 无 reference 时使用 arr 自身
                    if nd is not None:
                        ref_valid = np.isfinite(arr).all(axis=0) & ~np.any(arr == nd, axis=0)
                    else:
                        ref_valid = np.isfinite(arr).all(axis=0)

                # 只统计 reference valid area 中的 NaN/Inf
                nan_in_valid = int(np.sum(np.isnan(arr) & ref_valid[np.newaxis, :, :]))
                inf_in_valid = int(np.sum(np.isinf(arr) & ref_valid[np.newaxis, :, :]))
                total_nan = int(np.sum(np.isnan(arr)))
                total_inf = int(np.sum(np.isinf(arr)))
                valid_pixels = int(ref_valid.sum())

                # 只有 reference valid area 中出现 NaN/Inf 才判 fail
                if nan_in_valid > 0 or inf_in_valid > 0:
                    method_quality["status"] = "fail"
                    method_quality["issues"].append(
                        f"scene_{i}: NaN={nan_in_valid}, Inf={inf_in_valid} in valid area"
                    )

                method_quality[f"scene_{i}"] = {
                    "shape": list(bands_shape),
                    "n_bands": n_bands_out,
                    "nan_in_valid": nan_in_valid,
                    "inf_in_valid": inf_in_valid,
                    "total_nan": total_nan,
                    "total_inf": total_inf,
                    "valid_pixels": valid_pixels,
                }

            # Check transform and CRS consistency (use first scene)
            if transforms:
                method_quality["transform"] = str(transforms[0])
            method_quality["crs"] = crs
            method_quality["resolution"] = resolution
            method_quality["band_names"] = band_names

            quality_results[method] = method_quality
            if method_quality["status"] == "fail":
                logger.warning("  %s: 数据质量检查失败: %s", method, method_quality["issues"])
            else:
                logger.info("  %s: 数据质量检查通过", method)

        return quality_results

    # -----------------------------------------------------------------------
    # g. 完整管线
    # -----------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        执行完整的多景多波段处理管线。

        返回
        -------
        dict
            {
                'config': ExperimentConfig,
                'scene_data': dict,
                'overlaps': list,
                'registration': dict,
                'normalized': dict,
                'mosaics': dict,
                'metrics': dict,
                'elapsed_total': float,
                'pipeline_status': str,
                'registration_connected': bool,
                'n_scenes_requested': int,
                'n_scenes_processed': int,
                'unreachable_scenes': list,
                'failed_methods': list,
                'skipped_outputs': list,
            }
        """
        total_t0 = time.time()
        n_scenes_requested = len(self.config.scenes)
        logger.info("=" * 60)
        logger.info("管线启动: %s (请求 %d 景)", self.config.experiment_name, n_scenes_requested)
        logger.info("=" * 60)

        failed_methods: List[str] = []
        skipped_outputs: List[str] = []

        # 0. dry-run 提前返回（跳过文件校验）
        if self.dry_run:
            logger.info("[DRY-RUN] 以下为将要执行的步骤:")
            logger.info("  1. 加载 %d 景影像", n_scenes_requested)
            logger.info("  2. 检测重叠区域")
            logger.info("  3. 配准 (波段: %s)", self.config.registration_band)
            logger.info("  4. 辐射归一化: %s", self.config.ablation_methods)
            logger.info("  5. 生成镶嵌图")
            logger.info("  6. 计算评价指标")
            return {"dry_run": True, "config": self.config}

        # 0. 配置校验
        errors = validate_config(self.config)
        if errors:
            for e in errors:
                logger.error("配置校验失败: %s", e)
            raise ValueError(f"配置校验失败: {errors}")

        band_errors = validate_band_consistency(
            self.config.scenes,
            required_bands=set(self.config.selected_bands)
        )
        if band_errors:
            for e in band_errors:
                logger.error("波段一致性校验失败: %s", e)
            raise ValueError(f"波段一致性校验失败: {band_errors}")

        # 1. 加载场景（smoke 模式下先用全局中心裁剪）
        scene_data = self.load_scenes()

        # 2. 检测重叠
        overlaps = self.detect_overlaps(scene_data)
        if not overlaps:
            logger.warning("未检测到重叠区域，管线提前结束")
            return {
                "scene_data": scene_data,
                "overlaps": [],
                "registration": None,
                "normalized": None,
                "mosaics": None,
                "metrics": None,
                "elapsed_total": time.time() - total_t0,
                "pipeline_status": "failed",
                "registration_connected": False,
                "n_scenes_requested": n_scenes_requested,
                "n_scenes_processed": 0,
                "unreachable_scenes": [],
                "failed_methods": [],
                "skipped_outputs": ["no_overlaps"],
            }

        # 2.5 smoke 模式：从全幅重叠检测构建初始生成树，然后按树边重叠中心裁剪
        # 策略：先在全幅影像上完成配准（确保连通性），再裁剪到目标尺寸并复用全局位移
        if self.smoke:
            from src.multiband_pipeline import build_spanning_tree
            from src.coregistration import warp_multiband_with_displacement_field

            initial_tree = build_spanning_tree(overlaps, len(scene_data["arrays"]), self.control_idx)
            if initial_tree:
                # 先在全幅上配准，验证连通性并获取全局位移
                full_registration = self.register_scenes(scene_data, overlaps)
                full_global_shifts = full_registration["global_shifts"]

                # 裁剪到目标尺寸
                scene_data = self._smoke_recrop_by_spanning_tree(scene_data, initial_tree)
                overlaps = self.detect_overlaps(scene_data)
                logger.info("冒烟模式: 裁剪到 %dpx 后检测到 %d 对重叠",
                            self.smoke_crop_size, len(overlaps))

                # 对裁剪后的影像直接应用全幅配准的全局位移（不再重新配准）
                registered_arrays = []
                for idx in range(len(scene_data["arrays"])):
                    gdx = full_global_shifts[idx, 0]
                    gdy = full_global_shifts[idx, 1]
                    if abs(gdx) < 1e-6 and abs(gdy) < 1e-6:
                        registered_arrays.append(scene_data["arrays"][idx].astype(np.float64))
                    else:
                        h, w = scene_data["arrays"][idx].shape[1], scene_data["arrays"][idx].shape[2]
                        local_dx = np.zeros((h, w), dtype=np.float64)
                        local_dy = np.zeros((h, w), dtype=np.float64)
                        nd_val = scene_data["nodata_values"][idx]  # Keep None as None
                        warped = warp_multiband_with_displacement_field(
                            scene_data["arrays"][idx], gdx, gdy,
                            local_dx, local_dy, nd_val,
                        )
                        registered_arrays.append(warped)
                    logger.info("  [%d] 全局位移: dx=%.4f, dy=%.4f (复用全幅配准)", idx, gdx, gdy)

                # 构建 registration dict，复用全幅配准的连通性信息
                registration = {
                    "registered_arrays": registered_arrays,
                    "global_shifts": full_global_shifts,
                    "pair_matches": full_registration["pair_matches"],
                    "spanning_tree": full_registration["spanning_tree"],
                    "geometric_edges": full_registration["geometric_edges"],
                    "matching_edges": full_registration["matching_edges"],
                    "rejected_edges": full_registration["rejected_edges"],
                    "connected_components": full_registration["connected_components"],
                    "unreachable_scenes": full_registration["unreachable_scenes"],
                    "diagnostics": full_registration["diagnostics"],
                }
            else:
                registration = self.register_scenes(scene_data, overlaps)
        else:
            # 3. 配准（含连通性检查、匹配回退、详细日志）
            registration = self.register_scenes(scene_data, overlaps)

        # 将 transforms, bounds, nodata 传递到后续步骤
        registration["transforms"] = scene_data["transforms"]
        registration["bounds"] = scene_data["bounds"]
        registration["nodata_values"] = scene_data["nodata_values"]
        registration["crs"] = scene_data["crs"]

        n_scenes_processed = len(scene_data["arrays"])
        unreachable_scenes = registration.get("unreachable_scenes", [])
        registration_connected = len(unreachable_scenes) == 0

        # 4. 辐射归一化
        normalized = self.apply_radiometric_normalization(
            registration, overlaps,
        )

        # 检查哪些方法失败了
        method_arrays = {k: v for k, v in normalized.items()
                         if isinstance(v, list) and v and isinstance(v[0], np.ndarray)}
        for method_name in self.config.ablation_methods:
            if method_name not in method_arrays:
                failed_methods.append(method_name)

        output_dir = os.path.join(self.output_root, self.config.experiment_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save per-scene normalized GeoTIFFs
        for method, arrays_list in normalized.items():
            if not isinstance(arrays_list, list):
                continue
            method_dir = os.path.join(output_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            for i, arr in enumerate(arrays_list):
                if not isinstance(arr, np.ndarray):
                    continue
                if i >= len(scene_data["scene_ids"]):
                    skipped_outputs.append(f"{method}_scene_{i}_out_of_range")
                    continue
                scene_id = scene_data["scene_ids"][i]
                out_path = os.path.join(method_dir, f"{scene_id}_{method}.tif")
                if not self.dry_run:
                    write_geotiff(
                        out_path, arr,
                        scene_data["transforms"][i],
                        scene_data["crs"],
                        nodata=scene_data["nodata_values"][i],
                    )

        # 5. 生成镶嵌
        mosaics = self.compute_mosaics(
            method_arrays,
            scene_data["transforms"],
            scene_data["crs"],
            scene_data["nodata_values"],
            output_dir,
        )

        # 6. 计算指标
        metrics_dir = os.path.join(output_dir, "metrics")
        metrics = self.evaluate_metrics(
            normalized,
            overlaps,
            scene_data["nodata_values"],
            output_dir=metrics_dir,
        )

        # 7. 计算光谱保持指标（SAM, spectral RMSE, correlation）
        spectral = {}
        if self.config.enable_spectral_metrics:
            spectral_dir = os.path.join(output_dir, "spectral")
            original_arrays = normalized.get("original", None)
            if original_arrays is not None:
                spectral = self.evaluate_spectral(
                    normalized,
                    original_arrays,
                    scene_data["nodata_values"],
                    scene_data["band_names"],
                    overlaps,
                    output_dir=spectral_dir,
                )
        else:
            logger.info("光谱指标已禁用 (enable_spectral_metrics=False)")

        # 8. 数据质量检查
        quality = self.check_data_quality(normalized, scene_data)

        total_elapsed = time.time() - total_t0

        # 确定 pipeline_status - 使用请求的方法列表
        requested_methods = self._requested_normalization_methods()
        failed_methods = [
            m for m in requested_methods
            if m not in method_arrays
        ]
        quality_failed_methods = [
            m for m in requested_methods
            if quality.get(m, {}).get("status") == "fail"
        ]
        
        if registration_connected and not failed_methods and not quality_failed_methods:
            pipeline_status = "success"
        else:
            pipeline_status = "failed"
            if failed_methods:
                logger.warning("失败的方法: %s", failed_methods)
            if quality_failed_methods:
                logger.warning("质量检查失败的方法: %s", quality_failed_methods)

        logger.info("=" * 60)
        logger.info("管线完成: 总耗时 %.1fs, 状态=%s", total_elapsed, pipeline_status)
        logger.info("=" * 60)

        return {
            "config": self.config,
            "scene_data": scene_data,
            "overlaps": overlaps,
            "registration": registration,
            "normalized": normalized,
            "mosaics": mosaics,
            "metrics": metrics,
            "spectral": spectral,
            "quality": quality,
            "elapsed_total": total_elapsed,
            "pipeline_status": pipeline_status,
            "registration_connected": registration_connected,
            "n_scenes_requested": n_scenes_requested,
            "n_scenes_processed": n_scenes_processed,
            "unreachable_scenes": unreachable_scenes,
            "failed_methods": failed_methods,
            "skipped_outputs": skipped_outputs,
        }


# ===========================================================================
# 命令行入口
# ===========================================================================

def main():
    """命令行入口函数。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="BAGRN-VOLRN 多景多波段辐射归一化管线"
    )
    parser.add_argument(
        "config", type=str,
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印执行计划，不实际处理",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="冒烟测试模式（使用小区域裁剪）",
    )
    parser.add_argument(
        "--output-root", type=str, default=None,
        help="输出根目录（覆盖配置文件中的值）",
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=None,
        help="指定辐射归一化方法列表",
    )

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 加载配置
    from src.experiment_config import load_config
    config = load_config(args.config)

    # 命令行覆盖
    if args.dry_run:
        config.dry_run = True
    if args.smoke:
        config.smoke = True
    if args.output_root:
        config.output_root = args.output_root

    # 执行管线
    pipe = MultibandPipeline(config)
    results = pipe.run()

    # 输出摘要
    if not args.dry_run and results.get("metrics"):
        print("\n" + "=" * 60)
        print("指标摘要:")
        print("=" * 60)
        for method, m in results["metrics"].items():
            print(f"  {method}:")
            print(f"    ADM={m.get('adm', 0):.6f}  ADSD={m.get('adsd', 0):.6f}")
            print(f"    CD={m.get('cd', 0):.6f}  GL={m.get('gl', 0):.6f}")
            print(f"    RDOA={m.get('rdoa', 0):.6f}  Ave={m.get('ave', 0):.6f}")


if __name__ == "__main__":
    main()
