"""
实验配置模块

提供 BAGRN-VOLRN 实验的统一配置管理：
  - 数据类 ExperimentConfig 集中定义所有实验参数
  - YAML 格式的加载与保存
  - 配置验证（必填字段、波段一致性、文件存在性等）
  - 自动检测公共波段集合
  - 支持多分辨率策略（严格匹配 / 重采样到配准网格）
"""

import copy
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


# ---------------------------------------------------------------------------
# 默认参数值
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRATION_PARAMS: Dict[str, Any] = {
    "global_block_size": 512,
    "global_confidence_threshold": 0.50,
    "max_global_shift": 40.0,
    "robust_mad_scale": 3.0,
    "robust_residual_floor": 0.75,
    "robust_min_inliers": 5,
    "robust_min_inlier_ratio": 0.35,
    "global_refine_block_size": 384,
    "global_refine_max_iterations": 2,
    "global_refine_stop_magnitude": 0.15,
    "global_refine_max_correction": 5.0,
    "enable_local_refinement": True,
    "local_block_size": 256,
    "local_confidence_threshold": 0.60,
    "local_max_residual_shift": 3.0,
    "local_search_max_shift": 12.0,
    "local_outlier_mad_scale": 3.0,
    "local_hard_max_component": 8.0,
    "local_neighbor_k": 8,
    "local_min_controls": 12,
    "local_min_spatial_groups": 3,
    "local_max_component": 2.5,
    "local_hull_buffer": 128,
    "local_cv_min_rmse_improvement": 0.10,
    "local_cv_min_p95_improvement": 0.15,
    "local_smoothing_candidates": [0.01, 0.05, 0.1, 0.5, 1.0],
    "validation_block_size": 384,
    "validation_block_size_candidates": [384, 256, 192],
    "validation_step": 256,
    "validation_offset_row": 128,
    "validation_offset_col": 128,
    "validation_min_distance_from_training": 256,
    "enable_spatial_holdout": False,
    "holdout_fraction": 0.20,
    "holdout_seed": 42,
    "holdout_block_size": 512,
    "min_holdout_cells": 2,
    "holdout_buffer_pixels": 0,
    "validation_required_candidate_count": 10,
    "validation_confidence_threshold": 0.45,
    "validation_max_residual_shift": 3.0,
    "final_min_blocks": 5,
    "pass_min_mean_confidence": 0.50,
    "pass_max_median": 0.35,
    "pass_max_rmse": 0.60,
    "pass_max_p95": 1.00,
    "warn_min_mean_confidence": 0.45,
    "warn_max_median": 0.50,
    "warn_max_rmse": 0.75,
    "warn_max_p95": 1.25,
    "required_quality": "pass",
}


_DEFAULT_VOLRN_PARAMS: Dict[str, Any] = {
    "block_size": 400,
    "lambda": 0.5,
    "rho": 1.0,
    "max_iter": 200,
    "tol": 1e-4,
}

_DEFAULT_SENSITIVITY: Dict[str, Any] = {
    "block_size_candidates": [200, 400, 600],
    "lambda_candidates": [0.1, 0.3, 0.5, 0.7, 0.9],
    "rho_candidates": [0.5, 1.0, 2.0],
}

_DEFAULT_ABLATION_METHODS: List[str] = [
    "bagrn",
    "volrn_only",
    "bagrn_volrn",
    "histogram_matching",
    "moment_matching",
    "wallis",
]

_DEFAULT_FEATHER_WIDTHS: List[int] = [50, 100, 200]

# 合法的多分辨率策略
VALID_COMMON_BANDS_STRATEGIES = ("strict", "resample_to_registration_grid")

# 合法的镶嵌模式
VALID_MOSAIC_MODES = ("weighted", "source_selection", "narrow_feather")

# 镶嵌模式别名映射（旧名称 → 标准名称）
MOSAIC_MODE_ALIASES = {"feather": "narrow_feather", "hardcut": "source_selection"}


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class SceneConfig:
    """单景影像的配置。"""

    id: str
    datetime: str = ""
    sensor: str = ""
    bands: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """
    实验配置的顶层数据类。

    所有实验相关参数集中于此，便于序列化、校验和复现。
    """

    # ---- 基本信息 ----
    experiment_name: str = "default_experiment"
    output_root: str = "data/output"

    # ---- 波段设置 ----
    registration_band: str = "B14"
    control_scene: Optional[str] = "auto"
    selected_bands: List[str] = field(default_factory=list)
    rgb_bands: List[str] = field(default_factory=list)
    false_color_bands: List[str] = field(default_factory=list)

    # ---- 影像列表 ----
    scenes: List[Dict[str, Any]] = field(default_factory=list)

    # ---- 镶嵌参数 ----
    mosaic_modes: List[str] = field(default_factory=lambda: ["weighted", "source_selection", "narrow_feather"])
    feather_widths: List[int] = field(default_factory=lambda: [100])

    # ---- VOLRN 参数 ----
    registration_params: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(_DEFAULT_REGISTRATION_PARAMS))

    volrn_params: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(_DEFAULT_VOLRN_PARAMS))

    # ---- 消融 / 敏感性分析 ----
    ablation_methods: List[str] = field(default_factory=lambda: copy.deepcopy(_DEFAULT_ABLATION_METHODS))
    sensitivity: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(_DEFAULT_SENSITIVITY))
    scale_scene_counts: List[int] = field(default_factory=lambda: [2, 4, 6])

    # ---- 多分辨率策略 ----
    common_bands_strategy: str = "strict"

    # ---- 冒烟测试参数 ----
    smoke_crop_size: int = 512
    smoke_scene_count: int = 5

    # ---- 运行控制 ----
    seed: int = 42
    dry_run: bool = False
    smoke: bool = False
    enable_spectral_metrics: bool = True


# ---------------------------------------------------------------------------
# YAML 加载 / 保存
# ---------------------------------------------------------------------------

def load_config(path: str) -> ExperimentConfig:
    """
    从 YAML 文件加载实验配置。

    参数
    ----------
    path : str
        YAML 配置文件路径。

    返回
    -------
    ExperimentConfig
        解析后的配置对象。

    异常
    ------
    FileNotFoundError
        文件不存在。
    yaml.YAMLError
        YAML 格式错误。
    ValueError
        YAML 内容与 ExperimentConfig 结构不兼容。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raw = {}

    return _dict_to_config(raw)


def save_config(config: ExperimentConfig, path: str) -> str:
    """
    将实验配置保存为 YAML 文件。

    参数
    ----------
    config : ExperimentConfig
        要保存的配置对象。
    path : str
        输出 YAML 文件路径。

    返回
    -------
    str
        写入成功的文件路径。
    """
    d = asdict(config)
    # 清理默认工厂产生的空值，保持 YAML 简洁
    d = _clean_for_yaml(d)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(
            d,
            fh,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
    return path


# ---------------------------------------------------------------------------
# 配置验证
# ---------------------------------------------------------------------------

def validate_config(config: ExperimentConfig, skip_file_check: bool = False) -> List[str]:
    """
    校验实验配置的合法性。

    检查内容包括：
      1. 必填字段非空
      2. 波段名称格式合理
      3. scenes 列表非空且每景包含必要字段
      4. 每景的波段路径格式
      5. common_bands_strategy 合法
      6. volrn_params 中的参数范围
      7. rgb_bands / false_color_bands 中的波段应在 selected_bands 内
      8. registration_band 应在 selected_bands 内
      9. control_scene 若非 auto 则必须在 scenes 的 id 列表中

    返回
    ------
    list[str]
        校验错误列表。空列表表示全部通过。
    """
    errors: List[str] = []

    # ---- registration_params ----
    if hasattr(config, 'registration_params'):
        rq = config.registration_params.get("required_quality", "pass")
        if rq not in ("pass", "warn", "fail"):
            errors.append(f"registration_params.required_quality 必须是 'pass'/'warn'/'fail'，当前: '{rq}'")
        rp = config.registration_params
        holdout_fraction = float(rp.get("holdout_fraction", 0.20))
        if not 0.0 < holdout_fraction < 0.5:
            errors.append("registration_params.holdout_fraction 必须在 (0, 0.5) 内")
        sizes = rp.get("validation_block_size_candidates", [384, 256, 192])
        if (not isinstance(sizes, list) or not sizes
                or any(int(size) <= 0 for size in sizes)
                or len({int(size) for size in sizes}) != len(sizes)):
            errors.append(
                "registration_params.validation_block_size_candidates 必须是非空正整数且不重复"
            )
        for key in ("local_search_max_shift", "local_hard_max_component"):
            if float(rp.get(key, 0.0)) <= 0:
                errors.append(f"registration_params.{key} 必须大于 0")

    # ---- 基本字段 ----
    if not config.experiment_name.strip():
        errors.append("experiment_name 不能为空")
    if not config.output_root.strip():
        errors.append("output_root 不能为空")

    # ---- 波段 ----
    if not config.selected_bands:
        errors.append("selected_bands 列表不能为空")
    if config.registration_band and config.selected_bands:
        if config.registration_band not in config.selected_bands:
            errors.append(
                f"registration_band '{config.registration_band}' 不在 selected_bands 中"
            )

    # ---- rgb_bands / false_color_bands ----
    if config.rgb_bands and config.selected_bands:
        missing_rgb = [b for b in config.rgb_bands if b not in config.selected_bands]
        if missing_rgb:
            errors.append(f"rgb_bands 中有波段不在 selected_bands: {missing_rgb}")
    if config.false_color_bands and config.selected_bands:
        missing_fc = [b for b in config.false_color_bands if b not in config.selected_bands]
        if missing_fc:
            errors.append(
                f"false_color_bands 中有波段不在 selected_bands: {missing_fc}"
            )
    if config.rgb_bands and len(config.rgb_bands) != 3:
        errors.append(f"rgb_bands 必须恰好 3 个波段，当前: {config.rgb_bands}")
    if config.false_color_bands and len(config.false_color_bands) != 3:
        errors.append(f"false_color_bands 必须恰好 3 个波段，当前: {config.false_color_bands}")

    # ---- scenes ----
    if not config.scenes:
        errors.append("scenes 列表不能为空")
    else:
        scene_ids = set()
        for idx, scene in enumerate(config.scenes):
            prefix = f"scenes[{idx}]"
            if not isinstance(scene, dict):
                errors.append(f"{prefix}: 每景必须是字典类型")
                continue
            if "id" not in scene:
                errors.append(f"{prefix}: 缺少 'id' 字段")
            else:
                sid = scene["id"]
                if not sid or not str(sid).strip():
                    errors.append(f"{prefix}: scene id 不能为空")
                if sid in scene_ids:
                    errors.append(f"{prefix}: id '{sid}' 重复")
                scene_ids.add(sid)
            if "bands" not in scene:
                errors.append(f"{prefix}: 缺少 'bands' 字段")
            elif not isinstance(scene["bands"], dict):
                errors.append(f"{prefix}: 'bands' 必须是字典（波段名 → 路径）")
            else:
                for band_name, band_path in scene["bands"].items():
                    if not isinstance(band_path, str) or not band_path.strip():
                        errors.append(
                            f"{prefix}.bands.{band_name}: 路径不能为空"
                        )

        # control_scene 校验
        if config.control_scene and config.control_scene != "auto":
            scene_ids_valid = {s.get("id", "") for s in config.scenes if isinstance(s, dict)}
            if config.control_scene not in scene_ids_valid:
                errors.append(
                    f"control_scene '{config.control_scene}' 不在 scenes 的 id 列表中"
                )

        # File existence checks
        if not skip_file_check:
            for idx, scene in enumerate(config.scenes):
                if not isinstance(scene, dict):
                    continue
                sid = scene.get("id", f"scene_{idx}")
                bands = scene.get("bands", {})
                for band_name, band_path in bands.items():
                    if not isinstance(band_path, str) or not band_path.strip():
                        continue
                    full_path = os.path.abspath(band_path)
                    if not os.path.isfile(band_path):
                        errors.append(
                            f"scenes[{idx}] (id={sid}).bands.{band_name}: "
                            f"文件不存在: '{band_path}' (绝对路径: {full_path})"
                        )

    # ---- 多分辨率策略 ----
    if config.common_bands_strategy not in VALID_COMMON_BANDS_STRATEGIES:
        errors.append(
            f"common_bands_strategy 不合法: '{config.common_bands_strategy}'，"
            f"可选值: {VALID_COMMON_BANDS_STRATEGIES}"
        )

    # ---- volrn_params ----
    vp = config.volrn_params
    if not isinstance(vp, dict):
        errors.append("volrn_params 必须是字典")
    else:
        bs = vp.get("block_size", 400)
        if not isinstance(bs, int) or bs <= 0:
            errors.append(f"volrn_params.block_size 必须为正整数，当前值: {bs}")
        lam = vp.get("lambda", 0.5)
        if not isinstance(lam, (int, float)) or lam < 0:
            errors.append(f"volrn_params.lambda 必须为非负数，当前值: {lam}")
        rho = vp.get("rho", 1.0)
        if not isinstance(rho, (int, float)) or rho <= 0:
            errors.append(f"volrn_params.rho 必须为正数，当前值: {rho}")
        max_iter = vp.get("max_iter", 200)
        if not isinstance(max_iter, int) or max_iter <= 0:
            errors.append(f"volrn_params.max_iter 必须为正整数，当前值: {max_iter}")
        tol = vp.get("tol", 1e-4)
        if not isinstance(tol, (int, float)) or tol <= 0:
            errors.append(f"volrn_params.tol 必须为正数，当前值: {tol}")

    # ---- 冒烟测试参数 ----
    if config.smoke_crop_size <= 0:
        errors.append(f"smoke_crop_size 必须为正整数，当前值: {config.smoke_crop_size}")
    if config.smoke_scene_count <= 0:
        errors.append(f"smoke_scene_count 必须为正整数，当前值: {config.smoke_scene_count}")

    # ---- 镶嵌参数 ----
    if not config.mosaic_modes:
        errors.append("mosaic_modes 列表不能为空")
    for mode in config.mosaic_modes:
        if mode not in VALID_MOSAIC_MODES:
            errors.append(f"mosaic_modes 中有不合法的模式: '{mode}'，可选值: {VALID_MOSAIC_MODES}")
    if not config.feather_widths:
        errors.append("feather_widths 列表不能为空")
    for fw in config.feather_widths:
        if not isinstance(fw, int) or fw <= 0:
            errors.append(f"feather_widths 中的值必须为正整数，当前值: {fw}")

    # ---- scale_scene_counts ----
    for sc in config.scale_scene_counts:
        if not isinstance(sc, int) or sc <= 0:
            errors.append(f"scale_scene_counts 中的值必须为正整数，当前值: {sc}")

    return errors


# ---------------------------------------------------------------------------
# 公共波段检测
# ---------------------------------------------------------------------------

def get_common_bands(config: ExperimentConfig) -> List[str]:
    """
    自动检测所有景影像共有的波段名称。

    根据 common_bands_strategy 决定检测策略：
      - 'strict': 仅返回所有景都拥有的波段名（交集）
      - 'resample_to_registration_grid': 在 strict 基础上，
        若 registration_band 在某一景中缺失则报错，其余波段取交集

    返回
    ------
    list[str]
        排序后的公共波段列表。
    """
    if not config.scenes:
        return list(config.selected_bands)

    # 收集每景的波段名集合
    scene_band_sets: List[set] = []
    for scene in config.scenes:
        bands = scene.get("bands", {})
        scene_band_sets.append(set(bands.keys()))

    # 严格交集
    common = scene_band_sets[0]
    for s in scene_band_sets[1:]:
        common = common & s

    if config.common_bands_strategy == "strict":
        # 直接返回交集，并按 selected_bands 的顺序排列
        if config.selected_bands:
            ordered = [b for b in config.selected_bands if b in common]
        else:
            ordered = sorted(common)
        return ordered

    elif config.common_bands_strategy == "resample_to_registration_grid":
        # 检查配准波段是否在每一景中都存在
        reg_band = config.registration_band
        for idx, s in enumerate(scene_band_sets):
            if reg_band not in s:
                raise ValueError(
                    f"第 {idx} 景影像缺少配准波段 '{reg_band}'，"
                    f"在 'resample_to_registration_grid' 策略下无法重采样"
                )
        if config.selected_bands:
            ordered = [b for b in config.selected_bands if b in common]
        else:
            ordered = sorted(common)
        return ordered

    else:
        raise ValueError(f"未知的 common_bands_strategy: {config.common_bands_strategy}")


# ---------------------------------------------------------------------------
# 波段路径解析
# ---------------------------------------------------------------------------

def resolve_band_path(scene: Dict[str, Any], band_name: str) -> Optional[str]:
    """
    从场景字典中解析指定波段的文件路径。

    参数
    ----------
    scene : dict
        单景配置字典，需包含 'bands' 键。
    band_name : str
        波段名称（如 'B01'、'B14'）。

    返回
    ------
    str or None
        波段文件路径，若不存在则返回 None。
    """
    bands = scene.get("bands", {})
    path = bands.get(band_name)
    if path is None:
        return None
    return str(path).strip() if path else None


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------


def _parse_bool(value) -> bool:
    """
    Robust boolean parser for configuration values.
    
    Accepts: bool, str ("true"/"false"/"yes"/"no"/"1"/"0"), int (0/1)
    Raises ValueError for invalid inputs.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower in {"true", "yes", "1", "on"}:
            return True
        if value_lower in {"false", "no", "0", "off"}:
            return False
        raise ValueError(f"Cannot parse '{value}' as boolean")
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"Cannot parse {value} as boolean (must be 0 or 1)")
    raise ValueError(f"Cannot parse {type(value).__name__} as boolean")


def _merge_registration_params(raw: Any) -> Dict[str, Any]:
    """Merge user-provided registration_params with defaults."""
    merged = copy.deepcopy(_DEFAULT_REGISTRATION_PARAMS)
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k in merged:
                merged[k] = v
            else:
                print(f"[experiment_config] 警告: 未知 registration_params 字段 '{k}'", file=sys.stderr)
    return merged


def _dict_to_config(raw: Dict[str, Any]) -> ExperimentConfig:
    """
    将原始字典转换为 ExperimentConfig 数据类。

    对未知字段发出警告但不中断（向后兼容）。
    """
    cfg = ExperimentConfig()

    # 已知字段映射（字段名 → 类型转换）
    known_fields = {
        "experiment_name": str,
        "output_root": str,
        "registration_band": str,
        "control_scene": lambda v: v if v is None else str(v),
        "selected_bands": lambda v: list(v) if isinstance(v, list) else [str(v)],
        "rgb_bands": lambda v: list(v) if isinstance(v, list) else [str(v)],
        "false_color_bands": lambda v: list(v) if isinstance(v, list) else [str(v)],
        "scenes": lambda v: list(v) if isinstance(v, list) else [],
        "mosaic_modes": lambda v: list(v) if isinstance(v, list) else [],
        "feather_widths": lambda v: [int(x) for x in v] if isinstance(v, list) else [],
        "volrn_params": lambda v: dict(v) if isinstance(v, dict) else copy.deepcopy(_DEFAULT_VOLRN_PARAMS),
        "registration_params": _merge_registration_params,
        "ablation_methods": lambda v: list(v) if isinstance(v, list) else [],
        "sensitivity": lambda v: dict(v) if isinstance(v, dict) else copy.deepcopy(_DEFAULT_SENSITIVITY),
        "scale_scene_counts": lambda v: [int(x) for x in v] if isinstance(v, list) else [],
        "smoke_crop_size": int,
        "smoke_scene_count": int,
        "common_bands_strategy": str,
        "seed": int,
        "dry_run": _parse_bool,
        "smoke": _parse_bool,
        "enable_spectral_metrics": _parse_bool,
    }

    for key, value in raw.items():
        if key in known_fields:
            try:
                setattr(cfg, key, known_fields[key](value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"字段 '{key}' 的值 '{value}' 转换失败: {exc}"
                ) from exc
        else:
            print(f"[experiment_config] 警告: 未知字段 '{key}'，已忽略", file=sys.stderr)

    # Convert aliases in mosaic_modes
    raw_modes = getattr(cfg, 'mosaic_modes', [])
    converted = []
    for m in raw_modes:
        m_clean = m.strip()
        if m_clean in MOSAIC_MODE_ALIASES:
            print(f"[experiment_config] 警告: 镶嵌模式 '{m_clean}' 是别名，已转换为 '{MOSAIC_MODE_ALIASES[m_clean]}'", file=sys.stderr)
            converted.append(MOSAIC_MODE_ALIASES[m_clean])
        else:
            converted.append(m_clean)
    cfg.mosaic_modes = converted

    return cfg


def _clean_for_yaml(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理字典中可能产生的空值或无意义默认值，保持 YAML 输出简洁。
    """
    cleaned = {}
    for k, v in d.items():
        if isinstance(v, dict):
            cleaned[k] = _clean_for_yaml(v)
        elif isinstance(v, list) and len(v) == 0:
            # 保留空列表，因为某些场景下空列表有语义
            cleaned[k] = v
        else:
            cleaned[k] = v
    return cleaned


# ---------------------------------------------------------------------------
# 便捷构造函数（测试 / 演示用）
# ---------------------------------------------------------------------------

def make_test_config(
    scene_ids: Optional[List[str]] = None,
    n_bands: int = 4,
) -> ExperimentConfig:
    """
    快速构造一个用于单元测试或冒烟测试的最小配置。

    参数
    ----------
    scene_ids : list[str], optional
        每景的 id。默认为 ['scene_0', 'scene_1']。
    n_bands : int
        每景的波段数，默认 4。

    返回
    ------
    ExperimentConfig
    """
    if scene_ids is None:
        scene_ids = ["scene_0", "scene_1"]

    band_names = [f"B{str(i+1).zfill(2)}" for i in range(n_bands)]

    scenes = []
    for sid in scene_ids:
        bands_dict = {b: f"data/input/{sid}/{b}.tif" for b in band_names}
        scenes.append({
            "id": sid,
            "datetime": "2024-01-01",
            "sensor": "TEST",
            "bands": bands_dict,
        })

    return ExperimentConfig(
        experiment_name="test_experiment",
        output_root="data/output_test",
        registration_band=band_names[-1],
        control_scene="auto",
        selected_bands=band_names,
        rgb_bands=band_names[:3] if n_bands >= 3 else band_names,
        false_color_bands=band_names[:3] if n_bands >= 3 else band_names,
        scenes=scenes,
        common_bands_strategy="strict",
    )


# ---------------------------------------------------------------------------
# 直接运行时的简单演示
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = make_test_config()
    errors = validate_config(cfg)
    if errors:
        print("校验失败:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("配置校验通过")

    common = get_common_bands(cfg)
    print(f"公共波段: {common}")

    # 测试 YAML 保存与加载
    demo_path = "data/output_test/demo_config.yaml"
    save_config(cfg, demo_path)
    print(f"配置已保存至 {demo_path}")

    loaded = load_config(demo_path)
    print(f"加载成功: {loaded.experiment_name}, {len(loaded.scenes)} 景影像")
