"""
实验运行器 — 统一命令行入口

提供多种实验类型的调度、参数管理和结果汇总：
  - multiband:   完整多波段管线
  - ablation:    消融实验
  - sensitivity: 参数敏感性分析
  - scale:       多场景规模实验
  - diagnostics: VOLRN 系数 / RBF 位移场诊断
  - seam:        接缝评估
  - spectral:    光谱保真度验证

用法示例
--------
python -m src.experiment_runner --config configs/dz01_multiband.yaml multiband
python -m src.experiment_runner --config configs/dz01_multiband.yaml ablation --dry-run
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from src.experiment_config import (
    ExperimentConfig,
    load_config,
    validate_config,
    get_common_bands,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 环境信息收集
# ---------------------------------------------------------------------------

def _collect_environment_info(argv: List[str]) -> Dict[str, Any]:
    """收集运行环境信息，用于复现。"""
    info: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(argv),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
    }

    # 第三方库版本
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except Exception:
        info["numpy_version"] = "unknown"
    try:
        import scipy
        info["scipy_version"] = scipy.__version__
    except Exception:
        info["scipy_version"] = "unknown"
    try:
        import rasterio
        info["rasterio_version"] = rasterio.__version__
    except Exception:
        info["rasterio_version"] = "unknown"
    try:
        import skimage
        info["skimage_version"] = skimage.__version__
    except Exception:
        info["skimage_version"] = "unknown"

    # 可用内存（Linux / Windows）
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["available_memory_gb"] = round(mem.available / (1024 ** 3), 2)
        info["total_memory_gb"] = round(mem.total / (1024 ** 3), 2)
    except Exception:
        info["available_memory_gb"] = "unknown"
        info["total_memory_gb"] = "unknown"

    # Git 提交（如有）
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        info["git_commit"] = result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        info["git_commit"] = "unknown"

    return info


def _save_environment(output_dir: str, env_info: Dict[str, Any]) -> str:
    """将环境信息写入 environment.json。"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "environment.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2, ensure_ascii=False, default=str)
    return path


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _save_summary(output_dir: str, summary: Dict[str, Any]) -> str:
    """保存实验摘要到 summary.json。"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return path


def _setup_logging(log_dir: str, experiment_name: str) -> None:
    """配置日志：同时输出到控制台和文件。"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")

    root_logger = logging.getLogger()
    # Remove existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.INFO)

    # 控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(console_handler)

    # 文件
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)


def _parse_methods(methods_str: Optional[str]) -> Optional[List[str]]:
    """解析逗号分隔的方法列表。"""
    if methods_str is None:
        return None
    return [m.strip() for m in methods_str.split(",") if m.strip()]


def _csv_writerow(csv_path: str, header: List[str], rows: List[List[Any]]) -> None:
    """写入 CSV 文件（追加模式下先检查文件是否存在）。"""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _checkpoint_path(output_dir: str, step_name: str) -> str:
    """生成检查点文件路径。"""
    return os.path.join(output_dir, ".checkpoints", f"{step_name}.done")


def _is_step_done(output_dir: str, step_name: str) -> bool:
    """检查某步骤是否已完成。"""
    return os.path.exists(_checkpoint_path(output_dir, step_name))


def _mark_step_done(output_dir: str, step_name: str) -> None:
    """标记某步骤已完成。"""
    cp_dir = os.path.join(output_dir, ".checkpoints")
    os.makedirs(cp_dir, exist_ok=True)
    with open(_checkpoint_path(output_dir, step_name), "w") as f:
        f.write(f"completed at {datetime.now().isoformat()}\n")


# ===========================================================================
# 实验运行器
# ===========================================================================

def run_multiband(config: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
    """
    完整多波段管线实验。

    步骤：加载 → 重叠检测 → 配准 → 辐射归一化 → 镶嵌 → 指标。
    """
    from src.multiband_pipeline import MultibandPipeline
    from src.visualization import generate_rgb_quicklooks

    # Override methods from CLI if provided
    if args.methods:
        config.ablation_methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    output_dir = os.path.join(config.output_root, config.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("实验类型: multiband")
    logger.info("输出目录: %s", output_dir)
    logger.info("=" * 60)

    summary: Dict[str, Any] = {"experiment_type": "multiband", "steps": {}}
    results: Dict[str, Any] = {}

    # Step 1: 管线执行
    step = "pipeline"
    if args.resume and _is_step_done(output_dir, step):
        logger.info("跳过已完成步骤: %s", step)
    else:
        t0 = time.time()
        pipe = MultibandPipeline(config)
        results = pipe.run()
        summary["steps"][step] = {
            "elapsed_sec": time.time() - t0,
            "status": "completed",
        }
        _mark_step_done(output_dir, step)

    # Step 2: 生成 quicklook
    step = "quicklooks"
    if args.resume and _is_step_done(output_dir, step):
        logger.info("跳过已完成步骤: %s", step)
    elif results:
        try:
            ql_dir = os.path.join(output_dir, "quicklooks")
            os.makedirs(ql_dir, exist_ok=True)
            normalized = results.get("normalized", {})
            scene_data = results.get("scene_data", {})
            transforms = scene_data.get("transforms", [])
            nodata_values = scene_data.get("nodata_values", [])
            band_names = scene_data.get("band_names", [])

            # 将波段名列表转为索引元组
            rgb_idx = tuple(band_names.index(b) for b in config.rgb_bands if b in band_names)
            fc_idx = tuple(band_names.index(b) for b in config.false_color_bands if b in band_names)

            if normalized and transforms and rgb_idx:
                from src.visualization import compute_joint_stretch_params

                # Collect all arrays across methods for joint stretch
                all_arrays = []
                for method, arrays_list in normalized.items():
                    all_arrays.extend(arrays_list)

                # Compute shared stretch params
                rgb_stretch = compute_joint_stretch_params(all_arrays, rgb_idx)
                fc_bands = fc_idx if fc_idx else rgb_idx
                fc_stretch = compute_joint_stretch_params(all_arrays, fc_bands)

                # Save stretch parameters
                import json
                stretch_info = {
                    "rgb_bands": list(rgb_idx),
                    "rgb_stretch": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in rgb_stretch.items()},
                    "fc_bands": list(fc_bands),
                    "fc_stretch": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in fc_stretch.items()},
                }
                with open(os.path.join(ql_dir, "stretch_parameters.json"), "w") as f:
                    json.dump(stretch_info, f, indent=2)

                # Apply same stretch to each method
                for method, arrays_list in normalized.items():
                    arrays_dict = {i: arr for i, arr in enumerate(arrays_list)}
                    generate_rgb_quicklooks(
                        arrays_dict, transforms, nodata_values,
                        rgb_bands=rgb_idx,
                        false_color_bands=fc_bands,
                        output_dir=ql_dir,
                        method_name=method,
                        shared_rgb_stretch=rgb_stretch,
                        shared_fc_stretch=fc_stretch,
                    )
                summary["steps"][step] = {"status": "completed"}
            else:
                summary["steps"][step] = {"status": "skipped", "reason": "no normalized data"}
            _mark_step_done(output_dir, step)
        except Exception as exc:
            logger.error("quicklook 生成失败: %s", exc)
            summary["steps"][step] = {"status": "failed", "error": str(exc)}
    else:
        summary["steps"][step] = {"status": "skipped", "reason": "no results from pipeline"}

    # 汇总指标
    summary["metrics"] = {}
    if results.get("metrics"):
        for method, m in results["metrics"].items():
            summary["metrics"][method] = {
                k: v for k, v in m.items() if k in ("adm", "adsd", "cd", "gl", "rdoa", "ave")
            }

    # 补充管线状态字段
    summary["pipeline_status"] = results.get("pipeline_status", "unknown")
    summary["registration_connected"] = results.get("registration_connected", False)
    summary["n_scenes_requested"] = results.get("n_scenes_requested", 0)
    summary["n_scenes_processed"] = results.get("n_scenes_processed", 0)
    summary["unreachable_scenes"] = results.get("unreachable_scenes", [])
    summary["failed_methods"] = results.get("failed_methods", [])
    summary["skipped_outputs"] = results.get("skipped_outputs", [])

    # 汇总光谱指标
    spectral = results.get("spectral", {})
    summary["spectral"] = {}
    for method, sp in spectral.items():
        if isinstance(sp, dict) and "aggregate" in sp:
            agg = sp["aggregate"]
            summary["spectral"][method] = {
                "sam_mean": agg.get("sam", {}).get("mean", float("nan")),
                "sam_median": agg.get("sam", {}).get("median", float("nan")),
                "sam_p90": agg.get("sam", {}).get("p90", float("nan")),
                "sam_p95": agg.get("sam", {}).get("p95", float("nan")),
                "spectral_rmse_mean": agg.get("spectral_rmse", {}).get("mean", float("nan")),
                "relative_spectral_rmse_mean": agg.get("relative_spectral_rmse", {}).get("mean", float("nan")),
            }

    # 汇总数据质量
    quality = results.get("quality", {})
    summary["quality"] = {}
    for method, q in quality.items():
        summary["quality"][method] = {
            "status": q.get("status", "unknown"),
            "issues": q.get("issues", []),
        }

    return summary


def run_ablation(config: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
    """
    消融实验。

    对比方法：original, BAGRN-only, VOLRN-only, BAGRN-VOLRN,
    以及 BAGRN-VOLRN 不使用系数插值（diagnostic）、
    BAGRN-VOLRN 使用最近邻插值（optional）。
    """
    from src.multiband_pipeline import MultibandPipeline

    output_dir = os.path.join(config.output_root, config.experiment_name, "ablation")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("实验类型: ablation")
    logger.info("输出目录: %s", output_dir)
    logger.info("=" * 60)

    # 消融方法列表
    ablation_methods = [
        "original",
        "bagrn",
        "volrn_only",
        "bagrn_volrn",
    ]
    # 可选：BAGRN-VOLRN 不使用系数插值（通过禁用 VOLRN 实现）
    # 可选：BAGRN-VOLRN 最近邻插值
    if args.methods:
        ablation_methods = _parse_methods(args.methods)

    # 覆盖配置中的方法列表
    config_copy = copy.deepcopy(config)
    config_copy.ablation_methods = ablation_methods

    summary: Dict[str, Any] = {
        "experiment_type": "ablation",
        "methods": ablation_methods,
        "metrics": {},
    }

    # 执行管线
    t0 = time.time()
    pipe = MultibandPipeline(config_copy)
    results = pipe.run()
    summary["elapsed_sec"] = time.time() - t0

    # 保存消融指标 CSV
    if results.get("metrics"):
        metrics_csv = os.path.join(output_dir, "ablation_metrics.csv")
        header = ["method", "adm", "adsd", "cd", "gl", "rdoa", "ave"]
        rows = []
        for method in ablation_methods:
            m = results["metrics"].get(method, {})
            rows.append([
                method,
                m.get("adm", float("nan")),
                m.get("adsd", float("nan")),
                m.get("cd", float("nan")),
                m.get("gl", float("nan")),
                m.get("rdoa", float("nan")),
                m.get("ave", float("nan")),
            ])
        _csv_writerow(metrics_csv, header, rows)
        summary["ablation_metrics_csv"] = metrics_csv
        summary["metrics"] = {
            method: {k: v for k, v in results["metrics"].get(method, {}).items()
                     if k in ("adm", "adsd", "cd", "gl", "rdoa", "ave")}
            for method in ablation_methods
        }

    # 保存运行时间 CSV
    runtime_csv = os.path.join(output_dir, "ablation_runtime.csv")
    runtime_header = ["method", "elapsed_sec"]
    runtime_rows = []
    if results.get("normalized"):
        # 粗略计时（管线整体时间分摊）
        per_method = summary["elapsed_sec"] / max(len(ablation_methods), 1)
        for method in ablation_methods:
            runtime_rows.append([method, round(per_method, 2)])
    _csv_writerow(runtime_csv, runtime_header, runtime_rows)
    summary["ablation_runtime_csv"] = runtime_csv

    return summary


def run_sensitivity(config: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
    """
    参数敏感性分析。

    单因子实验：固定两个参数，扫描第三个。
    - block_size 扫描：[200, 300, 400, 500, 600, 800]
    - lambda 扫描：[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    - rho 扫描：[0.5, 1.0, 2.0]
    """
    from src.multiband_pipeline import MultibandPipeline

    output_dir = os.path.join(config.output_root, config.experiment_name, "sensitivity")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("实验类型: sensitivity")
    logger.info("输出目录: %s", output_dir)
    logger.info("=" * 60)

    # 默认参数扫描范围
    sens_cfg = config.sensitivity
    block_sizes = sens_cfg.get("block_size_candidates", [200, 300, 400, 500, 600, 800])
    lambdas = sens_cfg.get("lambda_candidates", [0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    rhos = sens_cfg.get("rho_candidates", [0.5, 1.0, 2.0])

    # 基准参数
    base_params = copy.deepcopy(config.volrn_params)

    summary: Dict[str, Any] = {
        "experiment_type": "sensitivity",
        "block_sizes": block_sizes,
        "lambdas": lambdas,
        "rhos": rhos,
        "raw_results": [],
    }

    raw_csv = os.path.join(output_dir, "sensitivity_raw.csv")
    raw_header = ["sweep_param", "param_value", "adm", "adsd", "cd", "gl", "rdoa", "ave", "elapsed_sec"]

    def _run_single(param_name: str, param_value: float) -> Dict[str, Any]:
        """运行单次敏感性实验。"""
        cfg = copy.deepcopy(config)
        cfg.volrn_params = copy.deepcopy(base_params)
        cfg.volrn_params[param_name] = param_value
        cfg.ablation_methods = ["bagrn_volrn"]

        t0 = time.time()
        try:
            pipe = MultibandPipeline(cfg)
            results = pipe.run()
            elapsed = time.time() - t0

            metrics = {}
            if results.get("metrics") and "bagrn_volrn" in results["metrics"]:
                metrics = results["metrics"]["bagrn_volrn"]

            row = {
                "sweep_param": param_name,
                "param_value": param_value,
                "adm": metrics.get("adm", float("nan")),
                "adsd": metrics.get("adsd", float("nan")),
                "cd": metrics.get("cd", float("nan")),
                "gl": metrics.get("gl", float("nan")),
                "rdoa": metrics.get("rdoa", float("nan")),
                "ave": metrics.get("ave", float("nan")),
                "elapsed_sec": round(elapsed, 2),
            }

            _csv_writerow(raw_csv, raw_header, [[
                row["sweep_param"], row["param_value"],
                row["adm"], row["adsd"], row["cd"], row["gl"],
                row["rdoa"], row["ave"], row["elapsed_sec"],
            ]])

            return row
        except Exception as exc:
            logger.error("敏感性实验失败 [%s=%s]: %s", param_name, param_value, exc)
            return {
                "sweep_param": param_name, "param_value": param_value,
                "error": str(exc),
            }

    # 1. 扫描 block_size（固定 lambda, rho）
    logger.info("扫描 block_size: %s", block_sizes)
    for bs in block_sizes:
        logger.info("  block_size=%d", bs)
        row = _run_single("block_size", bs)
        summary["raw_results"].append(row)

    # 2. 扫描 lambda（固定 block_size, rho）
    logger.info("扫描 lambda: %s", lambdas)
    for lam in lambdas:
        logger.info("  lambda=%s", lam)
        row = _run_single("lambda", lam)
        summary["raw_results"].append(row)

    # 3. 扫描 rho（固定 block_size, lambda）
    logger.info("扫描 rho: %s", rhos)
    for rho in rhos:
        logger.info("  rho=%s", rho)
        row = _run_single("rho", rho)
        summary["raw_results"].append(row)

    # 寻找 Pareto 最优参数范围
    valid = [r for r in summary["raw_results"] if "error" not in r]
    if valid:
        # 按 Ave 排序找最优
        sorted_by_ave = sorted(valid, key=lambda x: x.get("ave", float("inf")))
        best = sorted_by_ave[0]
        summary["recommended"] = {
            "best_param": best["sweep_param"],
            "best_value": best["param_value"],
            "best_ave": best["ave"],
        }

        # 保存推荐参数
        recommended_path = os.path.join(output_dir, "recommended_parameters.json")
        with open(recommended_path, "w", encoding="utf-8") as f:
            json.dump(summary["recommended"], f, indent=2, ensure_ascii=False)

    # 保存汇总
    summary_csv = os.path.join(output_dir, "sensitivity_summary.csv")
    summary_header = ["sweep_param", "min_ave", "max_ave", "mean_ave", "best_value"]
    summary_rows = []
    for param_name in ["block_size", "lambda", "rho"]:
        param_results = [r for r in valid if r["sweep_param"] == param_name]
        if param_results:
            aves = [r["ave"] for r in param_results]
            best_r = min(param_results, key=lambda x: x["ave"])
            summary_rows.append([
                param_name,
                min(aves), max(aves), np.mean(aves),
                best_r["param_value"],
            ])
    _csv_writerow(summary_csv, summary_header, summary_rows)
    summary["sensitivity_summary_csv"] = summary_csv
    summary["sensitivity_raw_csv"] = raw_csv

    return summary


def run_scale(config: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
    """
    多场景规模实验。

    测试 N=4, 8, 12 景场景下的运行时间、内存占用和指标。
    """
    from src.multiband_pipeline import MultibandPipeline

    output_dir = os.path.join(config.output_root, config.experiment_name, "scale")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("实验类型: scale")
    logger.info("输出目录: %s", output_dir)
    logger.info("=" * 60)

    scene_counts = config.scale_scene_counts
    if args.scene_count:
        scene_counts = [int(x) for x in args.scene_count.split(",")]

    all_scenes = config.scenes
    total_scenes = len(all_scenes)

    summary: Dict[str, Any] = {
        "experiment_type": "scale",
        "scene_counts": scene_counts,
        "total_available_scenes": total_scenes,
        "results": [],
    }

    scale_csv = os.path.join(output_dir, "scale_results.csv")
    scale_header = ["n_scenes", "elapsed_sec", "peak_memory_gb", "adm", "adsd", "cd", "gl", "rdoa", "ave"]

    for n in scene_counts:
        if n > total_scenes:
            logger.warning("请求 %d 景，但仅有 %d 景，跳过", n, total_scenes)
            continue

        logger.info("测试规模: N=%d 景", n)

        # 截取前 n 景
        cfg = copy.deepcopy(config)
        cfg.scenes = all_scenes[:n]

        t0 = time.time()
        peak_mem = 0.0

        try:
            pipe = MultibandPipeline(cfg)
            results = pipe.run()
            elapsed = time.time() - t0

            # 尝试获取内存占用
            try:
                import psutil
                peak_mem = psutil.Process().memory_info().rss / (1024 ** 3)
            except Exception:
                peak_mem = 0.0

            metrics = {}
            if results.get("metrics") and "bagrn_volrn" in results["metrics"]:
                metrics = results["metrics"]["bagrn_volrn"]
            elif results.get("metrics"):
                # 取第一个方法的指标
                first_method = next(iter(results["metrics"]))
                metrics = results["metrics"][first_method]

            row = {
                "n_scenes": n,
                "elapsed_sec": round(elapsed, 2),
                "peak_memory_gb": round(peak_mem, 3),
                "adm": metrics.get("adm", float("nan")),
                "adsd": metrics.get("adsd", float("nan")),
                "cd": metrics.get("cd", float("nan")),
                "gl": metrics.get("gl", float("nan")),
                "rdoa": metrics.get("rdoa", float("nan")),
                "ave": metrics.get("ave", float("nan")),
            }
            summary["results"].append(row)

            _csv_writerow(scale_csv, scale_header, [[
                row["n_scenes"], row["elapsed_sec"], row["peak_memory_gb"],
                row["adm"], row["adsd"], row["cd"], row["gl"],
                row["rdoa"], row["ave"],
            ]])

        except Exception as exc:
            logger.error("规模实验失败 (N=%d): %s", n, exc)
            summary["results"].append({"n_scenes": n, "error": str(exc)})

    summary["scale_results_csv"] = scale_csv
    return summary


def run_diagnostics(config: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
    """
    VOLRN 系数 / RBF 位移场诊断。

    分析 VOLRN 系数分布、异常值、RBF 位移场质量、局部过增强检测。
    """
    from src.multiband_pipeline import MultibandPipeline
    from src.diagnostics import (
        analyze_volrn_coefficients,
        check_local_overenhancement,
        analyze_rbf_displacement,
        save_diagnostics_outputs,
    )

    output_dir = os.path.join(config.output_root, config.experiment_name, "diagnostics")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("实验类型: diagnostics")
    logger.info("输出目录: %s", output_dir)
    logger.info("=" * 60)

    summary: Dict[str, Any] = {"experiment_type": "diagnostics"}

    # 先运行管线获取归一化结果
    try:
        pipe = MultibandPipeline(config)
        results = pipe.run()
    except Exception as exc:
        logger.error("管线运行失败: %s", exc)
        summary["error"] = str(exc)
        return summary

    if not results.get("normalized") or not results.get("overlaps"):
        summary["status"] = "skipped"
        summary["reason"] = "no normalized data or overlaps"
        return summary

    normalized = results["normalized"]
    overlaps = results["overlaps"]
    scene_data = results.get("scene_data", {})
    nodata_values = scene_data.get("nodata_values", [None] * len(config.scenes))

    # 1. VOLRN 系数分析（使用实际 VOLRN 模块返回的系数）
    volrn_diag_dir = os.path.join(output_dir, "volrn_coefficients")
    os.makedirs(volrn_diag_dir, exist_ok=True)
    summary["volrn_coefficients_dir"] = volrn_diag_dir

    block_coefficients = normalized.get("bagrn_volrn_block_coefficients")
    blocks_info_raw = results.get("registration", {}).get("blocks_info", [])
    if block_coefficients is not None and block_coefficients.size > 0 and blocks_info_raw:
        logger.info("分析 VOLRN 系数...")
        try:
            volrn_diag = analyze_volrn_coefficients(
                block_coefficients,
                blocks_info_raw,
                output_dir=volrn_diag_dir,
            )
            summary["volrn_coefficients"] = {
                "n_bands": volrn_diag.get("summary", {}).get("n_bands", 0),
                "n_blocks": volrn_diag.get("summary", {}).get("n_blocks", 0),
                "total_outliers": volrn_diag.get("summary", {}).get("total_outlier_blocks", 0),
                "global_a_rms": volrn_diag.get("summary", {}).get("global_a_rms", float("nan")),
                "global_b_rms": volrn_diag.get("summary", {}).get("global_b_rms", float("nan")),
            }
        except Exception as exc:
            logger.error("VOLRN 系数分析失败: %s", exc)
            summary["volrn_coefficients_error"] = str(exc)
    else:
        logger.warning("无法获取 VOLRN 块系数，跳过系数分析")
        summary["volrn_coefficients"] = {"skipped": True, "reason": "no block_coefficients"}

    # 2. 局部过增强检测
    if "original" in normalized and "bagrn_volrn" in normalized:
        logger.info("检测局部过增强...")
        try:
            overenh = check_local_overenhancement(
                normalized["original"],
                normalized["bagrn_volrn"],
                nodata_values,
                block_size=config.volrn_params.get("block_size", 400),
                output_dir=volrn_diag_dir,
            )
            summary["overenhancement"] = overenh.get("summary", {})
        except Exception as exc:
            logger.error("过增强检测失败: %s", exc)
            summary["overenhancement_error"] = str(exc)

    # 3. RBF 位移场诊断（如有配准位移信息）
    reg_info = results.get("registration", {})
    if reg_info and not reg_info.get("diagnostics", {}).get("skipped"):
        logger.info("RBF 位移场诊断...")
        # 使用全局位移作为简化诊断
        global_shifts = reg_info.get("global_shifts")
        if global_shifts is not None:
            rbf_diag_dir = os.path.join(output_dir, "rbf_displacement")
            os.makedirs(rbf_diag_dir, exist_ok=True)
            # 保存全局位移摘要
            shift_summary = {
                "global_shifts": global_shifts.tolist() if hasattr(global_shifts, 'tolist') else global_shifts,
                "n_images": len(global_shifts) if hasattr(global_shifts, '__len__') else 0,
            }
            with open(os.path.join(rbf_diag_dir, "global_shifts.json"), "w") as f:
                json.dump(shift_summary, f, indent=2)
            summary["rbf_displacement_dir"] = rbf_diag_dir

    # 保存诊断输出
    try:
        save_diagnostics_outputs(summary, output_dir)
    except Exception:
        pass

    return summary


def run_seam(config: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
    """
    接缝评估实验。

    测试不同羽化宽度下的接缝质量，寻找 Pareto 最优。
    """
    from src.multiband_pipeline import MultibandPipeline
    from src.seam_metrics import evaluate_mosaic_seams

    output_dir = os.path.join(config.output_root, config.experiment_name, "seam")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("实验类型: seam")
    logger.info("输出目录: %s", output_dir)
    logger.info("=" * 60)

    feather_widths = [2, 4, 8, 16, 32, 64]

    summary: Dict[str, Any] = {
        "experiment_type": "seam",
        "feather_widths": feather_widths,
        "results": [],
    }

    # 先运行管线获取归一化结果
    try:
        pipe = MultibandPipeline(config)
        results = pipe.run()
    except Exception as exc:
        logger.error("管线运行失败: %s", exc)
        summary["error"] = str(exc)
        return summary

    if not results.get("normalized"):
        summary["status"] = "skipped"
        summary["reason"] = "no normalized data"
        return summary

    # 使用 bagrn_volrn 结果评估接缝
    method = "bagrn_volrn"
    if method not in results["normalized"]:
        method = next(iter(results["normalized"]))

    arrays = results["normalized"][method]
    scene_data = results.get("scene_data", {})
    transforms = scene_data.get("transforms", [])
    nodata_values = scene_data.get("nodata_values", [None] * len(arrays))

    # 构建输出网格信息
    if transforms:
        from rasterio.transform import array_bounds
        h, w = arrays[0].shape[1], arrays[0].shape[2]
        bounds = array_bounds(h, w, transforms[0])
        grid_info = {"shape": (h, w), "bounds": bounds}
    else:
        h, w = arrays[0].shape[1], arrays[0].shape[2]
        grid_info = {"shape": (h, w)}

    # 评估不同羽化宽度
    logger.info("评估羽化宽度: %s", feather_widths)
    try:
        seam_results = evaluate_mosaic_seams(
            arrays, transforms, nodata_values, grid_info,
            feather_widths=feather_widths,
        )
        summary["results"] = seam_results
    except Exception as exc:
        logger.error("接缝评估失败: %s", exc)
        summary["error"] = str(exc)
        return summary

    # 保存接缝指标 CSV
    seam_csv = os.path.join(output_dir, "seam_metrics.csv")
    if seam_results:
        header = list(seam_results[0].keys())
        rows = [[r.get(h, "") for h in header] for r in seam_results]
        _csv_writerow(seam_csv, header, rows)
        summary["seam_metrics_csv"] = seam_csv

    # 寻找 Pareto 最优：接缝可见度 vs 清晰度保留
    valid_results = [r for r in seam_results if "avg_mean_jump" in r and "avg_sharpness_retention" in r]
    if valid_results:
        # Pareto 前沿：最小 jump 且最大 sharpness retention
        best_jump = min(valid_results, key=lambda x: x["avg_mean_jump"])
        best_sharp = max(valid_results, key=lambda x: x["avg_sharpness_retention"])
        pareto = {
            "best_for_seamlessness": {
                "feather_width": best_jump.get("feather_width"),
                "avg_mean_jump": best_jump.get("avg_mean_jump"),
            },
            "best_for_sharpness": {
                "feather_width": best_sharp.get("feather_width"),
                "avg_sharpness_retention": best_sharp.get("avg_sharpness_retention"),
            },
        }
        summary["pareto_front"] = pareto

        # 保存推荐镶嵌模式
        rec_path = os.path.join(output_dir, "recommended_mosaic_mode.json")
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(pareto, f, indent=2, ensure_ascii=False)

    return summary


def run_spectral(config: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
    """
    光谱保真度验证。

    计算 SAM、光谱 RMSE、波段比值误差，比较不同方法的光谱保持度。
    """
    from src.multiband_pipeline import MultibandPipeline
    from src.spectral_metrics import compute_all_spectral

    output_dir = os.path.join(config.output_root, config.experiment_name, "spectral")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("实验类型: spectral")
    logger.info("输出目录: %s", output_dir)
    logger.info("=" * 60)

    summary: Dict[str, Any] = {"experiment_type": "spectral", "per_method": {}}

    # 先运行管线
    try:
        pipe = MultibandPipeline(config)
        results = pipe.run()
    except Exception as exc:
        logger.error("管线运行失败: %s", exc)
        summary["error"] = str(exc)
        return summary

    if not results.get("normalized") or not results.get("overlaps"):
        summary["status"] = "skipped"
        summary["reason"] = "no normalized data or overlaps"
        return summary

    normalized = results["normalized"]
    overlaps = results["overlaps"]
    scene_data = results.get("scene_data", {})
    nodata_values = scene_data.get("nodata_values", [None] * len(config.scenes))
    band_names = scene_data.get("band_names", [])

    original = normalized.get("original")
    if original is None:
        summary["status"] = "skipped"
        summary["reason"] = "no original data for comparison"
        return summary

    # 逐方法计算光谱指标
    spectral_csv = os.path.join(output_dir, "spectral_metrics.csv")
    csv_header = [
        "method", "sam_mean", "sam_p90", "sam_p95",
        "spectral_rmse_mean", "spectral_rmse_p90",
        "relative_spectral_rmse_mean",
        "frobenius_diff",
    ]
    csv_rows = []

    for method, arrays in normalized.items():
        if method == "original":
            continue

        logger.info("计算光谱指标: %s", method)
        try:
            spectral = compute_all_spectral(
                original, arrays, nodata_values, overlaps,
                band_names=band_names, output_dir=output_dir,
            )

            agg = spectral.get("aggregate", {})
            sam = agg.get("sam", {})
            rmse = agg.get("spectral_rmse", {})
            rel_rmse = agg.get("relative_spectral_rmse", {})

            # 相关系数变化
            frob_diffs = []
            for corr in spectral.get("correlation", []):
                fd = corr.get("frobenius_diff", float("nan"))
                if not np.isnan(fd):
                    frob_diffs.append(fd)
            avg_frob = np.mean(frob_diffs) if frob_diffs else float("nan")

            method_metrics = {
                "sam": sam,
                "spectral_rmse": rmse,
                "relative_spectral_rmse": rel_rmse,
                "frobenius_diff": avg_frob,
            }
            summary["per_method"][method] = method_metrics

            csv_rows.append([
                method,
                sam.get("mean", float("nan")),
                sam.get("p90", float("nan")),
                sam.get("p95", float("nan")),
                rmse.get("mean", float("nan")),
                rmse.get("p90", float("nan")),
                rel_rmse.get("mean", float("nan")),
                avg_frob,
            ])

        except Exception as exc:
            logger.error("光谱指标计算失败 [%s]: %s", method, exc)
            summary["per_method"][method] = {"error": str(exc)}

    if csv_rows:
        _csv_writerow(spectral_csv, csv_header, csv_rows)
        summary["spectral_metrics_csv"] = spectral_csv

    return summary


# ===========================================================================
# 实验类型注册表
# ===========================================================================

EXPERIMENT_RUNNERS = {
    "multiband": run_multiband,
    "ablation": run_ablation,
    "sensitivity": run_sensitivity,
    "scale": run_scale,
    "diagnostics": run_diagnostics,
    "seam": run_seam,
    "spectral": run_spectral,
}


# ===========================================================================
# CLI 入口
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="experiment_runner",
        description="BAGRN-VOLRN 实验运行器 — 统一命令行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
实验类型:
  multiband    完整多波段管线（加载→重叠→配准→归一化→镶嵌→指标）
  ablation     消融实验（original / BAGRN / VOLRN / BAGRN-VOLRN）
  sensitivity  参数敏感性分析（block_size / lambda / rho 扫描）
  scale        多场景规模实验（N=4,8,12 景的时间、内存、指标）
  diagnostics  VOLRN 系数与 RBF 位移场诊断
  seam         接缝评估（羽化宽度 Pareto 分析）
  spectral     光谱保真度验证（SAM / RMSE / 波段比值）

示例:
  python -m src.experiment_runner --config configs/dz01.yaml multiband
  python -m src.experiment_runner --config configs/dz01.yaml ablation --dry-run
  python -m src.experiment_runner --config configs/dz01.yaml sensitivity --resume
        """,
    )

    parser.add_argument(
        "experiment_type",
        type=str,
        choices=list(EXPERIMENT_RUNNERS.keys()),
        help="实验类型",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印执行计划，不实际处理",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="冒烟测试模式（使用小区域裁剪）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从检查点恢复，跳过已完成步骤",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出（删除检查点）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行 worker 数（预留接口，暂未实现）",
    )
    parser.add_argument(
        "--scene-count",
        type=str,
        default=None,
        help="规模实验的场景数列表，逗号分隔（如 4,8,12）",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="指定方法列表，逗号分隔（如 bagrn,volrn_only,bagrn_volrn）",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
        help="冒烟测试裁剪大小（默认 512）",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="输出根目录（覆盖配置文件中的值）",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    主入口函数。

    参数
    ----------
    argv : list of str or None
        命令行参数。None 则使用 sys.argv[1:]。

    返回
    -------
    int
        退出码（0=成功，1=失败）。
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    # 设置随机种子
    np.random.seed(args.seed)

    # 加载配置
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"配置加载失败: {exc}", file=sys.stderr)
        return 1

    # 命令行覆盖
    if args.dry_run:
        config.dry_run = True
    if args.smoke:
        config.smoke = True
    if args.output_root:
        config.output_root = args.output_root
    if args.crop_size:
        config.smoke_crop_size = args.crop_size

    # 输出目录
    output_dir = os.path.join(config.output_root, config.experiment_name)

    # Only create output dir and setup logging for non-dry-run
    if not config.dry_run:
        os.makedirs(output_dir, exist_ok=True)
        _setup_logging(output_dir, config.experiment_name)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    logger.info("实验运行器启动")
    logger.info("实验类型: %s", args.experiment_type)
    logger.info("配置文件: %s", args.config)
    logger.info("输出目录: %s", output_dir)

    # Skip environment.json for dry-run
    if not config.dry_run:
        env_info = _collect_environment_info(sys.argv)
        env_info["experiment_type"] = args.experiment_type
        env_info["config_file"] = args.config
        _save_environment(output_dir, env_info)
        logger.info("环境信息已保存: %s", os.path.join(output_dir, "environment.json"))

    # 覆盖模式：删除检查点
    if args.overwrite:
        checkpoint_dir = os.path.join(output_dir, ".checkpoints")
        if os.path.exists(checkpoint_dir):
            import shutil
            shutil.rmtree(checkpoint_dir)
            logger.info("已清除检查点目录")

    # 配置校验
    errors = validate_config(config)
    if errors:
        for e in errors:
            logger.error("配置校验失败: %s", e)
        return 1

    # 调度到对应的实验运行器
    runner = EXPERIMENT_RUNNERS.get(args.experiment_type)
    if runner is None:
        logger.error("未知实验类型: %s", args.experiment_type)
        return 1

    t_start = time.time()
    try:
        summary = runner(config, args)
    except Exception as exc:
        logger.error("实验执行失败: %s", exc, exc_info=True)
        summary = {
            "experiment_type": args.experiment_type,
            "status": "failed",
            "error": str(exc),
        }
        # Save summary then return 1
        summary["elapsed_total_sec"] = round(time.time() - t_start, 2)
        summary["config_file"] = args.config
        summary["args"] = vars(args)
        _save_summary(output_dir, summary)
        return 1

    # 补充环境和时间信息
    summary["elapsed_total_sec"] = round(time.time() - t_start, 2)
    summary["config_file"] = args.config
    summary["args"] = vars(args)

    # 保存摘要
    summary_path = _save_summary(output_dir, summary)
    logger.info("摘要已保存: %s", summary_path)

    # 打印摘要
    logger.info("=" * 60)
    logger.info("实验完成: %s (耗时 %.1fs, 状态=%s)",
                args.experiment_type, summary["elapsed_total_sec"],
                summary.get("pipeline_status", "unknown"))
    if summary.get("metrics"):
        logger.info("指标摘要:")
        for method, m in summary["metrics"].items():
            if isinstance(m, dict) and "ave" in m:
                logger.info("  %s: Ave=%.6f", method, m["ave"])
    logger.info("=" * 60)

    # 返回非零退出码当管线失败
    if summary.get("pipeline_status") == "failed":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
