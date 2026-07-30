"""
真实多景数据验收验证脚本

用法:
  python scripts/validate_real_multiscene.py --config configs/dz01_multiband.yaml --scene-count 5 --crop-size 1024 --registration-only
  python scripts/validate_real_multiscene.py --config configs/dz01_multiband.yaml --scene-count 5 --crop-size 1024 --output-report report.json
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_config import load_config, ExperimentConfig
from src.multiband_pipeline import MultibandPipeline

logger = logging.getLogger(__name__)


def run_validation(args: argparse.Namespace) -> Dict[str, Any]:
    """执行验证并返回结果报告。"""
    t0 = time.time()

    # 加载配置
    config: ExperimentConfig = load_config(args.config)

    # 命令行覆盖
    config.smoke = True
    config.smoke_crop_size = args.crop_size
    config.smoke_scene_count = args.scene_count

    # 如果只保留部分景，裁剪 scenes 列表
    if args.scene_count and args.scene_count < len(config.scenes):
        config.scenes = config.scenes[: args.scene_count]

    report: Dict[str, Any] = {
        "config_file": args.config,
        "crop_size": args.crop_size,
        "scene_count": args.scene_count,
        "requested_scenes": [s.get("id", f"scene_{i}") for i, s in enumerate(config.scenes)],
    }

    # 构建管线
    pipe = MultibandPipeline(config)

    # Step 1: 加载场景
    logger.info("Step 1: 加载场景...")
    try:
        scene_data = pipe.load_scenes()
        report["loaded_scenes"] = scene_data["scene_ids"]
        report["n_bands"] = len(scene_data.get("band_names", []))
        report["resolution"] = scene_data.get("resolution")
        report["scene_bounds"] = {
            sid: list(b) for sid, b in zip(scene_data["scene_ids"], scene_data["bounds"])
        }
    except Exception as exc:
        report["error"] = f"加载场景失败: {exc}"
        report["status"] = "failed"
        return report

    # Step 2: 检测重叠
    logger.info("Step 2: 检测重叠...")
    try:
        overlaps = pipe.detect_overlaps(scene_data)
        report["n_overlaps"] = len(overlaps)
        report["overlap_pairs"] = [
            {"idx_i": ov["idx_i"], "idx_j": ov["idx_j"],
             "scene_i": scene_data["scene_ids"][ov["idx_i"]],
             "scene_j": scene_data["scene_ids"][ov["idx_j"]],
             "pixel_count": ov.get("pixel_count", 0)}
            for ov in overlaps
        ]
    except Exception as exc:
        report["error"] = f"检测重叠失败: {exc}"
        report["status"] = "failed"
        return report

    # Step 3: 配准（含连通性检查）
    logger.info("Step 3: 配准...")
    try:
        registration = pipe.register_scenes(scene_data, overlaps)
        report["registration"] = {
            "connected": len(registration.get("unreachable_scenes", [])) == 0,
            "unreachable_scenes": registration.get("unreachable_scenes", []),
            "geometric_edges": registration.get("geometric_edges", []),
            "matching_edges": registration.get("matching_edges", []),
            "rejected_edges": registration.get("rejected_edges", []),
            "spanning_tree": registration.get("spanning_tree", []),
            "connected_components": registration.get("connected_components", []),
            "n_pairs_matched": registration.get("diagnostics", {}).get("n_pairs_matched", 0),
            "n_rejected": registration.get("diagnostics", {}).get("n_rejected", 0),
        }
        # 逐边质量
        edge_quality = []
        for pm in registration.get("pair_matches", []):
            scene_i = scene_data["scene_ids"][pm["idx_i"]]
            scene_j = scene_data["scene_ids"][pm["idx_j"]]
            edge_quality.append({
                "pair": f"[{pm['idx_i']}]→[{pm['idx_j']}]",
                "scenes": f"{scene_i}<->{scene_j}",
                "method": pm.get("method", "unknown"),
                "shift_dx": round(pm["shift_dx"], 4),
                "shift_dy": round(pm["shift_dy"], 4),
                "confidence": round(pm["confidence"], 4),
                "n_blocks": pm["n_blocks"],
                "rmse": round(pm.get("rmse", 0), 4),
                "p95": round(pm.get("p95", 0), 4),
            })
        report["edge_quality"] = edge_quality
    except Exception as exc:
        report["error"] = f"配准失败: {exc}"
        report["status"] = "failed"
        return report

    # 检查连通性
    all_connected = len(report["registration"]["unreachable_scenes"]) == 0
    n_expected = args.scene_count
    n_actual = len(scene_data["scene_ids"])
    n_matching_edges = len(report["registration"]["matching_edges"])
    n_tree_edges = len(report["registration"]["spanning_tree"])

    report["connectivity_check"] = {
        "all_connected": all_connected,
        "n_expected": n_expected,
        "n_actual": n_actual,
        "n_matching_edges": n_matching_edges,
        "n_tree_edges": n_tree_edges,
        "tree_covers_all": n_tree_edges >= n_actual - 1 if n_actual > 0 else False,
    }

    if not all_connected:
        report["status"] = "failed"
        report["failure_reason"] = f"场景不连通: 不可达={report['registration']['unreachable_scenes']}"
    elif n_tree_edges < n_actual - 1:
        report["status"] = "failed"
        report["failure_reason"] = f"生成树不完整: {n_tree_edges} edges < {n_actual - 1} expected"
    else:
        report["status"] = "success"

    # Step 4: 如果不是 --registration-only，继续归一化
    if not args.registration_only:
        logger.info("Step 4: 辐射归一化...")
        try:
            registration["transforms"] = scene_data["transforms"]
            registration["bounds"] = scene_data["bounds"]
            registration["nodata_values"] = scene_data["nodata_values"]
            registration["crs"] = scene_data["crs"]
            normalized = pipe.apply_radiometric_normalization(registration, overlaps)
            method_arrays = {k: v for k, v in normalized.items()
                             if isinstance(v, list) and v and isinstance(v[0], __import__("numpy").ndarray)}
            report["methods_completed"] = list(method_arrays.keys())
            report["methods_failed"] = [m for m in config.ablation_methods if m not in method_arrays]
        except Exception as exc:
            report["error"] = f"归一化失败: {exc}"
            report["status"] = "failed"

    report["elapsed_sec"] = round(time.time() - t0, 2)
    return report


def main():
    parser = argparse.ArgumentParser(description="真实多景数据验收验证")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--crop-size", type=int, default=1024, help="smoke 裁剪尺寸")
    parser.add_argument("--scene-count", type=int, default=5, help="场景数量")
    parser.add_argument("--registration-only", action="store_true", help="仅执行配准验证")
    parser.add_argument("--output-report", type=str, default=None, help="输出报告 JSON 路径")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    report = run_validation(args)

    # 打印报告摘要
    print("\n" + "=" * 60)
    print("验收验证报告")
    print("=" * 60)
    print(f"状态: {report.get('status', 'unknown')}")
    print(f"请求场景: {report.get('requested_scenes', [])}")
    print(f"加载场景: {report.get('loaded_scenes', [])}")
    print(f"重叠对数: {report.get('n_overlaps', 0)}")
    reg = report.get("registration", {})
    print(f"连通: {reg.get('connected', False)}")
    print(f"不可达: {reg.get('unreachable_scenes', [])}")
    print(f"几何边: {reg.get('geometric_edges', [])}")
    print(f"匹配边: {reg.get('matching_edges', [])}")
    print(f"拒绝边: {reg.get('rejected_edges', [])}")
    print(f"生成树: {reg.get('spanning_tree', [])}")
    print(f"连通分量: {reg.get('connected_components', [])}")
    for eq in report.get("edge_quality", []):
        print(f"  {eq['scenes']}: method={eq['method']}, "
              f"dx={eq['shift_dx']:.4f}, dy={eq['shift_dy']:.4f}, "
              f"conf={eq['confidence']:.4f}, rmse={eq['rmse']:.4f}")
    if report.get("methods_completed"):
        print(f"完成方法: {report['methods_completed']}")
    if report.get("methods_failed"):
        print(f"失败方法: {report['methods_failed']}")
    print(f"耗时: {report.get('elapsed_sec', 0):.1f}s")
    if report.get("error"):
        print(f"错误: {report['error']}")
    print("=" * 60)

    # 保存报告
    if args.output_report:
        os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"报告已保存: {args.output_report}")

    # 返回退出码
    if report.get("status") == "failed":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
