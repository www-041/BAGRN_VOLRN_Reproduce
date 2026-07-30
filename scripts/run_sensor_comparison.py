"""
传感器对比实验：DZ01S vs DZ01V

为每个传感器创建仅包含该传感器目录的临时输入目录，再跑完整流程。
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from copy import deepcopy

import numpy as np

# 确保可以 import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import parse_args, run_pipeline


def _create_sensor_input_dir(input_dir: str, sensor: str, temp_base: str) -> str:
    """
    创建仅包含指定传感器(DZ01S 或 DZ01V)子目录的临时目录结构。
    
    原始结构:
      input_dir/20251114023841/DZ01S_...
      input_dir/20251114023841/DZ01V_...
    
    目标结构 (仅 sensor=DZ01S):
      temp_dir/20251114023841/DZ01S_... (复制或软链接)
    """
    sensor_dir = os.path.join(temp_base, f"sensor_{sensor}_input")
    if os.path.exists(sensor_dir):
        shutil.rmtree(sensor_dir)
    os.makedirs(sensor_dir)

    # 遍历所有时相目录
    for datetime_dir in sorted(os.listdir(input_dir)):
        dt_path = os.path.join(input_dir, datetime_dir)
        if not os.path.isdir(dt_path):
            continue
        
        # 找该时相下的传感器子目录
        for subdir in os.listdir(dt_path):
            if subdir.startswith(sensor + "_"):
                src = os.path.join(dt_path, subdir)
                dst = os.path.join(sensor_dir, datetime_dir, subdir)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                # Windows 下用复制（软链接需要管理员权限）
                shutil.copytree(src, dst)
                print(f"  已复制: {src} -> {dst}")
                break  # 每个时相只取一个该传感器的目录
    
    return sensor_dir


def run_sensor_pipeline(input_dir: str, output_dir: str, sensor: str, 
                        max_images: int = 2, crop_size: int = 512,
                        block_size: int = 100, lambda_param: float = 1.0,
                        rho: float = 10.0, max_iter: int = 30) -> dict:
    """针对单个传感器运行完整 BAGRN-VOLRN 流程。"""
    run_dir = os.path.join(output_dir, f"sensor_{sensor}")
    os.makedirs(run_dir, exist_ok=True)

    argv = [
        "--input_dir", input_dir,
        "--output_dir", run_dir,
        "--mode", "bagrn_volrn",
        "--max_images", str(max_images),
        "--block_size", str(block_size),
        "--lambda_param", str(lambda_param),
        "--rho", str(rho),
        "--max_iter", str(max_iter),
        "--tol", "1e-3",
        "--debug",
        "--debug_crop_size", str(crop_size),
        "--eval",
        "--compare", "histogram_matching,moment_matching,wallis",
    ]
    args = parse_args(argv)
    t0 = time.time()
    run_pipeline(args)
    elapsed = time.time() - t0

    # 读取日志
    log_path = os.path.join(run_dir, "run_log.json")
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        metrics = log.get("metrics", {})
        result = {
            "metrics": metrics,
            "time": round(elapsed, 2),
            "n_bands": log.get("n_bands", 0),
            "n_images": log.get("n_images", 0),
            "sensor": sensor,
        }
        for stage in ["baseline", "histogram_matching", "moment_matching", "wallis", "bagrn", "volrn"]:
            if stage in metrics:
                m = metrics[stage]
                print(f"  {stage}: Ave={m.get('ave', 'N/A')}")
        print(f"  耗时: {elapsed:.1f}s")
        return result
    except Exception as e:
        print(f"  读取日志失败: {e}")
        return {"error": str(e)}


def plot_sensor_comparison(results: dict, output_dir: str):
    """绘制传感器对比柱状图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib 未安装，跳过绘图")
        return

    sensors = list(results.keys())
    # 只画有的阶段
    all_stages = set()
    for r in results.values():
        all_stages.update(r.get("metrics", {}).keys())
    stages = [s for s in ["baseline", "histogram_matching", "moment_matching", "wallis", "bagrn", "volrn"] if s in all_stages]
    metrics_keys = ["adm", "adsd", "cd", "gl", "rdoa", "ave"]
    n_metrics = len(metrics_keys)

    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics))
    if n_metrics == 1:
        axes = [axes]
    x = np.arange(len(sensors))
    width = 0.8 / len(stages)

    for i, key in enumerate(metrics_keys):
        ax = axes[i]
        for j, stage in enumerate(stages):
            vals = [results[s].get("metrics", {}).get(stage, {}).get(key, 0) for s in sensors]
            ax.bar(x + j * width - width * (len(stages) - 1) / 2, vals, width, label=stage)
        ax.set_ylabel(key.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(sensors)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "sensor_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  图表已保存: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="DZ01S vs DZ01V 传感器对比")
    parser.add_argument("--input_dir", type=str, default="data/input")
    parser.add_argument("--output_dir", type=str, default="data/output/analysis")
    parser.add_argument("--crop_size", type=int, default=512, help="调试裁剪像素")
    parser.add_argument("--max_images", type=int, default=2, help="每传感器最多时相数")
    parser.add_argument("--block_size", type=int, default=100, help="VOLRN 网格块大小")
    parser.add_argument("--lambda_param", type=float, default=1.0, help="VOLRN 正则化权重")
    parser.add_argument("--rho", type=float, default=10.0, help="ADMM 惩罚参数")
    parser.add_argument("--max_iter", type=int, default=30, help="ADMM 最大迭代")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 创建临时目录用于存放传感器过滤后的输入
    with tempfile.TemporaryDirectory(prefix="sensor_input_") as temp_base:
        print(f"临时目录: {temp_base}")
        
        results = {}
        for sensor in ["DZ01S", "DZ01V"]:
            print(f"\n{'='*60}")
            print(f"  传感器对比: {sensor}")
            print(f"{'='*60}")
            
            # 创建该传感器专用输入目录
            sensor_input = _create_sensor_input_dir(args.input_dir, sensor, temp_base)
            
            # 运行流程
            res = run_sensor_pipeline(
                sensor_input, args.output_dir, sensor,
                max_images=args.max_images,
                crop_size=args.crop_size,
                block_size=args.block_size,
                lambda_param=args.lambda_param,
                rho=args.rho,
                max_iter=args.max_iter,
            )
            results[sensor] = res

        # 保存结果
        sensor_path = os.path.join(args.output_dir, "sensor_comparison.json")
        with open(sensor_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  传感器对比结果已保存: {sensor_path}")

        # 绘图
        if "DZ01S" in results and "DZ01V" in results:
            plot_sensor_comparison(results, args.output_dir)

    print(f"\n所有对比完成！结果在: {args.output_dir}")


if __name__ == "__main__":
    main()