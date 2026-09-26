"""Generate the legacy project report figures without import-time side effects."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def main(output_dir: str | Path | None = None) -> Path:
    """Generate the five report figures and return their output directory."""
    out_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[1] / "data" / "report_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    modules = ["BAGRN\n(bagrn.py)", "VOLRN\n(volrn.py)", "主流程\n(main.py)", "评价指标\n(metrics.py)", "IO工具\n(io_utils.py)", "重叠检测\n(overlap.py)", "测试代码\n(tests/)"]
    lines = [348, 696, 696, 333, 174, 152, 1057]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(lines, labels=modules, autopct="%1.1f%%", startangle=90, pctdistance=0.75,
           colors=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"])
    ax.set_title("BAGRN-VOLRN 项目代码分布", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_dir / "code_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["重叠检测", "BAGRN", "VOLRN", "指标", "主流程"]
    values = [7, 9, 8, 18, 5]
    ax.bar(labels, values, color=["#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"])
    ax.set_ylabel("测试用例数"); ax.set_title("各模块测试结果")
    fig.tight_layout(); fig.savefig(out_dir / "test_results.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5)); ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    steps = [(0.5, "读取\nGeoTIFF", "#45B7D1"), (2.0, "检测\n重叠区域", "#4ECDC4"), (3.5, "BAGRN\n全局归一", "#FF6B6B"), (5.0, "VOLRN\n局部归一", "#96CEB4"), (6.5, "写出\n归一化结果", "#DDA0DD"), (8.0, "评价指标\n& 日志", "#FFEAA7")]
    for x, label, color in steps:
        ax.add_patch(mpatches.FancyBboxPatch((x, 0.8), 1.2, 1.4, boxstyle="round,pad=0.15", facecolor=color, edgecolor="#333", alpha=0.85))
        ax.text(x + 0.6, 1.5, label, ha="center", va="center", fontsize=9, fontweight="bold")
    for first, second in zip(steps, steps[1:]):
        ax.annotate("", xy=(second[0], 1.5), xytext=(first[0] + 1.2, 1.5), arrowprops={"arrowstyle": "->", "lw": 2, "color": "#888"})
    ax.set_title("BAGRN-VOLRN 处理流水线", fontsize=14)
    fig.tight_layout(); fig.savefig(out_dir / "pipeline.png", dpi=150); plt.close(fig)

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].imshow(rng.random((6, 6)), cmap="Blues", aspect="auto"); axes[0].set_title("重叠区统计")
    axes[1].axis("off"); axes[1].text(0.5, 0.5, "D · theta = L\n\n加权最小二乘", ha="center", va="center", fontsize=15)
    x = np.linspace(0, 4, 100); axes[2].scatter(x, 0.3 * x + 0.5 + 0.05 * rng.standard_normal(100), s=5, label="原始"); axes[2].scatter(x, 0.3 * x + 0.8 + 0.05 * rng.standard_normal(100), s=5, label="归一化后"); axes[2].legend(fontsize=8)
    fig.suptitle("BAGRN 全局辐射归一化原理"); fig.tight_layout(); fig.savefig(out_dir / "bagrn_diagram.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].imshow(rng.random((3, 3)), cmap="Blues"); axes[0].set_title("规则网格分块")
    axes[1].axis("off"); axes[1].text(0.5, 0.5, "min 1/2 ||Bx||² + λ||Ax-b||₁", ha="center", va="center", fontsize=13)
    axes[2].axis("off"); axes[2].text(0.5, 0.5, "ADMM\nPCG → 软阈值 → 对偶更新", ha="center", va="center", fontsize=12)
    fig.suptitle("VOLRN 局部辐射归一化原理"); fig.tight_layout(); fig.savefig(out_dir / "volrn_diagram.png", dpi=150); plt.close(fig)
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None, type=Path)
    raise SystemExit(main(parser.parse_args().output_dir) or 0)
