"""Create a compact comparison table from completed pair diagnostic records."""
from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path


PAIRS = ("00_01", "00_10")
METRICS = ("mae", "rmse", "mamd", "msdd", "rdd")


def build_summary(output_root: Path) -> dict:
    root = Path(output_root)
    records = []
    for pair in PAIRS:
        pair_dir = root / pair
        payload = json.loads((pair_dir / "pair_metrics.json").read_text(encoding="utf-8"))
        meta = json.loads((pair_dir / "method_metadata.json").read_text(encoding="utf-8"))
        if payload.get("status") != "FULL_SUPPORT" or not payload.get("full_support"):
            raise ValueError(f"PAIR_NOT_FULL_SUPPORT:{pair}:{payload.get('status')}")
        if payload.get("coverage") != 1.0 or payload.get("missing_pixels") != 0:
            raise ValueError(f"PAIR_OUTPUT_COVERAGE_INVALID:{pair}")
        if meta.get("scope") != "1024_CROP_DIAGNOSTIC_ONLY":
            raise ValueError(f"PAIR_SCOPE_NOT_DIAGNOSTIC:{pair}")
        records.append({"pair_id": pair, "metrics": payload, "metadata": meta,
                        "regression": json.loads((pair_dir / "regression_audit.json").read_text(encoding="utf-8")),
                        "timing": payload["timing"], "output": payload["output"],
                        "preview": payload["preview"]})

    rows = []
    for record in records:
        m = record["metrics"]
        before, after = m["before"], m["after"]
        row = {"pair_id": record["pair_id"], "status": m["status"],
               "support_pixels": m["support_pixels"], "output_support_pixels": m["output_support_pixels"],
               "missing_pixels": m["missing_pixels"], "coverage": m["coverage"],
               "scale": record["regression"]["scale"], "offset": record["regression"]["offset"],
               "p_n": record["regression"]["actual_p_n"], "seed": record["regression"]["rng_seed"],
               "algorithm_s": record["timing"]["algorithm_s"], "total_wall_s": record["timing"]["total_wall_s"],
               "peak_rss_bytes": record["timing"]["peak_rss_bytes"],
               "rss_sample_count": record["timing"]["rss_sample_count"],
               "rss_status": record["timing"]["rss_status"],
               "corrected_tif": record["output"]["path"], "corrected_sha256": record["output"]["sha256"],
               "preview_png": record["preview"]["path"], "preview_sha256": record["preview"]["sha256"]}
        for metric in METRICS:
            row[f"before_{metric}"] = before["direct"].get(metric) if metric in {"mae", "rmse"} else before["distribution"].get(metric)
            row[f"after_{metric}"] = after["direct"].get(metric) if metric in {"mae", "rmse"} else after["distribution"].get(metric)
        rows.append(row)

    csv_path = root / "comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# LIRRN Phase 3 — 两对真实裁剪诊断结果", "",
             "范围：PAIR_ONLY / 1024×1024 DIAGNOSTIC_ONLY；不得作为13景正式排名。",
             "共同评价支持域为冻结掩膜交集；辐射单位 UNKNOWN_PIXEL_UNIT。", "",
             "| Pair | 支持像元 | MAE 前→后 | RMSE 前→后 | MAMD 前→后 | MSDD 前→后 | RDD 前→后 | scale / offset | 算法秒 | 总秒 | 峰值RSS / 样本 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        def pair(metric: str) -> str:
            return f"{row[f'before_{metric}']:.6f} → {row[f'after_{metric}']:.6f}"
        lines.append(f"| {row['pair_id']} | {row['support_pixels']:,} | {pair('mae')} | {pair('rmse')} | "
                     f"{pair('mamd')} | {pair('msdd')} | {pair('rdd')} | "
                     f"{row['scale']:.9f} / {row['offset']:.6f} | {row['algorithm_s']:.3f} | "
                     f"{row['total_wall_s']:.3f} | {row['peak_rss_bytes']} / {row['rss_sample_count']} ({row['rss_status']}) |")
    lines.extend(["", "## 判读限制", "",
                  "该评价是在裁剪内冻结有效交集上的像元一致性诊断，不是独立稳定地物真值；跨日期变化可能影响结果。",
                  "00_01/00_10使用同一P_N=1000、seed=20261014及NumPy RNG；不等同于MATLAB随机序列。",
                  "官方源码commit按计划固定，但本地精确源码SHA256因网络获取受阻未核验；MATLAB数值等价未验证。",
                  "LIRRN三个相同max-distance采样调用均保留；没有按结果调参。", ""])
    md_path = root / "comparison.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    result = {"scope": "PAIR_ONLY_DIAGNOSTIC_ONLY", "formal_ranking_eligible": False,
              "pair_count": len(rows), "pairs": rows,
              "comparison_csv": str(csv_path), "comparison_markdown": str(md_path)}
    (root / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_summary(args.output_root), indent=2, ensure_ascii=False))
