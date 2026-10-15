"""Task 10 summary tables and analysis-only radiometric figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.multiscene_sift.radiometric_protocol import GEOMETRY_SPECS, RADIOMETRIC_METHODS


def _read_run(run_dir: Path, geometry: str, method: str) -> dict:
    path = run_dir / geometry / method / "radiometric_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing radiometric run summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    metrics = summary.get("overlap_metrics", {}).get("summary", {})
    return {
        "geometry_run": geometry,
        "geometry_label": GEOMETRY_SPECS[geometry]["label"],
        "method": method,
        "overlap_mae": metrics.get("mae"),
        "overlap_rmse": metrics.get("rmse"),
        "overlap_bias": metrics.get("bias"),
        "histogram_mean_abs_difference": metrics.get("histogram_mean_abs_difference"),
        "histogram_std_abs_difference": metrics.get("histogram_std_abs_difference"),
        "histogram_tv_distance": metrics.get("histogram_tv_distance"),
        "seam_gradient": metrics.get("seam_gradient"),
        "seam_mae": metrics.get("seam_mae"),
        "seam_rmse": metrics.get("seam_rmse"),
        "seam_bias": metrics.get("seam_bias"),
        "run_dir": str(path.parent),
    }


def _write_figures(rows: list[dict], output_dir: Path) -> None:
    labels = [row["method"] for row in rows if row["geometry_run"] == next(iter(GEOMETRY_SPECS))]
    first_geometry = next(iter(GEOMETRY_SPECS))
    first_rows = [row for row in rows if row["geometry_run"] == first_geometry]
    x = np.arange(len(labels), dtype=float)
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, field, label in (
        (-width, "histogram_mean_abs_difference", "|mean difference|"),
        (0.0, "histogram_std_abs_difference", "|std difference|"),
        (width, "histogram_tv_distance", "histogram TV"),
    ):
        values = [row[field] if row[field] is not None else np.nan for row in first_rows]
        ax.bar(x + offset, values, width=width, label=label)
    ax.set_xticks(x, labels)
    ax.set_ylabel("difference")
    ax.set_title(f"Histogram change — {GEOMETRY_SPECS[first_geometry]['label']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "histogram_changes.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for geometry in GEOMETRY_SPECS:
        values = [
            row["seam_gradient"] if row["seam_gradient"] is not None else np.nan
            for row in rows if row["geometry_run"] == geometry
        ]
        ax.plot(labels, values, marker="o", label=GEOMETRY_SPECS[geometry]["label"])
    ax.set_ylabel("seam gradient metric")
    ax.set_title("Seam diagnostics under the fixed seam-zone rule")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "seam_diagnostics.png", dpi=150)
    plt.close(fig)

    for method in RADIOMETRIC_METHODS:
        fig, ax = plt.subplots(figsize=(8, 4))
        values = []
        labels = []
        for geometry in GEOMETRY_SPECS:
            row = next(row for row in rows if row["geometry_run"] == geometry and row["method"] == method)
            values.append(row["seam_gradient"] if row["seam_gradient"] is not None else np.nan)
            labels.append(geometry)
        ax.bar(labels, values, color="#4ECDC4")
        ax.set_ylabel("seam gradient metric")
        ax.set_title(f"{method} seam diagnostic")
        fig.tight_layout()
        fig.savefig(output_dir / f"{method}_seam.png", dpi=150)
        plt.close(fig)


def _write_markdown(rows: list[dict], answers: dict, path: Path) -> None:
    lines = [
        "# Task 10 Radiometric Audit",
        "",
        "All rows use the frozen B9 five-scene geometry and weighted-feather output grid.",
        "",
        "| Geometry | Method | Overlap MAE | Overlap RMSE | Bias | Seam Gradient |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['geometry_label']} | {row['method']} | {row['overlap_mae']} | "
            f"{row['overlap_rmse']} | {row['overlap_bias']} | {row['seam_gradient']} |"
        )
    lines.extend([
        "", "## Factual comparisons", "",
        f"- BAGRN overlap MAE is below RAW for every frozen geometry: `{answers['bagrn_mae_below_raw']}`.",
        f"- BAGRN+VOLRN overlap MAE is below BAGRN for every frozen geometry: `{answers['bagrn_volrn_mae_below_bagrn']}`.",
        f"- BAGRN overlap RMSE is below RAW for every frozen geometry: `{answers['bagrn_rmse_below_raw']}`.",
        f"- BAGRN+VOLRN seam gradient is below BAGRN for every frozen geometry: `{answers['bagrn_volrn_seam_below_bagrn']}`.",
        "",
        "These statements are comparisons of the recorded metrics; they do not claim that normalization improves every metric unless the corresponding value is lower in every geometry.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_radiometric_audit(run_root: str | Path, output_dir: str | Path) -> dict:
    """Build JSON, Markdown, histogram, and seam diagnostic artifacts."""
    run_root = Path(run_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _read_run(run_root, geometry, method)
        for geometry in GEOMETRY_SPECS
        for method in RADIOMETRIC_METHODS
    ]

    by_geometry = {
        geometry: {row["method"]: row for row in rows if row["geometry_run"] == geometry}
        for geometry in GEOMETRY_SPECS
    }

    def _below(left: str, right: str, field: str) -> bool:
        values = []
        for geometry in GEOMETRY_SPECS:
            a = by_geometry[geometry][left][field]
            b = by_geometry[geometry][right][field]
            values.append(a is not None and b is not None and a < b)
        return bool(values) and all(values)

    answers = {
        "bagrn_mae_below_raw": _below("BAGRN", "RAW", "overlap_mae"),
        "bagrn_volrn_mae_below_bagrn": _below("BAGRN_VOLRN", "BAGRN", "overlap_mae"),
        "bagrn_rmse_below_raw": _below("BAGRN", "RAW", "overlap_rmse"),
        "bagrn_volrn_seam_below_bagrn": _below("BAGRN_VOLRN", "BAGRN", "seam_gradient"),
    }
    result = {
        "schema_version": 1,
        "dataset": "B9",
        "manifest_indices": [2, 3, 5, 8, 10],
        "pixel_size_m": 14.0,
        "radiometric_methods": list(RADIOMETRIC_METHODS),
        "geometry_runs": list(GEOMETRY_SPECS),
        "rows": rows,
        "factual_answers": answers,
        "artifacts": {
            "markdown": "radiometric_audit.md",
            "histogram_changes": "histogram_changes.png",
            "seam_diagnostics": "seam_diagnostics.png",
            "seam_variants": [f"{method}_seam.png" for method in RADIOMETRIC_METHODS],
        },
    }
    (output_dir / "radiometric_audit_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(rows, answers, output_dir / "radiometric_audit.md")
    _write_figures(rows, output_dir)
    return result
