"""Analyze and plot VOLRN ADMM iteration history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HISTORY_FIELDS = (
    "objective", "primal_residual", "dual_residual", "x_update_norm",
    "z_update_norm", "dual_update_norm", "relative_change",
)


def _trend(values: np.ndarray) -> str:
    if values.size < 2:
        return "insufficient history"
    differences = np.diff(values)
    if np.all(differences <= 0.0):
        return "decreasing"
    if np.all(differences >= 0.0):
        return "increasing"
    return "oscillating"


def _load_history(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for band in payload.get("bands", []):
        for record in band.get("history", []):
            records.append({"band": int(band.get("band", 0)), **record})
    records.sort(key=lambda item: (item["band"], item["iteration"]))
    if not records:
        raise ValueError("VOLRN history contains no iteration records")
    return records


def _classify(objective: np.ndarray, primal: np.ndarray, dual: np.ndarray, relative: np.ndarray) -> str:
    dual_trend = _trend(dual)
    if dual_trend == "decreasing":
        return "slow convergence"
    if dual_trend == "oscillating":
        return "ADMM instability"
    if relative.size and float(np.median(relative[-max(1, len(relative) // 4):])) < 1e-5:
        return "update ineffective"
    return "ADMM instability" if _trend(objective) == "oscillating" else "update ineffective"


def _save_curve(path: Path, x: np.ndarray, series: dict[str, np.ndarray], title: str, log_scale: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for label, values in series.items():
        ax.plot(x, values, label=label, linewidth=1.2)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def analyze_volrn_history(history_path: str | Path, output_dir: str | Path) -> dict:
    history_path = Path(history_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_history(history_path)
    # The real gate has one band; preserve band labels if a caller supplies more.
    band_records = [record for record in records if record["band"] == records[0]["band"]]
    x = np.asarray([record["iteration"] for record in band_records], dtype=float)
    arrays = {
        field: np.asarray([float(record[field]) for record in band_records], dtype=float)
        for field in HISTORY_FIELDS
    }
    dual_trend = _trend(arrays["dual_residual"])
    objective_trend = _trend(arrays["objective"])
    primal_trend = _trend(arrays["primal_residual"])
    classification = _classify(
        arrays["objective"], arrays["primal_residual"], arrays["dual_residual"], arrays["relative_change"]
    )
    summary = {
        "schema_version": 1,
        "source_history": str(history_path),
        "band": int(records[0]["band"]),
        "iterations": int(len(band_records)),
        "classification": classification,
        "objective_trend": objective_trend,
        "primal_trend": primal_trend,
        "dual_trend": dual_trend,
        "objective_start": float(arrays["objective"][0]),
        "objective_end": float(arrays["objective"][-1]),
        "primal_residual_start": float(arrays["primal_residual"][0]),
        "primal_residual_end": float(arrays["primal_residual"][-1]),
        "dual_residual_start": float(arrays["dual_residual"][0]),
        "dual_residual_end": float(arrays["dual_residual"][-1]),
        "relative_change_end": float(arrays["relative_change"][-1]),
        "x_update_norm_end": float(arrays["x_update_norm"][-1]),
        "z_update_norm_end": float(arrays["z_update_norm"][-1]),
        "dual_update_norm_end": float(arrays["dual_update_norm"][-1]),
        "cg_iterations_start": int(band_records[0]["cg_iterations"]),
        "cg_iterations_end": int(band_records[-1]["cg_iterations"]),
    }
    _save_curve(output_dir / "objective_curve.png", x, {"objective": arrays["objective"]}, "VOLRN objective")
    _save_curve(output_dir / "primal_residual_curve.png", x, {"primal residual": arrays["primal_residual"]}, "VOLRN primal residual", log_scale=True)
    _save_curve(output_dir / "dual_residual_curve.png", x, {"dual residual": arrays["dual_residual"]}, "VOLRN dual residual", log_scale=True)
    _save_curve(
        output_dir / "update_norm_curve.png",
        x,
        {"x update": arrays["x_update_norm"], "z update": arrays["z_update_norm"], "dual update": arrays["dual_update_norm"]},
        "VOLRN update norms",
        log_scale=True,
    )
    (output_dir / "convergence_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = (
        "# Task10E VOLRN Convergence Analysis\n\n"
        f"Source history: `{history_path}`\n\n"
        f"- Iterations: `{summary['iterations']}`\n"
        f"- Classification: **{classification}**\n"
        f"- Objective: `{summary['objective_start']}` → `{summary['objective_end']}` ({objective_trend})\n"
        f"- Primal residual: `{summary['primal_residual_start']}` → `{summary['primal_residual_end']}` ({primal_trend})\n"
        f"- Dual residual: `{summary['dual_residual_start']}` → `{summary['dual_residual_end']}` ({dual_trend})\n"
        f"- Final relative change: `{summary['relative_change_end']}`\n\n"
        "The classification is diagnostic only. Frozen Task10D parameters were not changed.\n"
    )
    (output_dir / "2026-09-27-task10e-convergence-analysis.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    analyze_volrn_history(args.history, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
