"""Compare two completed five-scene runs without rerunning any experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REGISTRATION_METRICS = [
    "geographic_edges",
    "accepted_edges",
    "failed_edges",
    "total_raw_matches",
    "mean_raw_matches",
    "total_inliers",
    "mean_inliers",
    "mean_inlier_ratio",
    "mean_coverage",
    "mean_pairwise_rmse_px",
    "mean_pairwise_p95_px",
    "worst_pairwise_p95_px",
    "global_eval_scope",
    "global_eval_edges",
    "global_n_points",
    "global_rmse_mean_px",
    "global_p95_mean_px",
    "global_p95_worst_px",
    "global_max_px",
]

RUNTIME_METRICS = [
    "matching_runtime_sec",
    "geometry_runtime_sec",
    "pairwise_total_runtime_sec",
    "bagrn_runtime_sec",
    "volrn_runtime_sec",
    "total_runtime_sec",
]

COVERAGE_METRICS = [
    "union_valid_pixels",
    "final_valid_pixels",
    "union_but_final_invalid_pixels",
    "coverage_hole_ratio",
]

RADIOMETRIC_METRICS = ["ADM", "ADSD", "CD", "GL", "RDOA", "Ave"]
STAGES = ["Registered", "BAGRN", "VOLRN"]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _write_two_column_metric_table(
    path: Path,
    metrics: list[str],
    sift: dict,
    loftr: dict,
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "SIFT", "LoFTR"])
        writer.writeheader()
        for metric in metrics:
            writer.writerow({
                "metric": metric,
                "SIFT": sift.get(metric),
                "LoFTR": loftr.get(metric),
            })


def compare_runs(
    sift_dir: str | Path,
    loftr_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Read completed run directories and write four comparison CSVs."""
    sift_dir = Path(sift_dir)
    loftr_dir = Path(loftr_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sift_reg = _load_json(sift_dir / "registration_summary.json")
    loftr_reg = _load_json(loftr_dir / "registration_summary.json")
    sift_cov = _load_json(sift_dir / "mosaic_coverage_summary.json")
    loftr_cov = _load_json(loftr_dir / "mosaic_coverage_summary.json")
    sift_rad = _load_json(sift_dir / "radiometric_metrics.json")
    loftr_rad = _load_json(loftr_dir / "radiometric_metrics.json")

    _write_two_column_metric_table(
        out / "sift_vs_loftr_registration.csv",
        REGISTRATION_METRICS,
        sift_reg,
        loftr_reg,
    )
    _write_two_column_metric_table(
        out / "sift_vs_loftr_runtime.csv",
        RUNTIME_METRICS,
        sift_reg,
        loftr_reg,
    )
    _write_two_column_metric_table(
        out / "sift_vs_loftr_mosaic_coverage.csv",
        COVERAGE_METRICS,
        sift_cov,
        loftr_cov,
    )

    with open(out / "sift_vs_loftr_radiometric.csv", "w", newline="") as f:
        fields = ["matcher", "stage"] + RADIOMETRIC_METRICS
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, payload in (("SIFT", sift_rad), ("LoFTR", loftr_rad)):
            try:
                b14 = payload["per_band"]["B14"]
            except KeyError as exc:
                raise KeyError(
                    f"{label} radiometric_metrics.json has no B14 per-band results"
                ) from exc
            for stage in STAGES:
                metrics = b14.get(stage, {})
                row = {"matcher": label, "stage": stage}
                row.update({key: metrics.get(key) for key in RADIOMETRIC_METRICS})
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare completed SIFT and LoFTR five-scene runs"
    )
    parser.add_argument("--sift-dir", required=True)
    parser.add_argument("--loftr-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    compare_runs(args.sift_dir, args.loftr_dir, args.output_dir)


if __name__ == "__main__":
    main()
