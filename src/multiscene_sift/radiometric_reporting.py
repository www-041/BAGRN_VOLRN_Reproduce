"""Radiometric metrics reporting and normalization coefficient saving."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

from src.multiscene_sift.radiometric import BandRadiometricResult

logger = logging.getLogger(__name__)

METRIC_KEYS = ["ADM", "ADSD", "CD", "GL", "RDOA", "Ave"]
STAGE_NAMES = ["Registered", "BAGRN", "VOLRN"]


def collect_radiometric_metrics(
    band_results: dict[str, BandRadiometricResult],
) -> list[dict]:
    """Collect all radiometric metrics into rows for CSV/JSON.

    Returns:
        List of dicts, one per (band, stage) combination.
    """
    rows = []
    for band_name, result in band_results.items():
        metrics_for_band = [
            ("Registered", result.registered_metrics),
            ("BAGRN", result.bagrn_metrics),
            ("VOLRN", result.volrn_metrics),
        ]
        for stage_name, metrics in metrics_for_band:
            row = {"band": band_name, "stage": stage_name}
            for key in METRIC_KEYS:
                val = metrics.get(key.lower())
                if val is not None and np.isfinite(val):
                    row[key] = round(float(val), 6)
                else:
                    row[key] = None
            rows.append(row)
    return rows


def save_radiometric_metrics(
    band_results: dict[str, BandRadiometricResult],
    out_dir: Path,
) -> None:
    """Save radiometric_metrics.csv and radiometric_metrics.json."""
    rows = collect_radiometric_metrics(band_results)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "radiometric_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["band", "stage"] + METRIC_KEYS,
        )
        writer.writeheader()
        writer.writerows(rows)

    # JSON with per-band and mean summary
    per_band = {}
    for band_name, result in band_results.items():
        per_band[band_name] = {
            "Registered": _extract_metric_dict(result.registered_metrics),
            "BAGRN": _extract_metric_dict(result.bagrn_metrics),
            "VOLRN": _extract_metric_dict(result.volrn_metrics),
        }

    # Mean over B14/B8/B5
    means = {}
    for stage in STAGE_NAMES:
        means[stage] = {}
        for key in METRIC_KEYS:
            vals = []
            for band_name in band_results:
                metrics = {
                    "Registered": band_results[band_name].registered_metrics,
                    "BAGRN": band_results[band_name].bagrn_metrics,
                    "VOLRN": band_results[band_name].volrn_metrics,
                }[stage]
                v = metrics.get(key.lower())
                if v is not None and np.isfinite(v):
                    vals.append(float(v))
            means[stage][key] = round(float(np.mean(vals)), 6) if vals else None

    json_path = out_dir / "radiometric_metrics.json"
    with open(json_path, "w") as f:
        json.dump({
            "per_band": per_band,
            "mean_over_bands": means,
        }, f, indent=2)

    logger.info("Radiometric metrics saved: %s", csv_path)


def save_normalization_info(
    band_results: dict[str, BandRadiometricResult],
    radiometric_control_idx: int,
    radiometric_control_name: str,
    out_dir: Path,
) -> None:
    """Save radiometric_normalization_info.json."""
    info = {
        "radiometric_control_idx": radiometric_control_idx,
        "radiometric_control_name": radiometric_control_name,
        # Backward-compatible aliases. These refer to the radiometric control,
        # not the geometry reference.
        "reference_idx": radiometric_control_idx,
        "reference_name": radiometric_control_name,
        "per_band": {},
    }
    for band_name, result in band_results.items():
        info["per_band"][band_name] = {
            "overlap_pairs": [
                {"idx_i": ov["idx_i"], "idx_j": ov["idx_j"]}
                for ov in result.overlaps
            ],
            "n_overlap_pairs": len(result.overlaps),
            "bagrn_runtime_sec": round(result.bagrn_runtime_sec, 3),
            "volrn_runtime_sec": round(result.volrn_runtime_sec, 3),
            "block_size_pixels": 200,
            "lambda_param": 0.1,
            "rho": 1.0,
            "max_iter": 200,
            "tol": 1e-4,
        }
        # theta_mu and theta_sigma are arrays, save shapes/stats
        if result.theta_mu is not None:
            mu = np.asarray(result.theta_mu)
            info["per_band"][band_name]["theta_mu_shape"] = list(mu.shape)
            info["per_band"][band_name]["theta_mu_stats"] = {
                "mean": round(float(mu.mean()), 6) if mu.size > 0 else None,
                "std": round(float(mu.std()), 6) if mu.size > 0 else None,
            }
        if result.theta_sigma is not None:
            sg = np.asarray(result.theta_sigma)
            info["per_band"][band_name]["theta_sigma_shape"] = list(sg.shape)
            info["per_band"][band_name]["theta_sigma_stats"] = {
                "mean": round(float(sg.mean()), 6) if sg.size > 0 else None,
                "std": round(float(sg.std()), 6) if sg.size > 0 else None,
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "radiometric_normalization_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info("Normalization info saved")


def _extract_metric_dict(metrics: dict) -> dict:
    """Extract METRIC_KEYS from a flat metric dict."""
    return {
        key: round(float(metrics.get(key.lower(), float("nan"))), 6)
        if metrics.get(key.lower()) is not None
        and np.isfinite(metrics.get(key.lower()))
        else None
        for key in METRIC_KEYS
    }