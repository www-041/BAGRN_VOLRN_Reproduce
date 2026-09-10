"""Preflight and compare the fixed-HOLDOUT B14/B12 diagnostic runs."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.registration_band_ab import (
    compare_registration_band_runs,
    load_holdout_manifest,
    validate_band_ab_metadata,
)


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_comparison_files(comparison, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "registration_band_ab_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "registration_band_ab_stage_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "median", "rmse", "p95", "confidence", "n_blocks"])
        writer.writeheader()
        for name in ("b14", "b12"):
            quality = comparison[name]["quality"]
            writer.writerow({"run": name, **{key: quality.get(key) for key in writer.fieldnames[1:]}})
    with (output_dir / "registration_band_ab_holdout_pairs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx_i", "idx_j", "validation_row", "validation_col", "block_size"])
        for item in comparison["b14"]["holdout_keys"]:
            writer.writerow(item)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--config-b14", required=True)
    parser.add_argument("--config-b12", required=True)
    parser.add_argument("--scene-i", type=int, default=0)
    parser.add_argument("--scene-j", type=int, default=1)
    parser.add_argument("--b14-diagnostics")
    parser.add_argument("--b12-diagnostics")
    parser.add_argument("--output-dir", default="outputs/diagnose_band_c1_compare")
    args = parser.parse_args(argv)
    configs = _load_yaml(args.config_b14), _load_yaml(args.config_b12)
    metadata = validate_band_ab_metadata(
        *configs, scene_indices=(args.scene_i, args.scene_j)
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    if not metadata["valid"]:
        return 1
    if args.preflight_only:
        return 0
    if not args.b14_diagnostics or not args.b12_diagnostics:
        parser.error("comparison requires --b14-diagnostics and --b12-diagnostics")
    with Path(args.b14_diagnostics).open("r", encoding="utf-8") as handle:
        b14 = json.load(handle)
    with Path(args.b12_diagnostics).open("r", encoding="utf-8") as handle:
        b12 = json.load(handle)
    comparison = compare_registration_band_runs(b14, b12)
    _write_comparison_files(comparison, args.output_dir)
    return 0 if comparison["comparison_available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
