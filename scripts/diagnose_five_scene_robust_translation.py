"""Compare frozen five-scene translation-only robust adjustment variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.multiscene_sift.robust_translation_adjustment import (
    _json_safe,
    load_robust_translation_inputs,
    run_cycle_sensitivity,
    run_translation_variants,
    decide_robust_translation_candidate,
    synthetic_robust_translation_check,
    write_frozen_baseline,
    write_cycle_sensitivity,
    write_robust_translation_decision,
    write_synthetic_robustness,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five-scene-run-dir", required=True, type=Path)
    parser.add_argument("--inlier-recovery-dir", required=True, type=Path)
    parser.add_argument("--global-adjustment-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = load_robust_translation_inputs(
        args.five_scene_run_dir,
        args.inlier_recovery_dir,
        args.global_adjustment_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_frozen_baseline(inputs, args.output_dir)
    result = run_translation_variants(inputs, args.output_dir)
    sensitivity = run_cycle_sensitivity(inputs, result)
    write_cycle_sensitivity(sensitivity, args.output_dir)
    synthetic = synthetic_robust_translation_check()
    write_synthetic_robustness(synthetic, args.output_dir)
    equal_summary = dict(inputs["equal_l2_summary"])
    equal_summary["baseline_reproduction_status"] = result["baseline_reproduction_status"]
    decision = decide_robust_translation_candidate(
        equal_summary,
        {name: value for name, value in result["variants"].items() if name != "EQUAL_L2"},
        sensitivity,
    )
    write_robust_translation_decision(decision, args.output_dir)
    summary = {
        "baseline_reproduction_status": result["baseline_reproduction_status"],
        "baseline_reproduction": result["baseline_reproduction"],
        "huber_delta_px": result["huber_delta_px"],
        "cycle_sensitivity": sensitivity,
        "synthetic_robustness": synthetic,
        "decision": decision,
        "variants": {
            name: {
                "config": value["config"],
                "summary": value["summary"],
                "solution": value["solution"],
            }
            for name, value in result["variants"].items()
        },
    }
    (args.output_dir / "03_variants_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"baseline_reproduction={result['baseline_reproduction_status']}")
    for name, value in result["variants"].items():
        network = value["summary"]["edge_balanced"]
        print(
            f"{name}: zero_one_p95={value['summary']['zero_one']['p95_px']:.8f}, "
            f"mean_edge_rmse={network['mean_edge_rmse_px']:.8f}, "
            f"max_edge_p95={network['max_edge_p95_px']:.8f}"
        )
    return 0 if result["baseline_reproduction_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
