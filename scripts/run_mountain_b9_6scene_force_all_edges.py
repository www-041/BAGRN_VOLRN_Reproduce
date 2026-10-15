"""Run a deliberately permissive six-scene B9 experiment.

This launcher is isolated from the standard Task15 launcher.  It changes
registration acceptance gates only in this Python process, preserves every
weak edge that produces a finite affine model, and writes to a separate
output root.  It must not be interpreted as a standard scientific run.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import load_pipeline_config
from src.pipeline.runner import run_pipeline


CONFIG = ROOT / "configs/mountain_b9_6scene.yaml"
FORCE_OUTPUT_ROOT = Path("data/output/final_pipeline/mountain_b9_6scene_force_all_models")


def install_force_registration_gates() -> dict[str, object]:
    """Relax registration gates locally for this experimental launcher only."""
    from src.multiscene_sift import pairwise
    from src.registration_benchmark import geometry

    # Three correspondences are the mathematical minimum for an affine model.
    # The ratio gate is disabled so low-ratio, finite models are retained.
    geometry.MIN_INLIERS = 3
    geometry.MIN_INLIER_RATIO = 0.0
    geometry._is_geometry_valid = lambda *_args: True
    pairwise.MIN_INLIERS = 3
    pairwise.MIN_INLIER_RATIO = 0.0
    return {
        "minimum_inliers": 3,
        "minimum_inlier_ratio": 0.0,
        "residual_threshold_px": 2.0,
        "allow_invalid_geometry": True,
        "scope": "process-local experimental launcher only",
    }


def _experimental_config():
    config = load_pipeline_config(CONFIG)
    return replace(config, outputs=replace(config.outputs, root=FORCE_OUTPUT_ROOT))


def _write_force_metadata(output_root: Path, gates: dict[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "FORCE_ALL_FINITE_AFFINE_EDGES",
        "warning": "Experimental only; standard registration acceptance gates were relaxed.",
        "gates": gates,
        "expected_risk": "Weak edges, including edges with only a few inliers, may distort global transforms and radiometric normalization.",
        "standard_output_preserved": True,
    }
    (output_root / "force_mode.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the six-scene mountain B9 pipeline with permissive registration gates"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fresh", action="store_true", help="start the isolated force experiment")
    group.add_argument("--resume", action="store_true", help="resume the isolated force experiment")
    parser.add_argument("--from-stage", type=int, default=None)
    args = parser.parse_args(argv)

    gates = install_force_registration_gates()
    config = _experimental_config()
    output_root = ROOT / FORCE_OUTPUT_ROOT
    try:
        result = run_pipeline(
            config,
            mode="resume" if args.resume else "fresh",
            from_stage=args.from_stage,
        )
    except Exception:
        _write_force_metadata(output_root, gates)
        raise
    _write_force_metadata(output_root, gates)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
