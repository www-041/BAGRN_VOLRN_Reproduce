"""Recompute diagnostic mosaic coverage from preserved smoke artifacts only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.spectralmatch_phase2_worker import _coverage


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def recheck(run_root: Path, *, write: bool = False):
    root = run_root.resolve()
    expected_parent = (Path("data/output/benchmark_v1_unified")).resolve()
    if root.parent != expected_parent or not root.name.startswith("spectralmatch_phase2_"):
        raise ValueError("run root must be a task-scoped spectralmatch_phase2_* output directory")
    root_status_path = root / "phase2_run_status.json"
    root_status = _read(root_status_path)
    results = []
    for pair_record in root_status["pairs"]:
        pair_id = pair_record["pair_id"]
        smoke_path = Path(pair_record["smoke_result"])
        smoke = _read(smoke_path)
        pair_dir = smoke_path.parent
        if smoke.get("run_regime") == "DETERMINISTIC_1024_CROP_SMOKE":
            masks = [
                pair_dir / "spectralmatch_inputs" / f"scene_{int(pair_id[:2]):03d}_input_mask_crop.tif",
                pair_dir / "spectralmatch_inputs" / f"scene_{int(pair_id[3:5]):03d}_input_mask_crop.tif",
            ]
        else:
            masks = [Path(smoke["input_artifacts"][side]["frozen_mask_path"]) for side in ("i", "j")]
        mosaic = Path(smoke["stage_outputs"]["mosaic"])
        mosaic_mask = pair_dir / "output_masks" / "mosaic_mask.tif"
        coverage = _coverage(masks, mosaic, mosaic_mask)
        smoke["mosaic_coverage"] = coverage
        smoke["coverage_audit"] = {
            "status": "RECOMPUTED_FROM_PRESERVED_OUTPUTS",
            "method": "exact integer-pixel affine offset; no resampling; frozen union masks unchanged",
            "formal_ranking": "NOT_ELIGIBLE_CROP_DIAGNOSTIC_ONLY",
        }
        pair_record["smoke"] = smoke
        pair_record["benchmark_eligibility"] = "DIAGNOSTIC_ONLY_NOT_FORMAL_RANKING"
        pair_record["checker_interpretation"] = (
            "STRUCTURAL_SCHEMA_AND_FREEZE_VALIDATION_ONLY; does not make a crop smoke eligible for formal ranking"
        )
        results.append({"pair_id": pair_id, "coverage": coverage, "smoke_result": str(smoke_path)})
        if write:
            smoke_path.write_text(json.dumps(smoke, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            pair_status_path = root / pair_id / "pair_run_status.json"
            if pair_status_path.is_file():
                pair_status_path.write_text(json.dumps(pair_record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    root_status["pairs"] = root_status["pairs"]
    if write:
        root_status_path.write_text(json.dumps(root_status, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--write", action="store_true", help="update only derived smoke/status JSON, never raster outputs")
    args = parser.parse_args()
    results = recheck(args.run_root, write=args.write)
    print(json.dumps({"write": args.write, "pairs": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
