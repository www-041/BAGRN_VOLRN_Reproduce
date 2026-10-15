"""Write a separate, reproducible input/code hash attestation for one completed run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    status = json.loads((args.run / "run_status.json").read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    freeze = root.parent / "data/output/benchmark_v1_unified/freeze_20261013_v3"
    paths = []
    for scene in status["adaptation"]["scenes"]:
        paths.extend(Path(p) for p in scene["source_paths"].values())
        i = scene["scene_index"]
        paths.extend([freeze / "aligned_inputs" / f"scene_{i:03d}_B9_input.tif",
                      freeze / "aligned_inputs" / f"scene_{i:03d}_input_mask.tif"])
    paths.extend([freeze / "freeze_lock.json", freeze / "fixed_windows.csv",
                  root.parent / "data/output/final_pipeline/b9_13scene_formal_v2/stages/04_canonical_warp/task14_source_config.json",
                  root.parent / "data/output/final_pipeline/b9_13scene_formal_v2/stages/03_global_adjustment/global/translation_l2/global_transforms.json"])
    code = [root / "src/benchmark_v1_lsstaa" / name for name in
            ("__init__.py", "core.py", "diagnostics.py", "raster.py", "experiment.py", "scoring.py")]
    code.extend([root / "scripts/benchmark/run_lsstaa_three_scene_exploratory.py",
                 root / "scripts/benchmark/validate_lsstaa_official.py"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    out = {"status": status["three_scene_status"], "git_branch": "research/lsstaa-python-20261014",
           "git_head_base": head, "working_tree_code_uncommitted": True,
           "official_commit": "6bd81a1374927eecf5fd8b3843fa470e2e6c055d",
           "input_sha256": {str(p): digest(p) for p in paths},
           "code_sha256": {str(p.relative_to(root)): digest(p) for p in code}}
    target = args.run / "input_code_attestation.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"attested {len(paths)} inputs and {len(code)} code files: {target}")


if __name__ == "__main__":
    main()
