"""Execute Task16 without rerunning the frozen Task15 geometry/BAGRN stages."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task16_volrn_comparison import run_task16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/b9_13scene_task16.yaml")
    parser.add_argument("--task15-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="reuse a completed A baseline in a partially populated Task16 directory")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    source = config["source"]["task15_root"]
    output = config["outputs"]["root"]
    task15_root = args.task15_root or (ROOT / source if not Path(source).is_absolute() else Path(source))
    output_root = args.output_root or (ROOT / output if not Path(output).is_absolute() else Path(output))
    protocol = args.protocol or (ROOT / config["metrics"]["protocol"] if not Path(config["metrics"]["protocol"]).is_absolute() else Path(config["metrics"]["protocol"]))
    summary = run_task16(task15_root, output_root, protocol_path=protocol, resume=args.resume, task16_config=config)
    print(f"Task16 status: {summary['status']}")
    print(f"Output: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
