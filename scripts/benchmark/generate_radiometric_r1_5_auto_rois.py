from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmark_v1_radiometric_r15.cli import execute_generation  # noqa: E402
from benchmark_v1_radiometric_r15.proposals import ProposalConfig  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate immutable, unverified R1.5 radiometric ROI proposals.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pair", choices=("00_01", "02_03", "05_07", "08_10"))
    group.add_argument("--pairs", nargs="+", choices=("00_01", "02_03", "05_07", "08_10"))
    parser.add_argument("--max-candidates", type=int, default=15)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/radiometric_r1_5_auto_roi.yaml")
    args = parser.parse_args(argv)
    try:
        import yaml
        source = args.config if args.config.is_absolute() else ROOT / args.config
        config_data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        allowed = {field for field in ProposalConfig.__dataclass_fields__}
        kwargs = {key: value for key, value in config_data.items() if key in allowed}
        kwargs["max_candidates"] = args.max_candidates
        config = ProposalConfig(**kwargs)
        output = execute_generation(ROOT, [args.pair] if args.pair else args.pairs,
                                    args.run_id, max_candidates=args.max_candidates, config=config)
    except Exception as exc:
        print(f"R1.5_GENERATION_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"R1.5_GENERATION_READY run={args.run_id} pairs={','.join([args.pair] if args.pair else args.pairs)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
