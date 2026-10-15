from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_v1_unified.method_products import prepare_method_submission


def main():
    parser = argparse.ArgumentParser(description="Validate and package existing Unified V1 method products; creates no rasters.")
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--method-config", required=True, help="JSON containing already-produced paths and solver/runtime states")
    parser.add_argument("--output", required=True, help="Output metadata JSON; NOT_READY records contain no submission")
    args = parser.parse_args()
    config_path = Path(args.method_config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = prepare_method_submission(config, args.freeze, config_dir=config_path.parent)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "reasons": result["reasons"], "output": str(output)}, ensure_ascii=False))
    return 0 if result["status"] in {"READY_FOR_SCORING", "PAIR_ONLY_DIAGNOSTIC"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
