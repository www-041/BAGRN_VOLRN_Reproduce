"""Write Task 5 contributor and coverage diagnostics for all eight mosaics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.multiscene_sift.mosaic_diagnostics import run_coverage_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mosaic-root", required=True, type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = run_coverage_audit(args.mosaic_root, args.audit_dir)
    print(f"coverage rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
