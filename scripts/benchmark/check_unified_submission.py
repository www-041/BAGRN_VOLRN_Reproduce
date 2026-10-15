from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.benchmark_v1_unified.freeze import verify_lock
from src.benchmark_v1_unified.submission import validate_submission

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',required=True);ap.add_argument('--method-submission',required=True);args=ap.parse_args()
    root=Path(args.freeze);lock=verify_lock(root)
    if not lock.valid:print(json.dumps({'status':'FREEZE_INVALID','errors':lock.errors}));return 2
    frozen=json.loads((root/'frozen_manifest.json').read_text(encoding='utf-8'));submission=json.loads(Path(args.method_submission).read_text(encoding='utf-8'));audit=validate_submission(submission,{'input_freeze_sha256':frozen.get('input_freeze_sha256'),'geometry_freeze_sha256':frozen.get('geometry_freeze_sha256'),'band':'B9'})
    print(json.dumps(audit,ensure_ascii=False));return 0 if audit['eligible'] else 2
if __name__=='__main__':raise SystemExit(main())
