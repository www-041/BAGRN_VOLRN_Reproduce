from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.benchmark_v1_unified.freeze import verify_lock
from src.benchmark_v1_unified.submission import validate_submission
from src.benchmark_v1_unified.reporting import write_json_atomic

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',required=True);ap.add_argument('--method-submission',required=True);ap.add_argument('--regime',choices=['A','B1','B2','B3'],required=True);ap.add_argument('--output-root',required=True);args=ap.parse_args()
    fr=Path(args.freeze);verification=verify_lock(fr)
    if not verification.valid:raise SystemExit(f'freeze verification failed: {verification.errors}')
    frozen=json.loads((fr/'frozen_manifest.json').read_text(encoding='utf-8'));method=json.loads(Path(args.method_submission).read_text(encoding='utf-8'))
    audit=validate_submission(method,{'input_freeze_sha256':frozen.get('input_freeze_sha256'),'geometry_freeze_sha256':frozen.get('geometry_freeze_sha256'),'band':'B9'})
    if not audit['eligible']:raise SystemExit(f'method submission rejected: {audit["errors"]}')
    status='NOT_RUN_MISSING_REGIME_ARTIFACTS'
    if args.regime=='B3':status='NOT_RUN_SYNTHETIC_OUTPUT_ARTIFACT_REQUIRED'
    report={'protocol_id':frozen['protocol_id'],'regime':args.regime,'method_id':method['method_id'],'status':status,'submission_audit':audit,'scores':None,'reason':'Phase 1 does not run external methods; this entry point refuses to infer scores from missing artifacts.'}
    dest=write_json_atomic(Path(args.output_root)/'evaluation.json',report);print(json.dumps({'status':status,'report':str(dest)},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
