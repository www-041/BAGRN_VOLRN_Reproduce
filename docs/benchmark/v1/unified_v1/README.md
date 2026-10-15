# Unified Benchmark V1 Phase 1 usage

Run from repository root in `.venv-registration`:

```powershell
python scripts/benchmark/audit_unified_v1.py --config configs/benchmark_v1_unified.yaml
python scripts/benchmark/freeze_unified_v1.py --config configs/benchmark_v1_unified.yaml --output-root data/output/benchmark_v1_unified/freeze_20261013_v3
python scripts/benchmark/smoke_unified_v1.py --fixture-only
```

The freeze command refuses to overwrite an existing output directory. Use a new explicitly named directory for any rerun; do not edit a locked freeze. Check `freeze_readiness.json` and `freeze_lock.json` before accepting the input freeze.

Future submission validation/evaluation entry points are:

```powershell
python scripts/benchmark/check_unified_submission.py --freeze data/output/benchmark_v1_unified/freeze_20261013_v3 --method-submission path\to\method_submission.json
python scripts/benchmark/evaluate_unified_v1.py --freeze data/output/benchmark_v1_unified/freeze_20261013_v3 --method-submission path\to\method_submission.json --regime B1 --output-root data/output/benchmark_v1_unified\method_run
```

These future commands do not imply that any method has run. Phase 1 only creates a tiny S1–S4 synthetic fixture and evaluates the input freeze. Do not treat B1/B2 overlap consistency as stable-surface truth.
