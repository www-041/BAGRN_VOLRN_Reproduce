# Phase5 researcher-run commands

Do not launch these until reviewing [readiness](benchmark_phase5_readiness.md), resolving method/environment limitations, and confirming sufficient resources. Each invocation is one explicit pair and one method; outputs must be new directories. Development checkpoints are for development only and do not make results independent spatial-held-out generalization.

## Optional single-pair SIFT smoke

The following command was **not executed** because the resource preflight found only about 2.28 GiB available versus the frozen 12 GiB RSS ceiling. Run only after resources recover; do not alter the frozen input or estimator to force execution.

```powershell
.venv-registration/Scripts/python.exe -u scripts/run_benchmark_v1_phase5.py --config configs/benchmark_v1_phase5.yaml --pair 00_01 --method SIFT --smoke --output-root data/output/benchmark_v1_phase5/runs/00_01_sift_smoke
```

## Four-pair DEVELOPMENT run (manual, researcher-started only)

This loop issues 24 separate one-pair/one-method processes. It has **not** been executed. Review each method's availability first; do not interpret `ENV_UNAVAILABLE`, `WEIGHTS_UNAVAILABLE`, no overlap or algorithm failure as a successful score.

```powershell
$pairs = @('00_01','02_03','05_07','08_10')
$methods = @('SIFT','KARIOS','AROSICS_GLOBAL','LoFTR','EfficientLoFTR','LightGlue+DISK')
foreach ($pair in $pairs) {
  foreach ($method in $methods) {
    $out = "data/output/benchmark_v1_phase5/runs/dev_${pair}_${method}"
    .venv-registration/Scripts/python.exe -u scripts/run_benchmark_v1_phase5.py --config configs/benchmark_v1_phase5.yaml --pair $pair --method $method --development --output-root $out
    if ($LASTEXITCODE -ne 0) { Write-Warning "Run failed: $pair / $method (exit $LASTEXITCODE); inspect its status.json before continuing." }
  }
}
```

To prevent a later command from replacing prior results, each output root is pair/method-specific. The runner itself refuses an existing output directory. Inspect `status.json`, `method_result.json`, provenance and classified failure reason for every invocation. AROSICS global results use its native COREG solver and are not equivalent to feature-matcher RANSAC results. `GEOREF_ONLY` is written as a separate diagnostic baseline, not included in the six-method macro ranking.

The frozen DEVELOPMENT split is 43 points total across four pairs (11, 9, 9, 14). No TEST coordinates or second-review records enter adapters. Matcher workers may read only the declared common overlap pixels/masks; TEST labels remain isolated. The split is not a fully spatial-held-out image-pixel experiment.
