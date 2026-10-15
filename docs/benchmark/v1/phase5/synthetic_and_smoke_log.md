# Phase5 synthetic and real-smoke log

Snapshot: 2026-10-11. This report records tests and preflight only; it is not a benchmark result.

## Synthetic verification

- Frozen robust-fit fixture: known native displacement `(delta_row, delta_col) = (+2.25, -3.5)` with four injected outliers; deterministic recovery, reverse direction, tie-breaking and named minimum-count/inlier/coverage gates are covered by unit tests.
- Coordinate fixtures cover nonzero affine origins, pixel-center conversion, crop/resize round trips and per-scene masks.
- CLI synthetic runner verifies protocol lock, result persistence before scoring, status separation and truth isolation. External adapter tests are parser/mock/sign-contract tests, not third-party library runs.
- Latest Task1–7 focused regression: **105 passed**, 1205 warnings (primarily Rasterio/Affine synthetic reprojection/deprecation warnings). Runner-focused suite: **12 passed**, 1 Affine warning.
- `compileall` for Phase5 source and runner/worker scripts: exit 0.

## Read-only input/dependency preflight

Command:

```powershell
.venv-registration/Scripts/python.exe scripts/run_benchmark_v1_phase5.py --config configs/benchmark_v1_phase5.yaml --preflight
```

Observed: `PREFLIGHT_COMPLETE`; all 8 B9 identities required by the four frozen pairs verified; Phase3 assignment SHA256 `b4d11d86a0700dca5925d2294f3a426ed145b2f907e125fa0becc39ce9966e60`; config SHA256 `d402fa6296c28a9d3d213f3121be124de8d721e082329b2e2c005d610679046c`; inference false. The exact frozen configuration is recorded in `configs/benchmark_v1_phase5.yaml`.

## 00_01 SIFT smoke

**Not run: `BLOCKED_RESOURCE_PREFLIGHT`.** Before inference, available system memory was approximately 2.28 GiB, while the frozen worker RSS ceiling is 12 GiB. The smoke must not be made to fit by changing input size, estimator settings or resource limits. No alternate pair or method was substituted. No 00_01 Matcher result exists, and no smoke output directory was created.

Real common-view overlap preparation was therefore not validated end-to-end in this task. Reconsider the exact SIFT command only when resource preflight is safe under the frozen config; this is still a single-pair smoke, not a formal score.

## Scope assurance

No four-pair or 15-pair Matcher run, Task15, Task16, or radiometric experiment was executed. Phase3/4 frozen inputs and original B9 data were read-only.
