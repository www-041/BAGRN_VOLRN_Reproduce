# Implementation ledger — DZ01V Unified Benchmark V1 Phase 1

Plan source: `C:\Users\wang\Desktop\plan.md`; the repository copy at `docs/benchmark/v1/unified_v1/plan.md` is byte-identical (SHA256 `54215B6A5F118E03FF702F5EF27843500FBFDC962B775FF560EFC7B216E9DC25`).

## Checklist

- [x] P0 read-only source/formula/geometry/hash audit and crosswalk reports.
- [x] P1 isolated config, source identity/geometry audit, aligned input masks, 78-pair inventory, B2 windows and SHA freeze.
- [x] P2 metric/evaluation/report/runtime interfaces and fixture-only synthetic smoke.
- [x] P3 tests T01-T26, synthetic fixture, readiness report and final verification.
- [x] Whole-change review and immutable freeze-lock verification.

## Scope ruling

Work stayed in the new namespace and outputs: `src/benchmark_v1_unified/`, `scripts/benchmark/`, `configs/benchmark_v1_unified.yaml`, `tests/benchmark_v1_unified/`, `docs/benchmark/v1/unified_v1/`, and `data/output/benchmark_v1_unified/`. No Task15/16, R1/R1.5, Phase3/4/5 algorithm, frozen record or formal result was edited. No external radiometric method, Task15/16 rerun, registration Phase5 run or real-scene score was produced.

## Verified evidence

- Current B9 source hashes: 13/13 match `recorded_source_sha256_task15`.
- Canonical geometry: EPSG:32650, 14 m, affine `(14,0,633346,0,-14,3359874)`, 7992×12404; 13 Task15 transforms.
- Fresh aligned inputs reproduce all 13 Task15 input masks exactly. Initial blockwise reprojection differed by 16,824 pixels for scene 1; it was rejected, and Task15-equivalent local-window float64 bilinear reprojection is used in the final freeze.
- Pair inventory: 78 total, 48 positive support, 30 `NO_VALID_OVERLAP`. Native `00_01` support is 1,291,672 versus canonical-grid support 1,295,654; native and canonical `00_10` are both 1,374,076. Both definitions remain recorded.
- B2: 1,533 deterministic spatially dispersed 128×128 windows.
- Final freeze `data/output/benchmark_v1_unified/freeze_20261013_v3/`: status `FROZEN`, 54 locked files, 38-entry code/test manifest, zero lock/hash mismatches.
- Synthetic S1-S4 smoke passed on tiny arbitrary-unit fixtures; NPZ and generator hashes are locked. No actual normalization algorithm was run.
- Final full pytest: 1,283 passed, 4 skipped, 0 failed (175.99 s). Skips: two SIFT synthetic graph cases; EfficientLoFTR and LightGlue integration environment prerequisites. `compileall` and `git diff --check` passed.
- The baseline-only Phase5 runner failure did not recur in the final full suite. Phase5 source was not changed.
- Exact paper CD/GL formulas remain unverified; no W1-as-CD or gradient proxy-as-GL claim is made.
