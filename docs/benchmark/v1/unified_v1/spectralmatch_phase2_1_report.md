# Spectralmatch Phase 2.1 — implementation and diagnostic report

Date: 2026-10-13  
Method branch/base: `fix/spectralmatch-phase2-1` / `85429ba5e221b5ed3901569a86d5658de12603ad`  
Scope: implementation, synthetic verification, and read-only review of prior retry5 diagnostics. No new Spectralmatch crop algorithm run was allowed by the runtime gates.

## Executive result

Phase 2.1 adds a fail-closed process-tree RSS runner, an independent source-valid seam assignment variant, explicit PIF-mode plumbing and diagnostics, Git-identity-aware but content-strict freeze audit, and strict external A/B scoring. Synthetic/contract tests pass. New real-crop measurements are **blocked**, not successful or failed algorithm trials: the candidate runtime has mismatched GDAL libraries and no usable `psutil` monitor. The scoring entry point also refuses formal scoring because six files in the frozen code manifest do not match their locked SHA256 values.

No 13-scene experiment, formal ranking, Task15/Task16 run, or R1/R1.5 mutation occurred.

## Comparison table

“Retry5” rows below are historical measurements from `spectralmatch_phase2_20261013_retry5`; they are not new Phase 2.1 results. Metrics are unavailable where the current environment gate prevented execution.

| Run / variant | Status and scope | Coverage | 00_01 radiometry (MAE / RMSE / MAMD / MSDD / RDD) | Runtime / RSS | Interpretation |
|---|---|---:|---|---|---|
| Official Voronoi seam, retry5 00_10 | Historical, 1024×1024 diagnostic crop | 748,801 / 959,372 = **78.051%**; missing 210,571; extras 0 | N/A (00_10 seam audit) | Historical worker 96.75 s; runner 107.063 s; RSS samples 0, peak N/A; 2 GiB OS hard limit not evidenced | Incomplete coverage. |
| Source-valid coverage variant | `PRECONDITION_BLOCKED`; not run on either crop | N/A | N/A | N/A | Synthetic assignment tests only; no crop acceptance claim. |
| PIF `flood_from_match_points` | `PRECONDITION_BLOCKED`; 00_01 crop not run | N/A | N/A | N/A | PIF support/count/distribution and stage metrics unavailable. |
| PIF `entire` | `PRECONDITION_BLOCKED`; 00_01 crop not run | N/A | N/A | N/A | PIF support/count/distribution and stage metrics unavailable. |
| Input, retry5 00_01 | Historical, fixed 360,981-pixel crop support | N/A | **37.711 / 61.482 / 0.110 / 0.735 / 1.070** | Historical worker 93.218 s; runner 110.078 s; RSS samples 0, peak N/A; hard cap not evidenced | Input baseline only. |
| Global Regression, retry5 00_01 | Historical, same support | N/A | **746.452 / 749.928 / 746.451 / 37.605 / 746.451** | Included in above historical run; RSS unmeasured | Strong deterioration in this crop. |
| Local Block Adjustment, retry5 00_01 | Historical, same support | N/A | **57.624 / 83.493 / 0.054 / 44.859 / 29.552** | Included in above historical run; RSS unmeasured | Better than the global stage on MAE/RMSE but still worse than Input on those metrics; not a new trial. |

The retry5 per-scene means/std/P05/P95 are in `spectralmatch_phase2_20261013_retry5/diagnostics_pixel_audit_v2/00_01_stage_metrics.csv`. The recorded global adjustments were scene 000 scale `0.9550533295`, offset `-199.37361145`; scene 001 scale `1.0495390892`, offset `219.74446106`, from `00_01_radiometric_stage_audit.json`. The opposite offsets and distinct gains are consistent with the observed post-global mean separation. Why the upstream estimator selected those parameters is not established by this audit; the PIF sample mask/count/distribution was not exported, so no causal claim about its sample selection is made.

## Seam coverage finding

The retained retry5 pixel audit is `data/output/benchmark_v1_unified/spectralmatch_phase2_20261013_retry5/diagnostics_pixel_audit_v2/00_10_coverage_pixel_audit.json`.

| 00_10 stage | Valid pixels | Missing from frozen input union | Extra pixels |
|---|---:|---:|---:|
| Frozen input union | 959,372 | 0 | 0 |
| Footprints union | 959,372 | 0 | 0 |
| Voronoi seamline union | 748,801 | 210,571 | 0 |
| Seam-clipped union | 748,801 | 210,571 | 0 |
| Final mosaic | 748,801 | 210,571 | 0 |

Missing pixels classify as `scene_i_only=14,572`, `scene_j_only=0`, `both_overlap=195,999`. Therefore the gap already exists in the seamline assignment and is carried through clipping and final mosaicking; the audit does not implicate raster merge as the first point of loss. The replay test reproduces the saved official EMP/cutline selection and dropped-piece geometry exactly. This confirms the observed `_segment_emp()` piece-selection behavior for this crop, not that it is the only possible cause across inputs.

The upstream `weighted_seamline` strategy is present, but **was not run on the actual crops** because the runtime gate failed. No official alternative or coverage-retaining variant is accepted yet. The independent variant assigns only pixels present in a source-valid mask, tracks source labels, and passes synthetic one-source/overlap/no-source tests; real-crop 100% coverage, zero-extra coverage, time, and provenance remain unverified.

## Environment and resource monitoring

Candidate worker runtime audit: Python 3.11.17, `osgeo.gdal` 3.11.5, Rasterio-linked GDAL 3.10.3, PROJ 9.6.0. Python/PROJ satisfy declared ranges, but GDAL is inconsistent, so status is `INCOMPATIBLE`. No `conda`, `mamba`, or `micromamba` solver was found in PATH/common install locations, and no existing environment was modified. `psutil` is unavailable in the candidate worker environment.

The new runner refuses to start the worker if either compatible runtime or RSS monitor preconditions fail. Unit tests verify normal sampling, process-tree timeout/threshold termination, unavailable-monitor refusal, and telemetry that explicitly distinguishes polling from an OS hard limit. Polling is **not** a hard 2 GiB OS memory cap; `os_hard_memory_limit_applied=false`. Historical retry5 telemetry has zero samples and null peak, so it cannot establish actual peak RSS or the cause of any earlier whole-image termination.

The preserved retry5 monitor records are diagnostic evidence only; no fresh crop timing/RSS table is available for Phase 2.1.

## PIF comparison

Separate diagnostic run directories exist for `flood_from_match_points` and `entire`, but both statuses are `PRECONDITION_BLOCKED`, `algorithm_started=false`. The summary is `data/output/benchmark_v1_unified/spectralmatch_phase2_1_20261013/pif_sensitivity/pif_sensitivity_summary.json`.

| PIF mode | Run status | Evaluation support hash/count | PIF mask/count/distribution | Global/Local metrics |
|---|---|---|---|---|
| `flood_from_match_points` | PRECONDITION_BLOCKED | NOT_VERIFIED / unavailable | NOT_AVAILABLE | unavailable |
| `entire` | PRECONDITION_BLOCKED | NOT_VERIFIED / unavailable | NOT_AVAILABLE | unavailable |

The adapter records the selected mode, but the upstream API does not expose PIF mask, count, or spatial distribution in its current output contract. The diagnostic reports this limitation rather than inferring values. No mode comparison or winner selection can be claimed.

## Freeze audit and scoring

The method Git HEAD differs from the historical locked HEAD; Phase 2.1 now reports that separately and does not treat the Git identity difference alone as a content-lock failure. Strict Phase 1 content hashes remain enforced. The current audit is `FREEZE_INVALID` with six `CODE_FILE_HASH_MISMATCH` entries:

- `docs/benchmark/v1/unified_v1/benchmark_protocol_draft.md`
- `docs/benchmark/v1/unified_v1/plan.md`
- `src/benchmark_v1_unified/__init__.py`
- `src/benchmark_v1_unified/identity.py`
- `src/benchmark_v1_unified/structure.py`
- `tests/benchmark_v1_unified/test_core.py`

All 13 frozen raster/mask inputs checked PASS; pair inventory remains 78 total, 48 positive-overlap and 30 no-valid-overlap. The six content mismatches pre-exist this Phase 2.1 change set; none of the listed files was edited here. The Phase 2.1 scorer calls the freeze audit and therefore blocks real formal scoring while this state persists. Do not regenerate or weaken the freeze to make scoring pass.

The new A/B1/B2 scorer has synthetic tests for exact grid and support, no support shrinking, missing outputs, mask ambiguity, crop-only rejection, B2 frozen windows, and explicit B3 `NOT_RUN_B3`. It is not evidence of real Spectralmatch performance. Full-support real scoring, seam metrics requiring source provenance, and a formally valid score remain blocked.

## Verification

- Phase 2.1 focused suite after final review fixes: **42 passed, 2 warnings**.
- Full repository pytest: **collection interrupted with 16 errors**, all reported errors due missing `scikit-image` in the isolated worker environment (`ModuleNotFoundError: No module named 'skimage'`).
- Running the Phase 2.1 and frozen-core suites together additionally triggered Windows native exception `0xc06d007f` in the existing `benchmark_v1_unified/aligned.py` / Rasterio-GDAL path. No frozen source was modified to work around it.
- Real candidate environment: incompatible GDAL and unavailable RSS monitor; algorithm launch blocked.
- Phase 1 freeze audit: `FREEZE_INVALID`, six locked-code SHA mismatches as listed above.

### Final review fixes

An independent read-only review found three issues; all were reproduced and fixed with regression tests:

1. Zero process RSS samples while the worker is alive now yields `MONITOR_ERROR` and terminates the run (instead of waiting for timeout/completion).
2. Failure to create a `psutil.Process` handle now uses a Windows process-table snapshot and terminates/verifies the captured process tree; the regression test confirms the synthetic descendant exits.
3. The top-level scorer now preserves Experiment A's regime-specific eligibility flag instead of overwriting it with the B1/B2 `FULL_SUPPORT` status rule.

The focused tests for these cases pass as part of the 42-test suite.

## Decision and next gate

Do not begin full-pair resource preflight or 13-scene execution yet. First resolve the six frozen-code hash mismatches through the approved freeze governance process, then provision a new isolated environment with mutually consistent `osgeo.gdal` and Rasterio-linked GDAL (meeting upstream requirements) plus `psutil`. Re-run environment/freeze gates. Only after both crop gates pass should the two existing 1024×1024 diagnostic crops be run for official/variant seam coverage and both PIF modes. The 13-scene run remains unauthorized in this phase.

## Data-protection audit

Phase 2.1 changes are limited to the Phase 2 runner integration, Phase 2.1 modules/CLIs/tests, and this report. Phase 1 locked paths, frozen rasters, Task15/Task16, Ours, retry5 results, R1/R1.5 histories, and human review records were not edited. No new real-crop algorithm output is represented as a result.
