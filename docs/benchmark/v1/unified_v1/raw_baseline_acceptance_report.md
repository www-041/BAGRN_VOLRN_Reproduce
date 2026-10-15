# Unified Benchmark V1 — RAW Baseline Acceptance Report

Date: 2026-10-14  
Protocol: `DZ01V_UNIFIED_A_B_V1_20261013`  
Run status: **STOPPED — RAW full A/B1/B2 acceptance is NO-GO**

## Scope and preservation

The run used only the locked, unnormalized 13-scene B9 inputs and locked masks under `freeze_20261013_v3`. No Task15/16 code or artifacts, frozen protocol/formulas/windows/masks, or radiometric algorithm was changed. No BAGRN, VOLRN, or external method was run. B3 was not run.

The run output directory is `data/output/benchmark_v1_unified/raw_baseline_v1_20261014/`. It contains the preflight record, event/run logs, and a `STOPPED` solver status. The atomic mosaic writer removed incomplete temporary GeoTIFFs. No final mosaic, mask, source-label raster, method submission, or A/B score table was produced.

## Preflight evidence

| Check | Result |
|---|---|
| Freeze lock | PASS, 54 locked files |
| Canonical scenes | 13; all B9 input and mask paths present, locked, and grid headers valid |
| Grid | EPSG:32650, 14 m, 7992 × 12404 |
| Pair inventory | 78 total; 48 `OVERLAP`; 30 `NO_VALID_OVERLAP` |
| Frozen B2 windows | 1533; 1533 support hashes present |
| Available RAM at preflight | 1,717,846,016 bytes in the immediate read (about 1.60 GiB); the persisted preflight read 1,599,537,152 bytes |
| D: free space | 32,463,667,200 bytes (about 30.2 GiB) |
| Estimated pair spool / exact-order scratch | 3,028,558,800 / 1,586,124,288 bytes |
| Estimated total free-space requirement | 7,356,963,344 bytes, including output and reserve |
| Preflight result | PASS |

The persisted estimates passed; actual execution was still subject to a 512 MiB sampled process-tree RSS ceiling. This is a cooperative polling threshold, **not** an operating-system hard memory limit.

## Actual execution and failure

The RAW fixed-priority mosaic began, then stopped with:

```text
ResourceLimitExceeded:RSS_POLLING_LIMIT_TRIGGERED
```

The event log contains `PREFLIGHT_PASS` followed by `STOPPED`; it contains no `MOSAIC_COMPLETED` event. Therefore the stop occurred inside the mosaic stage, before any B1/B2 scoring. The atomic mosaic writer cleaned its partial GeoTIFF outputs. The freeze inputs were not modified.

The exception path in the current runner did **not** persist `ProcessResourceMonitor.report()`. Consequently, this run does not provide a defensible numeric peak RSS, sample count, scratch high-water mark, or threshold crossing value. These values are **UNKNOWN / NOT RECORDED**, not zero. The immediate post-run process RSS is not a substitute for peak RSS.

## Acceptance matrix

| Requirement | State |
|---|---|
| Freeze lock, scene identity and grid | PASS at preflight |
| 48 overlap + 30 no-overlap rows | Inventory verified; no full scoring output |
| 1533 B2 windows and support hashes | Inventory verified; window scoring not run |
| RAW mosaic union coverage, holes/extras | NOT EVALUATED |
| Mosaic source-label correspondence | NOT EVALUATED |
| A seam and single-source structure | NOT RUN |
| B1 exact seven metrics | NOT RUN |
| B2 1533-window exact seven metrics and structure | NOT RUN |
| Run RSS/scratch telemetry | INCOMPLETE: exception telemetry was not persisted |
| B3 | NOT RUN, as required |
| Overall RAW A/B1/B2 acceptance | **NO-GO** |

## Stop point

No threshold was raised and no retry was launched. The run is stopped at the first real mosaic attempt, as required by the resource guard. A later retry would first need an audited bounded-memory remedy for the mosaic stage and an exception-safe monitor report that persists sampled peak RSS, sample count, scratch peak, trigger state, and stage name. Any later run must use a new output/run ID and preserve this stopped run as history; this report does not authorize that retry.

The synthetic bounded-memory evaluator, mosaic, equivalence, and resource-guard tests are reported separately from this failed real-data run; they do not turn the real RAW acceptance into PASS.

## Verification after the stopped run

- Freeze lock re-check: **PASS** (`verify_lock`, all 54 locked files).
- `tests/benchmark_v1_unified`: **52 passed**.
- `compileall` for the new scorer, mosaic, monitor, runner, and benchmark tests: **PASS**.
- `git diff --check`: **PASS**. Git emitted line-ending normalization warnings for unrelated pre-existing Spectralmatch files; no whitespace error was reported.
- No real B1/B2 score, Mosaic acceptance, or resource peak is claimed by this report.
