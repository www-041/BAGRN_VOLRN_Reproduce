# Unified Benchmark V1 RAW Baseline Implementation Plan

> **For agentic workers:** Execute tasks inline with test-first changes. Do not modify frozen protocol/data, historical Task15/16, or prior experiment outputs.

**Goal:** Produce a resource-bounded, auditable real RAW A/B1/B2 baseline and deterministic provenance-labeled mosaic, or stop with a precise resource/implementation blocker.

**Architecture:** Keep the existing frozen array scorer and metric primitives unchanged as the reference. Add a separate windowed GeoTIFF scorer using exact on-disk order statistics for P95AE, exact W1, and global quantiles; add a deterministic fixed-priority RAW mosaic backend only if an existing backend fails the contract. Run real work only after freeze, input, RAM, disk, and scratch preflight passes.

**Tech Stack:** Python, rasterio, NumPy, SciPy, pytest, psutil (for measured process RSS), temporary disk-backed memmaps.

**Spec:** User request “Unified Benchmark V1：真实RAW基线与全量评分验收” (2026-10-14).

## Global Constraints

- Frozen formulas, support, windows, SHA256, ranking rules, Task15/16, raw inputs, and historical results are read-only.
- No BAGRN, VOLRN, or external method runs.
- All 78 pairs must be accounted for (48 overlap, 30 `NO_VALID_OVERLAP`); all 1,533 windows and support hashes must be validated.
- RDD and percentile metrics remain exact empirical/statistical definitions; no histograms or random sampling.
- Any real run is gated on measured available RAM, volume free space, scratch estimate, and declared resource budget.
- New artifacts use an independent run directory and never overwrite prior outputs.

## Review Focus

- Tiled Sobel must reproduce constant-boundary, 3x3-valid-center behavior including image borders; test halo seams and raster edges.
- Exact percentile/W1 order statistics must match SciPy/NumPy frozen primitives on ties, singleton, and nonuniform fixtures.
- Output nodata, dataset masks, explicit masks, and frozen input mask semantics must remain distinct and fail closed.
- Source labels must reference a valid contributing scene for every mosaic pixel; test missing labels and overlap priority.
- Insufficient RAM/disk or failed measurements must stop before opening a large scoring workload.

---

### Task 1: Freeze/resource/input preflight and backend audit

**Files:**
- Read only: `data/output/benchmark_v1_unified/freeze_20261013_v3/`, `src/benchmark_v1_unified/`, `src/`, `scripts/benchmark/`
- Create: `docs/benchmark/v1/unified_v1/raw_baseline_acceptance_report.md`

**Interfaces:** Record freeze verification, scene/input identity, pair/window counts, exact metric primitives, reusable mosaic backend decision, measured RAM/disk, and computed scratch budget. No algorithm or large raster run in this task.

- [ ] Verify locked freeze, all canonical inputs, all 78 pairs and 1,533 windows.
- [ ] Audit current mosaicking code for fixed priority, exact valid-union support, provenance labels, and reusable output grid.
- [ ] Measure RAM/disk with an available OS-backed tool; record unavailable measurements as blockers.
- [ ] Stop real execution if measured resources do not satisfy a conservative declared budget.

### Task 2: Exact bounded-memory metric primitives

**Files:**
- Create: `src/benchmark_v1_unified/streaming_metrics.py`
- Create: `tests/benchmark_v1_unified/test_streaming_metrics.py`
- Read only reference: `src/benchmark_v1_unified/radiometric.py`, `structure.py`, `seams.py`

**Interfaces:** Window iterators feed exact accumulators and disk-backed order-statistic stores; results preserve frozen metric dataclass/status fields and NumPy/SciPy quantile definitions.

- [ ] Add failing tests comparing block direct/distribution metrics to frozen array functions, including exact P95AE and empirical W1.
- [ ] Add failing tests for tiled Sobel halo/border equivalence and exact Q05/Q50/Q95 order statistics.
- [ ] Implement bounded accumulators, disk-backed exact sort/quantile, halo-aware Sobel, and scratch accounting/cleanup.
- [ ] Run focused tests and assert cleanup and status equivalence.

### Task 3: Streaming submission scorer and deterministic RAW mosaic

**Files:**
- Create: `src/benchmark_v1_unified/streaming_submission_scoring.py`
- Create if no compliant backend exists: `src/benchmark_v1_unified/raw_mosaic.py`
- Create/modify: `scripts/benchmark/score_unified_v1_submission.py` only for an explicit streaming mode, or add `scripts/benchmark/score_unified_v1_streaming.py`
- Create: `tests/benchmark_v1_unified/test_streaming_submission_scoring.py`, `test_raw_mosaic.py`

**Interfaces:** Public streaming entry accepts the same frozen root/submission/regime contract as `evaluate_submission`; it returns equivalent structures/statuses plus scratch and resource telemetry. RAW mosaic emits mosaic, explicit mask, int source labels, and a provenance/config record.

- [ ] Add fixture tests showing streaming A/B1/B2 agrees with array scorer within frozen precision and preserves statuses/window support hashes.
- [ ] Add mosaic tests for exact union, zero holes/extras, source validity, deterministic priority, and label-to-pixel correspondence.
- [ ] Implement pair-at-a-time B1; window-at-a-time B2; chunked A coverage/seam; per-scene structure/range with halo and exact quantiles.
- [ ] Keep exact RDD/P95/order statistics on disk; measure scratch high-water and clean temporary files.
- [ ] Audit and reuse a compliant existing mosaic backend, otherwise implement minimal fixed-priority hard-source selection.

### Task 4: Real RAW A/B1/B2 acceptance run

**Files:**
- Create only under `data/output/benchmark_v1_unified/raw_baseline_v1_20261014/`
- Update: `docs/benchmark/v1/unified_v1/raw_baseline_acceptance_report.md`

**Interfaces:** Run manifest binds protocol/freeze/input SHA, method submission, output hashes, evaluator report, logs, RSS samples, timings, scratch usage, and statuses.

- [ ] Repeat freeze/input/memory/disk preflight immediately before execution.
- [ ] Build independent RAW submission and deterministic mosaic, then verify output hashes/grid/masks/source labels.
- [ ] Execute A/B1/B2 only; emit all pair/window/scene rows, temporal groups, and exact support-hash audit.
- [ ] Verify 48/30 pair accounting, 1,533 window accounting, support hashes, coverage/hole/extra conditions, exact metric status, resource provenance, and no source-tree changes to locked inputs.
- [ ] Report B3 as `NOT_RUN`; do not start BAGRN/VOLRN or other methods.

**No commit/push:** The shared worktree contains unrelated historical user changes; stage/commit only if the user later requests it.
