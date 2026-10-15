# Spectralmatch Phase 2.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Phase 2 smoke execution fail-safe, diagnose and close seam coverage gaps in a separate variant, compare both official PIF modes, add strict frozen-protocol A/B scoring, and keep Phase 1 locks unchanged.

**Architecture:** Put all Phase 2.1 behavior in `src/benchmark_v1_spectralmatch_phase2/` and new Phase 2.1 CLIs. Reuse—but do not modify—Phase 1 frozen metric, grid, window, schema, and hash artifacts. Keep official Spectralmatch outputs and the coverage-retaining variant under separate run/method identities. Run real crop trials only in a Phase 2.1 isolated environment whose `osgeo.gdal` and Rasterio-linked GDAL are identical and meet upstream minimums.

**Tech Stack:** Python, pytest, rasterio, NumPy, GDAL bindings from the isolated Spectralmatch environment; `psutil` required for tree RSS polling.

**Spec:** `docs/superpowers/specs/2026-10-13-spectralmatch-phase2-1-design.md`

## Global Constraints

- Base commit: `85429ba5e221b5ed3901569a86d5658de12603ad`.
- Never modify Phase 1 frozen inputs, masks, windows, lock files, schema, frozen evaluator files, or files listed in `code_manifest.json`.
- Do not modify Spectralmatch upstream math, Task15/16, Ours, or prior smoke outputs.
- No 13-scene or all-48-pair formal runs; only synthetic tests and the existing `00_01`/`00_10` 1024×1024 crops.
- Crop outputs must carry `DIAGNOSTIC_CROP_ONLY` and cannot enter formal ranking.
- Never reduce frozen scoring support to match method output coverage.
- Distinguish sampled/polling RSS protection from a true OS hard memory limit; report the latter as not applied unless actually installed.

## Review Focus

1. Missing/failed RSS backend must prevent child launch; its test must assert a sentinel worker did not start.
2. Timeout/RSS termination must kill descendants, not only the root; its test must spawn and verify a child process exits.
3. Coverage fallback must never invent pixels outside valid source masks; its test must include one-source, overlap, and no-source cells.
4. B1/B2 must fail on any missing frozen support pixel instead of silently masking it away; tests must assert exact failure statuses and denominators.
5. New commits must not invalidate frozen code/input hashes; tests must distinguish locked-file hash drift from method Git HEAD changes.

---

### Task 1: Isolated environment compatibility audit

**Files:**
- Create: `src/benchmark_v1_spectralmatch_phase2/environment.py`
- Create: `scripts/benchmark/audit_spectralmatch_phase2_1_env.py`
- Test: `tests/benchmark_v1_spectralmatch_phase2/test_environment.py`

**Interfaces:**
- `audit_runtime() -> dict` reports Python, `osgeo.gdal.VersionInfo()`, `rasterio.__gdal_version__`, PROJ, minimum-version checks, GDAL consistency, and an overall `COMPATIBLE`/`INCOMPATIBLE` state.

- [ ] **Step 1: Add tests** for matching versions, mismatched Rasterio/GDAL, and each below-minimum version.
- [ ] **Step 2: Run tests and verify failures** against the absent Phase 2.1 audit function.
- [ ] **Step 3: Implement the version comparison and CLI**; do not edit `.venv-registration`, the existing Spectralmatch environment, upstream checkout, or freeze files.
- [ ] **Step 4: Inspect official environment declarations and perform a dependency solve in a new Phase 2.1-only prefix.** Create/use that environment only if GDAL ≥3.11, PROJ ≥9.3, Python 3.10–3.12, and Rasterio's linked GDAL matches `osgeo.gdal`; otherwise record the incompatibility and do not run real crop trials.
- [ ] **Step 5: Verify**: `python -m pytest tests/benchmark_v1_spectralmatch_phase2/test_environment.py -q`, then run the audit CLI inside the candidate environment and retain JSON evidence.

### Task 2: Fail-closed process-tree resource monitor

**Files:**
- Create: `src/benchmark_v1_spectralmatch_phase2/resource_monitor.py`
- Modify: `scripts/benchmark/run_spectralmatch_phase2.py`
- Test: `tests/benchmark_v1_spectralmatch_phase2/test_resource_monitor.py`
- Test: `tests/benchmark_v1_spectralmatch_phase2/test_runner_selection.py`

**Interfaces:**
- `run_monitored(command, *, cwd, env, timeout_s, rss_limit_bytes, sample_interval_s, output_dir) -> dict` returns `status`, `return_code`, `wall_seconds`, `peak_rss_bytes`, `rss_sample_count`, `process_tree_terminated`, `polling_limit_enforced`, and `os_hard_memory_limit_applied`.
- No monitor/backend available returns `MONITOR_UNAVAILABLE` before spawning the child.

- [ ] **Step 1: Write failing lifecycle tests** for unavailable psutil (sentinel child must not run), normal exit, timeout, RSS threshold trigger, and child-tree termination.
- [ ] **Step 2: Run the focused tests and confirm expected failures** with the missing module/function or current unsafe launch behavior.
- [ ] **Step 3: Implement the monitor** with preflight psutil validation, recursive process sampling, whole-tree termination, and explicit `os_hard_memory_limit_applied=False`.
- [ ] **Step 4: Wire the Phase 2.1 runner to fail closed** and preserve telemetry for blocked, completed, timeout, and resource-limit states.
- [ ] **Step 5: Run focused tests**: `python -m pytest tests/benchmark_v1_spectralmatch_phase2/test_resource_monitor.py tests/benchmark_v1_spectralmatch_phase2/test_runner_selection.py -q`.

### Task 3: Official seam strategy audit and source-valid coverage variant

**Files:**
- Create: `src/benchmark_v1_spectralmatch_phase2/coverage_seams.py`
- Create: `scripts/benchmark/run_spectralmatch_phase2_1_seams.py`
- Test: `tests/benchmark_v1_spectralmatch_phase2/test_coverage_seams.py`
- Test: `tests/benchmark_v1_spectralmatch_phase2/test_crop_seamline_replay.py`

**Interfaces:**
- `assign_complete_coverage(valid_i, valid_j, official_labels, priority) -> (labels, audit)` returns one source label for every pixel in `valid_i | valid_j`, `-1` elsewhere, and counts of official assignments/fallbacks/source-invalid pixels.
- `validate_assignment(labels, valid_i, valid_j, expected_union) -> audit` checks exact union coverage, no extras, and per-pixel source validity.

- [ ] **Step 1: Add synthetic tests** for one-source regions, overlap, missing official assignments, invalid official source labels, and no-source cells; require no-source support to fail explicitly.
- [ ] **Step 2: Add a deterministic replay test** using preserved 00_10 crop EMP/cutline inputs; assert official `_segment_emp()` selection reproduces the two saved seam geometries and dropped-piece areas.
- [ ] **Step 3: Audit upstream-supported seamline strategies** on existing 00_01/00_10 crops only; record each strategy, output coverage, runtime, and source provenance before selecting a fallback.
- [ ] **Step 4: Implement the independent coverage-retaining assignment** that preserves official valid labels and fills only unassigned cells from valid source masks using a fixed, deterministic priority; do not edit upstream source or official outputs.
- [ ] **Step 5: Run official and variant crop trials** into a new `spectralmatch_phase2_1_20261013/` directory; require both variants to report exact 100% input-union coverage, zero extras, and zero no-source pixels. Preserve per-pixel source labels and separate IDs.
- [ ] **Step 6: Run tests**: `python -m pytest tests/benchmark_v1_spectralmatch_phase2/test_coverage_seams.py tests/benchmark_v1_spectralmatch_phase2/test_crop_seamline_replay.py -q`.

### Task 4: 00_01 PIF-mode sensitivity diagnostic

**Files:**
- Modify: `src/benchmark_v1_spectralmatch_phase2/adapter.py`
- Modify: `scripts/benchmark/spectralmatch_phase2_worker.py`
- Modify: `scripts/benchmark/run_spectralmatch_phase2.py`
- Create: `scripts/benchmark/run_spectralmatch_phase2_1_pif.py`
- Test: `tests/benchmark_v1_spectralmatch_phase2/test_pif_mode_diagnostic.py`

**Interfaces:**
- Worker accepts explicit `pif_method` in `{flood_from_match_points, entire}` and emits it in metadata.
- Diagnostic output records the unchanged frozen support identity, PIF mask/hash and count when available, spatial distribution summary, readable global parameters, stage metrics/timings, and a reason for any unavailable artifact.

- [ ] **Step 1: Test PIF mode validation and support invariance** with synthetic worker metadata; reject unknown modes and any changed evaluation-mask hash.
- [ ] **Step 2: Run the focused test to confirm the missing mode/output behavior.**
- [ ] **Step 3: Thread the explicit mode through the Phase 2 adapter and worker** without changing upstream math or the default official retry5 artifacts.
- [ ] **Step 4: Run the two settings on only the preserved 00_01 1024 crop**, using identical frozen mask and parameters; write separate run IDs and a side-by-side diagnostic table. Do not select a winner or alter scoring support.
- [ ] **Step 5: Run test**: `python -m pytest tests/benchmark_v1_spectralmatch_phase2/test_pif_mode_diagnostic.py -q`.

### Task 5: Frozen Git identity audit without weakening content locks

**Files:**
- Modify: `src/benchmark_v1_spectralmatch_phase2/freeze_audit.py`
- Test: `tests/benchmark_v1_spectralmatch_phase2/test_freeze_audit.py`

**Interfaces:**
- Audit result retains `locked_git_head`, adds `method_git_head`, and keeps strict lock, manifest, and listed-file SHA256 checks.
- Current method HEAD mismatch alone is informational, not a freeze error.

- [ ] **Step 1: Add tests** showing a changed method HEAD passes when all freeze/file hashes match, while a changed locked source file still fails.
- [ ] **Step 2: Confirm the new test fails specifically on current `GIT_HEAD_MISMATCH` behavior.**
- [ ] **Step 3: Implement separate historical lock and current method commit reporting** without writing to the frozen directory.
- [ ] **Step 4: Run test**: `python -m pytest tests/benchmark_v1_spectralmatch_phase2/test_freeze_audit.py -q`.

### Task 6: Strict Phase 2.1 A/B external scoring entry point

**Files:**
- Create: `src/benchmark_v1_spectralmatch_phase2/scoring.py`
- Create: `scripts/benchmark/evaluate_unified_v1_phase2_1.py`
- Test: `tests/benchmark_v1_spectralmatch_phase2/test_scoring.py`
- Test: `tests/benchmark_v1_spectralmatch_phase2/test_scoring_cli.py`

**Interfaces:**
- `score_submission(freeze_root, submission, regime, *, expected_scope) -> dict` supports `A`, `B1`, `B2`; `B3` returns the existing explicit not-run state.
- Reuse frozen metric functions, exact schema and frozen windows. Do not edit `evaluate_unified_v1.py`, `src/benchmark_v1_unified/`, schema, or manifest.
- `scope=DIAGNOSTIC_CROP_ONLY` is never eligible for formal A/B ranking.

- [ ] **Step 1: Add synthetic tests** for A mosaic coverage/single-source structure, optional seam metrics requiring provenance, B1 full support, B2 frozen windows, missing output, wrong grid, ambiguous NoData/mask, crop scope rejection, and no support shrinking.
- [ ] **Step 2: Run tests and verify they fail** because the Phase 2.1 scorer/API does not exist.
- [ ] **Step 3: Implement raster identity/mask validation and explicit status accounting**; require exact frozen grid and report expected/observed pixel denominators.
- [ ] **Step 4: Implement A/B1/B2 CLI output** with per-pair/per-window detail and frozen macro aggregation; never produce formal scores from crop outputs.
- [ ] **Step 5: Run tests**: `python -m pytest tests/benchmark_v1_spectralmatch_phase2/test_scoring.py tests/benchmark_v1_spectralmatch_phase2/test_scoring_cli.py -q`.

### Task 7: Phase 2.1 aggregate report and regression verification

**Files:**
- Create: `docs/benchmark/v1/unified_v1/spectralmatch_phase2_1_report.md`
- Create: `data/output/benchmark_v1_unified/spectralmatch_phase2_1_20261013/` diagnostic JSON/CSV/masks only
- Test: all `tests/benchmark_v1_spectralmatch_phase2/` plus `tests/benchmark_v1_unified/`

- [ ] **Step 1: Produce the comparison table** for official seam, coverage-retaining seam, and both 00_01 PIF modes; include run status, crop scope, coverage, radiometric metrics, stage time, measured RSS state/peak, polling status, and hard-limit status. Preserve unsupported values as explicit unavailable statuses.
- [ ] **Step 2: Verify Phase 1 freeze integrity** with the unchanged locked artifact and code hashes.
- [ ] **Step 3: Run Phase 2.1 focused suite**, then the full repository pytest suite; report all failures/skips honestly.
- [ ] **Step 4: Run `python -m compileall` on new/modified Phase 2.1 modules and `git diff --check`.**
- [ ] **Step 5: Audit changed paths** to verify no Phase 1 locked file, input image, freeze artifact, old run directory, or unrelated worktree file changed.
- [ ] **Step 6: Stop**. Do not launch 13-scene work; only propose full-pair resource preflight if both crop acceptance gates pass.

## Self-review

- Coverage map: tasks 1–7 cover separate runtime versions/environment creation, fail-closed memory telemetry, process-tree termination, official seam strategy audit, coverage variant/provenance, both PIF settings, formal A/B scoring, B3 not-run, freeze hash/Git identity, isolated outputs, and final tests/report.
- Type/interface consistency: seam assignment labels use integer scene indices with `-1` invalid; scoring uses current schema artifact paths and an explicit scope; resource telemetry has distinct monitor, polling and OS-hard-limit fields.
- Frozen boundary: all scoring and audit extensions are outside the Phase 1 code manifest; imports of frozen functions are read-only.
- Review focus tests are attached to their owning tasks above.
