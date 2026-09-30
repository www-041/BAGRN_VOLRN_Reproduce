# Task14A.2 Valid-Mask Footprint Fix + Controlled Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Replace the 13-scene Stage07 bbox footprint input with the exact frozen five-scene actual-valid-mask footprint semantics, replay frozen source ownership, and produce isolated post-fix Stage08–12 outputs with all requested invariance and scientific gates.

**Architecture:** Extract the existing Task13A.1 polygonization body into a shared pure helper. Update the future Stage07 code path to use that helper, but never rerun Stage07 in this task. A new Task14A.2 driver reads frozen Stage06/07 artifacts, performs source-side-only replay, then delegates the unchanged Stage08–12 implementations into a new output root while adding scene-order, correction-order, V0, regression, and final-decision gates.

**Tech Stack:** Python 3.11, numpy, rasterio, scipy, shapely, pytest, existing `src.seam_local` Task13B/Task13B.1 modules, frozen GeoTIFF/CSV/JSON artifacts.

**Spec:** `docs/superpowers/specs/2026-09-29-task14a2-valid-footprint-fix-design.md`

## Global Constraints

- Do not rerun EfficientLoFTR, matcher, RANSAC, Translation-L2, canonical warp, EDT cache, BAGRN, VOLRN, Stage07 seam search, or Stage07 local-moment coefficient estimation.
- Do not change `resolve_source_sides()` or any Task13A.1/Task13B/Task13B.1 scientific rule.
- Do not force `02_05` or any true geometric failure to PASS.
- Keep `01_08` and `08_11` as `UNSTABLE_LOCAL_GAIN`; do not reconstruct missing coefficients.
- Keep all 12 `NO_FINAL_SHARED_SUPPORT` pairs out of seam, source-side, correction, and preference processing.
- Write formal outputs only below `data/output/b9_13scene_task14/task14a2_valid_footprint_fix/`.
- Preserve old Stage07–12 outputs and Task14A.1 audit outputs as read-only historical references.
- Do not commit, push, merge, or delete historical artifacts; task-level verification uses the working-tree diff.

## Review Focus

- Irregular valid masks and nodata holes: polygon geometry must preserve the exact five-scene semantics. Test in Task 1.
- A source-side replay with an initial/refined assignment disagreement: it must remain non-PASS and never create a preference edge. Test in Task 2.
- A saved coefficient pair whose ownership is unresolved: it may enter correction aggregation but not pairwise ownership labeling. Test in Task 2/3.
- A missing saved seam/coefficient pair (`UNSTABLE_LOCAL_GAIN`): it must remain unstable and never be reconstructed. Test in Task 2.
- A Stage06 V0 mosaic with wrong metadata or support: it must raise `HARD_STOP_V0_SEMANTIC_MISMATCH` before formal registration. Test in Task 3.

## File Map

- Create `src/seam_local/footprint.py`: pure shared valid-mask polygon helper.
- Modify `scripts/run_task13a1_source_side.py`: import the shared helper while preserving `_footprint_polygon` compatibility.
- Modify `scripts/run_task14a_resume_13.py`: use actual valid-mask footprints in the future `_stage07` path; do not invoke that path during Task14A.2.
- Create `scripts/run_task14a2_valid_footprint_fix.py`: frozen source-side replay, isolated Stage08–12 orchestration, V0/invariance gates, summaries, and report.
- Create `tests/test_task14a2_footprint.py`: helper and five-scene semantic-equivalence tests.
- Create `tests/test_task14a2_replay.py`: replay status, V0 gate, preference eligibility, and order-comparison unit tests.
- Create `data/output/b9_13scene_task14/task14a2_valid_footprint_fix/`: formal generated outputs only.

### Task 1: Extract exact shared footprint semantics

**Files:**
- Create: `src/seam_local/footprint.py`
- Modify: `scripts/run_task13a1_source_side.py`
- Modify: `scripts/run_task14a_resume_13.py`
- Test: `tests/test_task14a2_footprint.py`

**Interfaces:**
- Produces `footprint_polygon_from_valid_mask(mask: np.ndarray, transform: Affine) -> BaseGeometry`.
- Consumes existing Task13A.1 `_footprint_polygon` behavior and rasterio/shapely primitives.
- Later tasks import the helper and rely on exact polygon equality/bounds/area/centroid behavior.

- [ ] **Step 1: Write failing footprint tests**

  Add tests for: an irregular mask whose polygon bounds exceed neither the true support nor the bbox semantics; a mask with a nodata hole; empty-mask failure; frozen five-scene mask old-helper/new-helper geometry equivalence; and a synthetic two-scene bbox-overlap case where the actual footprints yield the expected resolver PASS.

- [ ] **Step 2: Run the focused tests and verify RED**

  Run: `.venv-registration/Scripts/python.exe -m pytest tests/test_task14a2_footprint.py -q`

  Expected: FAIL because the shared helper does not yet exist.

- [ ] **Step 3: Implement the shared helper**

  Copy the exact existing body of `scripts/run_task13a1_source_side.py::_footprint_polygon` into `src/seam_local/footprint.py` without changing rasterization, `mask`, transform, or `unary_union` semantics. Keep type validation and empty-mask errors explicit.

- [ ] **Step 4: Route both existing paths through the helper**

  Import the helper into `run_task13a1_source_side.py` and retain `_footprint_polygon = footprint_polygon_from_valid_mask` as a compatibility wrapper. In `run_task14a_resume_13.py`, add a valid-mask footprint loader for `_stage07` and replace `box(dataset_bounds)` as the resolver footprint input; retain bbox computation only for crop/window geometry.

- [ ] **Step 5: Run focused tests and verify GREEN**

  Run: `.venv-registration/Scripts/python.exe -m pytest tests/test_task14a2_footprint.py tests/test_task13a1_replay.py tests/test_task13a1_source_side.py -q`

  Expected: all selected tests pass, including old/new helper equality and existing five-scene behavior.

- [ ] **Step 6: Static-check the future Stage07 path**

  Run: `rg -n "box\(.*bounds|footprint_polygon_from_valid_mask|_footprint_polygon" scripts/run_task14a_resume_13.py scripts/run_task13a1_source_side.py src/seam_local/footprint.py`

  Expected: `box(dataset_bounds)` is absent from the resolver footprint construction; the shared helper is present in both source-side paths.

### Task 2: Implement frozen source-side replay and replay gate

**Files:**
- Create: `scripts/run_task14a2_valid_footprint_fix.py`
- Test: `tests/test_task14a2_replay.py`

**Interfaces:**
- Consumes: Task14A.1 `resolver_path_audit.json`, `source_side_failure_summary.csv`, Stage06 manifest/masks, Stage07 pair table, saved seams, and saved coefficient files.
- Produces: `source_side/source_side_replay_status.csv`, `source_side/source_side_summary.json`, `source_side/replay_gate.json`, and post-fix pair rows consumed by Task 3.
- Key pure interfaces: `classify_formal_pair_status(old_status, dryrun, initial, refined) -> str`, `validate_source_side_gate(rows) -> dict`, `validate_v0_semantics(...) -> dict`.

- [ ] **Step 1: Write failing replay-gate unit tests**

  Test that 45 mismatch rows become formal source-side `PASS`; `02_05` remains `REQUIRES_MULTISCENE_LABELING`; unstable/no-support rows remain unchanged; non-PASS rows do not become preference-eligible; and a V0 metadata/support mismatch raises `HARD_STOP_V0_SEMANTIC_MISMATCH`.

- [ ] **Step 2: Run focused replay tests and verify RED**

  Run: `.venv-registration/Scripts/python.exe -m pytest tests/test_task14a2_replay.py -q`

  Expected: FAIL because the replay driver/gate functions do not yet exist.

- [ ] **Step 3: Implement frozen artifact readers and source-side replay**

  Load all scene valid masks once, polygonize with the shared helper, parse each saved seam into its frozen local crop, and call the unchanged resolver for initial/refined seams. Copy old rows without write-back, update only source-side fields, preserve scientific `UNSTABLE_LOCAL_GAIN` and `NO_FINAL_SHARED_SUPPORT`, and use `formal_pair_status` separately from `new_source_side_status`.

- [ ] **Step 4: Implement and run the replay gate**

  Assert exactly 45 mismatch PASS results, one true geometric `REQUIRES_MULTISCENE_LABELING` (`02_05`), two unstable pairs, twelve no-support pairs, zero non-PASS preference eligibility, and unchanged SHA-256 hashes for all frozen Stage07 seam/coeff artifacts. Write the replay CSV/JSON before proceeding.

- [ ] **Step 5: Run focused tests and verify GREEN**

  Run: `.venv-registration/Scripts/python.exe -m pytest tests/test_task14a2_replay.py tests/test_task14a2_footprint.py -q`

  Expected: all focused tests pass.

### Task 3: Run isolated post-fix Stage08–12, invariance, and metrics

**Files:**
- Modify: `scripts/run_task14a2_valid_footprint_fix.py`
- Test: `tests/test_task14a2_replay.py`
- Generate: `data/output/b9_13scene_task14/task14a2_valid_footprint_fix/`

**Interfaces:**
- Consumes Task 2 post-fix pair rows and frozen Stage06/Stage05 inputs.
- Delegates unchanged `_stage08`, `_stage09`, `_stage10`, `_stage11`, `_stage12` primitives from `run_task14a_resume_13.py` into isolated output directories; Stage07 remains the frozen read-only input.
- Produces post-fix labels/methods/contributor counts/margins, corrected scenes, V1/V2 mosaics, formal metrics, scale summary, invariance JSON, and pre/post comparison.

- [ ] **Step 1: Write failing orchestration/gate tests**

  Add tests for preference eligibility (only formal PASS rows), label diagnostic key completeness, exact scene-identity remapping, correction order tolerance, V0 semantic registration, and final READY gating requiring all MAE/RDD/structural conditions.

- [ ] **Step 2: Run focused orchestration tests and verify RED**

  Run: `.venv-registration/Scripts/python.exe -m pytest tests/test_task14a2_replay.py -q`

  Expected: FAIL for missing post-fix diagnostics/gate functions.

- [ ] **Step 3: Implement V0 semantic validation**

  Validate the frozen Stage06 V0 weighted-feather semantic marker/metadata, CRS, transform, shape, dtype, nodata, finite valid support, and union support 62,033,096 without recomputing V0. Write a formal V0 record or stop with `HARD_STOP_V0_SEMANTIC_MISMATCH`.

- [ ] **Step 4: Delegate Stage08–12 into the isolated root**

  Reuse the existing label hierarchy and correction aggregation. Supply only PASS source-preference fields to Stage08; allow the stable unresolved geometric pair with saved coefficients into Stage09 correction aggregation; exclude unstable/no-support pairs. Add all required label diagnostic counts, score-margin min/median/P95, formal POST_FIX labels, and pre-fix diagnostic references.

- [ ] **Step 5: Implement scene-order invariance**

  Evaluate original order, reverse `[12, ..., 0]`, and one fixed documented permutation on the same frozen preference network. Map labels back to scene identity and record exact changed-pixel counts; stop with `HARD_STOP_13SCENE_SCENE_ORDER_DEPENDENCE` if either difference is nonzero.

- [ ] **Step 6: Implement correction pair-order invariance**

  Aggregate the same frozen correction deltas in normal and reversed pair order into diagnostic outputs. Record max/mean absolute differences and `allclose`; stop with `HARD_STOP_13SCENE_CORRECTION_ORDER_DEPENDENCE` when outside `rtol=1e-6`, `atol=1e-3`.

- [ ] **Step 7: Implement formal metrics, pre/post comparison, and decision gate**

  Record support, resolved ownership edges, cycle/tie/fallback/unresolved/invalid/two-scene disagreement counts, boundary MAE/RMSE/RDD weighted and median values, structural finite/NCC values, label-map changes, V2 differences, 5-scene replay gate, and the complete READY/MIXED/NOT_READY decision. READY is permitted only when every spec condition passes.

- [ ] **Step 8: Run focused tests and execute the controlled replay**

  Run: `.venv-registration/Scripts/python.exe -m pytest tests/test_task14a2_replay.py tests/test_task14a2_footprint.py -q`

  Then run: `.venv-registration/Scripts/python.exe scripts/run_task14a2_valid_footprint_fix.py`

  Expected: the source-side gate passes before Stage08; 45 mismatch pairs are source-side PASS, 02_05 remains unresolved, unstable/no-support counts are preserved, and all formal outputs are placed only in the new Task14A.2 directory. Any hard-stop gate terminates the run and writes its reason.

### Task 4: Full verification and report handoff

**Files:**
- Modify: `scripts/run_task14a2_valid_footprint_fix.py` only if verification exposes a code defect.
- Generate: `TASK14A2_VALID_FOOTPRINT_FIX_REPORT.md` and all required JSON/CSV/TIFF artifacts.

- [ ] **Step 1: Run targeted Task14A.2 and historical regressions**

  Run: `.venv-registration/Scripts/python.exe -m pytest tests/test_task14a2_footprint.py tests/test_task14a2_replay.py tests/test_task13a1_replay.py tests/test_task13a1_source_side.py tests/test_multiscene_adapter.py tests/test_task13b_labeling.py tests/test_task13b1_tie_resolution.py -q`

  Expected: zero failures.

- [ ] **Step 2: Compile and diff-check**

  Run: `.venv-registration/Scripts/python.exe -m compileall src scripts tests` and `git diff --check`

  Expected: exit 0 for both.

- [ ] **Step 3: Run the full pytest suite**

  Run: `.venv-registration/Scripts/python.exe -m pytest -q`

  Expected: zero failures; existing unrelated failures must be investigated, not hidden.

- [ ] **Step 4: Verify artifact isolation and historical preservation**

  Check that all new files are below `task14a2_valid_footprint_fix/`, old Stage07 pairwise JSON/CSV/markers and Task14A.1 audit files retain their pre-run hashes, and no Stage02–06 output has a new modification time or hash.

- [ ] **Step 5: Read the final report and hand off**

  Confirm all 23 questions are answered, old metrics are labeled `PRE_FIX_DIAGNOSTIC_REFERENCE`, new metrics `POST_FIX_FORMAL`, the final decision follows the full quality gate, and the process stops without starting any next-scale benchmark.

No commit is made because the user explicitly prohibited commit/push/merge for this task.
