# Final End-to-End Baseline Replay v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a paper-baseline package from frozen B9 geometry, reusing persisted per-scene BAGRN rasters when present and otherwise re-executing only BAGRN before regenerating weighted-feather mosaics and uniform metrics.

**Architecture:** A replay layer will authenticate the frozen scene selection, canonical grid, and persisted Global transforms but never invoke a matcher or Global solver. It will inspect each requested baseline for a complete per-scene BAGRN cache; missing or invalid caches cause a BAGRN-only run that saves normalised scene GeoTIFFs, its parameters, Task10D metrics, and a regenerated weighted-feather mosaic. A finalizer validates the replay outputs and builds the two method packages and paper tables.

**Tech Stack:** Python, NumPy, rasterio, SciPy, existing BAGRN and weighted-mosaic modules, standard library (`csv`, `json`, `hashlib`, `shutil`), pytest.

**Spec:** `C:\Users\wang\Desktop\Task11_Final_End_to_End_Baseline_Experiment_Plan.md`, revised by the user on 2026-09-27: Final End-to-End Baseline Replay v2; preserve frozen geometry; do not rerun matcher/Global; inspect BAGRN per-scene outputs and rerun BAGRN only if absent; regenerate weighted-feather mosaic and unified metrics; generate the final paper-baseline package.

## Global Constraints

- Use only frozen B9 selection `[2, 3, 5, 8, 10]`, canonical 14 m EPSG:32650 grid, and persisted `global_transforms.json`.
- Main baseline is `efficient_loftr/translation_l2`; traditional baseline is `sift/mst`; BAGRN radiometric control remains frozen at scene index `0`.
- Never invoke or import matcher, RANSAC, MST, Translation-L2, or any other Global optimization code in the replay execution path.
- Check `normalized_scenes/scene_*.tif` plus its BAGRN provenance before reuse. A missing, incomplete, grid-inconsistent, or provenance-inconsistent cache must trigger BAGRN-only regeneration in a new replay-cache directory.
- BAGRN regeneration must persist one normalized GeoTIFF per scene, `bagrn_parameters.npz`, `radiometric_summary.json`, provenance, and the regenerated `weighted_feather/mosaic.tif`; it must not overwrite historical frozen Task10 outputs.
- The final mosaic must be regenerated from the persisted/recreated BAGRN per-scene rasters and frozen Global geometry using the existing weighted-feather algorithm.
- Validate exact CRS, dimensions, affine transform, nodata footprint, finite valid data, artifact hashes, and source/provenance identity before publishing `final_results`.
- Write a new empty destination only; fail safely on an existing non-empty output root.
- Label MAMD, MSDD, RDD, local, seam, and CGL values as frozen Task10D metrics. Do not present them as resolving the separate paper CD/GL formula ambiguity.

## Review Focus

- A stale per-scene BAGRN cache with wrong geometry identity must not be reused; test the provenance rejection and BAGRN-only fallback.
- The fallback must never call matcher or Global code; test dependency boundaries and invocation log metadata.
- The regenerated mosaic must use BAGRN scene rasters, not RAW source scenes; test a deliberately distinct normalized fixture.
- A corrupt cache/mosaic must stop before final package materialization; test CRS/transform, nodata footprint, and non-finite values.
- The two baselines must retain their own geometry, metrics, and mosaic hashes in all tables; test distinct fixture identities.

---

### Task 1: Persist replayable per-scene BAGRN outputs

**Files:**
- Modify: `src/multiscene_sift/radiometric_runner.py`
- Create: `tests/multiscene_sift/test_final_baseline_replay_v2.py`

**Interfaces:**
- Consumes: the existing fixed-geometry BAGRN run inputs and an opt-in `persist_normalized_scenes: bool = False` parameter.
- Produces: `normalized_scenes/scene_000.tif` through `scene_004.tif` on the canonical grid, plus `normalized_scenes_manifest.json` containing source-config/global-transform hashes, geometry run, method, control index, grid identity, and per-file SHA-256 values.

- [ ] **Step 1: Write failing persistence tests**

Use small synthetic registered scenes and frozen-like transforms. Assert an opt-in BAGRN run writes one finite, canonical-grid TIFF per scene, preserves valid/nodata support, and writes a manifest with all required provenance values. Assert the default run writes no scene cache.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `pytest tests/multiscene_sift/test_final_baseline_replay_v2.py -v`

Expected: FAIL because the runner does not yet accept `persist_normalized_scenes` or write a scene cache.

- [ ] **Step 3: Implement cache persistence in `radiometric_runner.py`**

Add the opt-in parameter to `run_fixed_geometry_radiometric`. After BAGRN produces its final arrays and before mosaic creation, expand them to the canonical canvas, write float32 GeoTIFFs with the canonical transform/CRS and NaN nodata, and write a checksum/provenance manifest. Preserve the default behavior and never alter the BAGRN solver or geometry.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `pytest tests/multiscene_sift/test_final_baseline_replay_v2.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/multiscene_sift/radiometric_runner.py tests/multiscene_sift/test_final_baseline_replay_v2.py
git commit -m "feat: persist replayable BAGRN scene outputs"
```

### Task 2: BAGRN-only replay, mosaic regeneration, and metric collection

**Files:**
- Create: `src/multiscene_sift/final_baseline_replay.py`
- Create: `scripts/run_final_baseline_replay.py`
- Modify: `tests/multiscene_sift/test_final_baseline_replay_v2.py`

**Interfaces:**
- Consumes: `BaselineReplaySpec(name, geometry_run_dir, source_config, canonical_grid, replay_root)`, the frozen geometry transforms, and either a validated BAGRN scene cache or the raw source scenes.
- Produces: `replay_baseline(spec: BaselineReplaySpec) -> ReplayArtifact`, with `cache_status` (`REUSED` or `REGENERATED`), normalized scenes, BAGRN parameters/metrics, regenerated weighted-feather mosaic, and uniform Task10D metrics.

- [ ] **Step 1: Write failing replay tests**

Build main/traditional temporary fixtures with deliberately different frozen transforms and distinct normalized scene values. Assert a valid manifest is reused without calling the BAGRN runner; missing or mismatched caches invoke BAGRN with `method="BAGRN"` and `persist_normalized_scenes=True`; assert the mosaic is rebuilt from the normalized scenes and Task10D metrics include RAW and BAGRN snapshots.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `pytest tests/multiscene_sift/test_final_baseline_replay_v2.py -v`

Expected: FAIL because replay spec, cache validator, and BAGRN-only orchestrator do not exist.

- [ ] **Step 3: Implement the replay module and CLI**

Implement strict source/config/grid/global-transform hashing; cache validation; a BAGRN-only fallback through `run_fixed_geometry_radiometric`; and a mosaic regeneration routine that reads the persisted normalized TIFFs with their canonical transforms and calls only the existing weighted-feather mosaic implementation. The CLI accepts the frozen config, canonical grid, both Global run directories, a replay-cache root, and `--verify`; it must log which baselines were reused/regenerated and must expose no matcher/Global option.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `pytest tests/multiscene_sift/test_final_baseline_replay_v2.py -v`

Expected: PASS.

- [ ] **Step 5: Run affected tests and commit**

Run: `pytest tests/multiscene_sift/test_final_baseline_replay_v2.py tests/multiscene_sift/test_task10d_runner_schema.py tests/multiscene_sift/test_b9_weighted_mosaic.py -v`

Expected: PASS.

```bash
git add src/multiscene_sift/final_baseline_replay.py scripts/run_final_baseline_replay.py tests/multiscene_sift/test_final_baseline_replay_v2.py
git commit -m "feat: replay BAGRN baselines from frozen geometry"
```

### Task 3: Materialize final paper-baseline results

**Files:**
- Modify: `src/multiscene_sift/final_baseline_replay.py`
- Modify: `tests/multiscene_sift/test_final_baseline_replay_v2.py`

**Interfaces:**
- Consumes: the two validated `ReplayArtifact` values and an empty `final_results` root.
- Produces: `materialize_final_results(main, traditional, output_root) -> dict` and exactly the Task11 registration/radiometric/mosaic hierarchy, root manifest, and paper tables.

- [ ] **Step 1: Write failing materialization tests**

Assert the exact final hierarchy for both methods, pairwise CSV columns, Global JSON entries, RAW/BAGRN Task10D metric rows, BAGRN-derived final mosaic hash, preview, paper geometry/radiometric tables, and experiment summaries. Assert a non-empty output root causes no overwrite.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `pytest tests/multiscene_sift/test_final_baseline_replay_v2.py -v`

Expected: FAIL because the materializer does not exist.

- [ ] **Step 3: Implement strict finalization**

Create `EfficientLoFTR_Translation_BAGRN/` and `SIFT_MST_BAGRN/`, each with `01_registration`, `02_radiometric`, `03_mosaic`, and `experiment_summary.md`; use `final_mosaic.tif`, `preview.png`, and `mosaic_summary.json` from the regenerated replay stage. Add `manifest.json` with artifact hashes and cache status, plus `paper_tables.md` with the two specified comparisons and methodology limitations.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `pytest tests/multiscene_sift/test_final_baseline_replay_v2.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/multiscene_sift/final_baseline_replay.py tests/multiscene_sift/test_final_baseline_replay_v2.py
git commit -m "feat: package final replay baseline results"
```

### Task 4: Execute and verify the real B9 replay

**Files:**
- Create: `data/output/b9_five_scene_validation/final_replay_v2/**`
- Create: `data/output/b9_five_scene_validation/final_results/**`
- Create: `docs/experiments/TASK11_FINAL_BASELINE_REPLAY_V2.md`

**Interfaces:**
- Consumes: Task 2 CLI, frozen B9 config/grid/Global transforms, and finalization functions.
- Produces: verified main/traditional BAGRN scene caches, regenerated mosaics, final package, and a reproducible execution record.

- [ ] **Step 1: Run the v2 replay into new roots**

Run the CLI with `04_frozen_five_scene_config_1024.json`, `mosaic_runs_1024/protocol/canonical_output_grid.json`, `global_runs_1024/efficient_loftr/translation_l2`, `global_runs_1024/sift/mst`, replay root `final_replay_v2`, and final root `final_results`.

Expected: missing per-scene caches cause exactly two BAGRN-only runs; no matcher/Global solver run occurs; each cache contains five normalized scenes and one regenerated weighted-feather mosaic.

- [ ] **Step 2: Run delivery verification**

Run: `pytest tests/multiscene_sift/test_final_baseline_replay_v2.py -v` and the CLI `--verify` against both real roots.

Expected: PASS; all manifest hashes, canonical grid checks, finite valid pixels, and main/traditional labels validate.

- [ ] **Step 3: Record the result and commit**

Write the exact command, cache status, source hashes, output hashes, metrics references, and non-rerun guarantee into the experiment record. Do not claim algorithm superiority from this one fixed data set.

```bash
git add data/output/b9_five_scene_validation/final_replay_v2 data/output/b9_five_scene_validation/final_results docs/experiments/TASK11_FINAL_BASELINE_REPLAY_V2.md
git commit -m "docs: record final baseline replay v2"
```
