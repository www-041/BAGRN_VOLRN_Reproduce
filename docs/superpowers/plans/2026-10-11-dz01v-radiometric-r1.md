# DZ01V Radiometric Benchmark R1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Keep tests-first RED→GREEN for every behavior change.

**Goal:** Build the read-only DZ01V multi-temporal radiometric benchmark foundation, with a working human ROI review GUI, fixed-truth synthetic generator, independent metrics, and explicit R2 readiness boundaries.

**Architecture:** Add an isolated `src/benchmark_v1_radiometric_r1` package and `scripts/benchmark/` entry points. Read prior benchmark manifests, Task15/16 history, and source GeoTIFFs without modifying them. Write R1 reports under `docs/benchmark/v1/radiometric_r1/`; generated examples/run artifacts, if any, go only under `data/output/benchmark_v1_radiometric_r1/`.

**Tech Stack:** Python 3.11, rasterio, numpy, scipy, shapely, tkinter, Pillow, pytest; PyYAML for frozen scenario configuration. No model training or external package installation.

**Spec:** User-provided attachment `C:\Users\wang\.codex\attachments\14c6147f-9504-431a-8a83-b651fa6fcbce\已粘贴的文本.txt` (DZ01V Radiometric Benchmark R1, 12 chapters), treated as frozen request.

## Global Constraints

- Only the processed 13-scene DZ01V V sensor set, B1–B16; default development/evaluation band B9; physical pixel units remain `UNKNOWN` unless source evidence proves otherwise.
- Phase1–5 files, Phase3 10 km/500 m split rule, Task15/16 methods/config/results, and original imagery are read-only.
- No registration experiment, radiometric method run, full 13-scene run, model training, or full-image resampling/calibration.
- ROI labels `STABLE`/`REAL_CHANGE` remain `UNVERIFIED` until explicit human review; no approval or reviewer identity is fabricated.
- TEST ROI requires an independent second reviewer; calibration and evaluation ROIs remain distinct; role conflicts or unsafe geometry remain `UNASSIGNED`.
- R1 source code, tests, documentation, and output paths are separate from Phase5 and formal pipeline namespaces.
- Simulation cases and parameters are frozen before results; generated floats are synthetic numeric values, not claimed reflectance.
- Spectralmatch is compatibility review only; do not install or run it in this phase.

## Review Focus

1. 78 combinations vs 48 valid B9 overlaps: no-overlap pairs must be retained as `NO_VALID_OVERLAP`, not algorithm failures (Task 1 tests).
2. ROI polygon coordinates across two different source affines: both views and validity fractions must refer to the same EPSG:32650 geometry (Task 2 tests).
3. Reviewer labels and history: no automatic approval, same-person second review rejected, and TEST ROI eligibility separate from first approval (Task 2 tests).
4. Simulation clipping/NoData/real-change truth: save target, change mask/amplitude, observed perturbation, and evaluation support separately (Task 3 tests).
5. Metrics with empty/invalid masks and differing band scales: report explicit no-support status; never blend raw MAE across bands (Task 4 tests).

---

### Task 1: Read-only temporal inventory and deterministic pair shortlist

**Files:** Create `src/benchmark_v1_radiometric_r1/inventory.py`, `scripts/benchmark/build_radiometric_r1_inventory.py`, tests `tests/radiometric_r1/test_inventory.py`; create `radiometric_pair_inventory.csv`, `temporal_stratification_report.md`, `radiometric_pair_selection_draft.md`.

**Interfaces:** `build_pair_inventory(pair_candidates_csv, temporal_inventory_csv, scene_inventory_csv) -> list[dict]`; `stratify_pair(row) -> T0|T1|T2|T3|NO_VALID_OVERLAP`; `select_representative_pairs(rows, target=10) -> list[dict]`.

- [ ] RED: fixtures with 13 scenes/78 pairs, 48 valid overlaps (7 same-day, 41 cross-date) assert no-overlap state and fixed required pairs `00_01`, `04_11`, `02_10`, `00_10`; invalid/unknown date handling stays explicit.
- [ ] GREEN: join existing frozen manifests only; include all 78 pairs, effective overlap pixels and status; choose an 8–12 pair draft deterministically by time bin, overlap, quality and coverage, without reading algorithm scores.
- [ ] Verify CLI writes only to R1 docs, records source manifest hashes, and does not mutate prior CSVs. Run `pytest tests/radiometric_r1/test_inventory.py -q`.

### Task 2: ROI candidates, spatial eligibility, review store, and usable GUI

**Files:** Create `roi.py`, `roi_store.py`, `roi_gui.py`; create `scripts/benchmark/annotate_radiometric_rois.py`, tests `test_roi.py`, `test_roi_store.py`, `test_roi_gui.py`; create candidate GeoJSON, review CSV template, and annotation guide.

**Interfaces:** `build_roi_candidates(regions, split_regions) -> FeatureCollection`; `classify_roi_role(geometry, split_regions, cell_size=10000, margin=500) -> role/status`; `valid_fraction_for_geometry(dataset, geometry) -> float`; `ReviewStore.load/save/append_event`; `launch_roi_annotator(pair_id, candidate_path, output_path, band=9)`.

- [ ] RED: verify candidate features stay `UNKNOWN/UNVERIFIED`; polygon-to-grid conversion honors EPSG:32650 Phase3 formula and buffered safe interior; crossing cells/buffers or cross-pair spatial overlap cannot be assigned TEST; empty/invalid geometries fail closed.
- [ ] RED: store round trip, append-only review history, label schema, second reviewer must differ, no auto-approved ROI, TEST requires second review for formal eligibility.
- [ ] RED: window rendering of two scenes uses a common geospatial viewport and no full-raster read; draw/edit/delete/reload rectangular and polygon ROIs; mask valid fractions from source pixels.
- [ ] GREEN: implement tkinter/Pillow/rasterio window reader, band selector, synchronized pan/zoom and shared geospatial polygon; display-only percentile stretch; atomic review snapshot plus append-only event log. Default statuses and reviewer fields remain empty/UNVERIFIED.
- [ ] Verify `--check-env` and headless geometry/store tests; do not claim GUI visual approval without desktop display. Provide direct PowerShell launch command. Run R1 ROI tests.

### Task 3: Frozen controlled synthetic radiometric generator

**Files:** Create `synthetic.py`, `scripts/benchmark/generate_radiometric_r1_synthetic.py`, tests `test_synthetic.py`; create `synthetic_radiometric_protocol.md`, `synthetic_cases.yaml`, `synthetic_truth_schema.md`.

**Interfaces:** `generate_case(source, valid_mask, config, seed) -> SyntheticCase`; cases S1 global gain/offset, S2 spatial gain/offset field, S3 per-band known perturbation, S4 true-change plus observation perturbation with all truth layers.

- [ ] RED: exact synthetic arrays establish known inverse for S1, preserved fixed truth and deterministic field for S2, independent per-band parameter truth for S3, and S4 stores clean target/change mask/change amplitude/observed image/valid mask separately; test NoData, clipping and saturation flags.
- [ ] GREEN: freeze seed/ranges/formulas in YAML and create only bounded synthetic outputs under the dedicated R1 output root; never mutate inputs or call normalization algorithms.
- [ ] Verify repeatability and inverse relationships on tiny arrays. Run R1 synthetic tests.

### Task 4: Independent radiometric and structural metrics + legacy formula audit

**Files:** Create `metrics.py`, tests `test_metrics.py`; create `radiometric_metric_protocol.md`, `metric_compatibility_audit.md`.

**Interfaces:** `pair_metrics(reference, corrected, mask)`, `change_preservation_metrics(target_change, output_change, change_mask)`, `multiband_metrics(reference_cube, output_cube, mask, scale_policy)`, `aggregate_by_roi_pair_band_time(rows)`.

- [ ] RED: mathematical fixtures for MAMD, MSDD, exact Wasserstein RDD, MAE/RMSE/P95; no-difference, constant offset/gain, local difference, NoData, NaN/Inf, empty support, and macro-vs-pixel aggregation.
- [ ] RED: changed-region scores require independent supplied truth; SAM/SAC require caller-confirmed co-registration/scale/valid joint spectra and otherwise return `CONDITIONAL_NOT_EVALUABLE`; raw MAE is never pooled across bands.
- [ ] GREEN: implement independent metric code; audit unchanged Task15/16 helpers and exact support/aggregation differences, without editing legacy sources.
- [ ] Verify all definitions and test expected numbers. Run R1 metric tests.

### Task 5: V-band radiometric readiness and source identity audit

**Files:** Create `band_readiness.py`, tests `test_band_readiness.py`; create `v_band_radiometric_readiness.md`, `v_band_quality_inventory.csv`.

**Interfaces:** `summarize_v_bands(inventory_csv, scene_inventory_csv, identity_manifest) -> rows, summary`.

- [ ] RED: fixtures prove B1–B16 completeness, wavelength/units only populated when source metadata states them, UNKNOWN units stay UNKNOWN, scene grid differences and missing hashes remain explicit.
- [ ] GREEN: derive from existing 208-row metadata inventory and recorded B9 hashes; do not read all full image pixels or apply MTL calibration coefficients.
- [ ] Verify B9 + representative wavelength bands metadata and all 13 scenes; do not infer subpixel co-registration from equal Affine. Run R1 band tests.

### Task 6: Spectralmatch read-only compatibility review and R2 protocol handoff

**Files:** Create `spectralmatch_compatibility_review.md`, R2 experiment protocol section, `R1_SUMMARY.md`, `R1_READINESS_REPORT.md`.

- [ ] Verify official repository/JOSS/version/license/dependencies/input/output/API claims from primary sources; label claims official/source-inspected/unverified/incompatible. Do not clone/install/run.
- [ ] Define frozen-BAGRN local module ablation vs independent full radiometric pipeline comparison, common input/mask/ROI, support-domain/seam reporting, status denominators, wall time/RAM/VRAM, and no resume-as-first-run timing.
- [ ] Final readiness must be `R1_INFRASTRUCTURE_READY_MANUAL_REVIEW_PENDING` while no human ROI review exists; report approved stable/change count as 0, tools/tests/commands, known data limits, and explicit researcher checklist for R2.
- [ ] Run all `tests/radiometric_r1`, compileall for R1 modules/scripts, `git diff --check`, whitespace scan; verify Phase1–5, Task15/16 and source hashes/outputs unchanged. No formal algorithm or full-scene run.

## Execution note

Run tasks sequentially. The ROI GUI, synthetic generator and metrics are first-class deliverables, not placeholders. If a desktop display cannot be controlled in this environment, validate import/check-env and geometry/store behavior, and state that manual visual operation remains to be verified by the researcher.
