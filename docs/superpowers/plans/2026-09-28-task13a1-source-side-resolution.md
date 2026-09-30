# Task13A.1 Source-Side Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Resolve source ownership for the four frozen `AMBIGUOUS_SOURCE_SIDE` pairs using geometry-only evidence, replay only their V1/V2 composition, and produce a separate Task13A.1 audit package.

**Architecture:** Add a pure `SourceSideResult` resolver that partitions a saved monotonic seam into ordered sides, scores exclusive-valid boundary contact, and falls back to registered footprint centroid projection. Add an explicit assignment path to blending, then use a dedicated replay script that reads the saved Task13A seam GeoJSON/local coefficient CSV and frozen BAGRN rasters without recomputing upstream algorithms or overwriting Task13A artifacts.

**Tech Stack:** Python, NumPy, SciPy, rasterio, pytest, existing Task13A metrics and mosaic helpers.

**Spec:** `C:/Users/wang/.codex/attachments/dba5e95a-1445-4a7a-a720-ff39966602c2/已粘贴的文本.txt` (Task13A.1 Source-Side Resolution).

## Global Constraints

- Do not modify EfficientLoFTR, Translation-L2, BAGRN, seam cost, alpha/beta, P95 normalization, seam search, local moment, correction taper, refined seam, blend width, or metric definitions.
- Do not rerun matcher/global/BAGRN/VOLRN and do not recompute the ten-pair Task13A batch.
- Reuse saved `seam_initial.geojson`, `seam_refined.geojson`, `local_coefficients.csv`, and frozen BAGRN rasters.
- Process only `00_01`, `00_02`, `01_02`, and `01_04`; preserve all existing Task13A outputs.
- Resolver may use only geometry/valid-mask evidence, never MAE/RMSE/RDD or pair-ID hard-coding.
- Nested/identical/indistinguishable ownership must remain `REQUIRES_MULTISCENE_LABELING` (or the specified closed-seam equivalent), not a forced assignment.
- Do not commit, push, merge, or start Task13B.

## Review Focus

- Vertical and horizontal seams must map ordered side regions without scene-ID left/right assumptions; test both orientations.
- Swapping A/B inputs must swap ownership while preserving the spatial partition.
- Exclusive-contact ties must use centroid projection only when the relevant-axis separation is finite and non-negligible; nested/identical footprints remain unresolved.
- Resolver output must be independent of radiometric metrics and seam path must remain bitwise unchanged.
- Replayed mosaics must be finite and all frozen successful-pair artifacts must remain untouched.

### Task 1: Geometry-only source-side resolver

**Files:**
- Create: `src/seam_local/source_side.py`
- Test: `tests/test_task13a1_source_side.py`

**Interfaces:**
- `resolve_source_sides(seam, overlap_mask, valid_a, valid_b, scene_a_footprint, scene_b_footprint) -> SourceSideResult`
- `SourceSideResult` exposes `status`, `side_1_source`, `side_2_source`, `method`, `confidence`, and `diagnostics`.

- [ ] Write failing synthetic tests for vertical/horizontal contact, reversed spatial order, centroid fallback, nested footprints, identical centroids, A/B swap, and seam-path immutability.
- [ ] Run the focused test file and observe the expected missing resolver failure.
- [ ] Implement ordered seam-side masks, exclusive boundary-contact scores, centroid projection fallback, and explicit unresolved statuses without metric access.
- [ ] Run the focused tests and inspect diagnostics.

### Task 2: Explicit blend assignment and frozen replay

**Files:**
- Modify: `src/seam_local/blend.py`
- Create: `scripts/run_task13a1_source_side.py`
- Test: `tests/test_task13a1_replay.py`

**Interfaces:**
- Blend accepts an optional `SourceSideResult`/assignment override while preserving the existing default behavior.
- Replay reads the four saved pair artifacts, writes `data/output/b9_five_scene_validation/seam_local_task13a1_source_side/`, and emits `source_side_audit.csv`, `source_side_summary.json`, and `TASK13A1_SOURCE_SIDE_REPORT.md`.

- [ ] Add failing tests proving explicit assignments produce finite V1/V2 outputs and do not alter saved seam paths or coefficients.
- [ ] Run focused tests and observe failure before implementation.
- [ ] Implement explicit assignment in blending and the read-only four-pair replay/writer.
- [ ] Run focused tests; verify no Task13A output is overwritten.

### Task 3: Four-pair execution and verification

**Files:**
- Modify: `scripts/run_task13a1_source_side.py` only as needed for verified defects.
- Create: `data/output/b9_five_scene_validation/seam_local_task13a1_source_side/` artifacts.

- [ ] Run only the four-pair replay once using `.venv-registration/Scripts/python.exe`.
- [ ] Verify source-side audit classification, finite mosaics, seam/coefficient equality, and exact output counts.
- [ ] Run Task13A.1 focused tests, existing 37 Task13A tests, related radiometric/mosaic regressions, full pytest, compileall, and `git diff --check`.
- [ ] Summarize whether PASS reaches 8/10; otherwise retain `MIXED_NEEDS_REVIEW` and classify unresolved pairs as `REQUIRES_MULTISCENE_LABELING`.

