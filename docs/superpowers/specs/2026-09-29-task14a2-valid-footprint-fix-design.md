# Task14A.2 Valid-Mask Footprint Fix + 13-Scene Controlled Replay

## Purpose

Correct the 13-scene Stage07 source-side footprint input so that it has the
same actual-valid-mask polygon semantics already validated by the five-scene
Task13A.1 replay. Then replay only source-side ownership from frozen Stage07
artifacts and execute a new, isolated Stage08–12 post-fix pipeline.

The existing Stage02–06 artifacts and the existing Stage07–12 results remain
read-only historical references. No matcher, registration, canonical warp,
EDT, BAGRN, VOLRN, seam search, or local-moment coefficient estimation is
allowed in this task.

## Constraints

- Do not change `resolve_source_sides()` or any Task13A.1/Task13B/Task13B.1
  scientific rule.
- Do not change seam cost, search widths, correction widths, segment length,
  gain stability range, blend width, or label tie hierarchy.
- Do not force the three real geometric failures to PASS.
- Keep `01_08` and `08_11` as `UNSTABLE_LOCAL_GAIN`; their missing coefficient
  artifacts must not be reconstructed.
- Keep the 12 `NO_FINAL_SHARED_SUPPORT` pairs out of seam, source-side,
  correction, and preference processing.
- Write all Task14A.2 artifacts below
  `data/output/b9_13scene_task14/task14a2_valid_footprint_fix/`.
- Do not commit, push, merge, or delete historical artifacts.

## Shared footprint semantics

First characterize the existing five-scene
`scripts/run_task13a1_source_side.py::_footprint_polygon` behavior on frozen
masks. Extract that exact behavior into one small helper, preferably in
`src/seam_local/footprint.py`, named
`footprint_polygon_from_valid_mask(mask, transform)`. The helper must use the
same rasterio-shapes and unary-union semantics, preserving nodata holes and
disconnected valid components, and must raise on an empty mask. The refactor
must prove old-helper versus new-helper geometry equivalence on frozen
five-scene masks. The scientific contract is explicitly:

`13-scene footprint semantics == five-scene footprint semantics`.

`scripts/run_task13a1_source_side.py` and the Task14A.2 replay must import the
same helper. The frozen Stage07 implementation may be corrected to use this
helper for future runs, but Task14A.2 must not rerun Stage07. The resolver
continues to be `src.seam_local.source_side.resolve_source_sides`.

## Controlled source-side replay

The replay reads:

- Stage06 normalized scene GeoTIFFs and valid masks;
- Stage07 `pairwise_results.json`;
- saved `seam_initial.geojson` and `seam_refined.geojson`;
- saved `local_coefficients.csv` only as an opaque frozen artifact for later
  correction aggregation.

For each of the 45 `ADAPTER_LOGIC_MISMATCH` pair IDs from
`resolver_path_audit.json`, run the unchanged resolver on both saved seams
with actual valid-mask polygons. Record initial and refined assignments,
methods, shared support, and the old/new status separately.

The replay gate must assert:

- all 45 mismatch pairs have source-side status `PASS`;
- `02_05` remains `REQUIRES_MULTISCENE_LABELING`;
- `01_08` and `08_11` remain `UNSTABLE_LOCAL_GAIN`;
- all 12 no-support pairs remain `NO_FINAL_SHARED_SUPPORT`;
- no source preference is created for a non-PASS pair;
- no frozen Stage07 file is rewritten.

The formal table is `source_side/source_side_replay_status.csv`, with at least
`pair_id`, old Stage07 status, Task14A.1 dry-run status, new source-side
status, initial/refined assignments, resolver method, footprint method,
shared-valid pixels, local-gain status, and formal pair status.

## Post-fix Stage08–12

After the source-side gate passes, reconstruct the pairwise preference network
from only the resolved PASS pairs. Reuse the frozen Task13B + Task13B.1 order:

1. candidate scenes;
2. normalized pairwise preference score;
3. clipped normalized interiority;
4. non-saturated normalized interiority;
5. raw EDT;
6. unresolved.

Run Stage08–12 into the isolated Task14A.2 directory. Stage09 may aggregate
existing valid frozen local corrections for pairs with saved coefficients,
including the unresolved geometric pair, but must exclude the two unstable
pairs. Aggregation remains a weighted mean of pairwise deltas; no gain
multiplication or direct offset summation is allowed.

Required outputs include labels, label methods, contributor counts, score
margin, corrected scenes, V1/V2 mosaics, boundary metrics, structural metrics,
correction summary, invariance results, post/pre comparison, and scale
summary. Label diagnostics must explicitly include:

- `resolved_preference_edge_count`;
- `multiscene_pixels`;
- `cycle_pixels`, `cycle_fraction`;
- `pairwise_score_tie_pixels`;
- `clipped_interiority_fallback_pixels`;
- `unclipped_normalized_interiority_pixels`;
- `raw_edt_pixels`;
- `unresolved_pixels`, `invalid_label_pixels`,
  `two_scene_disagreement_pixels`;
- `label_pixels_per_scene`, `label_method_counts`;
- `score_margin.min`, `score_margin.median`, and `score_margin.p95`.

The old Stage11 V2 metrics are labeled `PRE_FIX_DIAGNOSTIC_REFERENCE`; new
metrics are labeled `POST_FIX_FORMAL`.

## Invariance and hard gates

Use the frozen 13-scene preference network for the original order,
`[12, ..., 0]`, and one fixed non-random permutation. After mapping labels
back to scene identity, require exact equality and record changed-pixel
counts.

Run correction aggregation with normal and reversed pair-table order on the
same frozen inputs. Require exact equality where possible, otherwise
`rtol=1e-6`, `atol=1e-3`, and record max/mean absolute differences.

Require:

- union support 62,033,096;
- unresolved labels 0;
- invalid labels 0;
- two-scene disagreement 0;
- scene-order differences 0;
- correction-order equivalence;
- V0/V1/V2 support 62,033,096;
- all structural values finite and all scene gradient NCC >= 0.99;
- no new numerical-invalid outputs.

Before registering the Stage06 mosaic as formal V0, verify frozen weighted
feather semantics, CRS/transform/shape, nodata convention, dtype, and union
support. Do not recompute V0. If any check fails, emit
`HARD_STOP_V0_SEMANTIC_MISMATCH` and stop.

The final decision is `READY_FOR_NEXT_SCALE_STAGE`, `MIXED_SCALE_REVIEW`, or
`NOT_READY_FOR_LARGE_SCALE`. `READY_FOR_NEXT_SCALE_STAGE` requires all of:

- all 45 adapter-logic mismatch pairs have source-side `PASS`;
- true geometric failures remain unresolved/multiscene as designed;
- unresolved labels = 0;
- invalid labels = 0;
- two-scene disagreement = 0;
- 13-scene scene-order differences = 0;
- correction-order equivalence = `PASS`;
- V0/V1/V2 support = 62,033,096;
- post-fix V2 weighted boundary MAE < BAGRN;
- post-fix V2 weighted boundary RDD < BAGRN;
- post-fix V2 median boundary MAE < BAGRN;
- post-fix V2 median boundary RDD < BAGRN;
- all structural metrics are finite;
- every evaluated scene gradient NCC >= 0.99;
- no new numerical-invalid outputs.

If any engineering gate passes but any scientific quality condition above
fails, the decision is `MIXED_SCALE_REVIEW` or `NOT_READY_FOR_LARGE_SCALE`,
never `READY_FOR_NEXT_SCALE_STAGE`. Even `READY_FOR_NEXT_SCALE_STAGE` is a
stopping point; no Task14S, 25/50/100-scene, 1000-scene, or Task15 stage
starts here.

## Verification

Add tests before implementation for irregular valid masks, nodata holes,
shared-helper equivalence, source-side replay status gates, five-scene
regression, V0 semantic validation, scene-order invariance,
correction-order invariance, and the existing radiometric/mosaic regressions.
Run targeted tests, then
`python -m compileall src scripts tests`, `git diff --check`, and the full
pytest suite using the repository environment.

## Historical traceability

The final report must preserve the chain:

`Stage07 bbox-footprint defect → Task14A.1 audit → root cause identified →
Task14A.2 controlled correction`.

The final report must explicitly answer:

1. What exactly was the Stage07 original bug?
2. Does the fix use exactly the same actual-valid-footprint semantics as the five-scene path?
3. Did all 45 adapter-logic mismatch pairs become source-side `PASS`?
4. How many true `REQUIRES_MULTISCENE_LABELING` pairs remain, and why?
5. How were the two `UNSTABLE_LOCAL_GAIN` pairs handled?
6. Did the 12 `NO_FINAL_SHARED_SUPPORT` pairs remain unchanged?
7. How many formal resolved ownership edges are there?
8. What are post-fix cycle pixels and cycle fraction?
9. How many pairwise-score ties occurred?
10. How many pixels used clipped interiority, unclipped normalized interiority, and raw EDT fallback?
11. Are unresolved labels zero?
12. Is the 13-scene scene-order replay exactly invariant?
13. Did correction pair-order replay pass?
14. Are V0/V1/V2 supports all 62,033,096?
15. What are post-fix BAGRN-to-V2 weighted MAE values?
16. What are post-fix BAGRN-to-V2 weighted RDD values?
17. Did median MAE/RDD improve?
18. Were all structural metrics preserved?
19. How many label pixels changed pre-fix to post-fix?
20. What changed between pre-fix and post-fix V2?
21. Did the five-scene replay continue to pass?
22. What is the final Task14 decision?
23. Is entry into the next scale benchmark permitted?

The report must explicitly state that the task stopped after completion without
beginning the next scale stage.
