# Task13B Five-Scene Multi-Scene Labeling + Final Mosaic

## Goal

Execute the attached Task13B plan on the frozen B9 five-scene geometry and BAGRN artifacts. Produce an authoritative source-label map, order-independent multi-label blending, V1/V2 mosaics, unified boundary metrics, diagnostics, figures, and the final report. Do not rerun matcher, global geometry, BAGRN, VOLRN, or alter Task13A seams/coefficients.

## Frozen constraints

- Use only the frozen Task13A protocol (`8232cfaf63b52803c1577a2502a23667376c1f4e8edf2e9ebb7323a82645ee36`), Task13A.1 source-side outputs, Task11 BAGRN weighted-feather baseline, and five frozen scenes.
- Preserve existing dirty worktree, Task12/Task13A/Task13A.1 artifacts, and all untracked files. Do not create a repo/worktree, commit, push, merge, or delete unrelated files.
- Write all new artifacts under `data/output/b9_five_scene_validation/multiscene_task13b/`.
- `00_02` remains `REQUIRES_MULTISCENE_LABELING`; it contributes no pairwise ownership vote. No scene-index, metric, or pair-specific hard-coded tie-break.
- Pairwise score normalization, P95-normalized EDT interiority fallback, 64 px blend width, 128 px correction corridor, 256 px segment/taper, order-independent correction averaging, and one-shot multi-label cosine blending must match the attached plan.
- Any provenance drift, two-scene disagreement, support mismatch, zero blend weight, nonfinite output, correction/scene-order dependence, or unresolved final labels is recorded as the specified hard stop and stops downstream composition.

## Task sequence

1. Freeze provenance and build the authoritative ten-pair table with hashes and read-only baseline references.
2. Implement/test pairwise preference fields and normalized global score aggregation, including cycle and consistency diagnostics.
3. Implement/test geometry-only P95 interiority tie-break and unresolved behavior.
4. Run the real five-scene label map and diagnostics; stop before mosaics if hard-stop conditions occur.
5. Implement/test reconstruction and weighted aggregation of frozen Task13A local deltas.
6. Implement/test order-independent multi-label cosine weights.
7. Stream V1/V2 mosaics from the shared labels/weights and validate grid/support/finiteness.
8. Compute fixed final-boundary radiometric metrics, structural preservation, and descriptive mosaic statistics.
9. Generate the four standardized audit figures and the fixed 14-section report.
10. Run focused Task13B tests, Task13A/13A.1 and radiometric/mosaic regressions, full suite, compileall, and git diff check. Stop at Task13B.

## Verification evidence

Record commands/results in `.superpowers/sdd/2026-09-28-task13b-five-scene-multiscene-labeling/progress.md`. The known baseline is 55 Task13A/13A.1 focused tests and 21 radiometric/mosaic regression tests; the previously disclosed two Task12/VOLRN full-suite failures must remain explicitly separated from Task13B results.
