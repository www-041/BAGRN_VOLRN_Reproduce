# Task13B.1 Deep-Overlap Tie Diagnosis + Non-Saturating Geometry Tie-Break

## Frozen scope

Use the existing Task13B labels, frozen scene masks, and Task13B pairwise score/interiority evidence. Do not rerun matcher/global/BAGRN/VOLRN/Task13A/Task13A.1 or regenerate pairwise fields. Preserve every old resolved label exactly; operate only on old `LABEL_UNRESOLVED` pixels. Do not use radiometry, scene ID, manifest index, file order, propagation, Graph Cut, or Poisson.

## Sequence

1. Audit all unresolved pixels and quantify clipped-interiority saturation, non-saturated normalized interiority, raw EDT, candidate sets, and coverage.
2. Gate on `all_clipped_equal_fraction >= 0.95`; if not met, stop without modifying the resolver.
3. On unresolved pixels only, apply level-3 `D_i / max(P95_i, 1e-6)` with margin `>1e-6`, then level-4 raw EDT with margin `>1e-6`; retain unresolved geometry ties.
4. Test old-label immutability, invalid candidates, radiometric isolation, and four scene-order permutations.
5. If all ties resolve, continue Task13B Task 5 onward in a new `multiscene_task13b1_tie_resolution/continuation/` tree. Otherwise record `PARTIAL_TIE_RESOLUTION` and stop.
6. Run focused, Task13B, Task13A/13A.1, radiometric/mosaic, full-suite, compileall, and diff-check verification.

## Deliverables

`data/output/b9_five_scene_validation/multiscene_task13b1_tie_resolution/` with diagnosis CSV/JSON, resolved label/method rasters, tie summary, and report. Continuation artifacts are written only if the tie-resolution gate passes. No existing Task13B artifact is overwritten.
