# Task16 — 13-scene BAGRN + VOLRN weighted-feather comparison

## Scope and fixed decisions

- Use the existing `data/output/final_pipeline/b9_13scene/` Task15 geometry and Stage06 BAGRN normalized scenes as read-only inputs.
- Do not rerun EfficientLoFTR, RANSAC, Translation-L2, canonical warp, distance-cache generation, or BAGRN.
- Use the canonical Task10D VOLRN implementation with exactly: block size 400 pixels, lambda 0.5, rho 1.0, max_iter 200, tol 1e-4, fixed rho, and the existing Jacobi preconditioner policy.
- Write all Task16 artifacts under `data/output/b9_13scene_volrn_comparison/`; never overwrite Task15 outputs.
- Preserve the user’s existing modified files and do not commit, merge, or push.

## Implementation sequence

1. Add a read-only provenance audit for the Task15 source config, canonical grid, Stage06 scene manifest/masks, and Task15 V2 reference. Stop with `HARD_STOP_PROVENANCE_MISMATCH` if the frozen 13-scene identity/grid/support checks fail.
2. Add a Task16 runner that builds the A baseline from the persisted Stage06 BAGRN scene rasters and the same weighted-feather mosaic implementation, then runs canonical fixed-parameter VOLRN on those same scene arrays for B.
3. Persist per-scene corrected rasters, VOLRN coefficients, complete solver histories, coefficient/application diagnostics, support masks, runtime records, and fixed-stretch previews.
4. Compute Task10D radiometric metrics and Task15-compatible A/B/C boundary/structure metrics on common support, including finite/invalid accounting and the required comparison CSVs.
5. Generate the three-method preview, plots, protocol/provenance manifest, and the 23-question Task16 report with strict-convergence versus finite-iter200 status stated separately.
6. Run focused unit tests and static/import checks, then execute the real 13-scene Task16 run. Verify required files, support equality, finite outputs, history length, and report consistency before claiming completion.

## Verification focus

- A support must equal the frozen 62,033,096-pixel Task15 union support.
- B must use the same scene order, transforms, masks, grid, and weighted-feather code as A.
- Every solver history must contain the actual iteration records up to 200; no fabricated records are permitted.
- All coefficients, solver states, corrected scene valid pixels, mosaics, and reported structural metrics must be finite.
- A non-converged but finite 200-iteration run is reported as `PASS_FINITE_ITER200_NONCONVERGED`; numerical invalidity stops formal B reporting.
