# Phase4 four-pair DEVELOPMENT adapter contract

Only frozen B9 pairs `00_01`, `02_03`, `05_07`, `08_10` enter the preparation manifest. Their 43 first-review-approved DEVELOPMENT points are distributed 11/9/9/14. No TEST candidate/event/coordinate is available to the DEVELOPMENT scoring interface.

The common main table has six method IDs: `SIFT`, `KARIOS`, `AROSICS_GLOBAL` (`COREG` global), `LoFTR`, `EfficientLoFTR`, `LightGlue+DISK`. `AROSICS_LOCAL` (`COREG_LOCAL`) is a separate local-warp experiment, not a global-translation ranking entry. KARIOS KLT output and AROSICS global shift must eventually be mapped to the same native B9 frame; their dependencies are presently absent and no end-to-end validity is claimed.

Each adapter result identifies a trial pair and method, `SUCCESS` or `FAIL`, a nonnegative elapsed time, failure reason when applicable, and input/preprocessing provenance. A success gives a finite `(delta_row, delta_col)` in the **native** `scene_i -> scene_j` direction. `provenance.frame` must be `native` and `translation_semantics` must be `full_native_i_to_j` (or `full_native_j_to_i` for a declared reverse input). This is the complete native-pixel mapping, **including any offset between georeferenced image origins**, not merely a residual correction after georeferenced alignment. If the upstream method returns a `scene_j -> scene_i` pure translation, negate both components during normalization and record `input_direction`; transformations from resized or common-grid inputs require their explicit inverse preprocessing chain before claiming a native result. The evaluator rejects unnormalized result dictionaries. Do not compare arbitrary raw matcher confidence scales. A failed pair keeps a `FAIL` row and null metrics; an unattempted pair keeps `NOT_ATTEMPTED`. The four-pair denominator remains visible for every method.

The DEVELOPMENT-only evaluator computes point RMSE, P95, max error and PCK@1/2/3 px for each successful pair, then a pair-level macro summary over successes together with fail/missing counts. This is preparation for a future small real trial, **not** the six-method trial itself. No TEST point may enter model fitting, RANSAC, threshold selection, tuning, or this evaluator.

Read-only local dependency/preparation check:

```powershell
python scripts/prepare_benchmark_v1_development.py --check-env
```

To write a small input manifest without running matchers:

```powershell
python scripts/prepare_benchmark_v1_development.py --check-env --write-manifest docs/benchmark/v1/phase4/development_input_manifest.json
```

There is deliberately **no formal six-algorithm run command in Phase4**. Before the later DEVELOPMENT trial, install/verify KARIOS and AROSICS in compatible environments, validate each adapter's native-coordinate transform on synthetic data, freeze actual method settings and timing scope, and give a separate experiment instruction. Import availability alone is not an end-to-end pass.
