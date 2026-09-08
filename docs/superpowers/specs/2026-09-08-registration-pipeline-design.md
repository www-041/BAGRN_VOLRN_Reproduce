# Registration Pipeline Reliability Design

## Purpose

This design makes the multi-scene registration stage reliable enough to gate BAGRN and VOLRN for the DZ01V B14 mosaic series. It addresses the implementation requirements in `DZ01V_B14_Claude_配准代码修复实施方案.docx` while preserving the existing Python-first pipeline and its public normalization interfaces.

The registration result must be decided from residuals measured on the final registered images, not from pre-warp pair-shift consistency. A scene may enter BAGRN/VOLRN only when the configured `required_quality` accepts the final quality classification.

## Scope and non-goals

In scope:

- robust block-level pair translation estimation with joint two-dimensional MAD filtering;
- network adjustment using the robust pair measurements;
- iterative post-global residual rematching and correction;
- optional local RBF refinement gated by control-point sufficiency, spatial coverage, and held-out cross-validation;
- one final displacement-field warp from each ORIGINAL scene;
- independent final validation and configured PASS/WARN/FAIL classification;
- a stable registration return schema and registration-only diagnostic outputs;
- regression and synthetic tests for the above behavior;
- documentation synchronization.

Out of scope:

- changes to the BAGRN or VOLRN mathematical models;
- changes to the paper's radiometric formulas;
- execution of real N=2, N=4, or N=6 experiments;
- changes to unrelated mosaic, metric, or report-generation behavior except where the new registration schema is consumed.

## Architecture and data flow

`MultibandPipeline.register_scenes()` remains the orchestration boundary. The lower-level numerical operations stay in `src/coregistration.py` and communicate through small dictionaries containing explicit diagnostics.

The registration flow is:

1. Match every geometric-overlap edge at block level.
2. Build one robust pair measurement per accepted edge. A pair measurement uses all accepted block matches, one joint XY inlier mask, confidence-weighted inlier means, and residual statistics. A phase-correlation result is allowed only as an explicit fallback when block matching cannot produce a robust result.
3. Check graph connectivity and solve `multi_image_network_adjustment()` using the robust pair shifts. Pair diagnostics remain edge-level diagnostics and are not the final quality result.
4. Create temporary global-only registered arrays by warping each scene from its ORIGINAL array with the current global shift. These arrays are used only for residual rematching.
5. Rematch critical spanning-tree edges and any configured usable overlap edges on the temporary global-only arrays. Solve the residual pair measurements as a delta network, update the global shifts, and repeat until the configured iteration limit or stop magnitude is reached. Corrections larger than the configured maximum are rejected and recorded as warnings.
6. For each target scene, build local residual controls from post-global matches using a single target/image coordinate convention. Only attempt RBF when local refinement is enabled, enough controls exist, enough spatial groups exist, and post-global residual matches are available.
7. Compare a translation-only baseline with the candidate RBF field on held-out controls. Apply RBF only when both configured held-out RMSE and P95 improvements pass. Otherwise record a global-only fallback.
8. Build the final registered arrays by applying the combined global and local displacement field exactly once to each ORIGINAL array. Valid zero-valued pixels remain valid when `nodata is None`; invalid samples are represented by the configured NoData behavior without treating zero as NoData.
9. Run independent validation grids on final registered target arrays. Validation must use configured confidence, residual-shift, minimum-block, and final quality thresholds, and must not reuse training shifts or training residual statistics.
10. Return the stable registration schema and stop the pipeline when the final quality is below `required_quality` or when the graph is not connected.

The reference/control scene remains fixed at zero global and local displacement. A scene's local field is defined in that scene's pixel coordinates and is clipped component-wise to `±local_max_component`; its influence fades to zero outside the control hull using `compute_hull_fade_mask()`.

## Interfaces

### Robust pair measurement

Add `build_robust_pair_measurement(matches, params)` in `src/coregistration.py`.

It consumes block match dictionaries containing `shift_dx`, `shift_dy`, and `confidence`, plus optional coordinates and screening diagnostics. It returns a dictionary containing at least:

```text
idx_i, idx_j,
shift_dx, shift_dy,
confidence,
n_blocks_total, n_blocks_inlier, inlier_ratio,
residual_median, rmse, p95,
status, matches, screening
```

The returned `rmse` and `p95` describe block-shift residuals around the robust pair translation. They are not shift magnitudes. Failure returns `status: "fail"` with zero confidence and an explanatory reason; callers may then try overlap phase correlation.

### Global refinement

Use a helper with an explicit original-array input, current global shifts, transforms, nodata values, and rematch edge list. It returns updated shifts plus an iteration history. Every temporary warp is generated from ORIGINAL arrays so interpolation error cannot accumulate.

### Local refinement

Reuse and correct the existing local-control, balancing, RBF-fitting, hull-fade, and cross-validation helpers where possible. The orchestration result records, per scene, whether RBF was used, why it was rejected, the number and distribution of controls, CV baseline/candidate metrics, clipping statistics, and field statistics.

### Registration result

`register_scenes()` returns:

```text
registered_arrays
global_shifts
pair_matches
connected: bool
local_refinement: {
  enabled: bool,
  used_for_scenes: list[int],
  fallback_scenes: list[int],
  cv_results: dict
}
final_validation: {edges: ..., overall: ...}
quality: {
  quality: "pass" | "warn" | "fail",
  rmse: float,
  p95: float,
  median: float,
  confidence: float,
  n_blocks: int
}
diagnostics
```

Existing graph fields such as `spanning_tree`, `geometric_edges`, `matching_edges`, `rejected_edges`, `connected_components`, and `unreachable_scenes` remain available for compatibility.

## Quality and failure behavior

The configured PASS thresholds remain the B14 thresholds:

- mean confidence ≥ 0.50;
- residual median ≤ 0.35 px;
- residual RMSE ≤ 0.60 px;
- residual P95 ≤ 1.00 px;
- accepted validation blocks ≥ 5.

WARN uses the existing configured WARN thresholds. `required_quality: pass` rejects WARN and FAIL. A missing critical validation edge is not silently treated as PASS; it is recorded as unavailable and the overall classification is conservative according to the configured requirement. A disconnected scene graph always blocks registration.

The pipeline must decide success using `registration["quality"]["quality"]` and the configured required quality, not merely `unreachable_scenes`. Normalization is not allowed to run after a failed registration gate.

## Diagnostic behavior

`scripts/diagnose_registration_pair.py` will select the two requested scenes, set `control_scene` to the selected first scene, run registration only, and serialize `registration_diagnostics.json`. The JSON will include scene IDs, raw block matches, robust pair results, initial and final global shifts, global-refinement history, local-refinement/CV decisions, local-field statistics, independent final validation, and PASS/WARN/FAIL.

The diagnostic will also write registered reference and target images plus a red-green overlay. Any diagnostic mosaic artifacts will be created from the registered ORIGINAL arrays. `source_selection` is used only as an explicit diagnostic mosaic mode; it will not be treated as the weighted mode.

## Testing strategy

Tests are added before implementation for each behavior:

- joint XY outlier rejection and block-level N=2 estimation;
- post-global residual refinement and rewarping from ORIGINAL arrays;
- RBF correction for smooth spatial residuals and rejection when held-out CV does not improve;
- final quality gate using post-warp residuals and configured thresholds;
- NoData-safe single warp preserving valid zeros;
- distance-field boundary analysis and configured confidence acceptance;
- diagnostic consumption of the actual registration schema.

Each focused test must be observed failing for the intended reason, then made green with the smallest implementation. The final verification includes focused tests, `pytest -q`, dry-run, configuration validation, and metadata-only preflight. Existing environment-level temporary-directory permission failures are reported separately from source regressions.

## Files expected to change

- `src/coregistration.py`: robust pair measurement, global residual refinement, local refinement corrections, warp NoData handling, independent validation, and threshold/dead-code fixes.
- `src/multiband_pipeline.py`: orchestration and stable registration schema/quality gate.
- `scripts/diagnose_registration_pair.py`: registration-only diagnostic and artifacts.
- `configs/dz01_mosaic_series_b14.yaml`: only if required keys are missing after implementation review.
- `docs/REGISTRATION_QUALITY_GATE.md`: synchronize the documented flow and thresholds.
- `tests/`: focused regression and synthetic tests listed above.

No real-data output or unrelated untracked artifact will be deleted.
