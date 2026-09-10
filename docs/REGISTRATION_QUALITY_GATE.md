# Registration Quality Gate

## Overview

The registration quality gate prevents low-quality geometric alignment from entering the time-consuming BAGRN/VOLRN pipeline. It uses common-valid overlap sampling, a spatial holdout reserved before any control extraction, robust shift estimation, and an independent validation pass on the final registered arrays for the quality decision.

## Quality Thresholds

### PASS (Required for B14 experiment)
- Mean confidence >= 0.50
- Final-validation median <= 0.35 px
- Final-validation RMSE <= 0.60 px
- Final-validation P95 <= 1.00 px
- Final-validation inliers >= 5

### WARN (Acceptable but not ideal)
- Mean confidence >= 0.45
- Final-validation median <= 0.50 px
- Final-validation RMSE <= 0.75 px
- Final-validation P95 <= 1.25 px
- Final-validation inliers >= 5

### FAIL (Pipeline stops)
- Any other condition

## B14 Experiment Configuration

For the DZ01V B14 N=2→4→6 experiment series, `required_quality: pass` is set. This means:
- Only PASS quality allows proceeding to BAGRN/VOLRN
- WARN quality will stop the pipeline
- This prevents wasting ~15 minutes on VOLRN when registration is insufficient

## Registration and final evidence

The initial block matches, robust pair measurements, and network-adjustment shifts are training and registration diagnostics. They are not final quality evidence. Each pair first builds a common-valid overlap grid and reserves exact validation windows from `[384, 256, 192]`, largest viable size first, using only footprint geometry and common-valid ratios. Global matching, global residual refinement, and local residual controls are restricted to TRAIN cells; the local RBF remains behind an internal spatial cross-validation gate. After global and optional gated local refinement, the pipeline builds the final registered arrays from the ORIGINAL arrays and runs bounded residual phase correlation only on those exact reserved HOLDOUT windows. The resulting `final_validation` and `quality` dictionaries are the authoritative quality evidence used by the gate.

The reservation also simulates the configured global, refinement, and local
training footprints before accepting a holdout layout. A partially valid TRAIN
footprint is screened by the existing joint-valid ratio; it is not rejected
merely because every pixel is not common-valid. The separate
`holdout_exclusion_mask` only prevents a training footprint from intersecting
the reserved validation union, and the phase-correlation fallback uses the
same train-only exclusion.

For local RBF refinement, the pipeline creates one deterministic spatial
cross-validation fold plan from the local controls and evaluates every value
in `local_smoothing_candidates` on exactly those folds. A candidate is
eligible only when all planned folds complete with finite predictions. The
selected value is the best candidate that clears both the existing held-out
RMSE and P95 improvement gates, ordered by candidate RMSE, candidate P95, and
then smoothing. If no available candidate clears both gates, the local field
is not enabled; the best available result is retained only as a diagnostic.
The final field fit receives that selected smoothing explicitly, so the
diagnostic `field_stats.smoothing` always identifies the model actually used.

When `local_cv_buffer_pixels` is configured, each blocked fold removes every
candidate training control within that Euclidean distance of the held-out
controls in scene-pixel coordinates. The buffer is applied to one fixed fold
plan shared by all smoothing candidates; it is not tuned from final HOLDOUT
residuals. Each fold records its pre/post-buffer training counts and minimum
train/test distance. If buffering leaves too few training controls or usable
folds, the CV result is unavailable and registration falls back to the
global-only local stage. The general default remains `0`; the DZ01V B14
configuration uses `256` pixels.

Every required validation edge must also have no `failure_reason` and at least
`final_min_blocks` accepted blocks. A passing pooled summary cannot mask a
failed or under-sampled required edge. If no candidate block size can provide
enough independent validation windows, the result is failed before matching
starts with `insufficient independent validation geometry`; PASS thresholds are
never reduced to compensate. The reservation diagnostics account for every
geometry candidate, including unused candidates and train-usable cells.

The diagnostic JSON separates `overlap`, `holdout`, `local_controls`,
`local_field`, and `final_validation`. This lets a failed run distinguish an
unusable common-valid footprint, insufficient holdout geometry, low-confidence
validation, and genuine post-warp residual error.

For each reserved HOLDOUT block, the diagnostic command also writes
`holdout_local_field_samples.csv`. Its `reference_center_*` columns are in
the validation reference grid, while `field_center_*` columns are in the
scene grid where the applied local field was sampled. The CSV records whether
the mapping succeeded, the actual faded-field value and hull support, nearest
local-control distance, predicted local correction, and final HOLDOUT
residual. This is diagnostic evidence only and does not participate in model
selection or quality gating.

The paired stage diagnostics add `global_only_validation` and
`stage_validation_comparison`. Both validation stages use the exact same
reserved windows, and every common block is aligned by its reserved row and
column rather than by list order. `rmse_improvement`, `p95_improvement`, and
`median_improvement` are defined as global-only minus final; a positive value
means that the selected local RBF improved the independent HOLDOUT. The
comparison is diagnostic only and never changes the formal `quality` decision,
which remains driven exclusively by final validation.

Interpret the paired result as follows:

- If global-only RMSE is larger than final RMSE, local RBF improved the
  independent HOLDOUT, although the result may still be below PASS/WARN.
- If global-only RMSE is smaller than final RMSE, local RBF worsened the true
  HOLDOUT despite internal buffered CV; investigate spatial overfit or local
  controls that do not represent the residual.
- If the two stages are nearly equal and both fail, inspect the sampled applied
  local field and HOLDOUT maps for hull fade, coverage, or controls that do not
  represent the residual.

The diagnostic command may write connected quality-FAIL results when explicitly
invoked for diagnosis. These files are evidence only: the command still
returns a nonzero exit code, writes JSON status `fail`, and never exposes those
arrays as formal BAGRN/VOLRN inputs. Blocked or disconnected results continue
to produce failure JSON without registered raster artifacts.

### HULL-C1 causal diagnostic

HULL-C1 is an opt-in counterfactual experiment enabled with
`--hull-causal-test`. Its only treatment is local-RBF support: the production
legacy branch uses the existing buffered-hull fade, while the strict branch
uses the same raw RBF field with inside-hull-only support. The raw RBF is fit
once per accepted scene; CV, smoothing, component cap, controls, global shifts,
and the reserved HOLDOUT are frozen and shared by all three stages.

`registered_arrays` and the main `quality/status` continue to represent the
legacy production final. Strict arrays are created from ORIGINAL inputs only
for counterfactual diagnostics and cannot enter BAGRN/VOLRN or alter the exit
code. The final HOLDOUT is never used to choose a branch or tune a model.
HULL-C1 writes `hull_causal_stage_metrics.csv`,
`hull_causal_window_stats.csv`, and strict-hull raster/overlay artifacts when
the diagnostic is available. The window table measures support and RBF
correction over each complete reserved window; `legacy_minus_strict` is the
causal comparison, and a positive value indicates strict improvement. The
integrity gate checks shared inputs and strict zero support outside the hull;
it does not judge whether strict improves RMSE or P95.

Whether strict support should become a production change is deliberately left
to the user checkpoint after the real N=2 HULL-C1 run. B12, affine residuals,
DEM terms, and mountainous local refinement are outside this diagnostic.

## Robust Estimation

The pipeline uses MAD-based robust estimation instead of simple weighted average:
1. Filter matches by confidence threshold (default: 0.50)
2. Iterative MAD-based outlier rejection (scale: 3.0)
3. Weighted mean of inliers
4. Residual statistics computation

## Diagnostic Command

For manual registration diagnosis (no BAGRN/VOLRN/mosaic):

```bash
python scripts/diagnose_registration_pair.py \
    --config configs/dz01_mosaic_series_b14.yaml \
    --scene-i 0 --scene-j 1
```

On success, or for an explicitly allowed connected quality-FAIL diagnosis, the
command writes `registration_diagnostics.json`, separate global-only and final
reference/target GeoTIFFs, two red-green overlays, a final HOLDOUT block map,
and a diagnostic mosaic beneath the requested output directory. Connectivity
or registration-blocked failures still write only structured failure JSON,
return a nonzero exit status, and do not label fallback arrays as registered.
The default mosaic mode is `weighted`; use `--mosaic-mode source_selection`
only when that diagnostic mode is explicitly requested.

## Manual Validation Procedure

After the code is merged, the user should:

1. Run preflight check
2. Run registration diagnostic for each scene pair
3. Verify the reserved HOLDOUT has enough accepted blocks and its final metrics meet PASS thresholds
4. Only then run full N=2 experiment

## References

- `src/coregistration.py::robust_shift_estimate()` - Robust estimation
- `src/coregistration.py::classify_registration_quality()` - Quality classification
- `src/multiband_pipeline.py::register_scenes()` - Quality gate integration
- `configs/dz01_mosaic_series_b14.yaml` - B14 experiment parameters
