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

The initial block matches, robust pair measurements, and network-adjustment shifts are training and registration diagnostics. They are not final quality evidence. Each pair first builds a common-valid mask and reserves blocked spatial HOLDOUT cells. Global matching, global residual refinement, and local residual controls are restricted to TRAIN cells; the local RBF remains behind an internal spatial cross-validation gate. After global and optional gated local refinement, the pipeline builds the final registered arrays from the ORIGINAL arrays and runs bounded residual phase correlation only on the reserved HOLDOUT cells. Validation block size is selected from `[384, 256, 192]` using geometric candidate counts, never by residual quality. The resulting `final_validation` and `quality` dictionaries are the authoritative quality evidence used by the gate.

Every required validation edge must also have no `failure_reason` and at least
`final_min_blocks` accepted blocks. A passing pooled summary cannot mask a
failed or under-sampled required edge. If no candidate block size can provide
enough holdout geometry, the result is explicitly reported as insufficient
geometry and remains FAIL; PASS thresholds are never reduced to compensate.

The diagnostic JSON separates `overlap`, `holdout`, `local_controls`,
`local_field`, and `final_validation`. This lets a failed run distinguish an
unusable common-valid footprint, insufficient holdout geometry, low-confidence
validation, and genuine post-warp residual error.

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

On success, the command writes `registration_diagnostics.json`, registered
reference and target GeoTIFFs, a red-green overlay, and a diagnostic mosaic
beneath the requested output directory. If connectivity or the configured
quality gate fails, it writes only a structured failure JSON record, returns a
nonzero exit status, and does not label fallback arrays as registered. The
default mosaic mode is `weighted`; use `--mosaic-mode source_selection` only
when that diagnostic mode is explicitly requested.

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
