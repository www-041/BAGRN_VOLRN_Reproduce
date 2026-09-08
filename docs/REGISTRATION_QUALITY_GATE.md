# Registration Quality Gate

## Overview

The registration quality gate prevents low-quality geometric alignment from entering the time-consuming BAGRN/VOLRN pipeline. It uses robust shift estimation and residual analysis to classify registration quality.

## Quality Thresholds

### PASS (Required for B14 experiment)
- Mean confidence >= 0.50
- Residual median <= 0.35 px
- Residual RMSE <= 0.60 px
- Residual P95 <= 1.00 px
- Validation inliers >= 5

### WARN (Acceptable but not ideal)
- Mean confidence >= 0.45
- Residual median <= 0.50 px
- Residual RMSE <= 0.75 px
- Residual P95 <= 1.25 px
- Validation inliers >= 5

### FAIL (Pipeline stops)
- Any other condition

## B14 Experiment Configuration

For the DZ01V B14 N=2→4→6 experiment series, `required_quality: pass` is set. This means:
- Only PASS quality allows proceeding to BAGRN/VOLRN
- WARN quality will stop the pipeline
- This prevents wasting ~15 minutes on VOLRN when registration is insufficient

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

## Manual Validation Procedure

After the code is merged, the user should:

1. Run preflight check
2. Run registration diagnostic for each scene pair
3. Verify quality metrics meet PASS thresholds
4. Only then run full N=2 experiment

## References

- `src/coregistration.py::robust_shift_estimate()` - Robust estimation
- `src/coregistration.py::classify_registration_quality()` - Quality classification
- `src/multiband_pipeline.py::register_scenes()` - Quality gate integration
- `configs/dz01_mosaic_series_b14.yaml` - B14 experiment parameters
