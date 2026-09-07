# DZ01V B14 Mosaic Series (N=2, 4, 6)

## Experiment Overview

This document describes the canonical experiment sequence for the DZ01V B14 mosaic series with nested 2/4/6 scene subsets.

**Configuration:**
- Sensor: DZ01V (same sensor for all scenes)
- Resolution: Same spatial resolution for all scenes
- Band: B14 only
- Control scene: scene_20251114 (fixed)
- Methods: original, bagrn, bagrn_volrn
- Mosaic mode: weighted only
- VOLRN parameters:
  - block_size: 200
  - lambda: 0.1
  - rho: 1.0
  - max_iter: 200
  - tol: 1e-4

**Scene Table:**

| N | Scene IDs |
|---|-----------|
| 2 | scene_20251114, scene_20251120 |
| 4 | N=2 + scene_20251208, scene_20251215 |
| 6 | N=4 + scene_20251214_025253, scene_20251214_025247 |

**Output Structure:**
```
data/output/mosaic_series/
  dz01v_mosaic_series_b14/
    scale/
      n2/
        original/
          mosaic_original_weighted.tif
        bagrn/
          mosaic_bagrn_weighted.tif
        bagrn_volrn/
          mosaic_bagrn_volrn_weighted.tif
      n4/
        [same structure as n2]
      n6/
        [same structure as n2]
```

## Preflight Check

**Before running any experiments**, verify the configuration:

```bash
python scripts/preflight_mosaic_series.py \
  --config configs/dz01_mosaic_series_b14.yaml \
  --scene-count 2,4,6
```

Expected output:
- ✓ All files exist
- ✓ All sensors are DZ01V
- ✓ CRS consistent across all scenes
- ✓ Resolution consistent across all scenes
- ✓ Overlap graph connected for N=2, 4, 6

**If preflight fails**, do not proceed with experiments. Fix the issues first.

## Formal commands — USER RUNS THESE MANUALLY

### N=2 (First)

```bash
python -m src.experiment_runner scale \
  --config configs/dz01_mosaic_series_b14.yaml \
  --scene-count 2
```

**Review criteria:**
- Registration: All edges connected, no rejected edges
- Metrics: ADM, ADSD, CD, GL should decrease from original → bagrn → bagrn_volrn
- Mosaics: Three weighted mosaics generated (original, bagrn, bagrn_volrn)
- Visual: No seams, no ghosting, no double edges
- Compare bagrn vs bagrn_volrn: VOLRN should show better local consistency

**Only proceed to N=4 if N=2 results are acceptable.**

### N=4 (Second)

```bash
python -m src.experiment_runner scale \
  --config configs/dz01_mosaic_series_b14.yaml \
  --scene-count 4
```

**Review criteria:**
- Registration: All 4 scenes connected in spanning tree
- Metrics: Should still show improvement from original → bagrn → bagrn_volrn
- Mosaics: Three weighted mosaics generated
- Visual: No new seams or artifacts compared to N=2
- Performance: Check execution time and memory usage

**Only proceed to N=6 if N=4 results are acceptable.**

### N=6 (Third)

```bash
python -m src.experiment_runner scale \
  --config configs/dz01_mosaic_series_b14.yaml \
  --scene-count 6
```

**Review criteria:**
- Registration: All 6 scenes connected, including the two 2025-12-14 scenes
- Metrics: BAGRN+VOLRN should remain stable as scene count grows
- Mosaics: Three weighted mosaics generated
- Visual: No severe seams or local over-correction
- Check: The two scenes from 2025-12-14 should integrate well

## Stop Conditions

**Stop and investigate if any of the following occur:**

1. **CLI exit code non-zero**
   - Check error messages in log
   - Verify input files exist and are valid

2. **pipeline_status != "success"**
   - Check scale_results.csv for details
   - Look for failed_methods or registration issues

3. **registration_connected != true**
   - Some scenes not connected in registration graph
   - Check overlap preflight results
   - May need to adjust control scene or add more overlaps

4. **failed_methods non-empty**
   - Some normalization methods failed
   - Check logs for specific errors

5. **Missing BAGRN+VOLRN mosaic**
   - Pipeline did not complete successfully
   - Check for errors in VOLRN stage

6. **Valid-area NaN/Inf in output**
   - Indicates numerical issues
   - Check for NoData handling problems

7. **Non-finite metrics**
   - ADM/ADSD/CD/GL are NaN or Inf
   - Indicates overlap or computation issues

8. **Visual artifacts:**
   - Obvious double edges/ghosting
   - Stronger seams after normalization (should be weaker)
   - Local over-correction (overshooting)
   - Color shifts in uniform areas

## Deferred Work

**The following are explicitly NOT part of this experiment series:**

- B01–B14 full multiband experiments
- Comparison methods: histogram matching, moment matching, Wallis
- Parameter sensitivity sweeps
- Seam width optimization
- Spectral metrics evaluation
- Diagnostics analysis
- Cross-sensor experiments (DZ01S ↔ DZ01V)
- Cross-resolution experiments

**These may be added in future experiment series after B14 results are validated.**

## Troubleshooting

### Registration fails for some edges

- Check overlap graph connectivity with preflight
- Verify scenes actually overlap geographically
- May need to adjust control scene if current one doesn't overlap well with all others

### VOLRN produces poor results

- Check block_size parameter (200 may be too large/small for your data)
- Verify lambda parameter (0.1 controls regularization strength)
- Check for NoData handling issues in overlap regions

### Mosaic has visible seams

- Check feather_width parameter (100 may need adjustment)
- Verify registration accuracy (large shifts cause seam issues)
- May need to use narrow_feather mode instead of weighted

### Out of memory

- Reduce smoke_crop_size in config
- Run with fewer scenes first (N=2 only)
- Check memory usage in scale_results.csv

## Success Criteria

**The experiment series is successful if:**

1. All N=2, 4, 6 runs complete with pipeline_status="success"
2. Registration graphs are connected for all N
3. Metrics show consistent improvement: original > bagrn > bagrn_volrn
4. Mosaics are visually acceptable (no severe artifacts)
5. BAGRN+VOLRN stability: metrics don't degrade as N increases
6. The two 2025-12-14 scenes integrate well in N=6

**If all criteria are met**, the B14 mosaic series is validated and can be used as a baseline for further experiments.
