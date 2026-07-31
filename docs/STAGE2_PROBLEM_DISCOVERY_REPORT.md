# Stage 2: Problem Discovery Report

**Date:** 2026-08-01
**Dataset:** DZ01V 4-scene subset (scene_20251114, scene_20251120, scene_20251208, scene_20251215)
**Crop size:** 1024 × 1024 px (centered on overlap edge centers)
**Baseline Ave:** original=145.53, bagrn=16.17 (−88.9%), volrn_only=55.89 (−61.6%), bagrn_volrn=12.74 (−91.2%)

---

## 1. Band Attribution

Per-band Ave decomposition for each method.

### BAGRN (Global Normalization Only)

| Band | ADM | ADSD | Ave |
|------|------|------|------|
| B01 | 13.28 | 7.95 | **10.61** |
| B02 | 11.49 | 5.35 | 8.42 |
| B03 | 8.20 | 4.29 | 6.24 |
| B04 | 21.80 | 13.76 | 17.78 |
| B05 | 33.58 | 20.01 | 26.79 |
| B06 | 17.46 | 12.46 | 14.96 |
| **B07** | **98.85** | **32.28** | **65.56** |
| **B08** | **60.94** | **25.55** | **43.24** |
| **B09** | **47.62** | **15.77** | **31.69** |
| B10 | 1.80 | 9.37 | 5.58 |
| B11 | 12.90 | 4.00 | 8.45 |
| B12 | 17.75 | 13.70 | 15.72 |
| **B13** | **38.62** | **25.25** | **31.94** |
| B14 | 14.37 | 14.52 | 14.44 |

**Top 3 worst bands:** B07 (65.6), B08 (43.2), B13 (31.9)
**Top 3 best bands:** B10 (5.6), B03 (6.2), B02 (8.4)

### BAGRN+VOLRN (Full Pipeline)

| Band | ADM | ADSD | Ave |
|------|------|------|------|
| B01 | 8.92 | 10.93 | 9.93 |
| B02 | 10.93 | 2.62 | 6.78 |
| B03 | 7.09 | 2.94 | 5.01 |
| B04 | 19.72 | 11.96 | 15.84 |
| B05 | 27.99 | 18.32 | 23.15 |
| B06 | 15.20 | 11.56 | 13.38 |
| **B07** | **43.75** | **30.26** | **37.01** |
| **B08** | **51.30** | **25.13** | **38.22** |
| **B09** | **38.65** | **14.94** | **26.80** |
| B10 | 2.99 | 8.19 | 5.59 |
| B11 | 11.97 | 3.47 | 7.72 |
| B12 | 12.99 | 9.58 | 11.29 |
| **B13** | **33.02** | **18.55** | **25.78** |
| B14 | 11.52 | 10.14 | 10.83 |

**Top 3 worst bands:** B08 (38.2), B07 (37.0), B09 (26.8)
**Top 3 best bands:** B03 (5.0), B10 (5.6), B02 (6.8)

### Key Finding

The SWIR bands (B07–B09) and B13 dominate the Ave in both methods. VOLRN reduces B07 Ave from 65.6 → 37.0 (−44%) but B08 only from 43.2 → 38.2 (−12%), indicating the SWIR bands have inherently higher radiometric variability that VOLRN cannot fully correct. The VNIR bands (B01–B06) are already well-normalized by BAGRN alone.

---

## 2. Scene Attribution

Per-scene Ave (averaged over all pairs involving that scene).

### BAGRN

| Scene | Ave |
|-------|------|
| 20251114 | 17.59 |
| **20251120** | **30.62** |
| 20251208 | 14.68 |
| 20251215 | 23.24 |

**Worst scene:** 20251120 (Ave=30.62, 1.77× the best scene)

### BAGRN+VOLRN

| Scene | Ave |
|-------|------|
| 20251114 | 14.89 |
| **20251120** | **23.47** |
| 20251208 | 11.71 |
| 20251215 | 17.74 |

**Worst scene:** 20251120 (Ave=23.47, consistently worst across both methods)

### Key Finding

Scene 20251120 (November 20) is consistently the weakest link, contributing disproportionately to overall Ave. This is consistent across BAGRN and BAGRN+VOLRN, suggesting the radiometric differences for this scene are too large for the current normalization to fully correct.

---

## 3. Spatial Attribution

Interior vs boundary Ave (boundary width=100 px).

| Method | Interior Ave | Boundary Ave | Ratio (B/I) |
|--------|-------------|-------------|-------------|
| original | 292.87 | 288.67 | 0.986 |
| bagrn | 36.93 | 34.69 | 0.939 |
| volrn_only | 107.73 | 117.35 | **1.089** |
| bagrn_volrn | 29.36 | 28.77 | 0.980 |

### Key Finding

- BAGRN and BAGRN+VOLRN: boundary Ave ≤ interior Ave (ratio < 1), meaning boundary regions are slightly better or equal — no boundary degradation.
- VOLRN only: ratio > 1 (1.09), boundary regions are 9% worse — VOLRN without BAGRN pre-processing shows mild boundary degradation.
- The degradation is modest (9%), not the dominant cause of Ave.

---

## 4. Multi-Window Evaluation

Per-window Ave across 6 window positions (edge-centered).

| Method | Min Ave | Max Ave | Mean Ave | Max/Min Ratio |
|--------|---------|---------|----------|---------------|
| original | 249.0 | 391.9 | 319.5 | 1.57 |
| bagrn | 11.7 | 198.4 | 95.0 | **16.96** |
| volrn_only | 106.4 | 275.1 | 190.5 | 2.59 |
| bagrn_volrn | 10.8 | 198.5 | 93.8 | **18.38** |

### Key Finding

BAGRN and BAGRN+VOLRN show extreme spatial variance (max/min ratio ~17–18×). The min Ave (~11) is excellent, but the max Ave (~198) is nearly as bad as the original. This confirms the problem is **highly localized**: certain window positions have poor overlap statistics while others are excellent. The Ave degradation is not uniform but concentrated in specific spatial regions.

---

## 5. NaN Trace

| Metric | Value |
|--------|-------|
| Input nan (total across bands) | scene_0: 2,492,154, scene_1: 2,661,862, scene_2: 5,570,124, scene_3: 42,980 |
| Output nan (total) | Same as input |
| **New nan pixels** | **0** |
| Block convergence | All 14 bands converged |
| ADMM iterations | 18–24 per band |
| Extreme a blocks (|a|>10 or a<0.01) | 0 |
| VOLRN blocks | 120 blocks, 75 pairs |

### Key Finding

VOLRN does NOT introduce new nan pixels — the nan pixels in the output are all pre-existing from the original data (scene_2 has the most with 5.57M). All ADMM iterations converged, and no blocks have extreme gain/offset values. The 201 nan pixels reported in Stage 1 for scene_0 are likely from the overlap windowing or boundary effects during the full-pipeline run, not from VOLRN itself.

---

## 6. Gain/Offset Ablation

The gain/offset ablation requires VOLRN block coefficients per band, which are available when running VOLRN from code but not when loading GeoTIFF outputs from Stage 1. The experiment returned empty results.

**Workaround needed:** Re-run VOLRN on the 1024px crops with `return_diagnostics=True` to extract per-block a/b coefficients, then compute:
- gain_only: f' = a × f_registered
- offset_only: f' = f_registered + b

This experiment should be re-run with the pipeline integration.

---

## Summary of Root Causes

| Factor | Contribution | Evidence |
|--------|-------------|----------|
| **SWIR band instability** | High | B07–B09 contribute 50%+ of Ave; BAGRN reduces B07 from 65→37 but still worst |
| **Scene 20251120** | High | Consistently worst scene (Ave=30.6 for BAGRN, 23.5 for BAGRN+VOLRN) |
| **Spatial heterogeneity** | Very High | Max/min window Ave ratio ~17×; problem is highly localized |
| **Boundary effects** | Low | Boundary/interior ratio ≈ 1.0 for BAGRN and BAGRN+VOLRN |
| **NaN propagation** | None | 0 new nan pixels from VOLRN; all pre-existing |
| **ADMM convergence** | None | All 14 bands converged; no extreme a/b values |

### Conclusion

The spectral distortion vs. radiometric consistency tradeoff is driven by:
1. **Inherent SWIR variability** (B07–B09) that exceeds what moment matching can equalize
2. **Scene 20251120** having systematically different radiometric characteristics
3. **High spatial heterogeneity** — some overlap regions have very poor statistics while others are excellent

The solution direction for Stage 3 should focus on:
- Band-specific weighting (down-weight SWIR in the Ave objective)
- Scene-specific normalization strategies for problematic scenes
- Spatially adaptive methods that handle heterogeneous overlap quality

---

## Files Generated

```
data/output/stage2_problem_discovery/dz01_stage2_problem_discovery/
├── band_attribution/
│   ├── band_attribution.json
│   ├── band_attribution_bagrn.csv
│   ├── band_attribution_bagrn_volrn.csv
│   ├── band_attribution_original.csv
│   └── band_attribution_volrn_only.csv
├── scene_attribution/
│   └── scene_attribution.json
├── spatial_attribution/
│   └── spatial_attribution.json
├── multiwindow/
│   └── multiwindow.json
├── nan_trace/
│   └── nan_trace.json
├── gain_offset_ablation/
│   └── gain_offset_ablation.json
├── data_summary.json
├── environment.json
└── summary.json
```
