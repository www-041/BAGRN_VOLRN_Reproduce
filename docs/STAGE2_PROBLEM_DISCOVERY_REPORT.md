# Stage 2: Problem Discovery Report (Corrected)

**Date:** 2026-08-01
**Last corrected:** 2026-08-01 (sensor/band metadata correction)
**Dataset:** DZ01 VNIR four-scene subset (scene_20251114, scene_20251120, scene_20251208, scene_20251215)
**Sensor:** DZ01 VNIR (B01–B14, approximately 410–860 nm)
**Crop size:** 1024 × 1024 px (centered on overlap edge centers)
**Baseline Ave:** original=145.53, bagrn=16.17 (−88.9%), volrn_only=55.89 (−61.6%), bagrn_volrn=12.74 (−91.2%)

> **Note on sensor bands:** DZ01 VNIR actually contains 16 bands (B01–B16, 410–1019 nm).
> The current experiment used B01–B14 only. B15 (960.5 nm) and B16 (1000.5 nm) were not
> included in Stage 1 or Stage 2. SWIR is a separate sensor (DZ01S, B01–B10, 1178–2468 nm,
> 30 m resolution) and is not included in the current experiments.

---

## 1. Band Attribution

Per-band Ave decomposition for each method. Bands are labeled by VNIR sensor ID and center wavelength.

### BAGRN (Global Normalization Only)

| Band | Center (nm) | Region | ADM | ADSD | Ave |
|------|-------------|--------|------|------|------|
| B01 | 421.0 | violet | 13.28 | 7.95 | 10.61 |
| B02 | 452.5 | blue | 11.49 | 5.35 | 8.42 |
| B03 | 490.0 | blue | 8.20 | 4.29 | 6.24 |
| B04 | 530.5 | green | 21.80 | 13.76 | 17.78 |
| B05 | 550.0 | green | 33.58 | 20.01 | 26.79 |
| B06 | 575.5 | yellow-green | 17.46 | 12.46 | 14.96 |
| **B07** | **633.5** | **red** | **98.85** | **32.28** | **65.56** |
| **B08** | **678.5** | **red** | **60.94** | **25.55** | **43.24** |
| **B09** | **701.5** | **red_to_nir_transition** | **47.62** | **15.77** | **31.69** |
| B10 | 722.5 | red_to_nir_transition | 1.80 | 9.37 | 5.58 |
| B11 | 743.5 | red_to_nir_transition | 12.90 | 4.00 | 8.45 |
| B12 | 769.0 | nir | 17.75 | 13.70 | 15.72 |
| **B13** | **814.0** | **nir** | **38.62** | **25.25** | **31.94** |
| B14 | 850.5 | nir | 14.37 | 14.52 | 14.44 |

**Top 3 worst bands:** B07 (633.5 nm, 65.6), B08 (678.5 nm, 43.2), B13 (814.0 nm, 31.9)
**Top 3 best bands:** B10 (722.5 nm, 5.6), B03 (490.0 nm, 6.2), B02 (452.5 nm, 8.4)

### BAGRN+VOLRN (Full Pipeline)

| Band | Center (nm) | Region | ADM | ADSD | Ave |
|------|-------------|--------|------|------|------|
| B01 | 421.0 | violet | 8.92 | 10.93 | 9.93 |
| B02 | 452.5 | blue | 10.93 | 2.62 | 6.78 |
| B03 | 490.0 | blue | 7.09 | 2.94 | 5.01 |
| B04 | 530.5 | green | 19.72 | 11.96 | 15.84 |
| B05 | 550.0 | green | 27.99 | 18.32 | 23.15 |
| B06 | 575.5 | yellow-green | 15.20 | 11.56 | 13.38 |
| **B07** | **633.5** | **red** | **43.75** | **30.26** | **37.01** |
| **B08** | **678.5** | **red** | **51.30** | **25.13** | **38.22** |
| **B09** | **701.5** | **red_to_nir_transition** | **38.65** | **14.94** | **26.80** |
| B10 | 722.5 | red_to_nir_transition | 2.99 | 8.19 | 5.59 |
| B11 | 743.5 | red_to_nir_transition | 11.97 | 3.47 | 7.72 |
| B12 | 769.0 | nir | 12.99 | 9.58 | 11.29 |
| **B13** | **814.0** | **nir** | **33.02** | **18.55** | **25.78** |
| B14 | 850.5 | nir | 11.52 | 10.14 | 10.83 |

**Top 3 worst bands:** B08 (678.5 nm, 38.2), B07 (633.5 nm, 37.0), B09 (701.5 nm, 26.8)
**Top 3 best bands:** B03 (490.0 nm, 5.0), B10 (722.5 nm, 5.6), B02 (452.5 nm, 6.8)

### Key Finding

VNIR B07–B09 and B13 show the largest residual Ave values in the current four-scene experiment.
B07 (633.5 nm, red) has the highest residual at 65.6 (BAGRN) and 37.0 (BAGRN+VOLRN).
VOLRN reduces B07 Ave from 65.6 → 37.0 (−44%) but B08 (678.5 nm) only from 43.2 → 38.2 (−12%).
The cause of the larger residuals in these bands has not yet been determined. Possible factors
include scene content, acquisition geometry, illumination differences, temporal surface changes,
overlap quality, and sensor response, but the current evidence cannot distinguish among them.

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

Scene 20251120 (November 20) is consistently the weakest link, contributing disproportionately
to overall Ave. This is consistent across BAGRN and BAGRN+VOLRN, suggesting the radiometric
differences for this scene are too large for the current normalization to fully correct. Without
the four scenes' individual VNIR metadata (sun angle, cloud cover, observation geometry), we
cannot determine whether this is caused by acquisition conditions, surface changes, or other factors.

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

BAGRN and BAGRN+VOLRN show boundary Ave ≤ interior Ave (ratio < 1), meaning boundary regions
are slightly better or equal — no boundary degradation. VOLRN only shows ratio > 1 (1.09),
boundary regions are 9% worse — VOLRN without BAGRN pre-processing shows mild boundary
degradation. The degradation is modest (9%), not the dominant cause of Ave.

> Status: partially_completed — only boundary/interior attribution is currently available.

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

BAGRN and BAGRN+VOLRN show extreme spatial variance (max/min ratio ~17–18×). The min Ave (~11)
is excellent, but the max Ave (~198) is nearly as bad as the original. This confirms the problem
is **highly localized**: certain window positions have poor overlap statistics while others are
excellent. The Ave degradation is not uniform but concentrated in specific spatial regions.

---

## 5. NaN Trace

| Metric | Value |
|--------|-------|
| Input nan (total across bands) | scene_0: 2,492,154, scene_1: 2,661,862, scene_2: 5,570,124, scene_3: 42,980 |
| Output nan (total) | Same as input |
| **New nan pixels** | **0** |
| Block convergence | All 14 bands converged |
| ADMM iterations | 18–24 per band |
| Extreme a blocks (\|a\|>10 or a<0.01) | 0 |
| VOLRN blocks | 120 blocks, 75 pairs |

### Key Finding

VOLRN does NOT introduce new nan pixels — the nan pixels in the output are all pre-existing
from the original data (scene_2 has the most with 5.57M). All ADMM iterations converged, and
no blocks have extreme gain/offset values.

> Status: completed_for_primary_question — no new NaN introduced by VOLRN.

---

## 6. Gain/Offset Ablation

| Status | incomplete |
|--------|-----------|
| Reason | BAGRN global gain/offset coefficients were not available in persisted Stage 1 outputs |

The gain/offset ablation requires per-block VOLRN coefficients (a, b) per band, which are
available when running VOLRN from code but not when loading GeoTIFF outputs from Stage 1.
The experiment returned empty results.

---

## Experiment Completion Status

| Experiment | Status | Note |
|------------|--------|------|
| band_attribution | completed | |
| scene_attribution | completed | |
| spatial_attribution | partially_completed | only boundary/interior attribution available |
| multiwindow | completed | |
| nan_trace | completed_for_primary_question | no new NaN introduced by VOLRN |
| gain_offset_ablation | incomplete | BAGRN coefficients not in persisted outputs |

---

## Summary of Root Causes

| Factor | Contribution | Evidence |
|--------|-------------|----------|
| **Red/transition/NIR residual differences** | High | B07 (633.5 nm), B08 (678.5 nm), B09 (701.5 nm), B13 (814.0 nm) have largest residuals |
| **Scene 20251120** | High | Consistently worst scene (Ave=30.6 for BAGRN, 23.5 for BAGRN+VOLRN) |
| **Spatial heterogeneity** | Very High | Max/min window Ave ratio ~17×; problem is highly localized |
| **Boundary effects** | Low | Boundary/interior ratio ≈ 1.0 for BAGRN and BAGRN+VOLRN |
| **NaN propagation** | None | 0 new nan pixels from VOLRN; all pre-existing |
| **ADMM convergence** | None | All 14 bands converged; no extreme a/b values |

### Conclusion

The residual radiometric differences in the current four-scene VNIR experiment are driven by:
1. **Larger residuals in red, red-transition, and near-infrared bands** (B07, B08, B09, B13)
2. **Scene 20251120** having systematically different radiometric characteristics
3. **High spatial heterogeneity** — some overlap regions have very poor statistics while others are excellent

The cause of the larger residuals in VNIR B07–B09 and B13 has not yet been determined.
Possible factors include scene content, acquisition geometry, illumination differences,
temporal surface changes, overlap quality, and sensor response, but the current evidence
cannot distinguish among them.

The solution direction for Stage 3 should focus on:
- Spatially quality-aware overlap weighting (down-weight regions with poor statistics)
- Scene-specific normalization strategies for problematic scenes
- VNIR red-to-near-infrared spectral preservation techniques

These are statistical observations from the current data, not sensor defects.

---

## Metadata Limitations

- Band center wavelengths are from the 2025-12-08 VNIR MTL file (reference only)
- The other three scenes' VNIR MTL files have not yet been obtained
- Scene-specific sun angle, cloud cover, observation geometry, and attitude parameters
  are **not available** for scenes 20251114, 20251120, and 20251215
- We cannot determine whether scene_20251120's poor performance is caused by
  acquisition conditions (sun angle, cloud cover, etc.) or surface properties
- The four scenes' individual VNIR MTL files are needed for further analysis

### Missing Metadata

| Scene | VNIR MTL Status |
|-------|----------------|
| scene_20251114 | missing |
| scene_20251120 | missing |
| scene_20251208 | available (reference only) |
| scene_20251215 | missing |

---

## Stage 3 Candidate Directions

### 1. Spatially quality-aware overlap normalization
**Evidence:** evidence supported
Different window Ave max/min ratio ~17–18×.

### 2. Scene- and acquisition-condition-aware normalization
**Evidence:** partially supported
scene_20251120 consistently worst. Limitation: missing four scenes' individual VNIR metadata.

### 3. VNIR red-to-near-infrared spectral preservation
**Evidence:** partially supported
B07 (633.5 nm), B08 (678.5 nm), B09 (701.5 nm), and B13 (814.0 nm) have larger residuals,
and BAGRN introduces larger overall SAM. Limitation: gain/offset ablation incomplete,
no ground truth land cover classification, incomplete scene metadata.

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

### Sensor-Qualified Band Names (for new CSV outputs)

| Band | Qualified Name | Center (nm) |
|------|---------------|-------------|
| B01 | VNIR_B01 | 421.0 |
| B02 | VNIR_B02 | 452.5 |
| B03 | VNIR_B03 | 490.0 |
| B04 | VNIR_B04 | 530.5 |
| B05 | VNIR_B05 | 550.0 |
| B06 | VNIR_B06 | 575.5 |
| B07 | VNIR_B07 | 633.5 |
| B08 | VNIR_B08 | 678.5 |
| B09 | VNIR_B09 | 701.5 |
| B10 | VNIR_B10 | 722.5 |
| B11 | VNIR_B11 | 743.5 |
| B12 | VNIR_B12 | 769.0 |
| B13 | VNIR_B13 | 814.0 |
| B14 | VNIR_B14 | 850.5 |
| B15 | (excluded) | 960.5 |
| B16 | (excluded) | 1000.5 |
