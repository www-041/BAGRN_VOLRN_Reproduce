# Two-Image Flat-Terrain Registration — Experiment 0 Results

> **Branch:** `exp/2026-09-17-two-image-flat-registration`
>
> **Date:** 2026-09-17
>
> **Output directory:** `data/output/two_image_flat_b14_v2`

## 1. Image Pair

| Property | Reference | Target |
|---|---|---|
| Path | `DZ01V_L2_E113.6_N36.3_20260616031133_01_T1_B14.TIF` | `DZ01V_L2_E113.4_N36.6_20260810030932_01_T1_B14.TIF` |
| Acquisition date | 2026-06-16 | 2026-08-10 |
| Band | B14 | B14 |
| CRS | EPSG:32649 | EPSG:32649 |
| Resolution | 14.0 m | 14.0 m |
| Dimensions | 4162×4149 | 4204×4076 |
| Common grid | 6458×5150 px | — |
| Overlap window | rows 2336–4150, cols 1454–3621 | — |

## 2. Benchmark Parameters

| Parameter | Value |
|---|---|
| `match_max_side` | 1600 px |
| `ransac_threshold` | 2.0 px |
| `device` | auto (CPU) |
| Matchers requested | phase, sift, loftr, lightglue |
| Mosaic mode | `source_selection` |

## 3. Environment

| Package | Version |
|---|---|
| torch | 2.14.0+cpu |
| kornia | 0.8.3 |
| CUDA | not available |
| lightglue | not installed (SSL certificate blocks git+https install) |
| Python | 3.14.0 |

## 4. Results Table

| Method | Status | Raw | Inliers | Inlier Ratio | Coverage | RMSE | P95 | NCC Before | NCC After | Verify Shift | Match Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase | TOO_FEW_INLIERS | 1 | 0 | 0.000 | 0.000 | — | — | 0.0453 | 0.0453 | — | 2.9 s |
| SIFT | OK | 176 | 70 | 0.398 | 0.328 | 1.269 | 1.855 | 0.0453 | 0.1394 | 1.41 px | 1.0 s |
| LoFTR | FAILED | — | — | — | — | — | — | — | — | — | — |
| LightGlue | not installed | — | — | — | — | — | — | — | — | — | — |

## 5. Phase Correlation Screening

| Metric | Count |
|---|---|
| Total blocks | 99 |
| low_valid | 0 |
| low_texture | 0 |
| **low_conf** | **98** |
| large_shift | 0 |
| accepted | 1 |

**Interpretation:** On this cross-temporal B14 pair (2-month gap), 98 of 99
phase-correlation blocks were rejected due to insufficient structural similarity
(low NCC confidence).  The gradient-based structural image used by phase
correlation is highly sensitive to radiometric differences between acquisition
dates.  B14 band reflectance changed significantly between June and August,
preventing reliable block matching at the default confidence threshold of 0.5.

## 6. Failure Reasons

- **Phase:** Dominant failure mode is `low_conf` (98/99 blocks).  Cross-temporal
  B14 radiometric differences exceed the sensitivity of gradient-based structural
  matching at the default parameters.  A sensitivity study (adjusting confidence
  threshold, block size, or using a different band) is deferred to a separate plan.
- **LoFTR:** SSL certificate verification failure prevents downloading pretrained
  weights from `cmp.felk.cvut.cz`.  This is a network/environment issue, not a
  code defect.
- **LightGlue:** Package cannot be installed via `git+https` due to SSL certificate
  issues on this machine.

## 7. Summary

On the current pair of flat-terrain cross-temporal B14 images, SIFT produced
70 reliable RANSAC inliers (inlier ratio 0.398, RMSE 1.27 px) under the fixed
benchmark settings.  The gradient NCC improved from 0.045 to 0.139 after
registration, and independent phase verification confirmed a residual shift of
only 1.41 px.  Phase correlation was not viable at the default parameters due
to inter-date radiometric differences in B14.