# Source / formula crosswalk — Unified Benchmark V1

Audit date: 2026-10-13. This is a source-code/protocol audit, not a fresh method run.

| New V1 identifier | Existing source | Existing computation / support | Relationship and V1 rule |
|---|---|---|---|
| `MAMD_V1` | `src/multiscene_sift/radiometric_metrics.py::compute_pair_mamd`; `src/benchmark_v1_radiometric_r1/metrics.py::pair_metrics` | `abs(mean(x)-mean(y))` on caller-provided common-valid samples | Same per-pair quantity. V1 primary aggregation is an unweighted pair macro; Task10D historical aggregation is shared-pixel weighted. Never merge the aggregate values. |
| `MSDD_V1` | `compute_pair_msdd`; R1 `pair_metrics` | `abs(std(x, ddof=0)-std(y, ddof=0))` on the supplied support | Same per-pair population-standard-deviation difference; aggregation differs as above. |
| `RDD_W1_EXACT_V1` | `compute_pair_rdd`; R1 `pair_metrics` | Exact one-dimensional empirical Wasserstein-1 distance on all supplied values | Same per-pair W1; it is not the paper's histogram `CD`. |
| `MAE_V1`, `RMSE_V1`, `BIAS_V1`, `P95AE_V1` | R1 `pair_metrics`; Task15/16 metric/report paths | Pixelwise differences on a declared common support; R1 P95 is percentile of absolute error | New V1 definitions explicitly require the frozen *input* pair support and full output coverage for a primary score. Output nodata/zero values cannot redefine the support. |
| `ADM_STYLE_MACRO_V1`, `ADSD_STYLE_MACRO_V1` | Paper metric names and the per-pair MAMD/MSDD quantities | Arithmetic mean across eligible pairs of per-pair absolute mean/std differences | Clearly named descriptive “style” summaries only. Not asserted to reproduce every paper sampling/aggregation detail. |
| `SOBEL_DIRECTION_LOSS_V1` | R1 `structure_metrics` / Task10D `compute_cgl` are distinct historical metrics | V1 uses the specified Sobel `/8`, input-only P50 edge threshold and `1-cos` direction loss | Neither historical gradient NCC nor Task10D circular orientation loss is silently relabeled as paper `GL`. |
| `GRADIENT_NCC_V1` (optional, not in V1 main protocol) | R1 `structure_metrics` | Pearson correlation of corresponding Sobel gradient-magnitude vectors after 3x3 support erosion | Retained only as separately named context; not interchangeable with orientation loss or paper `GL`. |
| Paper `CD`, `GL`, `RDOA`, `Ave` | Lin et al. 2024, Section 3 | Publisher abstract confirms the two-stage BAGRN global + VOLRN local method; accessible publisher page did not expose enough of Section 3's rendered formula details during this audit | Exact histogram bin functional and precise `GL` operator remain `FORMULA_UNVERIFIED`. No W1-as-CD, NCC-as-GL, or reconstructed paper composite is reported. |

## Fixture crosswalk result

The new package has a controlled-array test comparing new MAMD/MSDD/W1 with R1 `pair_metrics`; this checks formula-level per-pair agreement only. Historical weighted Task10D aggregation remains separate from V1 pair-macro aggregation. No historical Task15/Task16 numerical result was recomputed or imported as a V1 result.

## Paper record

Lin et al., *Joint block adjustment and variational optimization for global and local radiometric normalization toward multiple remote sensing image mosaicking*, ISPRS Journal of Photogrammetry and Remote Sensing 218 (2024), 187–203, DOI [10.1016/j.isprsjprs.2024.08.016](https://www.sciencedirect.com/science/article/pii/S0924271624003277). The publisher page is the primary source. Exact Section 3 formulas must be checked against an accessible authoritative full-text/rendering before any claim of exact ADM/ADSD/CD/GL/RDOA/Ave reproduction.
