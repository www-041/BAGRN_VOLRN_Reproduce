# B9 1024 Global Geometry Audit

Status: Task 8C analysis-only audit. Existing matcher, RANSAC, Global, radiometric, and mosaic code is not rerun or changed by this report.

## Frozen experiment

- Repository branch: `exp/2026-09-24-b9-five-scene-validation`
- Starting HEAD: `2b68c043ab2907c7bdab51246f714a91f34ee5a8`
- Dataset: frozen B9 five-scene manifest indices `[2, 3, 5, 8, 10]`
- Protocol: `match_max_side=1024`, `pixel_size_m=14.0`, coordinate frame `pair_common_grid`, transform direction `target_to_reference`
- Methods: `sift`, `loftr`, `efficient_loftr`, `lightglue_disk`; Global methods `MST`, `Translation-L2`
- Inputs are the existing `matcher_runs_1024_globalready/*` and `global_runs_1024/*` artifacts.

## Task 0 provenance

`08_global_method_comparison.json` and `.csv` contain exactly eight rows (four matchers × two Global methods); all eight have `status=PASS`, `accepted_edges=10`, and `connected=true`. Each run has `global_transforms.json`, `global_edge_consistency.{json,csv}`, and `global_registration.json`; MST runs additionally have `spanning_tree.json` and `01_mst_point_residuals.csv`, while Translation-L2 runs have `05_translation_point_residuals.csv` and `06_translation_solution.json`. The ten pair bundles are stored once per matcher under `matcher_runs_1024_globalready/<matcher>/geometry/pair_*.npz` with matching sidecars.

The downstream sections are filled by Tasks 1–10 from these frozen artifacts only.

## Edge-level geometry audit

The canonical table `geometry_audit_1024/01_edge_metrics.{csv,json}` contains 80 rows: 10 accepted edges for each of 4 matchers × 2 Global methods. Residuals use the existing evaluator definition `||G_i p_i - G_j p_j|| / 14m`; the point coordinates are transformed through each persisted `pair_common_transform` before applying the persisted Global transform.

| matcher | method | tree edges | non-tree edges | max edge P95 (px) | edge |
|---|---|---:|---:|---:|---|
| sift | MST | 4 | 6 | 1.6962 | 0-2 |
| sift | Translation-L2 | 4 | 6 | 1.6975 | 0-2 |
| loftr | MST | 4 | 6 | 1.3120 | 3-4 |
| loftr | Translation-L2 | 4 | 6 | 1.3124 | 3-4 |
| efficient_loftr | MST | 4 | 6 | 0.7211 | 0-4 |
| efficient_loftr | Translation-L2 | 4 | 6 | 0.7222 | 0-4 |
| lightglue_disk | MST | 4 | 6 | 2.2422 | 1-3 |
| lightglue_disk | Translation-L2 | 4 | 6 | 2.1622 | 1-3 |

Tree/non-tree labels come from each matcher's persisted `spanning_tree.json`. Translation-L2 reuses its matcher-specific MST tree only as the documented initialization/tree-label reference; it is not reclassified from pair ordering. The detailed grouping is in `02_tree_non_tree_summary.csv`.

## Point-weighted versus edge-balanced statistics

`05_residual_distribution_summary.csv` reports both pooled point statistics and equal-edge summaries. This matters because edge inlier counts differ substantially (for example, EfficientLoFTR has more than 100k accepted inliers while LightGlue has about 11k). No existing Global metric was replaced.

## MST to Translation-L2 redistribution

The canonical unordered-pair join and all per-edge deltas are in `03_translation_edge_delta.csv`; negative delta means the Translation-L2 residual decreased for that edge.

| matcher | improved | worsened | unchanged | pooled RMSE delta (px) | edge-balanced RMSE delta (px) |
|---|---:|---:|---:|---:|---:|
| sift | 4 | 6 | 0 | -0.001033 | -0.000966 |
| loftr | 5 | 5 | 0 | -0.003164 | -0.002725 |
| efficient_loftr | 5 | 5 | 0 | -0.000222 | -0.000249 |
| lightglue_disk | 1 | 6 | 3 | -0.003129 | +0.022236 |

These are constraint redistributions, not a global winner/ranking. LightGlue's pooled RMSE decrease and edge-balanced RMSE increase are an example of why both views are required.

## Cycle consistency

The existing `cycle_metric` is not computed: `src/multiscene_sift/global_comparison.py` explicitly writes `"cycle_metric": None` for every comparison row. This means unavailable/not implemented, not a numerical failure. The new `04_cycle_consistency.csv` therefore uses a separate name and reports raw pair-transform cycle closure only; it is not the post-Global point residual.

The accepted 5-node/10-edge graph has cycle rank `E - V + C = 6`; a deterministic fundamental cycle basis is used for each matcher. Worst/median translation closure in pixel coordinates are:

| matcher | cycles | worst closure (px) | median closure (px) |
|---|---:|---:|---:|
| sift | 6 | 0.6049 | 0.4449 |
| loftr | 6 | 0.4266 | 0.1533 |
| efficient_loftr | 6 | 0.1829 | 0.0790 |
| lightglue_disk | 6 | 1.2344 | ~0 |

## LightGlue+DISK anomaly audit

`06_lightglue_anomaly_audit.json` recomputes pooled percentiles from raw residual arrays. For MST, 11,154/11,263 points are at or below `1e-9` pixel, while edge `1-3` contributes the non-zero tail (`P95=2.2422 px`, 10.69% above 1 px). This explains how pooled P95 can be approximately zero while the maximum remains 2.2422 px. The duplicate diagnostic found zero duplicate ref points, target points, or correspondence pairs. Persisted world-unit P95 divided by 14 agrees with recomputed pixel P95 within `1.79e-6` pixel.

The near-zero MST tree residuals are structurally expected: MST transforms are propagated along their defining pair constraints, so tree edges are not independent validation accuracy. Translation-L2 redistributes small residuals over tree edges and reduces some non-tree tails; the per-edge evidence is retained rather than collapsed into a winner.

## Spatial residual and tie-point coverage audit

The `figures/` directory contains one residual CDF, per-edge P95, tree/non-tree, Translation delta, raw-cycle, residual spatial map, and coverage map family for each matcher/method, plus all-8 CDF and edge-balanced comparison figures. Spatial plots use the midpoint of the two transformed points in the persisted Global world frame; optional `>0.5 pixel` and `>1.0 pixel` tail maps are produced only when non-empty. Coverage maps use the same 20×20 occupancy grid over the common geographic frame and transform pair-common-grid points through each saved pair transform before aggregation.

Coverage totals/occupied cells are:

| matcher | accepted inliers | occupied / 400 cells |
|---|---:|---:|
| sift | 37,822 | 326 |
| loftr | 48,065 | 311 |
| efficient_loftr | 105,834 | 338 |
| lightglue_disk | 11,263 | 286 |

## Scope and stop point

This stage can identify measured residual behavior under the fixed B9/1024 protocol, spatial tail locations, MST-to-Translation constraint redistribution, raw pair-cycle closure, and tie-point occupancy. It cannot establish absolute geolocation accuracy, all-region/sensor generalization, or final algorithm superiority. No matcher, RANSAC, acceptance rule, MST, Translation-L2, Global transform, BAGRN, VOLRN, or mosaic was rerun or modified. The next stage is deliberately stopped before weighted feather mosaic.
