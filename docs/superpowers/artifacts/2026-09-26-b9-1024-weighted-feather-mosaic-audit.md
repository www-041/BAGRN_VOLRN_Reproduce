# B9 1024 Weighted-Feather Mosaic Audit

Status: Task 9 mosaic execution and seam diagnostics. This report uses the frozen B9 five-scene source set and the eight persisted Global solutions; it does not rerun matcher, RANSAC, MST, or Translation-L2.

## Frozen protocol

- Branch at start: `exp/2026-09-24-b9-five-scene-validation`
- Start HEAD: `8b91359c63082d65e1a5b0b17747eef82e2d1522`
- Source scenes: B9 manifest indices `[2, 3, 5, 8, 10]`
- Band: B9; CRS: EPSG:32650; source/output resolution: 14 m
- Global inputs: SIFT, LoFTR, EfficientLoFTR, LightGlue+DISK × MST/Translation-L2
- Radiometric normalization: NONE
- Blend: existing `src.mosaic.create_mosaic(mode="weighted")` contract
- Protocol artifacts: `data/output/b9_five_scene_validation/mosaic_runs_1024/protocol/`

Task 0 froze eight transform hashes, five source-raster inventories, and a resumable eight-row status ledger before any mosaic run. Remaining sections are filled by Tasks 1–11.

## Task 1 — frozen existing mosaic contract

The official mosaic path is `src.mosaic.create_mosaic`, called with `mode="weighted"` and the canonical output transform/width/height. It uses Rasterio `reproject` with `Resampling.bilinear`, same CRS for source and destination, source nodata mapped to `NaN`, and `init_dest_nodata=True`. A projected pixel is valid only when all projected bands are finite.

For each projected valid mask, the existing weight is `distance_transform_edt(mask)` cast to float64, with `1e-6` added on valid pixels. The weighted result is `sum(value * weight) / sum(weight)`, valid where `sum(weight) > 1e-12`. Outside the union and invalid results use the first declared nodata value; with these B9 uint16 rasters that value is `0.0` and the output dtype remains uint16. The output GeoTIFF uses LZW compression and preserves the supplied canonical transform, CRS, width, and height. No rounding or clipping beyond the final cast is introduced.

The runner's contributor/weight diagnostics reproduce only this projection and weight calculation for audit fields; the official `mosaic.tif` is still written by the existing weighted-feather core. No seamline, radiometric, graph-cut, multiband, or Poisson blending is introduced.

## Task 2 — canonical output grid

All eight runs use the immutable grid in `mosaic_runs_1024/protocol/canonical_output_grid.json`:

- CRS: EPSG:32650
- pixel size: 14 m
- width × height: 5116 × 6134
- bounds: `[641676.0, 3256834.0, 713300.0, 3342710.0]`
- transform: `[14.0, 0.0, 641676.0, 0.0, -14.0, 3342710.0]`

The transformed corner union is `[641687.7387459544, 3256841.409535937, 713300.0, 3342701.619884489]`; relative to the raw source union, the largest transform-induced edge movement is about 6.6 m and the deterministic 14 m snapping margin stays within the expected one-pixel safety margin. No raster resampling was used to construct this grid, and all eight transform sets resolve to the same grid identity.

## Task 3 — deterministic single-run runner

`scripts/run_b9_weighted_mosaic.py` accepts one frozen source config, one persisted Global-run directory, the canonical grid, and an output directory. It loads only source rasters and persisted transforms; it performs no matching, RANSAC, Global optimization, radiometric normalization, or transform inference. The official `mosaic.tif` is produced by the existing `src.mosaic.create_mosaic(mode="weighted")` implementation. The companion `valid_mask.tif`, `contributor_count.tif`, and `weight_sum.tif` use the same bilinear projection and distance-transform weight definition for auditability. `preview.png` is display-only percentile stretch, with its limits recorded in `run_summary.json`. A non-empty output directory is rejected to prevent accidental overwrite.

## Task 4 — eight-run execution

The sequential batch completed all eight frozen combinations with status `PASS`. Each row has its own log under `mosaic_audit_1024/logs/`, and every output was validated against the canonical transform, dimensions, CRS, 14 m resolution, finite valid pixels, and source dtype policy.

| matcher | Global method | valid pixels | mosaic dtype |
|---|---|---:|---|
| SIFT | MST | 22,166,642 | uint16 |
| SIFT | Translation-L2 | 22,166,580 | uint16 |
| LoFTR | MST | 22,167,910 | uint16 |
| LoFTR | Translation-L2 | 22,167,910 | uint16 |
| EfficientLoFTR | MST | 22,167,910 | uint16 |
| EfficientLoFTR | Translation-L2 | 22,167,910 | uint16 |
| LightGlue+DISK | MST | 22,167,910 | uint16 |
| LightGlue+DISK | Translation-L2 | 22,167,910 | uint16 |

## Task 5 — contributor and coverage audit

`mosaic_audit_1024/01_run_inventory.csv` contains all eight rows. Every row has the same denominator, `31,381,544` canonical-grid pixels, and the same maximum contributor count of 5. Invalid weight-sum pixels are zero for all runs. The SIFT runs have 22,166,642 and 22,166,580 valid pixels; the other six runs have 22,167,910 valid pixels. Contributor-count maps are in `mosaic_audit_1024/contributor_maps/` and use one shared display maximum (`5`) and the same raster extent/shape.

## Task 6 — overlap structural metrics

`mosaic_audit_1024/02_overlap_structural_metrics.csv` contains the expected 80 rows (4 matchers × 2 Global methods × 10 accepted pairs). All rows are `PASS` with finite primary structural metrics and shared-valid overlap masks. The primary fields are intensity ZNCC, gradient-magnitude NCC, and gradient-orientation cosine. The auxiliary intensity MAE/RMSE/mean-bias fields are explicitly labeled `RADIOMETRY_SENSITIVE`; they are descriptive and are not used as geometry scores. The synthetic tests confirm that a 1–2 pixel displacement lowers structural scores, while a constant brightness offset preserves ZNCC but changes MAE. No insufficient-overlap row occurred under the fixed minimum sample rule for this dataset.

## Task 7 — fixed feather seam-zone diagnostics

`mosaic_audit_1024/03_seam_zone_metrics.csv` contains 80 `PASS` rows. The seam diagnostic is not an optimized seamline: for each accepted pair it is the shared-valid region where pairwise normalized fixed-feather weights satisfy `min(w_i_norm, w_j_norm) >= 0.25`. The threshold is identical for all eight runs and is recorded in every row. Seam-zone pixel counts range from 1,675,636 to 8,732,548. Seam gradient metrics are structural; auxiliary seam intensity MAE/RMSE/bias fields remain labeled `RADIOMETRY_SENSITIVE`, so no geometry-only seam claim is made.

## Task 8 — standardized visual audit

The figure set is in `mosaic_audit_1024/figures/`, with selection and crop metadata in `07_figure_selection.json`:

- 8 official mosaic previews and 8 contributor-count maps, using the fixed runner preview policy and Task 5 shared contributor scale;
- 16 separately warped-source checkerboards and 16 gradient overlays for each matcher’s selected worst/stable edges, with LightGlue+DISK pair `1-3` included explicitly;
- 16 seam-zone zooms (highest mismatch and median region for both Global methods), reusing identical crop bounds between MST and Translation-L2 within each matcher;
- 4 MST-vs-Translation absolute mosaic-difference maps.

The difference maps are explicitly diagnostic only: they remain sensitive to radiometry and warp interpolation and are not geometric ground-truth error maps.
