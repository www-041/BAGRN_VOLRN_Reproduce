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
