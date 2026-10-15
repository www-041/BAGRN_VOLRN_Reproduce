# Global consistency metric unit audit

## Data flow

The modern matcher benchmark calls `global_consistency_diagnostics()` from
`src/multiscene_sift/modern_matcher_benchmark.py`. The diagnostic converts
pair-common-grid coordinates through the pair transform into CRS world
coordinates, then computes `dx_world` and `dy_world`. For the DZ01V B14
inputs, those world residuals are in metres because the raster transforms are
UTM transforms with 14 metre pixels.

The diagnostic divides those world residuals by `pixel_size_x` and
`pixel_size_y` to produce pixel residuals. The direct benchmark call currently
passes `pixel_size=1.0`, so the world residual is effectively left unchanged
while being labelled `*_px`. For example, the EfficientLoFTR non-tree RMSE
`755.4498` is 755.4498 metres under the current call, not 755.4498 pixels.

## Resolution source

`src/multiscene_sift/dataset.py` reads the B14 raster transform with rasterio
and records the resolution as `(abs(transform.a), abs(transform.e))`. The
current DZ01V manifest records B14 resolution `(14.0, 14.0)` for all five
scenes. The benchmark must propagate this metadata, or an explicit equivalent,
to the global consistency diagnostic.

## Correction point

The correction belongs at the benchmark-to-diagnostic call boundary in
`src/multiscene_sift/modern_matcher_benchmark.py`. The diagnostic should retain
separate world-metre and pixel-valued outputs; the benchmark must never use a
`pixel_size=1.0` placeholder for metre-valued raster data.

This audit intentionally does not modify matcher behavior, RANSAC, spanning
tree construction, global adjustment, radiometric normalization, or mosaic
generation.
