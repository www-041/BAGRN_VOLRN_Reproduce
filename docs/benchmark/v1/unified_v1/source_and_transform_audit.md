# Source and transform audit — Unified Benchmark V1

Audit date: 2026-10-13. Read-only inspection; no legacy method or result files were modified.

## Source identity

- Scope: 13 processed DZ01V V-sensor B9 GeoTIFFs, source root `D:/科研/地质一号/文献/BAGRN_VOLRN_Reproduce/data/input/B9_reference_13scenes_full`.
- Identity manifest: `docs/benchmark/v1/phase2/input_identity_manifest.csv`; recorded SHA field is `recorded_source_sha256_task15`.
- Current source-file hashes: **13/13 match** the recorded Task15 V2 hashes. The audit computes SHA256 from the actual source files. Source pixels are not edited.
- Pixel data unit: `UNKNOWN` for this benchmark; do not call the values surface reflectance or report physical reflectance error.

## Geometry and support

- Task15 canonical grid: EPSG:32650; affine `(14, 0, 633346, 0, -14, 3359874)`; 7992 columns x 12404 rows; 14 m pixel spacing.
- The 13 Task15 stage-06 `input_valid_masks/scene_000.tif` … `scene_012.tif` are UInt8, 0 nodata, on this exact grid. Pairwise intersections were counted in bounded blocks from these masks: 78 unordered pairs, **48 positive-support / 30 zero-support**.
- Two support definitions are explicitly retained. Native B9 `rasterio mask != 0` inventory says `00_01=1,291,672` and `00_10=1,374,076`. On the Task15 aligned canonical grid the corresponding intersections are `00_01=1,295,654` and `00_10=1,374,076`. The aligned count is the V1 scoring support; the native count remains historical inventory. The 3,982-pixel `00_01` delta is not silently “corrected”; it arises between native-grid and transformed canonical-grid support definitions, and this audit does not attribute every changed pixel to an individual mechanism.
- Pair/temporal reference: `docs/benchmark/v1/temporal_pair_inventory.csv`; positive pair temporal strata remain the frozen 7 / 13 / 6 / 22 counts (same-date / <7d / 7–<30d / >=30d). Zero-support pairs remain `NO_VALID_OVERLAP`, never matcher failures.

## Transform and resampling provenance

- Canonical grid record: `data/output/final_pipeline/b9_13scene_formal_v2/stages/04_canonical_warp/canonical_output_grid.json`.
- Scene world transforms: `data/output/final_pipeline/b9_13scene_formal_v2/stages/03_global_adjustment/global/translation_l2/global_transforms.json` (13 matrices).
- Task15 source configuration: `.../stages/04_canonical_warp/task14_source_config.json`.
- Code path `scripts/run_task14_13scene_scale.py::_task14_local_registered_inputs` applies `T' = G_i T_i`, reprojects intensity to the 14 m canonical grid with rasterio bilinear resampling, uses declared source nodata and NaN destination nodata, and derives validity from finite projected values. It casts the resulting projected scene values to float32 before downstream use. Mask reprojection helpers declare nearest-neighbor mask handling. V1 records both kernels and derives input support independently of method output.
- Stage 04 preserves grid and source config but does **not** preserve unnormalized aligned scene rasters. V1 therefore creates fresh aligned, unnormalized inputs only under its isolated output directory; it does not reuse normalized Stage06 scenes as `Input`.

## Historical Task15/Task16 artifact eligibility

- Existing BAGRN, Ours, and VOLRN outputs are historical, read-only products with different method stages and provenance. They are not fresh submissions to this protocol and are not compared/ranked here.
- Stage06 persists BAGRN-normalized scenes and input masks; Stage09 persists Ours outputs; Task16 persists VOLRN outputs and strict-ablation/E2E reports. Task16 also records `COMPLETED_FINITE_NONCONVERGED`; that status must remain visible for any later reuse.
- Historical runtime cannot be reconstructed as a common measured wrapper interval from all existing logs; missing values remain `NOT_MEASURED`, not zero.
- No Spectralmatch, Harmoni, RS-NormGAN, or other external radiometric algorithm was run in Phase 1.

## Freeze gate

The read-only identity/grid/transform audit passes. The final isolated aligned inputs passed exact mask comparison for all 13 scenes; the 78-pair inventory, 1,533 deterministic B2 windows, metric specification, synthetic generator/fixture provenance, and 38-entry code/test hash manifest are written and lock-verified. See `data/output/benchmark_v1_unified/freeze_20261013_v3/freeze_readiness.json` (54 locked files).
