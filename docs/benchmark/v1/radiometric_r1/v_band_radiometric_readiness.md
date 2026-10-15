# V-band radiometric readiness (metadata-only)

This report reads the existing V-band CSV and frozen B9 scene identity manifest only. It does not hash full rasters, read image pixels, apply MTL coefficients, resample, or modify any source file.

- Scenes: 13; metadata rows: 208; expected bands per scene: 16; complete scenes: 13.
- Missing bands: `none`.
- Center wavelength present: 208 / 208 records.
- Data unit labels in inventory: `{'UNKNOWN': 208}`; physical unit remains UNKNOWN unless the source metadata explicitly establishes it.
- QUALITY_STATUS=RISK scenes: 3 (20251214025253, 20251215023725, 20251215023737).
- Within-scene grid metadata discrepancies: none.
- B9 grid identical across scenes: False (distinct metadata signatures: 13).
- Existing SHA256 available: 13 records; missing: 195 records. No new whole-image hashing was performed; only hashes already present in the identity manifest are copied.

## Interpretation limits

The inventory contains wavelength center/bandwidth values from metadata, but this does not prove that MTL calibration coefficients were applied to stored pixels. Existing raster scale/offset and pixel type are retained as reported; no calibration was performed. Similar per-scene Affine/dimensions can support map-grid planning but do not prove detector-level or subpixel inter-band co-registration. A calibration/orthorectification provenance record and spatially independent inter-band check are still needed before physical reflectance metrics, SAM, or SAC.

The three flagged RISK scenes remain in the candidate population and must be reported as a quality stratum, not removed after seeing method results.
