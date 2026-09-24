# B9 loader audit

## Existing flat-scene loader

The existing `src.multiscene_sift.dataset.discover_five_scenes()` expects
scene directories directly under the input root:

```text
input_root/<scene_name>/*_B14.TIF
```

The modern matcher benchmark supplies a fixed five-scene name list and passes
`bands=(registration_band,)`; the established defaults are B14/B8/B5. This
loader does not search a `root/<time_dir>/<scene_dir>/` hierarchy and does not
parse MTL files.

## Reusable metadata interfaces

`discover_five_scenes()` already provides the reusable GeoTIFF metadata path:

- `rasterio.open()` reads CRS, transform, shape, nodata, and bounds;
- `Scene` stores band paths, transforms, shapes, CRS, and bounds;
- `build_geographic_overlap_graph()` reuses `Scene.bounds` for rectangle
  intersection and overlap ratios;
- `diagnostics.draw_overlap_graph()` can render an overlap graph.

The B9 adapter therefore only needs to discover the two-level directory,
validate unique B9/MTL files, parse the small set of MTL audit fields, and
return JSON-serialisable records. It does not replace the existing Scene or
matcher pipeline.

## B9 adapter boundary

The minimal supported input is:

```text
B9_reference_13scenes_full/<time_dir>/<scene_dir>/*_B9.TIF
B9_reference_13scenes_full/<time_dir>/<scene_dir>/*_MTL.txt
```

The B9 adapter will use the GeoTIFF as the authoritative source for CRS,
transform, bounds, resolution, width, and height. It will use MTL for
`PRODUCT_ID`, acquisition date/time, and cloud cover when present. A
directory with neither B9 nor MTL is treated as a non-scene directory and is
ignored; a candidate containing only one required artifact fails explicitly.

No production matcher, RANSAC, global connection, BAGRN, VOLRN, or mosaic code
is changed by the audit.
