# B9 Global Connection input contract audit

## Existing consumer contract

The existing global registration/adjustment code requires, per accepted edge:

- `inlier_ref_xy`: finite `(N, 2)` points in `pair_common_grid` pixel coordinates;
- `inlier_tgt_xy`: finite `(N, 2)` points in the same pair-common-grid frame and aligned row-for-row with `inlier_ref_xy`;
- `pair_pixel_matrix`: a finite `3x3` affine matrix mapping `tgt_xy` to `ref_xy`;
- `pair_common_transform`: the pair-common-grid affine transform mapping pixel coordinates to CRS world coordinates;
- explicit pixel size/CRS metadata for world and pixel metrics.

The shared geometry contract is:

```text
ref_xy ~= pair_pixel_matrix(tgt_xy)
pair_common_grid pixels --pair_common_transform--> CRS world metres
```

The current global evaluator converts both point arrays through the same
`pair_common_transform`, then applies the existing global scene transforms.
Reversed edge orientation is handled by the existing global registration code
by inverting the world affine; no new direction convention is introduced here.

## Current 1024 historical outputs

Audited directories:

```text
data/output/b9_five_scene_validation/matcher_runs_1024/sift/
data/output/b9_five_scene_validation/matcher_runs_1024/loftr/
data/output/b9_five_scene_validation/matcher_runs_1024/efficient_loftr/
data/output/b9_five_scene_validation/matcher_runs_1024/lightglue_disk/
```

Each contains only the pairwise summary, accepted graph, runtime, and run
configuration files. The pairwise summary contains scalar metrics and the
affine matrix, but no point arrays or pair-common transforms. No geometry
subdirectory, NPZ bundle, or point-level CSV was present.

Therefore the existing summary-only runs are explicitly classified as:

```text
POINT_DATA_UNRECOVERABLE_FROM_SUMMARY
GLOBAL_INPUT_INCOMPLETE
```

No point coordinates will be reconstructed from RMSE/P95, and no old 1600
point data will be substituted for the 1024 protocol.

## Serialization requirement for the next task

The new Global-ready rerun must persist the in-memory `PairwiseRegistration`
fields immediately after `register_pair()` returns and before pairwise summary
serialization. Accepted and rejected successful pairs retain real point
bundles; failed pairs retain status in the index only. The bundle must carry
the frozen B9/1024 provenance and exact coordinate/direction metadata.
