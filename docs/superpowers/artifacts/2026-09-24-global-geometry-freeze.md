# Five-Scene Global Geometry Freeze

Date: 2026-09-24

Source branch: `exp/2026-09-17-five-scene-sift-mosaic`

Source commit: `f43719428ae9451a2cbf98a5e4c93fc2fcc84eed`

## Frozen baseline

- SIFT MST `0-1` P95: approximately `53.65 px`.
- Equal-L2 Translation `0-1` P95: approximately `19.05 px`.
- Equal-L2 Translation maximum edge P95: approximately `19.05 px`.
- Equal-L2 Translation mean edge RMSE: approximately `10.8564 px`.
- Equal-L2 Translation tree-edge mean P95: approximately `10.0682 px`.

## Interpretation

Full Affine is not adopted as the current solution. Robust/Weighted Translation is
not adopted as the production or research baseline because the frozen five-scene
decision gate classified it as `ROBUST_WEIGHTED_UNSTABLE`.

The remaining limitation is mutually inconsistent pairwise geometry in the
five-scene tie-point network, especially the `0-1 / 0-4 / 1-4` triangle. Future
matcher experiments must therefore preserve the existing common-grid coordinate
contract, shared RANSAC/acceptance rules, and Equal-L2 Translation evaluation.

## Scope freeze

This note freezes the geometry-stage evidence only. It does not claim absolute
geolocation correctness or large-scene scalability. BAGRN, VOLRN, seamline,
mosaic, and production matcher integration are outside this freeze.
