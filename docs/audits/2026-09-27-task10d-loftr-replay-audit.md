# Task10D LoFTR Frozen-Protocol Replay Audit

Date: 2026-09-27

## Decision

`LOFTR_REPLAY_EQUIVALENT`

The existing replay artifact is geometry-equivalent to the original B9 1024
LoFTR run. No LoFTR global adjustment, mosaic, or other matcher was rerun.

## Frozen protocol checks

- Matcher: `loftr`; band: `B9`.
- Scene manifest indices: `[2, 3, 5, 8, 10]`.
- Common matching scale: `1024`.
- Coordinate frame: `pair_common_grid`.
- Transform direction: `target_to_reference`.
- Shared RANSAC: `AffineTransform`, residual threshold `2.0 px`, max trials
  `5000`, seed `0`, minimum inliers `20`, minimum ratio `0.3`.
- Replay validation reports `PASS`, `REPLAY_CONSISTENT`, ten processed pairs,
  ten accepted edges, and one connected component containing all five scenes.
- The replay configuration SHA-256 is
  `b9181afcc972b29716d309f7aa3e82d3fbdba57fd5d466e2c0aa04d25060c9ba`, which
  matches `04_frozen_five_scene_config_1024.json`.
- The replay runtime reports matcher device `auto` and connected status; no
  CPU fallback or CUDA failure is recorded.

## Pairwise replay comparison

The replay validator compares every pair's raw matches, inliers, residual RMSE,
residual P95, coverage, status, and accepted-edge set. All ten pairs are
identical to the original run. Across the network there are 50,000 raw matches,
48,065 inliers, and 1,935 removed matches after the validity/geometry filter.

Canonical SHA-256 hashes of each pair pixel matrix are equal between the old
`matcher_runs_1024/loftr` summary and the replay geometry sidecars:

| pair | transform SHA-256 |
|---|---|
| 00-01 | `0219f54dd2cfbeee624cc616838dfdeda49434a796a66da04390f641f012500e` |
| 00-02 | `52e3547651b9a4ec5e41d1239d0ba1f5c51d89926e7e8ff5f839d888c6e1642d` |
| 00-03 | `ecb6a36182e0f4df23fdcc8f3f335d674469c0204b98c0f749005a17d68c0e69` |
| 00-04 | `83255021912b859b1b9b5f1b7bb7b95c39795d4406ccc227a4c3adcf5b88b2d3` |
| 01-02 | `4f09f16aefafe9fa80b5ea5b51f173d871168e9867f3178f1df60182d0eb9b91` |
| 01-03 | `10eeba6ffb69d19caa543d89f4fde82d1852c7d3892effc3c163a459278e89cd` |
| 01-04 | `cb06cbc7011c9ba2bff67312d75d70df69f9d1df009120a0a2ecb9763d802ab8` |
| 02-03 | `244da83261dfe3efdda4ed510f5519ab81f1c6f6c8d0eb8418860a671d0fedb2` |
| 02-04 | `b6366e2dc6594126fa04882ef31a3d1738ddbc635bdb007cc592b03b42326079` |
| 03-04 | `42fb9ab235b444ebbbac47a8d378a0774d5d28c30c6874e4cec81d5a99dcc679` |

The hash is the SHA-256 of canonical JSON for the 3×3 `pixel_matrix` array
(`json.dumps(..., sort_keys=True, separators=(',', ':'))`).

## Evidence

- Replay validation: `data/output/b9_five_scene_validation/matcher_runs_1024_globalready/loftr/global_ready_validation.json`
- Replay configuration: `data/output/b9_five_scene_validation/matcher_runs_1024_globalready/loftr/run_config.json`
- Replay geometry index and pair sidecars: `data/output/b9_five_scene_validation/matcher_runs_1024_globalready/loftr/geometry_index.json` and `geometry/`
- Original comparison: `data/output/b9_five_scene_validation/matcher_runs_1024/loftr/pairwise_summary.json`

