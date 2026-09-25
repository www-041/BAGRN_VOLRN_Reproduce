# Overnight B9 global execution preflight

## Repository

- Branch: `exp/2026-09-24-b9-five-scene-validation`
- Start HEAD: `50acb1773e8ca0a72979f03ed036d386e05067c9`
- Tracked working tree: clean.
- Existing untracked temporary directories were preserved.
- Push: forbidden by the execution plan; none performed.

## Frozen protocol

- Config: `data/output/b9_five_scene_validation/04_frozen_five_scene_config_1024.json`
- SHA-256: `B9181AFCC972B29716D309F7AA3E82D3FBDBA57FD5D466E2C0AA04D25060C9BA`
- Band: `B9`
- Shared `match_max_side`: `1024`
- Manifest indices: `[2, 3, 5, 8, 10]`
- CRS: `EPSG:32650`
- Pixel size: `14.0 m`
- RANSAC: unchanged AffineTransform, threshold `2.0 px`, 5000 trials, seed 0, minimum 20 inliers, minimum ratio 0.3.
- Acceptance: unchanged from the frozen configuration.

## Environment

- Python environment: `.venv-registration`
- PyTorch: `2.14.0+cu126`
- CUDA runtime: `12.6`
- CUDA available: `True`
- Device: `NVIDIA GeForce RTX 4060 Laptop GPU`
- D: drive free space at preflight: approximately `61.5 GB`.
- Required free-space threshold: `20 GB`; preflight passed.

## Decision

`SAFE_TO_RUN = YES`

The run may proceed with the plan's sequential matcher order. Existing
`matcher_runs_1024/*` outputs are historical pairwise-only inputs and will not
be overwritten. New reruns will use
`matcher_runs_1024_globalready/*`; global outputs will use
`global_runs_1024/*`.
