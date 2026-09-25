# B9 unified 1024 protocol verification and GPU gate

## Protocol

- Existing historical config: `data/output/b9_five_scene_validation/04_frozen_five_scene_config.json`
- Existing historical scale: `registration.match_max_side=1600`
- New config: `data/output/b9_five_scene_validation/04_frozen_five_scene_config_1024.json`
- New shared scale: `registration.match_max_side=1024`
- Frozen manifest indices: `[2, 3, 5, 8, 10]`
- Local experiment indices: `0..4` in that frozen order.
- Coordinate frame: `pair_common_grid`
- RANSAC and acceptance settings: unchanged from the old config.
- Gate output root: `data/output/b9_five_scene_validation/gpu_gate_1024/`

## Deterministic worst-pair selection

The shape-only 1024 audit found a tie for the maximum coarse-token count and estimated single FP32 similarity matrix:

- Manifest pair `(2,3)` → local experiment pair `(0,1)`: model input `1024x992`, coarse `128x124`, `15872` tokens, approximately `961.0 MiB` for one FP32 `sim_matrix`.
- Manifest pair `(2,5)` → local experiment pair `(0,2)`: model input `992x1024`, coarse `124x128`, `15872` tokens, approximately `961.0 MiB` for one FP32 `sim_matrix`.

The gate uses `(2,3)` / `(0,1)` deterministically, matching the Task 7C worst-pair reference. This is a one-pair gate, not a formal B9 run.

## User-run EfficientLoFTR GPU gate

Run from the repository root in PowerShell. The following command is prepared but was **not executed by Codex**:

```powershell
$env:EFFICIENT_LOFTR_REPO = "D:\科研\地质一号\文献\EfficientLoFTR"

.\.venv-registration\Scripts\python.exe -m scripts.run_b9_registration `
  --config data/output/b9_five_scene_validation/04_frozen_five_scene_config_1024.json `
  --output-dir data/output/b9_five_scene_validation/gpu_gate_1024/efficient_loftr_worst_pair `
  --matcher efficient_loftr `
  --device cuda `
  --pair 0 1
```

In a second PowerShell window, monitor peak VRAM:

```powershell
nvidia-smi -l 1
```

### Gate PASS

Report `1024_GPU_GATE = PASS` only if the requested pair completes with:

- no CUDA out-of-memory error;
- no CPU fallback;
- RANSAC completed;
- `raw_matches > 0` and `inliers > 0`;
- finite RMSE and P95;
- the output is under the dedicated `gpu_gate_1024` directory.

### Gate FAIL

If the pair fails, especially with CUDA OOM, report:

```text
TASK_7D_1024_GPU_GATE_FAILED
```

Then stop. Do not automatically switch to 960/800, FP16/BF16, model optimization, or matcher-specific scales.

## Scope status

- Formal five-scene matcher experiments: not run.
- Four-matcher benchmark: not run.
- MST/global translation/mosaic/BAGRN/VOLRN: not run.
- Old 1600 config and historical outputs: untouched.
