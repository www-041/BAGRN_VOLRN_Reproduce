# Registration CUDA verification

## PyTorch CUDA identity

- torch: `2.14.0+cu126`
- torch.version.cuda: `12.6`
- torch.cuda.is_available(): `True`
- device_count: `1`
- device_name: `NVIDIA GeForce RTX 4060 Laptop GPU`

## Real CUDA tensor operation

- operation: `2048 x 2048` CUDA matrix multiplication
- result device: `cuda:0`
- allocated VRAM: `40.12 MB`
- reserved VRAM: `52.0 MB`
- result: PASS

## CUDA autocast

- operation: `1024 x 1024` CUDA matrix multiplication under `torch.autocast(device_type="cuda", dtype=torch.float16)`
- result dtype: `torch.float16`
- result device: `cuda:0`
- allocated VRAM: `14.12 MB`
- reserved VRAM: `20.0 MB`
- result: PASS

## Package integrity

- pip check after Task 2: `No broken requirements found.`

This artifact proves PyTorch CUDA execution only. EfficientLoFTR device routing is verified separately in Tasks 5-6.

## Task 4 regression evidence

- Task 7A geometry and memory stress: `30 passed in 5.17s` (`test_memory_safe_affine.py` + `test_geometry.py`, including 20k/50k stress).
- B9 runner/adapter focused tests: `17 passed, 1 skipped, 7 warnings in 25.92s`.
- Matcher contract regression: `35 passed, 2 skipped, 1 warning in 27.65s`.
- Post-regression `pip check`: `No broken requirements found.`
- Skips were pre-existing optional integration skips (`EFFICIENT_LOFTR_REPO` not set in the pytest process and LightGlue integration condition); no tests were altered to remove skips.

## Task 5 device routing

- Existing `_resolve_device("auto")` already returns `"cuda"` when `torch.cuda.is_available()` is true.
- Existing EfficientLoFTR model path calls `.eval().to(device)` and inference tensors call `.to(resolved_device)`.
- Added regression test `test_auto_device_resolves_to_cuda_when_cuda_is_available`.
- Focused adapter result after test addition: `10 passed, 1 skipped, 1 warning in 12.31s`.
- Result: `NO_PRODUCTION_CODE_CHANGE_REQUIRED`.

## Task 6 EfficientLoFTR GPU smoke

- Existing local repository: `D:\科研\地质一号\文献\EfficientLoFTR`
- Existing checkpoint: `D:\科研\地质一号\文献\EfficientLoFTR\weights\eloftr_outdoor.ckpt` (192,797,577 bytes)
- Smoke input: synthetic 128x128 grayscale pair; no real B9 scene data.
- Result: `status=SMOKE_COMPLETE`
- Adapter metadata device: `cuda`
- Match count: `143`
- Runtime: `5.327 sec`
- CUDA allocated/reserved: `8.12 MB / 366.0 MB`
- `CUDA is not available` warning: `ABSENT`
- Other warning: existing `torch.jit.script` deprecation FutureWarning only.
