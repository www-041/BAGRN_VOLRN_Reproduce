# Task10C Preflight

Date: 2026-09-26

## Repository identity

- Branch: `exp/2026-09-24-b9-five-scene-validation`
- HEAD: `8bbf830e69f2ede66398ad226f103a12672075a3`
- HEAD descends from the required starting commit `8bbf830`.
- Remote: `https://github.com/www-041/BAGRN_VOLRN_Reproduce.git`
- `push_performed=false`.

## Environment

- Python 3.11.6
- NumPy 2.4.6; SciPy 1.17.1; rasterio 1.4.4; OpenCV 5.0.0
- PyTorch 2.14.0+cu126; CUDA 12.6 available
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8,585,216,000 bytes VRAM

## Baseline

Command: `.venv-registration/Scripts/python.exe -m pytest -q`

Result: `767 passed, 4 skipped, 9205 warnings in 135.61s`; exit code 0.

The skipped tests are the existing synthetic disconnected-SIFT cases and two
optional matcher integration cases. No destructive cleanup was required.
