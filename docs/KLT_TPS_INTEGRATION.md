# KLT-TPS Registration Backend

Source algorithm: senior-provided `register_images.py`  
SHA256: `4ff4ff68f52bb58764755c3efd14fcd15fadf6a05f636d784c562855cc98fbac`

Preserved algorithm semantics:
- local normalization
- GFTT corners
- forward/backward Pyramid-LK
- FB threshold filtering
- thin-plate-spline RBFInterpolator
- coarse field + cubic upsampling
- Jacobian folding + max-shift safety
- output-to-source flow
- one cubic resample from ORIGINAL moving DN

Project adaptations:
- Rasterio/in-memory arrays replace GDAL standalone I/O
- geographic overlap context replaces same-shape/same-GeoTransform `check_grid()`
- accepted correspondences are mapped back to moving native pixel coordinates
- one estimated B12 flow is applied to all loaded bands

Phase-I limitation: `klt_tps` supports exactly N=2.

Manual N=2 validation (run by the user after this branch is pushed):

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_registration_pair.py `
  --config configs/dz01_klt_tps_n2_b12.yaml `
  --scene-i 0 --scene-j 1 --validation-band B14 `
  --output-dir outputs/klt_tps_n2_b12
```

This branch does not run real DZ01V data and does not implement N=4/N=6,
multi-scene TPS composition, parameter search, or BAGRN/VOLRN changes.
