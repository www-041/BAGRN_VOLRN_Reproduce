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
