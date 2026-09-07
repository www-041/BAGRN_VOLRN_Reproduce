"""Synthetic end-to-end smoke test.

Task 19 Step 4 of reliability-fixes plan:
Maximum automated end-to-end experiment allowed:
- 2 synthetic images
- 1 band
- <=128x128 each
- small overlap
- BAGRN
- VOLRN with very small block size / limited iterations
- mosaic
- metrics
- Assert: shapes valid, NoData preserved, all metrics finite
"""

import numpy as np
import tempfile
import os
from rasterio.transform import from_origin

from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.mosaic import create_mosaic
from src.metrics import compute_all
from src.overlap import get_overlap_window


def test_synthetic_end_to_end_smoke():
    """Full synthetic pipeline: BAGRN -> VOLRN -> mosaic -> metrics."""
    # 2 synthetic images, 1 band, 64x64 each
    np.random.seed(42)
    img_a = np.random.rand(1, 64, 64).astype(np.float64) * 100 + 100
    img_b = np.random.rand(1, 64, 64).astype(np.float64) * 100 + 110
    
    # Add NoData region
    img_a[:, :8, :] = -9999.0
    img_b[:, :8, :] = -9999.0
    
    transforms = [
        from_origin(0, 64, 1, 1),
        from_origin(32, 64, 1, 1),
    ]
    bounds = [
        (0, 0, 64, 64),
        (32, 0, 96, 64),
    ]
    nodata = [-9999.0, -9999.0]
    
    # Compute overlap
    overlap = get_overlap_window(bounds[0], transforms[0], bounds[1], transforms[1])
    assert overlap is not None
    (ri_s, ri_e, ci_s, ci_e), (rj_s, rj_e, cj_s, cj_e) = overlap
    overlaps = [{
        'idx_i': 0, 'idx_j': 1,
        'window_i': (ri_s, ri_e, ci_s, ci_e),
        'window_j': (rj_s, rj_e, cj_s, cj_e),
    }]
    
    # BAGRN
    bagrn_result, theta_mu, theta_sigma = bagrn_normalize(
        [img_a, img_b], nodata, overlaps, control_idx=0
    )
    assert len(bagrn_result) == 2
    assert bagrn_result[0].shape == img_a.shape
    assert np.all(bagrn_result[0][:, :8, :] == -9999.0)  # NoData preserved
    
    # VOLRN
    volrn_result, volrn_coeffs = volrn_normalize(
        bagrn_result, transforms, bounds, nodata,
        block_size_pixels=16, max_iter=3, tol=1e-3,
    )
    assert len(volrn_result) == 2
    assert volrn_result[0].shape == img_a.shape
    assert np.all(volrn_result[0][:, :8, :] == -9999.0)  # NoData preserved
    
    # Mosaic
    with tempfile.TemporaryDirectory() as tmpdir:
        mosaic_path = os.path.join(tmpdir, "mosaic.tif")
        create_mosaic(
            volrn_result, transforms, "EPSG:4326", nodata, mosaic_path,
            mode="weighted",
        )
        assert os.path.exists(mosaic_path)
        
        # Metrics
        metrics = compute_all(
            [img_a, img_b], volrn_result, nodata, overlaps, [0]
        )
        
        # All metrics should be finite
        for key, val in metrics.items():
            assert np.isfinite(val), f"Metric {key} is not finite: {val}"
