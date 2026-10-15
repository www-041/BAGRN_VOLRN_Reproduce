import numpy as np
import rasterio
from rasterio.transform import Affine

from src.cloud_mask import CloudMaskRecord
from src.multiscene_sift.models import Scene, MosaicGrid


def _write_tif(path, data, transform):
    with rasterio.open(
        path, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1],
        count=1, dtype='float32', crs='EPSG:32649', transform=transform,
        nodata=0.0,
    ) as dst:
        dst.write(data.astype('float32'), 1)


def test_radiometric_pipeline_accepts_source_cloud_masks_and_preserves_cloud(tmp_path):
    from src.multiscene_sift.radiometric import normalize_registered_band

    tf0 = Affine(1, 0, 0, 0, -1, 20)
    tf1 = Affine(1, 0, 5, 0, -1, 20)
    d0 = np.ones((20, 20), dtype=float) * 10
    d1 = np.ones((20, 20), dtype=float) * 12
    d0[3, 3] = 1000.0
    p0 = tmp_path / 's0_B14.TIF'
    p1 = tmp_path / 's1_B14.TIF'
    _write_tif(p0, d0, tf0)
    _write_tif(p1, d1, tf1)

    scenes = [
        Scene(0, 's0', str(tmp_path), {'B14': str(p0)}, rasterio.crs.CRS.from_epsg(32649), {'B14': tf0}, {'B14': (20,20)}, {'B14': 0.0}, {'B14': rasterio.transform.array_bounds(20,20,tf0)}),
        Scene(1, 's1', str(tmp_path), {'B14': str(p1)}, rasterio.crs.CRS.from_epsg(32649), {'B14': tf1}, {'B14': (20,20)}, {'B14': 0.0}, {'B14': rasterio.transform.array_bounds(20,20,tf1)}),
    ]
    cm0 = np.zeros((20, 20), dtype=bool); cm0[3,3] = True
    cm1 = np.zeros((20, 20), dtype=bool)
    masks = [
        CloudMaskRecord(cm0, tf0, rasterio.crs.CRS.from_epsg(32649), 1/400, None),
        CloudMaskRecord(cm1, tf1, rasterio.crs.CRS.from_epsg(32649), 0.0, None),
    ]
    grid = MosaicGrid(rasterio.crs.CRS.from_epsg(32649), Affine(1,0,0,0,-1,20), 25, 20, 1.0)
    G = [np.eye(3), np.eye(3)]

    result = normalize_registered_band(
        scenes, G, 'B14', radiometric_control_idx=1,
        registration_grid=grid, block_size_pixels=10,
        lambda_param=0.1, rho=1.0, max_iter=20, tol=1e-4,
        cloud_masks=masks,
    )
    assert np.isfinite(result.normalized_arrays[0][0, 3, 3])
    assert result.cloud_masks_registered[0][3, 3]
