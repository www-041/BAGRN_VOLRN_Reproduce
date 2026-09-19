import numpy as np
from rasterio.transform import Affine


def test_parse_dz01_mtl_reads_calibration_and_cloud_cover(tmp_path):
    from src.cloud_mask import parse_dz01_mtl

    p = tmp_path / "scene_MTL.txt"
    p.write_text(
        "\n".join([
            'RADIANCE_MULT_BAND_2 = 2.26E-02',
            'RADIANCE_ADD_BAND_2 = 1.0',
            'RADIANCE_MULT_BAND_14 = 1.56E-02',
            'RADIANCE_ADD_BAND_14 = 0.0',
            'CLOUD_COVER = 0.0148',
        ]), encoding='utf-8'
    )
    meta = parse_dz01_mtl(p)
    assert meta['radiance_mult']['B2'] == 2.26e-2
    assert meta['radiance_add']['B2'] == 1.0
    assert meta['radiance_mult']['B14'] == 1.56e-2
    assert meta['cloud_cover'] == 0.0148


def test_detect_cloud_mask_finds_bright_neutral_patch_but_not_vegetation():
    from src.cloud_mask import detect_cloud_mask

    shape = (20, 20)
    b2 = np.full(shape, 100.0)
    b5 = np.full(shape, 110.0)
    b7 = np.full(shape, 120.0)
    b14 = np.full(shape, 140.0)

    # Cloud: bright in visible + NIR and spectrally neutral.
    sl_cloud = (slice(2, 6), slice(2, 6))
    for arr, value in [(b2, 900), (b5, 920), (b7, 910), (b14, 950)]:
        arr[sl_cloud] = value

    # Vegetation: strong NIR but not bright/neutral in visible.
    sl_veg = (slice(12, 16), slice(12, 16))
    b2[sl_veg] = 80
    b5[sl_veg] = 140
    b7[sl_veg] = 100
    b14[sl_veg] = 900

    mask = detect_cloud_mask(
        {'B2': b2, 'B5': b5, 'B7': b7, 'B14': b14},
        radiance_mult={'B2': 1.0, 'B5': 1.0, 'B7': 1.0, 'B14': 1.0},
        radiance_add={'B2': 0.0, 'B5': 0.0, 'B7': 0.0, 'B14': 0.0},
        visible_percentile=90.0,
        nir_percentile=80.0,
        max_whiteness=0.35,
        max_ndvi=0.35,
        min_component_size=4,
        dilation_iterations=0,
    )

    assert mask[sl_cloud].mean() == 1.0
    assert mask[sl_veg].mean() == 0.0


def test_generate_scene_cloud_mask_discovers_files_and_saves_geotiff(tmp_path):
    from types import SimpleNamespace
    import rasterio
    from rasterio.transform import Affine
    from src.cloud_mask import generate_scene_cloud_mask

    scene_dir = tmp_path / 'scene'
    scene_dir.mkdir()
    tf = Affine(1, 0, 0, 0, -1, 20)
    base = {
        'B2': np.full((20,20), 100, dtype=np.uint16),
        'B5': np.full((20,20), 110, dtype=np.uint16),
        'B7': np.full((20,20), 120, dtype=np.uint16),
        'B14': np.full((20,20), 140, dtype=np.uint16),
    }
    for arr, value in [(base['B2'],900),(base['B5'],920),(base['B7'],910),(base['B14'],950)]:
        arr[2:6,2:6] = value
    for band, arr in base.items():
        with rasterio.open(
            scene_dir / f'TEST_{band}.TIF', 'w', driver='GTiff', height=20, width=20,
            count=1, dtype='uint16', crs='EPSG:32649', transform=tf, nodata=0,
        ) as dst:
            dst.write(arr, 1)
    (scene_dir / 'TEST_MTL.txt').write_text('\n'.join([
        'RADIANCE_MULT_BAND_2 = 1.0', 'RADIANCE_ADD_BAND_2 = 0',
        'RADIANCE_MULT_BAND_5 = 1.0', 'RADIANCE_ADD_BAND_5 = 0',
        'RADIANCE_MULT_BAND_7 = 1.0', 'RADIANCE_ADD_BAND_7 = 0',
        'RADIANCE_MULT_BAND_14 = 1.0', 'RADIANCE_ADD_BAND_14 = 0',
        'CLOUD_COVER = 0.04',
    ]), encoding='utf-8')
    scene = SimpleNamespace(directory=str(scene_dir), name='TEST')
    out_dir = tmp_path / 'masks'
    rec = generate_scene_cloud_mask(
        scene, output_dir=out_dir, visible_percentile=90, nir_percentile=80,
        min_component_size=4, dilation_iterations=0,
    )
    assert rec.mask[2:6,2:6].mean() == 1.0
    assert rec.metadata_cloud_cover == 0.04
    assert (out_dir / 'TEST_cloud_mask.tif').exists()
