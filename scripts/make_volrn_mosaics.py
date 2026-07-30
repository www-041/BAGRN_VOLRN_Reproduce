"""
VOLRN pipeline: BAGRN → VOLRN → mosaic.
Uses coregistered images from final_v2/*/coregistered/.
"""
import os, sys, glob, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import read_geotiff, write_geotiff
from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.overlap import get_overlap_window
from src.mosaic import create_mosaic
from src.metrics import compute_all

OUTPUT_BASE = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\final_v2'
BANDS = ['B5', 'B8', 'B14']


def compute_bounds(arr, tr):
    rows, cols = arr.shape
    return (tr.c, tr.f + tr.e * rows, tr.c + tr.a * cols, tr.f)


def load_images_from_dir(d):
    files = sorted(glob.glob(os.path.join(d, '*.TIF')))
    images = []
    for f in files:
        arr, tr, crs, nd = read_geotiff(f)
        while arr.ndim > 2:
            arr = arr[0]
        images.append({'arr': arr, 'transform': tr, 'crs': crs, 'nodata': nd,
                        'name': os.path.basename(f), 'path': f})
    return images


def find_overlaps(images):
    overlaps = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            bi = compute_bounds(images[i]['arr'], images[i]['transform'])
            bj = compute_bounds(images[j]['arr'], images[j]['transform'])
            windows = get_overlap_window(bi, images[i]['transform'],
                                          bj, images[j]['transform'])
            if windows is not None:
                wi, wj = windows
                n = (wi[1] - wi[0]) * (wi[3] - wi[2])
                if n > 100:
                    overlaps.append({'idx_i': i, 'idx_j': j,
                                     'window_i': wi, 'window_j': wj})
    return overlaps


for band in BANDS:
    coreg_dir = os.path.join(OUTPUT_BASE, band, 'coregistered')
    if not os.path.isdir(coreg_dir):
        print(f"Skip {band}: no coregistered dir")
        continue

    print(f"\n{'='*60}")
    print(f"  {band} — VOLRN")
    print(f"{'='*60}")

    images = load_images_from_dir(coreg_dir)
    print(f"  {len(images)} coregistered images")
    overlaps = find_overlaps(images)
    print(f"  {len(overlaps)} overlap pairs")

    arrs_3d = [img['arr'][np.newaxis, :, :] for img in images]
    nodatas = [img['nodata'] for img in images]
    transforms = [img['transform'] for img in images]
    bounds_list = [compute_bounds(img['arr'], img['transform']) for img in images]

    # BAGRN first
    print("  Running BAGRN...")
    t0 = time.time()
    bagrn_result, _, _ = bagrn_normalize(arrs_3d, nodatas, overlaps, control_idx=0)
    print(f"  BAGRN done ({time.time()-t0:.1f}s)")

    # VOLRN
    print("  Running VOLRN...")
    t0 = time.time()
    volrn_result, coeffs = volrn_normalize(
        bagrn_result, transforms, bounds_list, nodatas,
        block_size_pixels=200, lambda_param=0.5, rho=1.0,
        max_iter=200, tol=1e-4, verbose=False)
    t_volrn = time.time() - t0
    print(f"  VOLRN done ({t_volrn:.1f}s)")

    # Metrics
    metrics_bagrn = compute_all(bagrn_result, arrs_3d, nodatas, overlaps, [0])
    metrics_volrn = compute_all(volrn_result, arrs_3d, nodatas, overlaps, [0])
    print(f"  BAGRN: {metrics_bagrn}")
    print(f"  VOLRN: {metrics_volrn}")

    # Save VOLRN individual images
    volrn_dir = os.path.join(OUTPUT_BASE, band, 'volrn')
    os.makedirs(volrn_dir, exist_ok=True)
    for i, img in enumerate(images):
        out = os.path.join(volrn_dir, img['name'].replace('.TIF', '_volrn.tif'))
        write_geotiff(out, volrn_result[i][0], img['transform'], img['crs'],
                      nodata=img['nodata'])

    # VOLRN mosaic
    crs = images[0]['crs']
    volrn_arrs = [volrn_result[i] for i in range(len(images))]
    out_path = os.path.join(OUTPUT_BASE, band, f'mosaic_{band}_volrn.tif')
    print(f"  Creating VOLRN mosaic...")
    t0 = time.time()
    create_mosaic(volrn_arrs, transforms, crs, nodatas, out_path)
    size = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved: {out_path} ({size:.1f} MB, {time.time()-t0:.1f}s)")

    # Save metrics
    with open(os.path.join(OUTPUT_BASE, band, 'metrics.json'), 'w') as f:
        json.dump({'bagrn': metrics_bagrn, 'volrn': metrics_volrn}, f, indent=2)

    print(f"\n  {band} VOLRN done!")
