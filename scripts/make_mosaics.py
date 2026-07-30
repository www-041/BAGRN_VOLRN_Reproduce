"""
Simplified pipeline: BAGRN + mosaic only (VOLRN omitted due to performance).
Uses already coregistered images from final_v2/*/coregistered/.
"""
import os, sys, glob, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import read_geotiff, write_geotiff
from src.bagrn import bagrn_normalize
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
    print(f"  {band}")
    print(f"{'='*60}")

    images = load_images_from_dir(coreg_dir)
    print(f"  {len(images)} coregistered images")
    for img in images:
        print(f"    {img['name']}")

    overlaps = find_overlaps(images)
    print(f"  {len(overlaps)} overlap pairs")

    # BAGRN
    arrs = [img['arr'][np.newaxis, :, :] for img in images]
    nodatas = [img['nodata'] for img in images]

    t0 = time.time()
    bagrn_result, _, _ = bagrn_normalize(arrs, nodatas, overlaps, control_idx=0)
    t_bagrn = time.time() - t0
    print(f"  BAGRN done ({t_bagrn:.1f}s)")

    # Metrics
    metrics = compute_all(bagrn_result, arrs, nodatas, overlaps, [0])
    print(f"  Metrics: {metrics}")

    # Save metrics
    with open(os.path.join(OUTPUT_BASE, band, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save BAGRN individual images
    bagrn_dir = os.path.join(OUTPUT_BASE, band, 'bagrn')
    os.makedirs(bagrn_dir, exist_ok=True)
    for i, img in enumerate(images):
        out = os.path.join(bagrn_dir, img['name'].replace('.TIF', '_bagrn.tif'))
        write_geotiff(out, bagrn_result[i][0], img['transform'], img['crs'],
                      nodata=img['nodata'])

    # Mosaics
    transforms = [img['transform'] for img in images]
    crs = images[0]['crs']
    nodatas = [img['nodata'] for img in images]

    for label, arrs_list in [
        ('original', [img['arr'][np.newaxis, :, :] for img in images]),
        ('bagrn', [bagrn_result[i] for i in range(len(images))]),
    ]:
        out_path = os.path.join(OUTPUT_BASE, band, f'mosaic_{band}_{label}.tif')
        print(f"  Creating {label} mosaic...")
        t0 = time.time()
        create_mosaic(arrs_list, transforms, crs, nodatas, out_path)
        size = os.path.getsize(out_path) / 1024 / 1024
        print(f"  Saved: {out_path} ({size:.1f} MB, {time.time()-t0:.1f}s)")

    print(f"\n  {band} done!")
