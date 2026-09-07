"""
Full pipeline: co-registration → BAGRN → VOLRN → mosaic
For DZ01V sensor, bands B5, B8, B14.
"""
import os, sys, glob, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import read_geotiff, write_geotiff
from src.coregistration import coregister_pair, compute_shifts_from_overlap, phase_correlation
from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.overlap import get_overlap_window
from src.mosaic import create_mosaic
from src.metrics import compute_all

INPUT_BASE = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\input'
OUTPUT_BASE = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\final_v2'
os.makedirs(OUTPUT_BASE, exist_ok=True)

SENSOR = 'DZ01V'
BANDS = ['B5', 'B8', 'B14']


def load_images(band, sensor):
    """Load all images for a given band and sensor."""
    pattern = os.path.join(INPUT_BASE, '**', f'*{sensor}*{band}*.TIF')
    files = sorted(glob.glob(pattern, recursive=True))
    files = [f for f in files if '_PAN' not in f.upper()]

    images = []
    for f in files:
        arr, tr, crs, nd = read_geotiff(f)
        # Squeeze to 2D
        while arr.ndim > 2:
            arr = arr[0]
        images.append({'path': f, 'arr': arr, 'transform': tr, 'crs': crs,
                        'nodata': nd, 'name': os.path.basename(f)})
    return images


def compute_bounds(arr, tr):
    """Compute geographic bounds (left, bottom, right, top)."""
    rows, cols = arr.shape
    left = tr.c
    right = tr.c + tr.a * cols
    top = tr.f
    bottom = tr.f + tr.e * rows
    return (left, bottom, right, top)


def find_overlaps(images):
    """Find all overlapping pairs."""
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
                                     'window_i': wi, 'window_j': wj,
                                     'n_pixels': n})
    return overlaps


def step1_coregister(images, overlaps, out_dir):
    """Step 1: Co-register all images to the first one using overlap patches."""
    print("\n=== Step 1: Co-registration ===")
    coreg_dir = os.path.join(out_dir, 'coregistered')
    os.makedirs(coreg_dir, exist_ok=True)

    ref = images[0]
    # Copy reference as-is
    import shutil
    ref_out = os.path.join(coreg_dir, ref['name'])
    shutil.copy2(ref['path'], ref_out)
    coreg_images = [images[0]]

    shifts = []
    for i in range(1, len(images)):
        tgt = images[i]
        print(f"\n  Co-registering {tgt['name']}...")

        # Find overlap pairs involving this image
        best_shift_y, best_shift_x = 0.0, 0.0
        best_conf = 0.0
        n_found = 0

        for ov in overlaps:
            # Only look at overlaps between ref (0) and target (i)
            if ov['idx_i'] == 0 and ov['idx_j'] == i:
                wi = ov['window_j']  # window for target image i
                wj = ov['window_i']  # window for reference image 0
            elif ov['idx_i'] == i and ov['idx_j'] == 0:
                wi = ov['window_i']  # window for target image i
                wj = ov['window_j']  # window for reference image 0
            else:
                continue

            # Extract overlap patches: window = (row_start, row_end, col_start, col_end)
            patch_self = tgt['arr'][wi[0]:wi[1], wi[2]:wi[3]].astype(np.float64)
            patch_ref = images[0]['arr'][wj[0]:wj[1], wj[2]:wj[3]].astype(np.float64)

            # Crop to same size
            h = min(patch_self.shape[0], patch_ref.shape[0])
            w = min(patch_self.shape[1], patch_ref.shape[1])
            if h < 10 or w < 10:
                continue
            patch_self = patch_self[:h, :w]
            patch_ref = patch_ref[:h, :w]

            # Phase correlation
            sy, sx, conf = phase_correlation(patch_ref, patch_self)
            print(f"    Pair ({ov['idx_i']},{ov['idx_j']}): shift=({sy:.4f},{sx:.4f}) conf={conf:.4f}")

            if conf > best_conf:
                best_conf = conf
                best_shift_y = sy
                best_shift_x = sx
                n_found += 1

        shifts.append({'shift_y': best_shift_y, 'shift_x': best_shift_x,
                        'confidence': best_conf, 'n_pairs': n_found})
        print(f"  => Best: dy={best_shift_y:.4f} dx={best_shift_x:.4f} conf={best_conf:.4f}")

        # Apply shift and save
        out_path = os.path.join(coreg_dir, tgt['name'])
        if abs(best_shift_y) > 0.01 or abs(best_shift_x) > 0.01:
            from scipy.ndimage import shift as ndimage_shift
            arr_shifted = ndimage_shift(tgt['arr'], [best_shift_y, best_shift_x],
                                         order=1, mode='constant', cval=tgt['nodata'])
        else:
            arr_shifted = tgt['arr']

        write_geotiff(out_path, arr_shifted, tgt['transform'], tgt['crs'],
                      nodata=tgt['nodata'])

        arr2, tr2, crs2, nd2 = read_geotiff(out_path)
        while arr2.ndim > 2:
            arr2 = arr2[0]
        coreg_images.append({'path': out_path, 'arr': arr2, 'transform': tr2,
                              'crs': crs2, 'nodata': nd2, 'name': tgt['name']})

    return coreg_images, shifts


def step2_bagrn(images, overlaps, out_dir):
    """Step 2: BAGRN global normalization."""
    print("\n=== Step 2: BAGRN normalization ===")
    # BAGRN expects 3D arrays: (band, rows, cols)
    arrs = [img['arr'][np.newaxis, :, :] for img in images]
    nodatas = [img['nodata'] for img in images]
    bands = [0]

    bagrn_result, _, _ = bagrn_normalize(arrs, nodatas, overlaps, control_idx=0)
    metrics = compute_all(arrs, bagrn_result, nodatas, overlaps, bands)  # Fixed: before=original, after=bagrn
    print(f"  BAGRN metrics: {metrics}")

    # Save BAGRN individual images (squeeze back to 2D)
    bagrn_dir = os.path.join(out_dir, 'bagrn')
    os.makedirs(bagrn_dir, exist_ok=True)
    for i, img in enumerate(images):
        out_path = os.path.join(bagrn_dir, img['name'].replace('.TIF', '_bagrn.tif'))
        write_geotiff(out_path, bagrn_result[i][0], img['transform'], img['crs'],
                      nodata=img['nodata'])

    return bagrn_result, metrics


def step3_volrn(images, bagrn_result, overlaps, out_dir):
    """Step 3: VOLRN local normalization."""
    print("\n=== Step 3: VOLRN normalization ===")
    nodatas = [img['nodata'] for img in images]
    transforms = [img['transform'] for img in images]
    bounds_list = [compute_bounds(img['arr'], img['transform']) for img in images]

    # bagrn_result is already 3D: (n_images, bands, rows, cols)
    volrn_result, _ = volrn_normalize(
        bagrn_result, transforms, bounds_list, nodatas,
        block_size_pixels=200, lambda_param=0.5, rho=1.0,
        max_iter=200, tol=1e-4, verbose=False)

    # Compute metrics (need 3D arrays)
    arrs_3d = [img['arr'][np.newaxis, :, :] for img in images]
    metrics = compute_all(arrs_3d, volrn_result, nodatas, overlaps, [0])  # Fixed: before=original, after=volrn
    print(f"  VOLRN metrics: {metrics}")

    # Save VOLRN individual images (squeeze band dim)
    volrn_dir = os.path.join(out_dir, 'volrn')
    os.makedirs(volrn_dir, exist_ok=True)
    for i, img in enumerate(images):
        out_path = os.path.join(volrn_dir, img['name'].replace('.TIF', '_volrn.tif'))
        write_geotiff(out_path, volrn_result[i][0], img['transform'], img['crs'],
                      nodata=img['nodata'])

    return volrn_result, metrics


def step4_mosaic(images, arrs_dict, out_dir):
    """Step 4: Create mosaics for original, BAGRN, and VOLRN."""
    print("\n=== Step 4: Mosaic creation ===")
    transforms = [img['transform'] for img in images]
    crs = images[0]['crs']
    nodatas = [img['nodata'] for img in images]

    for label, arrs in arrs_dict.items():
        out_path = os.path.join(out_dir, f'mosaic_{label}.tif')
        print(f"  Creating mosaic: {label}...")
        # arrs is list of 3D (band, rows, cols) or 2D (rows, cols)
        # create_mosaic expects list of 2D or 3D arrays
        create_mosaic(arrs, transforms, crs, nodatas, out_path)
        size = os.path.getsize(out_path) / 1024 / 1024
        print(f"  Saved: {out_path} ({size:.1f} MB)")


def process_band(band):
    """Full pipeline for one band."""
    print(f"\n{'='*60}")
    print(f"  Processing {SENSOR} - {band}")
    print(f"{'='*60}")

    band_dir = os.path.join(OUTPUT_BASE, f'{band}')
    os.makedirs(band_dir, exist_ok=True)

    # Load images
    images = load_images(band, SENSOR)
    print(f"\nLoaded {len(images)} images:")
    for img in images:
        print(f"  {img['name']}: {img['arr'].shape}")

    # Find overlaps
    overlaps = find_overlaps(images)
    print(f"\n{len(overlaps)} overlap pairs found")

    # Step 1: Co-register
    coreg_images, shifts = step1_coregister(images, overlaps, band_dir)

    # Step 2: BAGRN (expects 3D arrays: band, rows, cols)
    bagrn_result, bagrn_metrics = step2_bagrn(coreg_images, overlaps, band_dir)

    # Step 3: VOLRN
    volrn_result, volrn_metrics = step3_volrn(coreg_images, bagrn_result, overlaps, band_dir)

    # Step 4: Mosaics
    # Mosaic needs 2D arrays, squeeze band dimension
    arrs_orig = [img['arr'] for img in coreg_images]
    arrs_bagrn = [bagrn_result[i][0] for i in range(len(coreg_images))]
    arrs_volrn = [volrn_result[i][0] for i in range(len(coreg_images))]

    arrs_dict = {
        'original': arrs_orig,
        'bagrn': arrs_bagrn,
        'volrn': arrs_volrn,
    }
    step4_mosaic(coreg_images, arrs_dict, band_dir)

    # Save shift info
    shift_info = {
        'reference': coreg_images[0]['name'],
        'shifts': shifts,
    }
    with open(os.path.join(band_dir, 'coreg_shifts.json'), 'w') as f:
        json.dump(shift_info, f, indent=2)

    # Save metrics
    metrics_info = {
        'bagrn': bagrn_metrics,
        'volrn': volrn_metrics,
    }
    with open(os.path.join(band_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_info, f, indent=2)

    print(f"\n  Summary for {band}:")
    print(f"  {'Metric':<12} {'Original':>12} {'BAGRN':>12} {'VOLRN':>12}")
    print(f"  {'-'*50}")
    arrs_3d = [img['arr'][np.newaxis, :, :] for img in coreg_images]
    nodatas = [img['nodata'] for img in coreg_images]
    baseline = compute_all(arrs_3d, arrs_3d, nodatas, overlaps, [0])
    for k in baseline:
        print(f"  {k:<12} {baseline[k]:>12.4f} {bagrn_metrics[k]:>12.4f} {volrn_metrics[k]:>12.4f}")

    return baseline, bagrn_metrics, volrn_metrics


if __name__ == '__main__':
    t_start = time.time()

    all_results = {}
    for band in BANDS:
        try:
            b, bg, vr = process_band(band)
            all_results[band] = {'baseline': b, 'bagrn': bg, 'volrn': vr}
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n  ERROR processing {band}: {e}")

    elapsed = time.time() - t_start
    print(f"\n\n{'='*60}")
    print(f"  ALL DONE in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\n  Output: {OUTPUT_BASE}")
    for band in BANDS:
        band_dir = os.path.join(OUTPUT_BASE, band)
        if os.path.isdir(band_dir):
            tifs = sorted([f for f in os.listdir(band_dir) if f.endswith('.tif')])
            print(f"\n  {band}/ ({len(tifs)} TIF files)")
            for f in tifs:
                size = os.path.getsize(os.path.join(band_dir, f)) / 1024 / 1024
                print(f"    {f} ({size:.1f} MB)")
