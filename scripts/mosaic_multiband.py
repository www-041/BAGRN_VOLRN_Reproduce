"""Multi-band mosaic: B5, B8, B14 — BAGRN only (baseline)."""
import os, sys, glob, time
import numpy as np
import rasterio
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import read_geotiff, write_geotiff
from src.bagrn import bagrn_normalize
from src.overlap import get_overlap_window
from src.mosaic import create_mosaic
from src.metrics import compute_all

OUTPUT_BASE = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\mosaic_baseline'
os.makedirs(OUTPUT_BASE, exist_ok=True)

BANDS = ['B5', 'B8', 'B14']
SENSORS = ['DZ01V', 'DZ01S']
INPUT_BASE = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\input'


def load_band(band_name, sensor_filter=None):
    pattern = os.path.join(INPUT_BASE, '**', f'*{band_name}*.TIF')
    files = sorted(glob.glob(pattern, recursive=True))
    files = [f for f in files if '_PAN' not in f.upper()]
    if sensor_filter:
        files = [f for f in files if sensor_filter.upper() in os.path.basename(f).upper()]
    print(f"\nFound {len(files)} {band_name} files" + (f" ({sensor_filter})" if sensor_filter else ""))

    arrays, transforms, bounds_list, nodata_list, filenames = [], [], [], [], []
    crs = None
    for f in files:
        arr, tr, c, nd = read_geotiff(f)
        arrays.append(arr)
        transforms.append(tr)
        nodata_list.append(nd)
        filenames.append(os.path.basename(f))
        crs = c
        bnd = (tr.c, tr.f + tr.e * arr.shape[1], tr.c + tr.a * arr.shape[2], tr.f)
        bounds_list.append(bnd)
        print(f"  {os.path.basename(f)}: {arr.shape[1]}x{arr.shape[2]}, nodata={nd}")
    return arrays, transforms, bounds_list, nodata_list, filenames, crs


def compute_overlaps(arrays, transforms, bounds_list):
    overlaps = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            windows = get_overlap_window(
                bounds_list[i], transforms[i], bounds_list[j], transforms[j])
            if windows is not None:
                win_i, win_j = windows
                n_pix = (win_i[1] - win_i[0]) * (win_i[3] - win_i[2])
                if n_pix > 0:
                    overlaps.append({'idx_i': i, 'idx_j': j,
                                     'window_i': win_i, 'window_j': win_j})
    return overlaps


def process_band(band_name, sensor_filter=None):
    sensor_label = f"{band_name}_{sensor_filter}" if sensor_filter else band_name
    print(f"\n{'='*60}")
    print(f"  Processing {sensor_label}")
    print(f"{'='*60}")

    out_dir = os.path.join(OUTPUT_BASE, sensor_label)
    os.makedirs(out_dir, exist_ok=True)

    arrays, transforms, bounds_list, nodata_list, filenames, crs = load_band(band_name, sensor_filter)
    if len(arrays) < 2:
        print(f"  Only {len(arrays)} image(s), skipping")
        return None

    overlaps = compute_overlaps(arrays, transforms, bounds_list)
    print(f"  {len(overlaps)} overlap pairs")
    bands = [0]

    # Baseline metrics
    baseline = compute_all(arrays, arrays, nodata_list, overlaps, bands)
    print(f"  Baseline: {baseline}")

    # BAGRN
    print(f"\n  --- BAGRN ---")
    t0 = time.time()
    bagrn_result, _, _ = bagrn_normalize(arrays, nodata_list, overlaps, control_idx=0)
    t_bagrn = time.time() - t0
    bagrn_metrics = compute_all(bagrn_result, arrays, nodata_list, overlaps, bands)
    print(f"  BAGRN ({t_bagrn:.1f}s): {bagrn_metrics}")

    # Save BAGRN individual images
    for idx in range(len(bagrn_result)):
        out_path = os.path.join(out_dir, f"{filenames[idx].replace('.TIF', '_bagrn.tif')}")
        write_geotiff(out_path, bagrn_result[idx], transforms[idx], crs, nodata=nodata_list[idx])

    # Mosaic BAGRN
    mosaic_bagrn = os.path.join(out_dir, f'mosaic_{sensor_label}_bagrn.tif')
    t0 = time.time()
    create_mosaic(bagrn_result, transforms, crs, nodata_list, mosaic_bagrn)
    t_mos = time.time() - t0
    print(f"  Mosaic BAGRN ({t_mos:.1f}s): {mosaic_bagrn}")

    # Original mosaic
    mosaic_orig = os.path.join(out_dir, f'mosaic_{sensor_label}_original.tif')
    t0 = time.time()
    create_mosaic(arrays, transforms, crs, nodata_list, mosaic_orig)
    t_mos = time.time() - t0
    print(f"  Mosaic original ({t_mos:.1f}s): {mosaic_orig}")

    # Summary
    print(f"\n  {'='*50}")
    print(f"  {sensor_label} Summary")
    print(f"  {'='*50}")
    print(f"  {'Metric':<12} {'Original':>12} {'BAGRN':>12}")
    print(f"  {'-'*40}")
    for k in baseline:
        print(f"  {k:<12} {baseline[k]:>12.4f} {bagrn_metrics[k]:>12.4f}")

    return baseline, bagrn_metrics


if __name__ == '__main__':
    all_results = {}
    for band in BANDS:
        for sensor in SENSORS:
            key = f"{band}_{sensor}"
            try:
                result = process_band(band, sensor)
                if result:
                    all_results[key] = {'baseline': result[0], 'bagrn': result[1]}
            except Exception as e:
                print(f"\n  ERROR: {e}")

    # Final summary
    print(f"\n\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    for key, r in all_results.items():
        print(f"\n  {key}:")
        for k in r['baseline']:
            imp = (r['baseline'][k] - r['bagrn'][k]) / r['baseline'][k] * 100 if r['baseline'][k] != 0 else 0
            print(f"  {k:<12} {r['baseline'][k]:>12.4f} -> {r['bagrn'][k]:>12.4f}  ({imp:+.1f}%)")

    print(f"\n  Output: {OUTPUT_BASE}")
    for d in sorted(os.listdir(OUTPUT_BASE)):
        band_dir = os.path.join(OUTPUT_BASE, d)
        if os.path.isdir(band_dir):
            tifs = [f for f in sorted(os.listdir(band_dir)) if f.endswith('.tif')]
            print(f"  {d}/ ({len(tifs)} files)")
            for f in tifs:
                size = os.path.getsize(os.path.join(band_dir, f))
                print(f"    {f} ({size/1024/1024:.1f} MB)")
