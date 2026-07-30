"""B14 visible band mosaic baseline: load → BAGRN → mosaic."""
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

OUTPUT_DIR = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\mosaic_B14'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load all B14 files
print("="*60)
print("  Step 1: Loading B14 bands")
print("="*60)

input_base = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\input'
b14_files = sorted(glob.glob(os.path.join(input_base, '**', '*B14*.TIF'), recursive=True))
print(f"Found {len(b14_files)} B14 files")

arrays = []
transforms = []
bounds_list = []
nodata_list = []
filenames = []

for f in b14_files:
    arr, tr, crs, nd = read_geotiff(f)
    arrays.append(arr)
    transforms.append(tr)
    nodata_list.append(nd)
    filenames.append(os.path.basename(f))
    bnd = (tr.c, tr.f + tr.e * arr.shape[1], tr.c + tr.a * arr.shape[2], tr.f)
    bounds_list.append(bnd)
    print(f"  {os.path.basename(f)}: {arr.shape[1]}x{arr.shape[2]}, "
          f"bounds=({bnd[0]:.0f},{bnd[1]:.0f})-({bnd[2]:.0f},{bnd[3]:.0f}), nodata={nd}")

# 2. Compute all overlap pairs
print(f"\n{'='*60}")
print("  Step 2: Detecting overlaps")
print("="*60)

overlaps = []
for i in range(len(arrays)):
    for j in range(i + 1, len(arrays)):
        windows = get_overlap_window(
            bounds_list[i], transforms[i], bounds_list[j], transforms[j])
        if windows is not None:
            win_i, win_j = windows
            n_pix = (win_i[1] - win_i[0]) * (win_i[3] - win_i[2])
            if n_pix > 0:
                overlaps.append({
                    'idx_i': i, 'idx_j': j,
                    'window_i': win_i, 'window_j': win_j,
                })
                print(f"  Pair ({i},{j}): {n_pix} pixels, "
                      f"win_i={win_i}, win_j={win_j}")

print(f"Total: {len(overlaps)} overlap pairs")

# 3. Compute baseline metrics (before normalization)
print(f"\n{'='*60}")
print("  Step 3: Baseline metrics (before normalization)")
print("="*60)

bands = [0]
baseline_metrics = compute_all(arrays, arrays, nodata_list, overlaps, bands)
print(f"  Baseline: {baseline_metrics}")

# 4. BAGRN normalization
print(f"\n{'='*60}")
print("  Step 4: BAGRN normalization")
print("="*60)

t0 = time.time()
bagrn_result, theta_mu, theta_sigma = bagrn_normalize(
    arrays, nodata_list, overlaps, control_idx=0)
t_bagrn = time.time() - t0
print(f"  BAGRN completed in {t_bagrn:.2f}s")

# Print compensation parameters
print(f"  theta_mu shape: {np.array(theta_mu).shape}, theta_sigma shape: {np.array(theta_sigma).shape}")
print(f"  theta_mu: {np.array(theta_mu).ravel()[:6]}")
print(f"  theta_sigma: {np.array(theta_sigma).ravel()[:6]}")

# 5. Compute metrics after BAGRN
print(f"\n{'='*60}")
print("  Step 5: Metrics after BAGRN")
print("="*60)

bagrn_metrics = compute_all(bagrn_result, arrays, nodata_list, overlaps, bands)
print(f"  BAGRN: {bagrn_metrics}")

# Comparison table
print(f"\n  {'Metric':<12} {'Before':>12} {'After BAGRN':>14} {'Improvement':>14}")
print(f"  {'-'*54}")
for k in baseline_metrics:
    b = baseline_metrics[k]
    a = bagrn_metrics[k]
    imp = (b - a) / b * 100 if b != 0 else 0
    print(f"  {k:<12} {b:>12.4f} {a:>14.4f} {imp:>+13.1f}%")

# 6. Save normalized images
print(f"\n{'='*60}")
print("  Step 6: Saving normalized images")
print("="*60)

for idx in tqdm(range(len(bagrn_result)), desc="Saving"):
    out_name = f"{filenames[idx].replace('.TIF', '_bagrn.tif')}"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    write_geotiff(out_path, bagrn_result[idx], transforms[idx], crs, nodata=nodata_list[idx])
    print(f"  Saved: {out_name}")

# 7. Create mosaic
print(f"\n{'='*60}")
print("  Step 7: Creating mosaic")
print("="*60)

t0 = time.time()
mosaic_path = os.path.join(OUTPUT_DIR, 'mosaic_B14_bagrn.tif')
create_mosaic(
    bagrn_result, transforms, crs, nodata_list,
    mosaic_path, resolution=None)
t_mosaic = time.time() - t0
print(f"  Mosaic saved to: {mosaic_path}")
print(f"  Mosaic created in {t_mosaic:.2f}s")

# 8. Also create mosaic from original (before BAGRN) for comparison
print(f"\n{'='*60}")
print("  Step 8: Creating original mosaic (for comparison)")
print("="*60)

mosaic_orig_path = os.path.join(OUTPUT_DIR, 'mosaic_B14_original.tif')
create_mosaic(
    arrays, transforms, crs, nodata_list,
    mosaic_orig_path, resolution=None)
print(f"  Original mosaic saved to: {mosaic_orig_path}")

print(f"\n{'='*60}")
print("  Done!")
print("="*60)
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  Files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"    {f} ({size/1024/1024:.1f} MB)")
