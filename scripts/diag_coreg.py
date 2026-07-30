"""Diagnostic: check if images actually need co-registration."""
import sys, os, numpy as np
sys.path.insert(0, '.')
from src.io_utils import read_geotiff
from src.overlap import get_overlap_window
from scipy.ndimage import shift as ndimage_shift
from scipy.signal import correlate

IMG1 = r'data/input/20251208025051/DZ01V_L2_E118.9_N30.3_20251208025051_01_T1/DZ01V_L2_E118.9_N30.3_20251208025051_01_T1_B5.TIF'
IMG2 = r'data/input/20251214025247/DZ01V_L2_E119.0_N30.3_20251214025247_01_T1/DZ01V_L2_E119.0_N30.3_20251214025247_01_T1_B5.TIF'

arr1, tr1, crs, nd1 = read_geotiff(IMG1)
arr2, tr2, _, nd2 = read_geotiff(IMG2)
while arr1.ndim > 2: arr1 = arr1[0]
while arr2.ndim > 2: arr2 = arr2[0]

rows1, cols1 = arr1.shape
rows2, cols2 = arr2.shape
b1 = (tr1.c, tr1.f + tr1.e * rows1, tr1.c + tr1.a * cols1, tr1.f)
b2 = (tr2.c, tr2.f + tr2.e * rows2, tr2.c + tr2.a * cols2, tr2.f)

win = get_overlap_window(b1, tr1, b2, tr2)
wi, wj = win
print(f"Overlap: img1[{wi[0]}:{wi[1]}, {wi[2]}:{wi[3]}]  img2[{wj[0]}:{wj[1]}, {wj[2]}:{wj[3]}]")
print(f"Overlap size: {wi[1]-wi[0]}x{wi[3]-wi[2]}")

patch1 = arr1[wi[0]:wi[1], wi[2]:wi[3]].astype(np.float64)
patch2 = arr2[wj[0]:wj[1], wj[2]:wj[3]].astype(np.float64)

# Same size
h = min(patch1.shape[0], patch2.shape[0])
w = min(patch1.shape[1], patch2.shape[1])
patch1 = patch1[:h, :w]
patch2 = patch2[:h, :w]

print(f"Patch size: {h}x{w}")

# Test shift on FULL overlap
cc = correlate(patch2, patch1, mode='full', method='fft')
peak = np.unravel_index(np.argmax(cc), cc.shape)
cy, cx = peak
shift_y_full = cy - (h - 1)
shift_x_full = cx - (w - 1)
print(f"\nFull overlap shift: dy={shift_y_full:.4f} dx={shift_x_full:.4f}")

# Test shift on CENTER only (4 quadrants)
for name, r1, r2, c1, c2 in [
    ('center', h//4, 3*h//4, w//4, 3*w//4),
    ('top-left', 0, h//2, 0, w//2),
    ('top-right', 0, h//2, w//2, w),
    ('bottom-left', h//2, h, 0, w//2),
    ('bottom-right', h//2, h, w//2, w),
]:
    p1 = patch1[r1:r2, c1:c2]
    p2 = patch2[r1:r2, c1:c2]
    ph, pw = p1.shape
    if ph < 10 or pw < 10:
        continue
    mask = (p1 > 0) & (p2 > 0)
    if mask.sum() < 100:
        continue
    cc2 = correlate(p2, p1, mode='full', method='fft')
    peak2 = np.unravel_index(np.argmax(cc2), cc2.shape)
    sy = peak2[0] - (ph - 1)
    sx = peak2[1] - (pw - 1)
    print(f"  {name:12s}: dy={sy:.4f} dx={sx:.4f}  (size {ph}x{pw})")

# Check: what if we DON'T shift at all?
print("\n--- Without co-registration ---")
diff_noshift = patch1 - patch2
mask = (patch1 > 0) & (patch2 > 0)
print(f"  Mean diff: {diff_noshift[mask].mean():.2f}")
print(f"  Std diff: {diff_noshift[mask].std():.2f}")
print(f"  ADM: {abs(diff_noshift[mask].mean()):.2f}")

# Check: with 5-pixel shift
print("\n--- With dx=5 shift ---")
patch2_shifted = ndimage_shift(patch2, [0, 5], order=1, mode='constant', cval=0)
diff_shift5 = patch1 - patch2_shifted
print(f"  Mean diff: {diff_shift5[mask].mean():.2f}")
print(f"  Std diff: {diff_shift5[mask].std():.2f}")
print(f"  ADM: {abs(diff_shift5[mask].mean()):.2f}")

# Check: with various dx values
print("\n--- ADM vs dx shift ---")
for dx in range(-2, 8):
    patch2_s = ndimage_shift(patch2, [0, dx], order=1, mode='constant', cval=0)
    diff = patch1 - patch2_s
    adm = abs(diff[mask].mean())
    print(f"  dx={dx:+d}: ADM={adm:.2f}")
