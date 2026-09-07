"""Diagnostic: check weight maps in mosaic."""
import sys, os, numpy as np
sys.path.insert(0, '.')
from src.io_utils import read_geotiff
from src.overlap import get_overlap_window
from src.mosaic import _compute_weight_map, _combined_bounds
from scipy.ndimage import distance_transform_edt
import rasterio
from rasterio.warp import reproject, Resampling

IMG1 = r'data/input/20251208025051/DZ01V_L2_E118.9_N30.3_20251208025051_01_T1/DZ01V_L2_E118.9_N30.3_20251208025051_01_T1_B5.TIF'
IMG2 = r'data/input/20251214025247/DZ01V_L2_E119.0_N30.3_20251214025247_01_T1/DZ01V_L2_E119.0_N30.3_20251214025247_01_T1_B5.TIF'

arr1, tr1, crs, nd1 = read_geotiff(IMG1)
arr2, tr2, _, nd2 = read_geotiff(IMG2)
while arr1.ndim > 2: arr1 = arr1[0]
while arr2.ndim > 2: arr2 = arr2[0]

# Compute overlap
rows1, cols1 = arr1.shape
rows2, cols2 = arr2.shape
b1 = (tr1.c, tr1.f + tr1.e * rows1, tr1.c + tr1.a * cols1, tr1.f)
b2 = (tr2.c, tr2.f + tr2.e * rows2, tr2.c + tr2.a * cols2, tr2.f)
win = get_overlap_window(b1, tr1, b2, tr2)

if win:
    wi, wj = win
    print(f"Overlap: img1[{wi[0]}:{wi[1]}, {wi[2]}:{wi[3]}]  img2[{wj[0]}:{wj[1]}, {wj[2]}:{wj[3]}]")
    print(f"Overlap size: {wi[1]-wi[0]}x{wi[3]-wi[2]}")
    
    # Compute weight maps
    mask1 = np.ones(arr1.shape, dtype=bool)
    mask2 = np.ones(arr2.shape, dtype=bool)
    
    w1 = _compute_weight_map(mask1)
    w2 = _compute_weight_map(mask2)
    
    # Check weights in overlap region
    w1_overlap = w1[wi[0]:wi[1], wi[2]:wi[3]]
    w2_overlap = w2[wj[0]:wj[1], wj[2]:wj[3]]
    
    print(f"\nWeight map 1 in overlap:")
    print(f"  min={w1_overlap.min():.6f}, max={w1_overlap.max():.6f}, mean={w1_overlap.mean():.6f}")
    
    print(f"\nWeight map 2 in overlap:")
    print(f"  min={w2_overlap.min():.6f}, max={w2_overlap.max():.6f}, mean={w2_overlap.mean():.6f}")
    
    # Check the ratio
    ratio = w1_overlap / (w1_overlap + w2_overlap + 1e-12)
    print(f"\nWeight ratio (w1/(w1+w2)) in overlap:")
    print(f"  min={ratio.min():.6f}, max={ratio.max():.6f}, mean={ratio.mean():.6f}")
    
    # Show weight distribution
    print(f"\nWeight 1 histogram (overlap):")
    hist1, bins1 = np.histogram(w1_overlap, bins=10)
    for i in range(10):
        print(f"  [{bins1[i]:.4f}, {bins1[i+1]:.4f}): {hist1[i]}")
    
    print(f"\nWeight 2 histogram (overlap):")
    hist2, bins2 = np.histogram(w2_overlap, bins=10)
    for i in range(10):
        print(f"  [{bins2[i]:.4f}, {bins2[i+1]:.4f}): {hist2[i]}")
    
    # Check image values in overlap
    p1 = arr1[wi[0]:wi[1], wi[2]:wi[3]].astype(np.float64)
    p2 = arr2[wj[0]:wj[1], wj[2]:wj[3]].astype(np.float64)
    mask = np.isfinite(p1) & np.isfinite(p2)  # Fixed: check finite, not > 0
    
    print(f"\nImage values in overlap:")
    print(f"  Image 1: min={p1[mask].min():.1f}, max={p1[mask].max():.1f}, mean={p1[mask].mean():.1f}")
    print(f"  Image 2: min={p2[mask].min():.1f}, max={p2[mask].max():.1f}, mean={p2[mask].mean():.1f}")
    
    # Simulate weighted average
    weighted_avg = (p1 * w1_overlap + p2 * w2_overlap) / (w1_overlap + w2_overlap + 1e-12)
    print(f"  Weighted avg: min={weighted_avg[mask].min():.1f}, max={weighted_avg[mask].max():.1f}, mean={weighted_avg[mask].mean():.1f}")
    
    # Check what the seam looks like
    # Find the boundary where one image ends and the other begins
    # In overlap, check row by row
    h, w = w1_overlap.shape
    seam_row = h // 2
    print(f"\nRow {seam_row} profile (10 sample points):")
    for c in range(0, w, w//10):
        print(f"  col={c}: img1={p1[seam_row, c]:.1f}, img2={p2[seam_row, c]:.1f}, "
              f"w1={w1_overlap[seam_row, c]:.4f}, w2={w2_overlap[seam_row, c]:.4f}, "
              f"blend={weighted_avg[seam_row, c]:.1f}")
