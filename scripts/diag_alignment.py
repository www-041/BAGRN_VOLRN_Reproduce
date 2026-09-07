"""Diagnostic script: check pixel-level alignment between two BAGRN images.

Loads img1.tif and img2.tif from BAGRN output, reprojects to a common grid,
and measures cross-correlation shifts at full resolution and in small patches.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from scipy.signal import correlate
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\two_image\B5'
IMG1 = os.path.join(BASE, 'bagrn', 'img1.tif')
IMG2 = os.path.join(BASE, 'bagrn', 'img2.tif')


def load_info(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        tr = src.transform
        crs = src.crs
        nd = src.nodata
        h, w = arr.shape
    return arr, tr, crs, nd, h, w


def bounds(tr, h, w):
    left = tr.c
    top = tr.f
    right = tr.c + tr.a * w
    bottom = tr.f + tr.e * h
    return left, bottom, right, top


def phase_corr_shift(ref, tgt):
    """Full cross-correlation shift (subpixel) between two 2D arrays.
    Returns (dy, dx, confidence) in pixels (target relative to ref).
    ref and tgt must be same shape.
    """
    h, w = ref.shape
    r = ref.copy()
    t = tgt.copy()

    mask = np.isfinite(r) & np.isfinite(t)  # Fixed: check finite, not > 0
    valid = mask.sum()
    if valid < 100:
        return 0.0, 0.0, 0.0, valid

    r[~mask] = 0
    t[~mask] = 0

    win_y = np.hanning(h)
    win_x = np.hanning(w)
    win = win_y[:, None] * win_x[None, :]

    r *= win
    t *= win

    cc = correlate(t, r, mode='full', method='fft')
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    cy, cx = peak
    center_y, center_x = h - 1, w - 1

    # Subpixel via 5x5 centroid
    dy_sub, dx_sub = 0.0, 0.0
    if 2 <= cy < cc.shape[0]-2 and 2 <= cx < cc.shape[1]-2:
        patch = cc[cy-2:cy+3, cx-2:cx+3]
        total = patch.sum()
        if total > 0:
            ys = np.arange(-2, 3).reshape(-1, 1)
            xs = np.arange(-2, 3).reshape(1, -1)
            wy = np.broadcast_to(ys, (5, 5)).astype(np.float64)
            wx = np.broadcast_to(xs, (5, 5)).astype(np.float64)
            dy_sub = (patch * wy).sum() / total
            dx_sub = (patch * wx).sum() / total

    shift_y = (cy - center_y) + dy_sub
    shift_x = (cx - center_x) + dx_sub

    cc_sorted = np.sort(cc.ravel())[-2:][::-1]
    conf = cc_sorted[0] / max(cc_sorted[1], 1e-10)
    conf = min(conf, 10.0)

    return shift_y, shift_x, conf, valid


def main():
    print("=" * 70)
    print("  ALIGNMENT DIAGNOSTIC: BAGRN img1 vs img2 (B5)")
    print("=" * 70)

    # --- Load ---
    arr1, tr1, crs1, nd1, h1, w1 = load_info(IMG1)
    arr2, tr2, crs2, nd2, h2, w2 = load_info(IMG2)

    b1 = bounds(tr1, h1, w1)
    b2 = bounds(tr2, h2, w2)

    print(f"\nImage 1: {h1}x{w1}, nodata={nd1}")
    print(f"  Transform: {tr1}")
    print(f"  Bounds: left={b1[0]:.4f}, bottom={b1[1]:.4f}, right={b1[2]:.4f}, top={b1[3]:.4f}")
    print(f"  Resolution: dx={tr1.a:.6f}, dy={tr1.e:.6f}")

    print(f"\nImage 2: {h2}x{w2}, nodata={nd2}")
    print(f"  Transform: {tr2}")
    print(f"  Bounds: left={b2[0]:.4f}, bottom={b2[1]:.4f}, right={b2[2]:.4f}, top={b2[3]:.4f}")
    print(f"  Resolution: dx={tr2.a:.6f}, dy={tr2.e:.6f}")

    # --- Geographic overlap ---
    ov_left = max(b1[0], b2[0])
    ov_right = min(b1[2], b2[2])
    ov_top = min(b1[3], b2[3])
    ov_bottom = max(b1[1], b2[1])

    print(f"\n--- Geographic Overlap ---")
    print(f"  left={ov_left:.6f}, right={ov_right:.6f}")
    print(f"  top={ov_top:.6f}, bottom={ov_bottom:.6f}")

    # With UTM: top (north) > bottom (south), so overlap exists when top > bottom
    if ov_left >= ov_right or ov_top <= ov_bottom:
        print("  ERROR: No geographic overlap!")
        return

    print(f"  Width:  {ov_right - ov_left:.6f} m")
    print(f"  Height: {ov_top - ov_bottom:.6f} m")

    # --- Reproject both to common grid in overlap region ---
    res = min(abs(tr1.a), abs(tr2.a))
    out_w = max(1, int(round((ov_right - ov_left) / res)))
    out_h = max(1, int(round((ov_top - ov_bottom) / res)))
    out_tr = Affine(res, 0, ov_left, 0, -res, ov_top)

    print(f"\n--- Common Grid ---")
    print(f"  Size: {out_w}x{out_h}, res={res:.6f}")

    proj1 = np.zeros((out_h, out_w), dtype=np.float64)
    proj2 = np.zeros((out_h, out_w), dtype=np.float64)

    reproject(
        source=arr1, destination=proj1,
        src_transform=tr1, src_crs=crs1,
        dst_transform=out_tr, dst_crs=crs1,
        resampling=Resampling.bilinear,
    )
    reproject(
        source=arr2, destination=proj2,
        src_transform=tr2, src_crs=crs2,
        dst_transform=out_tr, dst_crs=crs2,
        resampling=Resampling.bilinear,
    )

    # Mask valid pixels
    mask1 = proj1 != nd1 if nd1 is not None else proj1 != 0
    mask2 = proj2 != nd2 if nd2 is not None else proj2 != 0
    valid_mask = mask1 & mask2

    print(f"  Valid overlap pixels: {valid_mask.sum()} / {out_h * out_w} "
          f"({100*valid_mask.sum()/(out_h*out_w):.1f}%)")

    if valid_mask.sum() < 100:
        print("  ERROR: Too few valid pixels in overlap.")
        return

    # --- Full-resolution shift ---
    print(f"\n--- Full-Resolution Cross-Correlation ---")
    # Use only the valid bounding box
    rows_valid = np.any(valid_mask, axis=1)
    cols_valid = np.any(valid_mask, axis=0)
    r0, r1_idx = np.where(rows_valid)[0][[0, -1]]
    c0, c1_idx = np.where(cols_valid)[0][[0, -1]]

    # Extract valid subregion (pad a bit for correlation)
    pad = 10
    r0p = max(0, r0 - pad)
    r1p = min(out_h, r1_idx + pad + 1)
    c0p = max(0, c0 - pad)
    c1p = min(out_w, c1_idx + pad + 1)

    sub1 = proj1[r0p:r1p, c0p:c1p]
    sub2 = proj2[r0p:r1p, c0p:c1p]
    sub_mask = valid_mask[r0p:r1p, c0p:c1p]

    # Zero out non-overlap in both
    s1 = sub1.copy()
    s2 = sub2.copy()
    s1[~sub_mask] = 0
    s2[~sub_mask] = 0

    dy, dx, conf, nvalid = phase_corr_shift(s1, s2)
    print(f"  Shift (target relative to ref): dy={dy:.4f} px, dx={dx:.4f} px")
    print(f"  Confidence: {conf:.2f}")
    print(f"  Valid pixels used: {nvalid}")
    print(f"  Shift in geographic units: dy={dy*res:.6f}, dx={dx*res:.6f}")

    # --- Local patch analysis ---
    print(f"\n--- Local Patch Shifts (256x256) ---")
    patch_size = 256
    step = 128  # 50% overlap

    local_shifts = []
    patches_checked = 0

    for pr in range(r0, min(r1_idx - patch_size + 1, out_h - patch_size + 1), step):
        for pc in range(c0, min(c1_idx - patch_size + 1, out_w - patch_size + 1), step):
            p1 = proj1[pr:pr+patch_size, pc:pc+patch_size]
            p2 = proj2[pr:pr+patch_size, pc:pc+patch_size]
            pm = valid_mask[pr:pr+patch_size, pc:pc+patch_size]

            if pm.sum() < patch_size * patch_size * 0.3:
                continue

            #地理坐标 of patch center
            geo_x = out_tr.c + (pc + patch_size/2) * out_tr.a
            geo_y = out_tr.f + (pr + patch_size/2) * out_tr.e

            p1z = p1.copy(); p1z[~pm] = 0
            p2z = p2.copy(); p2z[~pm] = 0

            dy_p, dx_p, conf_p, nv = phase_corr_shift(p1z, p2z)
            local_shifts.append((geo_x, geo_y, dy_p, dx_p, conf_p, nv))
            patches_checked += 1

    print(f"  Patches checked: {patches_checked}")

    if local_shifts:
        shifts_arr = np.array([(s[2], s[3]) for s in local_shifts])
        print(f"\n  {'Patch center (lon, lat)':<35} {'dy(px)':>8} {'dx(px)':>8} {'conf':>6} {'nvalid':>7}")
        print(f"  {'-'*70}")
        for gx, gy, dy_p, dx_p, conf_p, nv in local_shifts:
            print(f"  ({gx:.4f}, {gy:.4f}){'':>17} {dy_p:8.4f} {dx_p:8.4f} {conf_p:6.2f} {nv:7d}")

        print(f"\n  Statistics across {len(local_shifts)} patches:")
        print(f"    dy: mean={shifts_arr[:,0].mean():.4f}, std={shifts_arr[:,0].std():.4f}, "
              f"min={shifts_arr[:,0].min():.4f}, max={shifts_arr[:,0].max():.4f}")
        print(f"    dx: mean={shifts_arr[:,1].mean():.4f}, std={shifts_arr[:,1].std():.4f}, "
              f"min={shifts_arr[:,1].min():.4f}, max={shifts_arr[:,1].max():.4f}")

        # Check if shifts are spatially varying
        if shifts_arr[:,0].std() > 0.2 or shifts_arr[:,1].std() > 0.2:
            print(f"\n  WARNING: Shifts vary significantly across the overlap!")
            print(f"  This suggests non-rigid distortion or residual geometric error.")
        else:
            print(f"\n  Shifts are relatively uniform (< 0.2 px std).")
            print(f"  Remaining misalignment likely from: resampling artifacts, intensity differences,")
            print(f"  or the cutline placement in the mosaic.")
    else:
        print("  No patches could be analyzed.")

    # --- Difference image stats ---
    print(f"\n--- Pixel Difference Stats (overlap area) ---")
    diff = proj1[valid_mask] - proj2[valid_mask]
    print(f"  Mean diff:  {diff.mean():.4f}")
    print(f"  Std diff:   {diff.std():.4f}")
    print(f"  Abs mean:   {np.abs(diff).mean():.4f}")
    print(f"  RMSE:       {np.sqrt(np.mean(diff**2)):.4f}")
    print(f"  Max abs:    {np.abs(diff).max():.4f}")

    # --- Mosaic cutline analysis ---
    print(f"\n--- Mosaic Cutline Analysis ---")
    overlap_cols = np.where(valid_mask.any(axis=0))[0]
    if len(overlap_cols) > 0:
        col_start = overlap_cols[0]
        col_end = overlap_cols[-1]
        mid_col = (col_start + col_end) // 2
        mid_geo_x = out_tr.c + mid_col * out_tr.a
        print(f"  Overlap column range: {col_start} to {col_end}")
        print(f"  Midline column: {mid_col} (geo x={mid_geo_x:.6f})")
        print(f"  Cutline at: {mid_geo_x:.6f} longitude")
    else:
        print("  No overlap columns found.")

    print(f"\n{'='*70}")
    print("  DIAGNOSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
