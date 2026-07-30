"""Test VOLRN normalization fix on DZ01V and DZ01S data."""
import os
import sys
import glob
import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import read_geotiff
from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.overlap import get_overlap_window, intersection_bounds
from src.metrics import compute_all


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


def run_two_images(name, dir1, dir2, band_idx=0, crop_size=512):
    """Test on two overlapping images from different acquisitions."""
    print(f'\n{"="*60}')
    print(f'  {name}')
    print(f'{"="*60}')

    arrays = []
    transforms = []
    bounds_list = []
    nodata_list = []

    for d in [dir1, dir2]:
        tifs = sorted(glob.glob(os.path.join(d, '*.TIF')))
        tifs = [f for f in tifs if '_PAN' not in os.path.basename(f).upper()]
        f = tifs[min(band_idx, len(tifs) - 1)]
        arr, tr, nd, crs = read_geotiff(f)
        arrays.append(arr)
        transforms.append(tr)
        nodata_list.append(nd)
        bnd = (tr.c, tr.f + tr.e * arr.shape[1], tr.c + tr.a * arr.shape[2], tr.f)
        bounds_list.append(bnd)
        print(f'  {os.path.basename(f)}: {arr.shape}, bounds=({bnd[0]:.0f},{bnd[1]:.0f})-({bnd[2]:.0f},{bnd[3]:.0f})')

    # Crop to intersection
    common = bounds_list[0]
    for i in range(1, len(bounds_list)):
        common = intersection_bounds(common, bounds_list[i])
    print(f'  Intersection: ({common[0]:.0f},{common[1]:.0f})-({common[2]:.0f},{common[3]:.0f})')

    for i in range(len(arrays)):
        arr, tr = arrays[i], transforms[i]
        rows, cols = arr.shape[1], arr.shape[2]
        c_min, r_min = ~tr * (common[0], common[3])
        c_max, r_max = ~tr * (common[2], common[1])
        r_s = max(0, int(round(min(r_min, r_max))))
        r_e = min(rows, int(round(max(r_min, r_max))))
        c_s = max(0, int(round(min(c_min, c_max))))
        c_e = min(cols, int(round(max(c_min, c_max))))
        if r_e - r_s > crop_size:
            mid = (r_s + r_e) // 2
            r_s = max(0, mid - crop_size // 2)
            r_e = min(rows, r_s + crop_size)
        if c_e - c_s > crop_size:
            mid = (c_s + c_e) // 2
            c_s = max(0, mid - crop_size // 2)
            c_e = min(cols, c_s + crop_size)
        arrays[i] = arr[:, r_s:r_e, c_s:c_e]
        transforms[i] = rasterio.Affine(tr.a, tr.b, tr.c + c_s * tr.a,
                                        tr.d, tr.e, tr.f + r_s * tr.e)
        bounds_list[i] = (transforms[i].c,
                          transforms[i].f + transforms[i].e * (r_e - r_s),
                          transforms[i].c + transforms[i].a * (c_e - c_s),
                          transforms[i].f)

    print(f'  Cropped: {arrays[0].shape}')
    overlaps = compute_overlaps(arrays, transforms, bounds_list)
    print(f'  {len(overlaps)} overlap pairs')
    if not overlaps:
        return

    bands = list(range(arrays[0].shape[0]))

    # Check B matrix structure
    from src.volrn import image_blocking, _build_volrn_system
    blocks, pairs = image_blocking(
        arrays, transforms, bounds_list, nodata_list, 200, bands)
    print(f'\n  Blocks: {len(blocks)}, Pairs: {len(pairs)}')
    if blocks:
        B, A, b_vec, mu_scale = _build_volrn_system(blocks, pairs, 0)
        print(f'  B shape: {B.shape}, nnz: {B.nnz}')
        print(f'  A shape: {A.shape}, nnz: {A.nnz}')
        print(f'  mu_scale: {mu_scale:.4f}')
        # Check what Bx looks like for identity
        x_id = np.ones(2 * len(blocks))
        x_id[1::2] = 0.0
        Bx = B @ x_id
        Ax_b = A @ x_id - b_vec
        print(f'  ||Bx|| (identity): {np.linalg.norm(Bx):.6f}')
        print(f'  ||Ax-b|| (identity): {np.linalg.norm(Ax_b):.6f}')
        print(f'  Bx range: [{Bx.min():.6f}, {Bx.max():.6f}]')
        print(f'  Ax-b range: [{Ax_b.min():.6f}, {Ax_b.max():.6f}]')
        # Check block stats
        mu_vals = [blocks[k].mu[0] for k in range(len(blocks))]
        sg_vals = [blocks[k].sigma[0] for k in range(len(blocks))]
        print(f'  Block mu range: [{min(mu_vals):.1f}, {max(mu_vals):.1f}]')
        print(f'  Block sigma range: [{min(sg_vals):.1f}, {max(sg_vals):.1f}]')

    # Baseline
    baseline = compute_all(arrays, arrays, nodata_list, overlaps, bands)

    # VOLRN direct on original
    print('\n  VOLRN direct on original...')
    vd, _ = volrn_normalize(
        [a.copy() for a in arrays], transforms, bounds_list, nodata_list,
        block_size_pixels=200, lambda_param=0.5, rho=1.0,
        max_iter=200, tol=1e-4, verbose=True,
    )
    vd_m = compute_all(vd, arrays, nodata_list, overlaps, bands)

    # BAGRN
    print('  BAGRN...')
    bg, _, _ = bagrn_normalize(arrays, nodata_list, overlaps, control_idx=0)
    bg_m = compute_all(bg, arrays, nodata_list, overlaps, bands)

    # VOLRN on BAGRN
    print('  VOLRN on BAGRN...')
    vr, _ = volrn_normalize(
        bg, transforms, bounds_list, nodata_list,
        block_size_pixels=200, lambda_param=0.5, rho=1.0,
        max_iter=200, tol=1e-4, verbose=True,
    )
    vr_m = compute_all(vr, arrays, nodata_list, overlaps, bands)

    # Table
    print(f'\n  {"Metric":<12} {"Baseline":>12} {"VOLRN_direct":>14} {"BAGRN":>12} {"VOLRN_on_BAGRN":>16}')
    print(f'  {"-"*68}')
    for k in baseline:
        print(f'  {k:<12} {baseline[k]:>12.4f} {vd_m[k]:>14.4f} {bg_m[k]:>12.4f} {vr_m[k]:>16.4f}')


if __name__ == '__main__':
    base = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\input'

    # DZ01V: 20251208 and 20251214 overlap well
    run_two_images(
        'DZ01V Band 0 (2 images)',
        os.path.join(base, '20251208025051', 'DZ01V_L2_E118.9_N30.3_20251208025051_01_T1'),
        os.path.join(base, '20251214025247', 'DZ01V_L2_E119.0_N30.3_20251214025247_01_T1'),
        band_idx=0, crop_size=512,
    )

    # DZ01S: same dates
    run_two_images(
        'DZ01S Band 0 (2 images)',
        os.path.join(base, '20251208025051', 'DZ01S_L2_E118.8_N30.3_20251208025051_01_T1'),
        os.path.join(base, '20251214025247', 'DZ01S_L2_E119.0_N30.3_20251214025247_01_T1'),
        band_idx=0, crop_size=512,
    )
