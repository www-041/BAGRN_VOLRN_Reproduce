"""
Two-image B14 pipeline: co-register → BAGRN → VOLRN → mosaic
"""
import os, sys, time, json, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import read_geotiff, write_geotiff
from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.overlap import get_overlap_window
from src.mosaic import create_mosaic
from src.metrics import compute_all

OUTPUT = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\two_image'
os.makedirs(OUTPUT, exist_ok=True)

IMG1 = r'data/input/20251208025051/DZ01V_L2_E118.9_N30.3_20251208025051_01_T1/DZ01V_L2_E118.9_N30.3_20251208025051_01_T1_{}.TIF'
IMG2 = r'data/input/20251214025247/DZ01V_L2_E119.0_N30.3_20251214025247_01_T1/DZ01V_L2_E119.0_N30.3_20251214025247_01_T1_{}.TIF'
BANDS = ['B14']


def compute_overlap(arr1, tr1, arr2, tr2):
    rows1, cols1 = arr1.shape
    rows2, cols2 = arr2.shape
    b1 = (tr1.c, tr1.f + tr1.e * rows1, tr1.c + tr1.a * cols1, tr1.f)
    b2 = (tr2.c, tr2.f + tr2.e * rows2, tr2.c + tr2.a * cols2, tr2.f)
    return get_overlap_window(b1, tr1, b2, tr2)


def compute_bounds(tr, shape):
    h, w = shape
    return (tr.c, tr.f + tr.e * h, tr.c + tr.a * w, tr.f)


def save_csv(path, rows, header):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def process_band(band):
    print(f"\n{'='*70}")
    print(f"  Processing {band}")
    print(f"{'='*70}")

    # ---- Load ----
    arr1, tr1, crs1, nd1 = read_geotiff(IMG1.format(band))
    arr2_orig, tr2_orig, crs2, nd2 = read_geotiff(IMG2.format(band))
    while arr1.ndim > 2: arr1 = arr1[0]
    while arr2_orig.ndim > 2: arr2_orig = arr2_orig[0]

    print(f"  Image 1 (ref): {arr1.shape}, nodata={nd1}")
    print(f"  Image 2 (tgt): {arr2_orig.shape}, nodata={nd2}")

    if crs1 != crs2:
        raise ValueError(f"CRS不一致: {crs1} vs {crs2}")
    crs = crs1

    # ================================================================
    # Step 1: Co-registration (existing logic, abbreviated)
    # ================================================================
    print("\n  --- Co-registration ---")
    from src.coregistration import (
        collect_block_matches, compute_model_residual_stats,
        fit_affine_ransac, compute_shifts_from_overlap,
        build_local_residual_controls, fit_local_rbf,
        spatial_cross_validate, warp_with_displacement_field,
    )

    matches, screening = collect_block_matches(arr1, tr1, arr2_orig, tr2_orig, nd1, nd2)
    lag_y, lag_x, conf, _ = compute_shifts_from_overlap(arr1, tr1, arr2_orig, tr2_orig, nd1, nd2)
    print(f"  Translation: dy={lag_y:.4f}, dx={lag_x:.4f}")

    ctrl = build_local_residual_controls(matches, lag_x, lag_y, confidence_threshold=0.75)
    print(f"  Local control points: {ctrl['n_valid']}")

    use_local = False
    local_rbf_dx = local_rbf_dy = local_coord_range = None
    best_smoothing = None

    if ctrl['n_valid'] >= 30:
        local_cv_summary, _ = spatial_cross_validate(
            ctrl['points_xy'], ctrl['residual_dx'], ctrl['residual_dy'],
            lag_x, lag_y, matches)
        if local_cv_summary:
            trans_cv_p95 = local_cv_summary.get('translation', {}).get('p95', float('inf'))
            best_model = 'translation'
            best_p95 = trans_cv_p95
            for m_name in ['affine', 'rbf']:
                if m_name in local_cv_summary and local_cv_summary[m_name]['p95'] < best_p95:
                    best_p95 = local_cv_summary[m_name]['p95']
                    best_model = m_name
            if best_model == 'rbf':
                improvement = 1.0 - best_p95 / max(trans_cv_p95, 1e-10)
                if improvement >= 0.10:
                    best_rbf_p95 = float('inf')
                    for sm in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
                        try:
                            rbf_dx, rbf_dy, cmin, cmax = fit_local_rbf(
                                ctrl['points_xy'], ctrl['residual_dx'], ctrl['residual_dy'],
                                smoothing=sm, neighbors=min(20, ctrl['n_valid']))
                            cv_p95 = spatial_cross_validate(
                                ctrl['points_xy'], ctrl['residual_dx'], ctrl['residual_dy'],
                                lag_x, lag_y, matches)[0].get('rbf', {}).get('p95', float('inf'))
                            if cv_p95 < best_rbf_p95:
                                best_rbf_p95 = cv_p95
                                best_smoothing = sm
                                local_rbf_dx = rbf_dx
                                local_rbf_dy = rbf_dy
                                local_coord_range = (cmin[0], cmin[1], cmax[0], cmax[1])
                        except Exception:
                            continue
                    if local_rbf_dx is not None:
                        use_local = True
                        print(f"  => LOCAL RBF (smoothing={best_smoothing})")

    # Apply transform
    h2, w2 = arr2_orig.shape
    local_dx_field = np.zeros((h2, w2), dtype=np.float64)
    local_dy_field = np.zeros((h2, w2), dtype=np.float64)

    if use_local and ctrl['n_valid'] >= 3:
        from scipy.spatial import Delaunay
        hull = Delaunay(ctrl['points_xy'])
        yy, xx = np.mgrid[0:h2, 0:w2]
        test_points = np.column_stack([xx.ravel(), yy.ravel()])
        chunk = 50000
        for i in range(0, len(test_points), chunk):
            cp = test_points[i:i+chunk]
            in_hull = hull.find_simplex(cp) >= 0
            tx = (cp[:, 0] - local_coord_range[0]) / max(local_coord_range[2] - local_coord_range[0], 1e-10)
            ty = (cp[:, 1] - local_coord_range[1]) / max(local_coord_range[3] - local_coord_range[1], 1e-10)
            pts_norm = np.column_stack([tx, ty])
            cdx = np.clip(local_rbf_dx(pts_norm), -2.5, 2.5)
            cdy = np.clip(local_rbf_dy(pts_norm), -2.5, 2.5)
            cdx[~in_hull] = 0
            cdy[~in_hull] = 0
            local_dx_field.ravel()[i:i+chunk] = cdx
            local_dy_field.ravel()[i:i+chunk] = cdy
        arr2_coreg = warp_with_displacement_field(arr2_orig, lag_x, lag_y, local_dx_field, local_dy_field, nd2)
    else:
        from scipy.ndimage import shift as ndimage_shift
        arr2_coreg = arr2_orig.astype(np.float64)
        if abs(lag_y) > 0.01 or abs(lag_x) > 0.01:
            arr2_coreg = ndimage_shift(arr2_coreg, [lag_y, lag_x], order=1, mode='constant', cval=nd2, prefilter=False)

    tr2_coreg = tr2_orig

    # ================================================================
    # Step 2: BAGRN + VOLRN + Mosaic
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  B14 radiometric normalization and mosaic")
    print(f"{'='*70}")

    arrs_registered = [arr1[np.newaxis, :, :], arr2_coreg[np.newaxis, :, :]]
    nodatas = [nd1, nd2]
    transforms_all = [tr1, tr2_coreg]
    bounds_all = [compute_bounds(tr1, arr1.shape), compute_bounds(tr2_coreg, arr2_coreg.shape)]

    win2 = compute_overlap(arr1, tr1, arr2_coreg, tr2_coreg)
    overlaps = []
    if win2 is not None:
        wi2, wj2 = win2
        n = (wi2[1] - wi2[0]) * (wi2[3] - wi2[2])
        if n > 100:
            overlaps = [{'idx_i': 0, 'idx_j': 1, 'window_i': wi2, 'window_j': wj2}]
    print(f"  Overlap pairs: {len(overlaps)}")

    # --- Original metrics ---
    original_metrics = compute_all(arrs_registered, arrs_registered, nodatas, overlaps, [0])

    # --- BAGRN ---
    t_bagrn = time.time()
    bagrn_result, bagrn_coeffs, bagrn_info = bagrn_normalize(
        arrs_registered, nodatas, overlaps, control_idx=0)
    t_bagrn = time.time() - t_bagrn
    bagrn_metrics = compute_all(arrs_registered, bagrn_result, nodatas, overlaps, [0])

    # --- VOLRN ---
    t_volrn = time.time()
    volrn_result, volrn_coeffs = volrn_normalize(
        bagrn_result, transforms_all, bounds_all, nodatas,
        block_size_pixels=200, lambda_param=0.1, rho=1.0,
        max_iter=200, tol=1e-4, verbose=False)
    t_volrn = time.time() - t_volrn
    volrn_metrics = compute_all(arrs_registered, volrn_result, nodatas, overlaps, [0])

    # --- Print table ---
    def _v(m, key):
        return m.get(key, float('nan'))

    print(f"\n{'='*70}")
    print(f"  B14 radiometric normalization and mosaic")
    print(f"{'='*70}")
    print(f"  {'Metric':<12s} {'Original':>12s} {'BAGRN':>12s} {'VOLRN':>12s}")
    print(f"  {'-'*48}")
    for key, label in [('adm', 'ADM'), ('adsd', 'ADSD'), ('rdoa', 'RDOA'), ('ave', 'Ave')]:
        v0 = _v(original_metrics, key)
        v1 = _v(bagrn_metrics, key)
        v2 = _v(volrn_metrics, key)
        print(f"  {label:<12s} {v0:12.6f} {v1:12.6f} {v2:12.6f}")
    print(f"\n  BAGRN time: {t_bagrn:.1f}s")
    print(f"  VOLRN time: {t_volrn:.1f}s")

    # --- Mosaics ---
    out_dir = os.path.join(OUTPUT, band)

    path_bagrn = os.path.join(out_dir, f'mosaic_{band}_bagrn.tif')
    t0 = time.time()
    create_mosaic(bagrn_result, transforms_all, crs, nodatas, path_bagrn)
    # NaN/Inf check
    _check_mosaic(path_bagrn)
    print(f"\n  BAGRN mosaic: {path_bagrn} ({os.path.getsize(path_bagrn)/1024/1024:.1f} MB, {time.time()-t0:.1f}s)")

    path_volrn = os.path.join(out_dir, f'mosaic_{band}_volrn.tif')
    t0 = time.time()
    create_mosaic(volrn_result, transforms_all, crs, nodatas, path_volrn)
    _check_mosaic(path_volrn)
    print(f"  VOLRN mosaic: {path_volrn} ({os.path.getsize(path_volrn)/1024/1024:.1f} MB, {time.time()-t0:.1f}s)")

    return {
        'quality_status': 'DONE',
        'metrics': {
            'original': original_metrics,
            'bagrn': bagrn_metrics,
            'volrn': volrn_metrics,
        },
    }


def _check_mosaic(path):
    """检查镶嵌图中是否存在 NaN 或 Inf，若存在则抛出错误。"""
    from src.io_utils import read_geotiff
    arr, _, _, nd = read_geotiff(path)
    if arr.ndim > 2:
        arr = arr[0]
    valid = np.isfinite(arr) & (arr != nd)
    nan_count = np.sum(~np.isfinite(arr))
    inf_count = np.sum(np.isinf(arr))
    if nan_count > 0 or inf_count > 0:
        raise RuntimeError(f"Mosaic contains NaN={nan_count}, Inf={inf_count}: {path}")


if __name__ == '__main__':
    t_start = time.time()
    all_results = {}
    for band in BANDS:
        try:
            r = process_band(band)
            if r:
                all_results[band] = r
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}")

    with open(os.path.join(OUTPUT, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nTotal: {time.time()-t_start:.1f}s")
    print(f"Output: {OUTPUT}")
