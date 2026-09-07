"""
λ 参数敏感性分析脚本

测试不同 λ 值对 VOLRN 结果的影响，输出每个 λ 的指标和块系数分布。
用法: E:\\python3\\python3.exe scripts/param_sensitivity.py
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import read_geotiff, write_geotiff
from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.overlap import get_overlap_window
from src.metrics import compute_all
from src.mosaic import create_mosaic

OUTPUT = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\param_sensitivity'
os.makedirs(OUTPUT, exist_ok=True)

IMG1 = r'data/input/20251208025051/DZ01V_L2_E118.9_N30.3_20251208025051_01_T1/DZ01V_L2_E118.9_N30.3_20251208025051_01_T1_{}.TIF'
IMG2 = r'data/input/20251214025247/DZ01V_L2_E119.0_N30.3_20251214025247_01_T1/DZ01V_L2_E119.0_N30.3_20251214025247_01_T1_{}.TIF'

BANDS = ['B5', 'B8', 'B14']
LAMBDAS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def process_band(band):
    print(f"\n{'='*60}")
    print(f"  Processing {band}")
    print(f"{'='*60}")

    arr1, tr1, crs, nd1 = read_geotiff(IMG1.format(band))
    arr2, tr2, _, nd2 = read_geotiff(IMG2.format(band))
    while arr1.ndim > 2: arr1 = arr1[0]
    while arr2.ndim > 2: arr2 = arr2[0]

    # Co-registration
    from src.coregistration import compute_shifts_from_overlap
    shift_y, shift_x, conf, block_stats = compute_shifts_from_overlap(
        arr1, tr1, arr2, tr2, nd1, nd2)
    if abs(shift_y) > 0.05 or abs(shift_x) > 0.05:
        from scipy.ndimage import shift as ndimage_shift
        arr2 = ndimage_shift(arr2, [shift_y, shift_x], order=1, mode='constant', cval=nd2)
        from rasterio.transform import Affine
        tr2 = Affine(tr2.a, tr2.b, tr2.c + shift_x * tr2.a, tr2.d, tr2.e, tr2.f + shift_y * tr2.e)

    arrs_3d = [arr1[np.newaxis, :, :], arr2[np.newaxis, :, :]]
    nodatas = [nd1, nd2]

    win2 = get_overlap_window(
        (tr1.c, tr1.f + tr1.e*arr1.shape[0], tr1.c + tr1.a*arr1.shape[1], tr1.f), tr1,
        (tr2.c, tr2.f + tr2.e*arr2.shape[0], tr2.c + tr2.a*arr2.shape[1], tr2.f), tr2)

    overlaps = []
    if win2 is not None:
        wi2, wj2 = win2
        n = (wi2[1]-wi2[0])*(wi2[3]-wi2[2])
        if n > 100:
            overlaps = [{'idx_i': 0, 'idx_j': 1, 'window_i': wi2, 'window_j': wj2}]

    # BAGRN
    t0 = time.time()
    bagrn_result, _, _ = bagrn_normalize(arrs_3d, nodatas, overlaps, control_idx=0)
    bagrn_metrics = compute_all(arrs_3d, bagrn_result, nodatas, overlaps, [0])  # Fixed: before=original
    bagrn_time = time.time() - t0
    print(f"  BAGRN ({bagrn_time:.1f}s): RDOA={bagrn_metrics['rdoa']:.6f}")

    transforms = [tr1, tr2]
    def get_bounds(arr, tr):
        r, c = arr.shape
        return (tr.c, tr.f + tr.e*r, tr.c + tr.a*c, tr.f)
    bounds = [get_bounds(arr1, tr1), get_bounds(arr2, tr2)]

    results = []

    for lam in LAMBDAS:
        print(f"\n  --- λ = {lam} ---")
        t0 = time.time()
        volrn_result, volrn_coeffs = volrn_normalize(
            bagrn_result, transforms, bounds, nodatas,
            block_size_pixels=200, lambda_param=lam, rho=1.0,
            max_iter=200, tol=1e-4, verbose=False)
        volrn_time = time.time() - t0

        volrn_metrics = compute_all(arrs_3d, volrn_result, nodatas, overlaps, [0])  # Fixed: before=original

        # Block coefficient diagnostics
        a_diag = ""
        b_diag = ""
        if volrn_coeffs is not None and volrn_coeffs.size > 0:
            a_vals = volrn_coeffs[0, :, 0]
            b_vals = volrn_coeffs[0, :, 1]
            a_diag = f"a=[{a_vals.min():.4f},{np.median(a_vals):.4f},{a_vals.max():.4f}]"
            b_diag = f"b=[{b_vals.min():.4f},{np.median(b_vals):.4f},{b_vals.max():.4f}]"

        print(f"  VOLRN ({volrn_time:.1f}s): RDOA={volrn_metrics['rdoa']:.6f}  "
              f"ADM={volrn_metrics['adm']:.6f}  ADSD={volrn_metrics['adsd']:.6f}  "
              f"CD={volrn_metrics['cd']:.10f}  GL={volrn_metrics['gl']:.10f}")
        print(f"  Coefficients: {a_diag}  {b_diag}")

        results.append({
            'lambda': lam,
            'time': volrn_time,
            'metrics': volrn_metrics,
            'volrn_result': volrn_result,
        })

    # Print comparison table
    print(f"\n  --- λ Comparison for {band} ---")
    print(f"  {'λ':>6}  {'RDOA':>12}  {'ADM':>12}  {'ADSD':>12}  {'CD':>12}  {'GL':>12}")
    print(f"  {'-'*70}")
    print(f"  {'BAGRN':>6}  {bagrn_metrics['rdoa']:>12.6f}  {bagrn_metrics['adm']:>12.6f}  "
          f"{bagrn_metrics['adsd']:>12.6f}  {bagrn_metrics['cd']:>12.10f}  {bagrn_metrics['gl']:>12.10f}")
    for r in results:
        m = r['metrics']
        print(f"  {r['lambda']:>6}  {m['rdoa']:>12.6f}  {m['adm']:>12.6f}  "
              f"{m['adsd']:>12.6f}  {m['cd']:>12.10f}  {m['gl']:>12.10f}")

    return results


if __name__ == '__main__':
    t_start = time.time()
    all_results = {}
    for band in BANDS:
        try:
            r = process_band(band)
            if r:
                all_results[band] = r
        except Exception as e:
            print(f"\n  ERROR: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Total time: {time.time()-t_start:.1f}s")
    print(f"  Output: {OUTPUT}")
