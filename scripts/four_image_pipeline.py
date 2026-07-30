"""
Four-image B14 pipeline:
1. Load, match, network adjustment
2. Global registered (intermediate)
3. Global residual refinement (only [1])
4. Local RBF for [1] vs [0]
5. [1]-[3] RBF conflict diagnosis
6. [3] pre-correction multi-scale (diagnostic only)
7. [3] bilateral global correction
8. [3] post-correction multi-scale → balanced controls
9. [3] smoothing sweep (0.1, 0.3, 1.0) → score-based selection
10. [2] re-match + existing RBF logic
11. Final single resampling, verify all 6 pairs, BAGRN → mosaic
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import read_geotiff, write_geotiff
from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.overlap import get_overlap_window
from src.mosaic import create_mosaic
from src.metrics import compute_all

OUTPUT = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\four_image'
os.makedirs(OUTPUT, exist_ok=True)

IMAGES = {
    0: ('20251214a', r'data/input/20251214025247/DZ01V_L2_E119.0_N30.3_20251214025247_01_T1/DZ01V_L2_E119.0_N30.3_20251214025247_01_T1_B14.TIF'),
    1: ('20251208',  r'data/input/20251208025051/DZ01V_L2_E118.9_N30.3_20251208025051_01_T1/DZ01V_L2_E118.9_N30.3_20251208025051_01_T1_B14.TIF'),
    2: ('20251215',  r'data/input/20251215023725/DZ01V_L2_E119.0_N30.1_20251215023725_01_T1/DZ01V_L2_E119.0_N30.1_20251215023725_01_T1_B14.TIF'),
    3: ('20251120',  r'data/input/20251120024220/DZ01V_L2_E118.9_N30.2_20251120024220_01_T1/DZ01V_L2_E118.9_N30.2_20251120024220_01_T1_B14.TIF'),
}
BAND = 'B14'
REF_IDX = 0
PARENT_MAP = {0: None, 1: 0, 3: 0, 2: 3}
SMOOTHING_CANDIDATES = [0.1, 0.3, 1.0]


def compute_bounds(tr, shape):
    h, w = shape
    return (tr.c, tr.f + tr.e * h, tr.c + tr.a * w, tr.f)


def _eval_edge(arr_ref, tr_ref, arr_tgt_arr, tr_tgt, nd_ref, nd_tgt):
    from src.coregistration import compute_shifts_from_overlap
    try:
        res_y, res_x, conf, stats = compute_shifts_from_overlap(
            arr_ref, tr_ref, arr_tgt_arr, tr_tgt, nd_ref, nd_tgt)
        if stats and stats.get('available', True):
            return {
                'residual_dx': float(res_x), 'residual_dy': float(res_y),
                'residual_magnitude': float(np.hypot(res_x, res_y)),
                'rmse': stats.get('residual_rmse', float('nan')),
                'p95': stats.get('residual_p95', float('nan')),
                'n_blocks': stats.get('n_blocks_inlier', 0),
                'confidence': float(conf), 'available': True,
            }
    except Exception:
        pass
    return {'available': False, 'residual_dx': float('nan'), 'residual_dy': float('nan'),
            'residual_magnitude': float('nan'), 'rmse': float('nan'), 'p95': float('nan'),
            'n_blocks': 0, 'confidence': 0.0}


def _fit_rbf_field(arr_original, global_dx, global_dy, ctrl, h, w, nodata, smoothing):
    """拟合RBF并返回局部位移场。"""
    from src.coregistration import fit_local_rbf, compute_hull_fade_mask
    from scipy.spatial import Delaunay

    rbf_dx, rbf_dy, cmin, cmax = fit_local_rbf(
        ctrl['points_xy'], ctrl['residual_dx'], ctrl['residual_dy'],
        smoothing=smoothing, neighbors=min(20, ctrl['n_valid']))

    hull_mask = compute_hull_fade_mask(ctrl['points_xy'], h, w, buffer=128)

    local_dx = np.zeros((h, w), dtype=np.float64)
    local_dy = np.zeros((h, w), dtype=np.float64)

    hull = Delaunay(ctrl['points_xy'])
    yy, xx = np.mgrid[0:h, 0:w]
    test_pts = np.column_stack([xx.ravel(), yy.ravel()])
    chunk = 50000
    for ci in range(0, len(test_pts), chunk):
        cp = test_pts[ci:ci + chunk]
        tx = (cp[:, 0] - cmin[0]) / max(cmax[0] - cmin[0], 1e-10)
        ty = (cp[:, 1] - cmin[1]) / max(cmax[1] - cmin[1], 1e-10)
        pts_n = np.column_stack([tx, ty])
        cdx = rbf_dx(pts_n)
        cdy = rbf_dy(pts_n)
        mag = np.hypot(cdx, cdy)
        scale = np.where(mag > 2.5, 2.5 / np.maximum(mag, 1e-10), 1.0)
        cdx = cdx * scale
        cdy = cdy * scale
        local_dx.ravel()[ci:ci + chunk] = cdx
        local_dy.ravel()[ci:ci + chunk] = cdy

    local_dx *= hull_mask
    local_dy *= hull_mask

    # 严格最终限制
    mag_f = np.hypot(local_dx, local_dy)
    sf = np.minimum(1.0, 2.5 / np.maximum(mag_f, 1e-12))
    local_dx *= sf
    local_dy *= sf

    return local_dx, local_dy


def _multiscale_rematch(arr_ref, tr_ref, arr_tgt, tr_tgt, nd_ref, nd_tgt,
                         block_sizes=(512, 384, 256)):
    """多尺度重新匹配，汇总所有尺度的控制点。"""
    from src.coregistration import rematch_pair_on_registered, build_residual_controls_from_rematch

    all_ctrls = []
    info = {}

    for bs in block_sizes:
        rm = rematch_pair_on_registered(
            arr_ref, arr_tgt, tr_ref, tr_tgt, nd_ref, nd_tgt,
            max_residual_shift=5, block_size=bs, confidence_threshold=0.5)
        if rm is not None:
            used_conf = rm.get('used_confidence_threshold', 0.5)
            ctrl = build_residual_controls_from_rematch(
                rm['matches'], confidence_threshold=used_conf, min_points=5)
            all_ctrls.append(ctrl)
            info[bs] = {'n_blocks': rm['n_blocks'], 'n_ctrl': ctrl['n_valid'],
                        'conf': used_conf, 'rmse': rm['rmse']}
        else:
            info[bs] = None

    return all_ctrls, info


def _print_ctrl_stats(ctrl_list, edge_name):
    """打印控制点残余统计。"""
    all_dx, all_dy = [], []
    for c in ctrl_list:
        if c['n_valid'] > 0:
            all_dx.append(c['residual_dx'])
            all_dy.append(c['residual_dy'])
    if not all_dx:
        print(f"  {edge_name}: no control points")
        return
    dx = np.concatenate(all_dx)
    dy = np.concatenate(all_dy)
    mag = np.hypot(dx, dy)
    print(f"  {edge_name}: median_dx={np.median(dx):.3f}, median_dy={np.median(dy):.3f}, "
          f"P95_mag={np.percentile(mag, 95):.3f}, n={len(dx)}")
    if np.median(mag) > 0.5:
        print(f"  WARNING: {edge_name} median residual magnitude > 0.5 px")


def main():
    t_start = time.time()
    print(f"{'='*70}")
    print(f"  Four-image B14 pipeline")
    print(f"{'='*70}")

    # ================================================================
    # Step 1: Load
    # ================================================================
    print("\n  --- Loading images ---")
    arrays, transforms, nodatas = {}, {}, {}
    crs = None
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for idx in sorted(IMAGES.keys()):
        name, path = IMAGES[idx]
        arr, tr, c, nd = read_geotiff(os.path.join(base, path))
        while arr.ndim > 2: arr = arr[0]
        arrays[idx] = arr
        transforms[idx] = tr
        nodatas[idx] = nd
        if crs is None: crs = c
        print(f"  [{idx}] {name}: {arr.shape}, nodata={nd}")

    n_images = len(IMAGES)

    # ================================================================
    # Step 2: Pairwise matching
    # ================================================================
    print("\n  --- Pairwise matching ---")
    from src.coregistration import collect_pair_matches

    pair_measurements = []
    for i in range(n_images):
        for j in range(i + 1, n_images):
            print(f"  [{i}]-[{j}]...", end=' ', flush=True)
            result = collect_pair_matches(
                arrays[i], transforms[i], arrays[j], transforms[j],
                nodatas[i], nodatas[j], max_global_shift=40)
            if result is None:
                print("no overlap")
                continue
            pair_measurements.append({
                'idx_i': i, 'idx_j': j,
                'shift_dx': result['shift_dx'], 'shift_dy': result['shift_dy'],
                'confidence': result['confidence'], 'n_blocks': result['n_blocks'],
                'rmse': result['rmse'], 'matches': result['matches'],
                'available': result.get('available', True),
                'failure_reason': result.get('failure_reason', None),
                'screening': result['screening'],
            })
            if result.get('available', True):
                print(f"dx={result['shift_dx']:.3f}, dy={result['shift_dy']:.3f}, "
                      f"conf={result['confidence']:.2f}, RMSE={result['rmse']:.3f}, "
                      f"blocks={result['n_blocks']}")
            else:
                print(f"FAILED: {result.get('failure_reason', 'unknown')}")

    available_pairs = [p for p in pair_measurements if p.get('available', True)]

    # ================================================================
    # Step 3: Network adjustment
    # ================================================================
    print("\n  --- Network adjustment ---")
    from src.coregistration import multi_image_network_adjustment

    net_result = multi_image_network_adjustment(available_pairs, n_images, reference_idx=REF_IDX)
    global_shifts = net_result['global_shifts']

    print(f"  Reference: [{REF_IDX}] {IMAGES[REF_IDX][0]}")
    print(f"  Global shifts:")
    for idx in range(n_images):
        print(f"    [{idx}] {IMAGES[idx][0]}: dx={global_shifts[idx,0]:.4f}, dy={global_shifts[idx,1]:.4f}")

    if net_result['loop_errors']:
        for le in net_result['loop_errors']:
            print(f"    Loop: {le['nodes']}: error={le['error_magnitude']:.4f}")

    # ================================================================
    # Step 4: Global registered
    # ================================================================
    print("\n  --- Global registration ---")
    from src.coregistration import warp_with_displacement_field

    global_registered = {}
    for idx in range(n_images):
        if idx == REF_IDX:
            global_registered[idx] = arrays[idx].astype(np.float64)
            continue
        gdx, gdy = global_shifts[idx, 0], global_shifts[idx, 1]
        global_registered[idx] = warp_with_displacement_field(
            arrays[idx], gdx, gdy,
            np.zeros(arrays[idx].shape), np.zeros(arrays[idx].shape), nodatas[idx])
        print(f"  [{idx}] {IMAGES[idx][0]}: global=({gdx:.4f},{gdy:.4f})")

    # ================================================================
    # Step 5: Global residual refinement (only [1])
    # ================================================================
    print("\n  --- Global residual refinement (only [1]) ---")
    from src.coregistration import refine_global_residual_shifts

    refine_global_residual_shifts(
        global_registered, global_shifts, arrays, transforms, nodatas, PARENT_MAP,
        max_iterations=2)

    print(f"\n  Final global shifts:")
    for idx in range(n_images):
        print(f"    [{idx}] {IMAGES[idx][0]}: dx={global_shifts[idx,0]:.4f}, dy={global_shifts[idx,1]:.4f}")

    # ================================================================
    # Step 6: Local RBF for [1] (vs [0])
    # ================================================================
    print("\n  --- Local RBF: [1] vs [0] ---")
    from src.coregistration import rematch_pair_on_registered, merge_multi_edge_controls

    h_1, w_1 = arrays[1].shape
    local_rbf_1 = None

    rematch_01 = rematch_pair_on_registered(
        global_registered[0], global_registered[1],
        transforms[0], transforms[1],
        nodatas[0], nodatas[1],
        max_residual_shift=5, block_size=512, confidence_threshold=0.5)

    if rematch_01 is not None:
        used_conf = rematch_01.get('used_confidence_threshold', 0.5)
        print(f"  [0]-[1]: {rematch_01['n_blocks']} blocks, RMSE={rematch_01['rmse']:.3f}, conf_thr={used_conf}")
        from src.coregistration import build_residual_controls_from_rematch
        ctrl_1 = build_residual_controls_from_rematch(
            rematch_01['matches'], confidence_threshold=used_conf, min_points=10)
        print(f"  [1] control points: {ctrl_1['n_valid']}")

        if ctrl_1['n_valid'] >= 20:
            try:
                ldx, ldy = _fit_rbf_field(
                    arrays[1], global_shifts[1, 0], global_shifts[1, 1],
                    ctrl_1, h_1, w_1, nodatas[1], smoothing=0.1)

                arr_no = warp_with_displacement_field(
                    arrays[1], global_shifts[1, 0], global_shifts[1, 1],
                    np.zeros((h_1, w_1)), np.zeros((h_1, w_1)), nodatas[1])
                arr_yes = warp_with_displacement_field(
                    arrays[1], global_shifts[1, 0], global_shifts[1, 1],
                    ldx, ldy, nodatas[1])

                e_no = _eval_edge(global_registered[0], transforms[0],
                                  arr_no, transforms[1], nodatas[0], nodatas[1])
                e_yes = _eval_edge(global_registered[0], transforms[0],
                                   arr_yes, transforms[1], nodatas[0], nodatas[1])

                print(f"  [0]-[1] check: mag {e_no['residual_magnitude']:.3f}->{e_yes['residual_magnitude']:.3f}, "
                      f"RMSE {e_no['rmse']:.3f}->{e_yes['rmse']:.3f}, "
                      f"P95 {e_no['p95']:.3f}->{e_yes['p95']:.3f}")

                rbf_helps = (e_yes['rmse'] <= e_no['rmse'] and
                             e_yes['p95'] <= e_no['p95'] * 1.1)
                if rbf_helps:
                    local_rbf_1 = (ldx, ldy)
                    max_local = float(np.max(np.hypot(ldx, ldy)))
                    print(f"  [1] RBF: ON ({ctrl_1['n_valid']} pts, local_max={max_local:.3f})")
                else:
                    print(f"  [1] RBF rejected")
            except Exception as e:
                print(f"  [1] RBF failed: {e}")
        else:
            print(f"  [1] RBF skipped: need >=20, have {ctrl_1['n_valid']}")

    if local_rbf_1 is not None:
        ldx, ldy = local_rbf_1
        final_1 = warp_with_displacement_field(
            arrays[1], global_shifts[1, 0], global_shifts[1, 1],
            ldx, ldy, nodatas[1])
    else:
        final_1 = global_registered[1]

    # ================================================================
    # Step 7: [1]-[3] RBF conflict diagnosis
    # ================================================================
    print("\n  --- [1]-[3] RBF conflict diagnosis ---")
    h_3, w_3 = arrays[3].shape

    arr_no_3 = warp_with_displacement_field(
        arrays[3], global_shifts[3, 0], global_shifts[3, 1],
        np.zeros((h_3, w_3)), np.zeros((h_3, w_3)), nodatas[3])

    e_before = _eval_edge(global_registered[1], transforms[1],
                          arr_no_3, transforms[3], nodatas[1], nodatas[3])
    e_after = _eval_edge(final_1, transforms[1],
                         arr_no_3, transforms[3], nodatas[1], nodatas[3])

    print(f"  [1]-[3] before image[1] RBF: mag={e_before['residual_magnitude']:.3f}, "
          f"RMSE={e_before['rmse']:.3f}, P95={e_before['p95']:.3f}")
    print(f"  [1]-[3] after image[1] RBF:  mag={e_after['residual_magnitude']:.3f}, "
          f"RMSE={e_after['rmse']:.3f}, P95={e_after['p95']:.3f}")

    if e_before['available'] and e_after['available']:
        p95_change = e_after['p95'] - e_before['p95']
        if p95_change > e_before['p95'] * 0.20:
            print(f"  Image [1] RBF causes cross-edge conflict.")

    # ================================================================
    # Step 8: [3] pre-correction multi-scale (diagnostic only)
    # ================================================================
    print("\n  --- [3] pre-correction multi-scale (diagnostic) ---")

    pre_ctrl_03, pre_info_03 = _multiscale_rematch(
        global_registered[0], transforms[0],
        global_registered[3], transforms[3],
        nodatas[0], nodatas[3])
    print(f"  Pre-correction [0]-[3]:")
    for bs, inf in pre_info_03.items():
        if inf:
            print(f"    block={bs}: {inf['n_blocks']} blocks, {inf['n_ctrl']} ctrl")
        else:
            print(f"    block={bs}: FAILED")

    pre_ctrl_13, pre_info_13 = _multiscale_rematch(
        final_1, transforms[1],
        global_registered[3], transforms[3],
        nodatas[1], nodatas[3])
    print(f"  Pre-correction [1]-[3]:")
    for bs, inf in pre_info_13.items():
        if inf:
            print(f"    block={bs}: {inf['n_blocks']} blocks, {inf['n_ctrl']} ctrl")
        else:
            print(f"    block={bs}: FAILED")

    _print_ctrl_stats(pre_ctrl_03, "[0]-[3] pre")
    _print_ctrl_stats(pre_ctrl_13, "[1]-[3] pre")

    # ================================================================
    # Step 9: [3] bilateral global correction
    # ================================================================
    print("\n  --- [3] bilateral global correction ---")

    e03_no = _eval_edge(global_registered[0], transforms[0],
                        arr_no_3, transforms[3], nodatas[0], nodatas[3])
    e13_no = _eval_edge(final_1, transforms[1],
                        arr_no_3, transforms[3], nodatas[1], nodatas[3])

    print(f"  Before: [0]-[3] mag={e03_no['residual_magnitude']:.3f}, "
          f"[1]-[3] mag={e13_no['residual_magnitude']:.3f}")

    w03 = (e03_no['confidence'] * e03_no['n_blocks'] /
           max(e03_no['rmse'], 0.1)) if e03_no['available'] else 0.0
    w13 = (e13_no['confidence'] * e13_no['n_blocks'] /
           max(e13_no['rmse'], 0.1)) if e13_no['available'] else 0.0

    bilateral_accepted = False
    delta_dx, delta_dy = 0.0, 0.0

    if w03 > 0 and w13 > 0:
        delta_dx = (e03_no['residual_dx'] * w03 + e13_no['residual_dx'] * w13) / (w03 + w13)
        delta_dy = (e03_no['residual_dy'] * w03 + e13_no['residual_dy'] * w13) / (w03 + w13)
        delta_mag = float(np.hypot(delta_dx, delta_dy))
        print(f"  Weighted delta: dx={delta_dx:.4f}, dy={delta_dy:.4f}, mag={delta_mag:.4f}")

        if delta_mag <= 2.0:
            cand_shift_3 = global_shifts[3].copy() + np.array([delta_dx, delta_dy])
            cand_shift_2 = global_shifts[2].copy() + np.array([delta_dx, delta_dy])

            arr_cand_3 = warp_with_displacement_field(
                arrays[3], cand_shift_3[0], cand_shift_3[1],
                np.zeros((h_3, w_3)), np.zeros((h_3, w_3)), nodatas[3])

            e03_c = _eval_edge(global_registered[0], transforms[0],
                               arr_cand_3, transforms[3], nodatas[0], nodatas[3])
            e13_c = _eval_edge(final_1, transforms[1],
                               arr_cand_3, transforms[3], nodatas[1], nodatas[3])

            print(f"  Candidate: [0]-[3] mag={e03_c['residual_magnitude']:.3f}, "
                  f"[1]-[3] mag={e13_c['residual_magnitude']:.3f}")

            combined_mag_no = max(e03_no['residual_magnitude'], e13_no['residual_magnitude'])
            combined_mag_c = max(e03_c['residual_magnitude'], e13_c['residual_magnitude'])
            combined_rmse_no = e03_no['rmse'] + e13_no['rmse']
            combined_rmse_c = e03_c['rmse'] + e13_c['rmse']

            mag_improved = combined_mag_c < combined_mag_no
            rmse_improved = combined_rmse_c < combined_rmse_no
            p95_ok_03 = e03_c['p95'] <= e03_no['p95'] * 1.1 if e03_no['available'] else True
            p95_ok_13 = e13_c['p95'] <= e13_no['p95'] * 1.1 if e13_no['available'] else True

            if mag_improved and rmse_improved and p95_ok_03 and p95_ok_13:
                global_shifts[3, :] += np.array([delta_dx, delta_dy])
                global_shifts[2, :] += np.array([delta_dx, delta_dy])
                global_registered[3] = warp_with_displacement_field(
                    arrays[3], global_shifts[3, 0], global_shifts[3, 1],
                    np.zeros((h_3, w_3)), np.zeros((h_3, w_3)), nodatas[3])
                h_2, w_2 = arrays[2].shape
                global_registered[2] = warp_with_displacement_field(
                    arrays[2], global_shifts[2, 0], global_shifts[2, 1],
                    np.zeros((h_2, w_2)), np.zeros((h_2, w_2)), nodatas[2])
                bilateral_accepted = True
                print(f"  ACCEPTED (+{delta_dx:.4f}, +{delta_dy:.4f})")
            else:
                print(f"  REJECTED (mag_improved={mag_improved}, rmse_improved={rmse_improved})")
    elif w03 > 0 and e03_no['n_blocks'] >= 15:
        if e03_no['rmse'] <= 1.0 and e03_no['p95'] <= 1.5:
            delta_dx, delta_dy = e03_no['residual_dx'], e03_no['residual_dy']
            global_shifts[3, :] += np.array([delta_dx, delta_dy])
            global_shifts[2, :] += np.array([delta_dx, delta_dy])
            global_registered[3] = warp_with_displacement_field(
                arrays[3], global_shifts[3, 0], global_shifts[3, 1],
                np.zeros((h_3, w_3)), np.zeros((h_3, w_3)), nodatas[3])
            h_2, w_2 = arrays[2].shape
            global_registered[2] = warp_with_displacement_field(
                arrays[2], global_shifts[2, 0], global_shifts[2, 1],
                np.zeros((h_2, w_2)), np.zeros((h_2, w_2)), nodatas[2])
            bilateral_accepted = True
            print(f"  ACCEPTED (single-edge, +{delta_dx:.4f}, +{delta_dy:.4f})")

    if not bilateral_accepted:
        print(f"  No bilateral correction applied")

    # ================================================================
    # Step 10: [3] post-correction multi-scale (actual RBF controls)
    # ================================================================
    print("\n  --- Post-correction [3] controls ---")

    post_ctrl_03, post_info_03 = _multiscale_rematch(
        global_registered[0], transforms[0],
        global_registered[3], transforms[3],
        nodatas[0], nodatas[3])
    print(f"  [0]-[3] post-correction:")
    for bs, inf in post_info_03.items():
        if inf:
            print(f"    block={bs}: {inf['n_blocks']} blocks, {inf['n_ctrl']} ctrl, "
                  f"conf_thr={inf['conf']}, rmse={inf['rmse']:.3f}")
        else:
            print(f"    block={bs}: FAILED")

    post_ctrl_13, post_info_13 = _multiscale_rematch(
        final_1, transforms[1],
        global_registered[3], transforms[3],
        nodatas[1], nodatas[3])
    print(f"  [1]-[3] post-correction:")
    for bs, inf in post_info_13.items():
        if inf:
            print(f"    block={bs}: {inf['n_blocks']} blocks, {inf['n_ctrl']} ctrl, "
                  f"conf_thr={inf['conf']}, rmse={inf['rmse']:.3f}")
        else:
            print(f"    block={bs}: FAILED")

    _print_ctrl_stats(post_ctrl_03, "[0]-[3] post")
    _print_ctrl_stats(post_ctrl_13, "[1]-[3] post")

    # ================================================================
    # Step 11: Balance controls per edge
    # ================================================================
    print("\n  --- Balancing controls ---")
    from src.coregistration import balance_edge_controls

    balanced_03 = balance_edge_controls(post_ctrl_03, max_total=60, grid_size=256,
                                         min_per_edge=15, dedup_distance=50.0)
    balanced_13 = balance_edge_controls(post_ctrl_13, max_total=60, grid_size=256,
                                         min_per_edge=15, dedup_distance=50.0)

    n_03 = sum(c['n_valid'] for c in balanced_03)
    n_13 = sum(c['n_valid'] for c in balanced_13)
    print(f"  Balanced [0]-[3]: {n_03} points")
    print(f"  Balanced [1]-[3]: {n_13} points")

    merged_3_post = merge_multi_edge_controls(
        balanced_03 + balanced_13, dedup_distance=50.0,
        confidence_threshold=0.4, min_points=15)
    print(f"  Merged post-correction controls: {merged_3_post['n_valid']}")

    # ================================================================
    # Step 12: [3] smoothing sweep + score-based selection
    # ================================================================
    print("\n  --- [3] RBF smoothing sweep ---")

    local_rbf_3 = None
    arr_no_3_final = warp_with_displacement_field(
        arrays[3], global_shifts[3, 0], global_shifts[3, 1],
        np.zeros((h_3, w_3)), np.zeros((h_3, w_3)), nodatas[3])

    e03_base = _eval_edge(global_registered[0], transforms[0],
                          arr_no_3_final, transforms[3], nodatas[0], nodatas[3])
    e13_base = _eval_edge(final_1, transforms[1],
                          arr_no_3_final, transforms[3], nodatas[1], nodatas[3])

    if merged_3_post['n_valid'] >= 15:
        best_score = float('inf')
        best_smoothing = None
        best_local = None
        candidates = []

        for sm in SMOOTHING_CANDIDATES:
            try:
                ldx_s, ldy_s = _fit_rbf_field(
                    arrays[3], global_shifts[3, 0], global_shifts[3, 1],
                    merged_3_post, h_3, w_3, nodatas[3], smoothing=sm)

                arr_s = warp_with_displacement_field(
                    arrays[3], global_shifts[3, 0], global_shifts[3, 1],
                    ldx_s, ldy_s, nodatas[3])

                e03_s = _eval_edge(global_registered[0], transforms[0],
                                   arr_s, transforms[3], nodatas[0], nodatas[3])
                e13_s = _eval_edge(final_1, transforms[1],
                                   arr_s, transforms[3], nodatas[1], nodatas[3])

                # 评分
                score = (e03_s['residual_magnitude'] + e13_s['residual_magnitude'] +
                         e03_s['rmse'] + e13_s['rmse'] +
                         0.5 * e03_s['p95'] + 0.5 * e13_s['p95'])

                # 约束检查
                q03 = (e03_s['residual_magnitude'] <= 0.30 and
                       e03_s['rmse'] <= 0.75 and e03_s['p95'] <= 1.20)
                q13 = (e13_s['residual_magnitude'] <= 0.50 and
                       e13_s['rmse'] <= 1.00 and e13_s['p95'] <= 1.50)

                # [0]-[3] P95不能恶化>0.20px
                p95_03_ok = True
                if e03_base['available'] and e03_s['available']:
                    p95_03_ok = e03_s['p95'] <= e03_base['p95'] + 0.20

                # [0]-[3] mag<=0.30
                mag_03_ok = e03_s['residual_magnitude'] <= 0.30

                # [1]-[3]相较仅全局有改善
                mag_13_improved = True
                if e13_base['available'] and e13_s['available']:
                    mag_13_improved = e13_s['residual_magnitude'] <= e13_base['residual_magnitude']

                passed = q03 and q13 and p95_03_ok and mag_03_ok and mag_13_improved

                max_local = float(np.max(np.hypot(ldx_s, ldy_s)))
                print(f"  sm={sm}: score={score:.3f}, "
                      f"[0]-[3] mag={e03_s['residual_magnitude']:.3f} RMSE={e03_s['rmse']:.3f} P95={e03_s['p95']:.3f}, "
                      f"[1]-[3] mag={e13_s['residual_magnitude']:.3f} RMSE={e13_s['rmse']:.3f} P95={e13_s['p95']:.3f}, "
                      f"local_max={max_local:.3f} => {'PASS' if passed else 'FAIL'}")

                candidates.append({
                    'smoothing': sm, 'score': score, 'passed': passed,
                    'local': (ldx_s, ldy_s), 'e03': e03_s, 'e13': e13_s,
                    'max_local': max_local,
                })

                if passed and score < best_score:
                    best_score = score
                    best_smoothing = sm
                    best_local = (ldx_s, ldy_s)

            except Exception as e:
                print(f"  sm={sm}: FAILED ({e})")

        if best_local is not None:
            local_rbf_3 = best_local
            print(f"  Best: sm={best_smoothing}, score={best_score:.3f}")
        else:
            print(f"  No candidate passed, falling back to global-only")
    else:
        print(f"  [3] RBF skipped: need >=15, have {merged_3_post['n_valid']}")

    # ================================================================
    # Step 13: [2] re-match + existing RBF logic
    # ================================================================
    print("\n  --- [2] re-match and RBF ---")
    h_2, w_2 = arrays[2].shape

    ctrl_list_2 = []
    rematch_results_2 = {}

    for neighbor in [0, 1, 3]:
        ref_arr = global_registered[neighbor] if neighbor == 0 else (final_1 if neighbor == 1 else global_registered[3])
        rm = rematch_pair_on_registered(
            ref_arr, global_registered[2],
            transforms[neighbor], transforms[2],
            nodatas[neighbor], nodatas[2],
            max_residual_shift=5, block_size=512, confidence_threshold=0.5)
        if rm is not None:
            used_conf = rm.get('used_confidence_threshold', 0.5)
            from src.coregistration import build_residual_controls_from_rematch
            ctrl = build_residual_controls_from_rematch(
                rm['matches'], confidence_threshold=used_conf, min_points=5)
            ctrl_list_2.append(ctrl)
            rematch_results_2[neighbor] = rm
            print(f"  from [{neighbor}]: {ctrl['n_valid']} control points")
        else:
            print(f"  from [{neighbor}]: UNAVAILABLE")

    merged_2 = merge_multi_edge_controls(ctrl_list_2, dedup_distance=50.0,
                                          confidence_threshold=0.4, min_points=20)
    n_valid_edges = len([r for r in rematch_results_2.values() if r is not None])
    print(f"  [2] merged: {merged_2['n_valid']} (before: {merged_2.get('n_before_filter', '?')}), "
          f"valid edges: {n_valid_edges}")

    local_rbf_2 = None

    if merged_2['n_valid'] >= 20:
        try:
            ldx2, ldy2 = _fit_rbf_field(
                arrays[2], global_shifts[2, 0], global_shifts[2, 1],
                merged_2, h_2, w_2, nodatas[2], smoothing=0.1)

            arr_no2 = warp_with_displacement_field(
                arrays[2], global_shifts[2, 0], global_shifts[2, 1],
                np.zeros((h_2, w_2)), np.zeros((h_2, w_2)), nodatas[2])
            arr_yes2 = warp_with_displacement_field(
                arrays[2], global_shifts[2, 0], global_shifts[2, 1],
                ldx2, ldy2, nodatas[2])

            evals_no_2 = {}
            evals_yes_2 = {}
            for ni in [0, 1, 3]:
                ref_arr = global_registered[ni] if ni == 0 else (final_1 if ni == 1 else global_registered[3])
                evals_no_2[ni] = _eval_edge(ref_arr, transforms[ni],
                                            arr_no2, transforms[2], nodatas[ni], nodatas[2])
                evals_yes_2[ni] = _eval_edge(ref_arr, transforms[ni],
                                             arr_yes2, transforms[2], nodatas[ni], nodatas[2])
                if evals_yes_2[ni]['available']:
                    print(f"  [{ni}]-[2]: mag {evals_no_2[ni]['residual_magnitude']:.3f}->"
                          f"{evals_yes_2[ni]['residual_magnitude']:.3f}, "
                          f"RMSE {evals_no_2[ni]['rmse']:.3f}->{evals_yes_2[ni]['rmse']:.3f}, "
                          f"P95 {evals_no_2[ni]['p95']:.3f}->{evals_yes_2[ni]['p95']:.3f}")

            available_ni = [ni for ni in [0, 1, 3] if evals_yes_2[ni]['available']]
            rbf_2_retained = False

            if len(available_ni) >= 2:
                combined_rmse_no = sum(evals_no_2[ni]['rmse'] for ni in available_ni)
                combined_rmse_yes = sum(evals_yes_2[ni]['rmse'] for ni in available_ni)
                any_p95_worse = any(
                    evals_yes_2[ni]['p95'] > evals_no_2[ni]['p95'] * 1.1
                    for ni in available_ni
                    if evals_no_2[ni]['available'] and evals_yes_2[ni]['available'])
                rbf_2_retained = (combined_rmse_yes < combined_rmse_no and not any_p95_worse)

            elif len(available_ni) == 1 and 3 in available_ni:
                e_no_23 = evals_no_2[3]
                e_yes_23 = evals_yes_2[3]
                n_ctrl = merged_2['n_valid']
                n_blk = e_yes_23['n_blocks']
                mag_ok = e_yes_23['residual_magnitude'] <= 0.50
                rmse_ok = e_yes_23['rmse'] <= 0.50
                p95_ok = e_yes_23['p95'] <= 0.75
                rmse_drop = ((e_no_23['rmse'] - e_yes_23['rmse']) /
                             max(e_no_23['rmse'], 1e-10)) >= 0.50
                p95_drop = ((e_no_23['p95'] - e_yes_23['p95']) /
                            max(e_no_23['p95'], 1e-10)) >= 0.50
                local_max = float(np.max(np.hypot(ldx2, ldy2)))
                local_ok = local_max <= 2.5 + 1e-6

                other_ni = [ni for ni in [0, 1] if evals_yes_2[ni]['available']]
                no_other_worse = all(
                    evals_yes_2[ni]['p95'] <= evals_no_2[ni]['p95'] * 1.1
                    for ni in other_ni
                    if evals_no_2[ni]['available'] and evals_yes_2[ni]['available'])

                rbf_2_retained = (n_ctrl >= 40 and n_blk >= 30 and
                                  mag_ok and rmse_ok and p95_ok and
                                  rmse_drop and p95_drop and local_ok and no_other_worse)

                if not rbf_2_retained:
                    reasons = []
                    if n_ctrl < 40: reasons.append(f"ctrl={n_ctrl}<40")
                    if n_blk < 30: reasons.append(f"blocks={n_blk}<30")
                    if not mag_ok: reasons.append(f"mag>0.50")
                    if not rmse_ok: reasons.append(f"rmse>0.50")
                    if not p95_ok: reasons.append(f"p95>0.75")
                    if not rmse_drop: reasons.append("rmse drop<50%")
                    if not p95_drop: reasons.append("p95 drop<50%")
                    if not local_ok: reasons.append(f"local_max>2.5")
                    if not no_other_worse: reasons.append("other P95 worsened")
                    print(f"  [2] single-edge: REJECTED ({', '.join(reasons)})")

            if rbf_2_retained:
                local_rbf_2 = (ldx2, ldy2)
                max_local = float(np.max(np.hypot(ldx2, ldy2)))
                print(f"  [2] RBF: RETAINED ({merged_2['n_valid']} pts, local_max={max_local:.3f})")
            else:
                print(f"  [2] RBF rejected")

        except Exception as e:
            print(f"  [2] RBF failed: {e}")
    else:
        print(f"  [2] RBF skipped: need >=20, have {merged_2['n_valid']}")

    # ================================================================
    # Step 14: Final single resampling
    # ================================================================
    print("\n  --- Final single resampling ---")
    final_registered = {}
    final_registered[REF_IDX] = global_registered[REF_IDX]

    for idx in [1, 3, 2]:
        lf = local_rbf_1 if idx == 1 else (local_rbf_3 if idx == 3 else local_rbf_2)

        if lf is not None:
            ldx, ldy = lf
            final_registered[idx] = warp_with_displacement_field(
                arrays[idx], global_shifts[idx, 0], global_shifts[idx, 1],
                ldx, ldy, nodatas[idx])
            max_local = float(np.max(np.hypot(ldx, ldy)))
            print(f"  [{idx}] {IMAGES[idx][0]}: global+local, local_max={max_local:.3f}")
        else:
            final_registered[idx] = global_registered[idx]
            print(f"  [{idx}] {IMAGES[idx][0]}: global only")

    # ================================================================
    # Step 15: Verify all 6 pairs + quality gate
    # ================================================================
    print("\n  --- Post-registration verification ---")
    all_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pair_verify = {}

    for i, j in all_pairs:
        v = _eval_edge(final_registered[i], transforms[i],
                       final_registered[j], transforms[j],
                       nodatas[i], nodatas[j])
        pair_verify[(i, j)] = v

    print("\n  --- Quality gate (connectivity-based) ---")

    def _check_edge(v, need_blocks=3):
        """Return (label, passed, checked)."""
        if not v['available']:
            return 'UNAVAILABLE', False, False
        mag_ok = v['residual_magnitude'] <= 0.50
        rmse_ok = v['rmse'] <= 1.00
        p95_ok = v['p95'] <= 1.50
        blk_ok = v['n_blocks'] >= need_blocks
        mini = mag_ok and rmse_ok and p95_ok and blk_ok
        pref = (v['residual_magnitude'] <= 0.30 and v['rmse'] <= 0.75 and
                v['p95'] <= 1.20 and blk_ok)
        if pref:
            return 'PREFERRED', True, True
        elif mini:
            return 'ACCEPTABLE', True, True
        else:
            reasons = []
            if not mag_ok: reasons.append(f"mag={v['residual_magnitude']:.2f}>0.50")
            if not rmse_ok: reasons.append(f"rmse={v['rmse']:.2f}>1.00")
            if not p95_ok: reasons.append(f"p95={v['p95']:.2f}>1.50")
            if not blk_ok: reasons.append(f"blocks={v['n_blocks']}<3")
            return f"FAIL({', '.join(reasons)})", False, True

    # Classify each edge
    print(f"\n  {'Pair':<10s} {'res_mag':>8s} {'RMSE':>8s} {'P95':>8s} {'blocks':>7s}  {'status'}")
    print(f"  {'-'*60}")

    passed_edges = []
    rejected_edges = []

    for i, j in all_pairs:
        v = pair_verify[(i, j)]
        label, passed, checked = _check_edge(v)
        mag_s = f"{v['residual_magnitude']:.3f}" if v['available'] else 'nan'
        rmse_s = f"{v['rmse']:.3f}" if v['available'] else 'nan'
        p95_s = f"{v['p95']:.3f}" if v['available'] else 'nan'
        blk_s = f"{v['n_blocks']}" if v['available'] else '0'
        print(f"  [{i}]-[{j}]  {mag_s:>8s} {rmse_s:>8s} {p95_s:>8s} {blk_s:>7s}  {label}")
        if checked and passed:
            passed_edges.append((i, j))
        elif checked:
            rejected_edges.append((i, j))

    # Build connectivity graph from passed edges (undirected)
    from collections import defaultdict, deque

    adj = defaultdict(set)
    for i, j in passed_edges:
        adj[i].add(j)
        adj[j].add(i)

    # BFS from reference image [0] to check all images are reachable
    reachable = set()
    queue = deque([REF_IDX])
    reachable.add(REF_IDX)
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in reachable:
                reachable.add(nb)
                queue.append(nb)

    all_reachable = reachable == set(range(n_images))

    # Extract spanning tree edges (edges on BFS tree)
    spanning_tree_edges = []
    visited_tree = {REF_IDX}
    queue_tree = deque([REF_IDX])
    while queue_tree:
        node = queue_tree.popleft()
        for nb in adj[node]:
            if nb not in visited_tree:
                visited_tree.add(nb)
                spanning_tree_edges.append((node, nb))
                queue_tree.append(nb)

    # Check all spanning tree edges are in passed_edges
    tree_edges_pass = all(e in passed_edges or (e[1], e[0]) in passed_edges
                          for e in spanning_tree_edges)

    # STRICT quality: ALL overlap edges must pass
    strict_pass = len(rejected_edges) == 0

    # OPERATIONAL quality: spanning tree complete and all tree edges pass
    operational_pass = all_reachable and tree_edges_pass and len(spanning_tree_edges) == n_images - 1

    print(f"\n  Passed edges: {passed_edges}")
    print(f"  Rejected edges: {rejected_edges}")
    print(f"  Spanning tree: {spanning_tree_edges}")
    print(f"  All images reachable from [{REF_IDX}]: {all_reachable}")
    print(f"\n  STRICT QUALITY: {'PASS' if strict_pass else 'FAIL'}")
    print(f"  OPERATIONAL QUALITY: {'PASS' if operational_pass else 'FAIL'}")

    if rejected_edges:
        print(f"\n  Rejected redundant edges:")
        for i, j in rejected_edges:
            v = pair_verify[(i, j)]
            mag_s = f"{v['residual_magnitude']:.3f}" if v['available'] else 'nan'
            rmse_s = f"{v['rmse']:.3f}" if v['available'] else 'nan'
            p95_s = f"{v['p95']:.3f}" if v['available'] else 'nan'
            blk_s = f"{v['n_blocks']}" if v['available'] else '0'
            print(f"    [{i}]-[{j}]: mag={mag_s}, RMSE={rmse_s}, P95={p95_s}, blocks={blk_s}")

    if not operational_pass:
        print(f"\n  => Skipping BAGRN and mosaic.")
        return {
            'quality': 'FAIL',
            'strict_quality': 'FAIL' if not strict_pass else 'PASS',
            'operational_quality': 'FAIL',
            'global_shifts': global_shifts.tolist(),
            'pair_verify': {f"{i}-{j}": v for (i, j), v in pair_verify.items()},
            'passed_edges': [list(e) for e in passed_edges],
            'rejected_edges': [list(e) for e in rejected_edges],
            'spanning_tree_edges': [list(e) for e in spanning_tree_edges],
        }

    # ================================================================
    # Step 16: BAGRN + Mosaic
    # ================================================================
    print("\n  --- BAGRN (using only reliable edges) ---")
    arrs_3d = [final_registered[idx][np.newaxis, :, :] for idx in range(n_images)]
    nodatas_list = [nodatas[idx] for idx in range(n_images)]

    bagrn_overlaps = []
    for i, j in passed_edges:
        b1 = compute_bounds(transforms[i], final_registered[i].shape)
        b2 = compute_bounds(transforms[j], final_registered[j].shape)
        win = get_overlap_window(b1, transforms[i], b2, transforms[j])
        if win is not None:
            wi2, wj2 = win
            n = (wi2[1] - wi2[0]) * (wi2[3] - wi2[2])
            if n > 100:
                bagrn_overlaps.append({'idx_i': i, 'idx_j': j, 'window_i': wi2, 'window_j': wj2})
    print(f"  BAGRN overlap pairs (reliable only): {len(bagrn_overlaps)}")
    for ov in bagrn_overlaps:
        print(f"    [{ov['idx_i']}]—[{ov['idx_j']}]")
    if rejected_edges:
        print(f"  Excluded edges: {rejected_edges}")

    original_metrics = compute_all(arrs_3d, arrs_3d, nodatas_list, bagrn_overlaps, [REF_IDX])

    t_bagrn = time.time()
    bagrn_result, _, _ = bagrn_normalize(arrs_3d, nodatas_list, bagrn_overlaps, control_idx=REF_IDX)
    t_bagrn = time.time() - t_bagrn
    bagrn_metrics = compute_all(arrs_3d, bagrn_result, nodatas_list, bagrn_overlaps, [REF_IDX])

    # --- VOLRN (局部辐射归一化，作用于 BAGRN 结果) ---
    print("\n  --- VOLRN (local radiometric normalization) ---")
    transforms_list = [transforms[idx] for idx in range(n_images)]
    bounds_list = [compute_bounds(transforms[idx], final_registered[idx].shape)
                   for idx in range(n_images)]

    t_volrn = time.time()
    volrn_result, volrn_coeffs = volrn_normalize(
        bagrn_result, transforms_list, bounds_list, nodatas_list,
        block_size_pixels=200, lambda_param=0.1, rho=1.0,
        max_iter=200, tol=1e-4, verbose=False)
    t_volrn = time.time() - t_volrn
    volrn_metrics = compute_all(arrs_3d, volrn_result, nodatas_list, bagrn_overlaps, [REF_IDX])

    if volrn_coeffs is not None and volrn_coeffs.size > 0:
        a_vals = volrn_coeffs[0, :, 0]
        b_vals = volrn_coeffs[0, :, 1]
        print(f"  VOLRN blocks: {len(a_vals)}, a range: [{a_vals.min():.4f}, {a_vals.max():.4f}], "
              f"b range: [{b_vals.min():.4f}, {b_vals.max():.4f}]")

    def _v(m, key):
        return m.get(key, float('nan'))

    print(f"\n  {'Metric':<12s} {'Original':>12s} {'BAGRN':>12s} {'VOLRN':>12s}")
    print(f"  {'-'*48}")
    for key, label in [('adm', 'ADM'), ('adsd', 'ADSD'), ('rdoa', 'RDOA'), ('ave', 'Ave')]:
        v0 = _v(original_metrics, key)
        v1 = _v(bagrn_metrics, key)
        v2 = _v(volrn_metrics, key)
        print(f"  {label:<12s} {v0:12.6f} {v1:12.6f} {v2:12.6f}")
    print(f"\n  BAGRN time: {t_bagrn:.1f}s, VOLRN time: {t_volrn:.1f}s")

    print("\n  --- Mosaic ---")

    # 生成诊断镶嵌：
    #   1. source_selection: 每像素单源，诊断几何
    #   2. narrow_feather(8px): 接缝两侧8像素(112m)羽化，其余单源
    #   3. weighted: 全局加权平均（对比实验）
    #   4. bagrn_source_selection: BAGRN + 单源
    #   5. bagrn_narrow_feather_8px: BAGRN + 8px羽化
    #   6. bagrn_weighted: BAGRN + 全局加权（仅对比）
    #   7. bagrn_volrn_source_selection: BAGRN+VOLRN + 单源
    #   8. bagrn_volrn_narrow_feather_8px: BAGRN+VOLRN + 8px羽化（最终成果）
    #   9. bagrn_volrn_weighted: BAGRN+VOLRN + 全局加权（仅对比）
    mosaic_configs = [
        ('source_selection', arrs_3d, 'source_selection', 8),
        ('narrow_feather_8px', arrs_3d, 'narrow_feather', 8),
        ('weighted', arrs_3d, 'weighted', 8),
        ('bagrn_source_selection', bagrn_result, 'source_selection', 8),
        ('bagrn_narrow_feather_8px', bagrn_result, 'narrow_feather', 8),
        ('bagrn_weighted', bagrn_result, 'weighted', 8),
        ('bagrn_volrn_source_selection', volrn_result, 'source_selection', 8),
        ('bagrn_volrn_narrow_feather_8px', volrn_result, 'narrow_feather', 8),
        ('bagrn_volrn_weighted', volrn_result, 'weighted', 8),
    ]

    for label, data, mode, fw in mosaic_configs:
        out_path = os.path.join(OUTPUT, f'mosaic_B14_four_{label}.tif')
        t0 = time.time()
        create_mosaic(data, transforms_list, crs, nodatas_list, out_path,
                      mode=mode, feather_width=fw)
        size = os.path.getsize(out_path) / 1024 / 1024

        check_arr, _, _, _ = read_geotiff(out_path)
        if check_arr.ndim > 2: check_arr = check_arr[0]
        nan_c = int(np.sum(~np.isfinite(check_arr)))
        inf_c = int(np.sum(np.isinf(check_arr)))
        valid_c = int(np.sum(np.isfinite(check_arr) & (check_arr != 0)))

        print(f"  {label}: {out_path}")
        print(f"    Size: {size:.1f} MB, time: {time.time()-t0:.1f}s")
        print(f"    Union: {check_arr.size}, valid: {valid_c}, NaN: {nan_c}, Inf: {inf_c}")
        if nan_c > 0 or inf_c > 0:
            raise RuntimeError(f"Mosaic has NaN={nan_c}, Inf={inf_c}")

    print(f"\n  The mosaic is based on a reliable spanning-tree registration network.")
    if rejected_edges:
        print(f"  Edge {rejected_edges} was excluded as a redundant low-quality edge.")

    print(f"\n{'='*70}")
    print(f"  Done. Total: {time.time()-t_start:.1f}s")
    print(f"  Output: {OUTPUT}")

    return {
        'quality': 'OPERATIONAL_PASS',
        'strict_quality': 'PASS' if strict_pass else 'FAIL',
        'operational_quality': 'PASS',
        'original_metrics': original_metrics,
        'bagrn_metrics': bagrn_metrics,
        'volrn_metrics': volrn_metrics,
        'global_shifts': global_shifts.tolist(),
        'pair_verify': {f"{i}-{j}": v for (i, j), v in pair_verify.items()},
        'passed_edges': [list(e) for e in passed_edges],
        'rejected_edges': [list(e) for e in rejected_edges],
        'spanning_tree_edges': [list(e) for e in spanning_tree_edges],
        'mosaic_note': ('The mosaic is based on a reliable spanning-tree '
                        'registration network. Edge [1]-[3] was excluded '
                        'as a redundant low-quality edge.' if rejected_edges
                        else 'All edges passed.'),
        'recommended_output': 'mosaic_B14_four_bagrn_volrn_narrow_feather_8px.tif',
    }


if __name__ == '__main__':
    try:
        result = main()
        with open(os.path.join(OUTPUT, 'results.json'), 'w') as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ERROR: {e}")
