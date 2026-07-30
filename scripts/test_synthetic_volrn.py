"""Synthetic test: create spatially varying radiometric differences and test VOLRN."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bagrn import bagrn_normalize
from src.volrn import volrn_normalize
from src.overlap import get_overlap_window
from src.metrics import compute_all


def create_synthetic_pair(rows=512, cols=512, overlap_cols=256):
    """
    Create two overlapping images with spatially varying radiometric differences.

    Image 1: base values ~3000
    Image 2: base values ~3000 + spatially varying gain/bias
      - Left half: gain=1.1, bias=+100
      - Right half: gain=0.9, bias=-100
    """
    np.random.seed(42)

    # Create base image with some texture
    y, x = np.mgrid[0:rows, 0:cols]
    base = 3000.0 + 200.0 * np.sin(2 * np.pi * x / cols) * np.cos(2 * np.pi * y / rows)
    base += np.random.normal(0, 10, (rows, cols))

    # Image 1: base + noise
    img1 = base + np.random.normal(0, 5, (rows, cols))

    # Image 2: spatially varying gain/bias
    # Gain varies from 0.85 to 1.15 across x
    gain = 0.85 + 0.30 * x / cols  # 0.85 to 1.15
    # Bias varies from -150 to +150 across y
    bias = -150.0 + 300.0 * y / rows  # -150 to +150

    img2 = gain * base + bias + np.random.normal(0, 5, (rows, cols))

    # Make single-band arrays: shape (1, rows, cols)
    arr1 = img1[np.newaxis, :, :]
    arr2 = img2[np.newaxis, :, :]

    # Overlap: img1 is left, img2 is right, overlapping in middle
    # Total scene: cols + overlap_cols wide
    # img1 occupies [0, cols)
    # img2 occupies [overlap_cols, overlap_cols + cols)
    scene_cols = cols + overlap_cols
    scene1 = np.zeros((1, rows, scene_cols))
    scene2 = np.zeros((1, rows, scene_cols))
    scene1[:, :, :cols] = arr1
    scene2[:, :, overlap_cols:overlap_cols + cols] = arr2

    # Use identity transform (1m/pixel)
    import rasterio.transform
    tr1 = rasterio.transform.from_bounds(0, 0, scene_cols, rows, scene_cols, rows)
    tr2 = rasterio.transform.from_bounds(0, 0, scene_cols, rows, scene_cols, rows)

    bounds1 = (0, 0, scene_cols, rows)
    bounds2 = (0, 0, scene_cols, rows)

    print(f'Scene: {scene1.shape}, overlap: {overlap_cols} cols')

    # Overlap window
    windows = get_overlap_window(bounds1, tr1, bounds2, tr2)
    print(f'Full overlap window: {windows}')

    # We want partial overlap: img1 right side overlaps img2 left side
    # Manually define overlap
    win1 = (0, rows, cols - overlap_cols, cols)  # right part of img1
    win2 = (0, rows, 0, overlap_cols)  # left part of img2

    # Create proper overlap arrays
    arr1_overlap = scene1[:, :, cols-overlap_cols:cols]
    arr2_overlap = scene2[:, :, overlap_cols:overlap_cols+overlap_cols]

    print(f'Overlap regions: img1 {arr1_overlap.shape}, img2 {arr2_overlap.shape}')

    return arr1, arr2, tr1, tr2, bounds1, bounds2, win1, win2


def run_synthetic():
    print('='*60)
    print('  Synthetic Test: Spatially Varying Radiometric Differences')
    print('='*60)

    rows, cols, overlap_cols = 512, 512, 256

    # Create images
    np.random.seed(42)
    y, x = np.mgrid[0:rows, 0:2*cols-overlap_cols]
    scene_cols = 2*cols - overlap_cols

    base = 3000.0 + 200.0 * np.sin(2*np.pi*x/cols) * np.cos(2*np.pi*y/rows)
    base += np.random.normal(0, 10, base.shape)

    # Image 1: left part of scene (cols pixels wide)
    img1_data = base[:, :cols].copy()
    img1_data += np.random.normal(0, 5, img1_data.shape)

    # Image 2: right part of scene (cols pixels wide)
    # Spatially varying gain/bias relative to base
    gain = 0.85 + 0.30 * x[:, cols:] / cols  # 0.85 to 1.15
    bias = -150.0 + 300.0 * y[:, cols:] / rows  # -150 to +150
    img2_data = gain * base[:, cols:] + bias + np.random.normal(0, 5, base[:, cols:].shape)

    # Make into (1, rows, cols) arrays
    arr1 = img1_data[np.newaxis, :, :].astype(np.float64)
    arr2 = img2_data[np.newaxis, :, :].astype(np.float64)

    print(f'Image 1: {arr1.shape}, range [{arr1.min():.1f}, {arr1.max():.1f}]')
    print(f'Image 2: {arr2.shape}, range [{arr2.min():.1f}, {arr2.max():.1f}]')

    # Define transforms and bounds
    import rasterio.transform
    tr1 = rasterio.transform.from_bounds(0, 0, cols, rows, cols, rows)
    tr2 = rasterio.transform.from_bounds(cols - overlap_cols, 0,
                                          2*cols - overlap_cols, rows, cols, rows)

    bounds1 = (0, 0, cols, rows)
    bounds2 = (cols - overlap_cols, 0, 2*cols - overlap_cols, rows)

    print(f'Bounds 1: {bounds1}')
    print(f'Bounds 2: {bounds2}')

    # Compute overlap
    windows = get_overlap_window(bounds1, tr1, bounds2, tr2)
    if windows is None:
        print('No overlap!')
        return
    win1, win2 = windows
    n_pix = (win1[1]-win1[0])*(win1[3]-win1[2])
    print(f'Overlap: win1={win1}, win2={win2}, pixels={n_pix}')

    arrays = [arr1, arr2]
    transforms = [tr1, tr2]
    bounds_list = [bounds1, bounds2]
    nodata_list = [None, None]
    overlaps = [{'idx_i': 0, 'idx_j': 1, 'window_i': win1, 'window_j': win2}]
    bands = [0]

    # Check overlap stats
    p1 = arr1[0, win1[0]:win1[1], win1[2]:win1[3]]
    p2 = arr2[0, win2[0]:win2[1], win2[2]:win2[3]]
    print(f'\nOverlap pixel stats:')
    print(f'  img1: mean={p1.mean():.1f}, std={p1.std():.1f}')
    print(f'  img2: mean={p2.mean():.1f}, std={p2.std():.1f}')
    print(f'  mean diff: {abs(p1.mean()-p2.mean()):.1f}')

    # Baseline
    baseline = compute_all(arrays, arrays, nodata_list, overlaps, bands)
    print(f'\nBaseline: {baseline}')

    # BAGRN
    print('\nRunning BAGRN...')
    bg, _, _ = bagrn_normalize(arrays, nodata_list, overlaps, control_idx=0)
    bg_m = compute_all(bg, arrays, nodata_list, overlaps, bands)
    print(f'BAGRN: {bg_m}')

    # Check BAGRN correction
    bg1 = bg[0][0, win1[0]:win1[1], win1[2]:win1[3]]
    bg2 = bg[1][0, win2[0]:win2[1], win2[2]:win2[3]]
    print(f'After BAGRN overlap stats:')
    print(f'  img1: mean={bg1.mean():.1f}, std={bg1.std():.1f}')
    print(f'  img2: mean={bg2.mean():.1f}, std={bg2.std():.1f}')
    print(f'  mean diff: {abs(bg1.mean()-bg2.mean()):.1f}')

    # VOLRN on BAGRN output - test different lambda values
    for lam in [0.5, 0.1, 0.01, 0.001]:
        print(f'\nRunning VOLRN (lambda={lam}) on BAGRN...')
        vr, coeffs = volrn_normalize(
            bg, transforms, bounds_list, nodata_list,
            block_size_pixels=128, lambda_param=lam, rho=1.0,
            max_iter=200, tol=1e-4, verbose=False,
        )
        vr_m = compute_all(vr, arrays, nodata_list, overlaps, bands)

        # Check block coefficients
        a_vals = coeffs[:, 0]
        b_vals = coeffs[:, 1]
        print(f'  a: [{a_vals.min():.4f}, {a_vals.max():.4f}], mean={a_vals.mean():.4f}')
        print(f'  b: [{b_vals.min():.1f}, {b_vals.max():.1f}], mean={b_vals.mean():.1f}')
        print(f'  ADM: {vr_m["adm"]:.4f}, ADSD: {vr_m["adsd"]:.4f}, AVE: {vr_m["ave"]:.4f}')
        print(f'  GL: {vr_m["gl"]:.6f}')

    # Best lambda
    print('\nRunning VOLRN (lambda=0.01) on BAGRN...')
    vr, coeffs = volrn_normalize(
        bg, transforms, bounds_list, nodata_list,
        block_size_pixels=128, lambda_param=0.01, rho=1.0,
        max_iter=200, tol=1e-4, verbose=True,
    )
    vr_m = compute_all(vr, arrays, nodata_list, overlaps, bands)
    print(f'VOLRN: {vr_m}')

    # Detailed overlap analysis
    print('\n--- Detailed Overlap Analysis ---')
    # Before any correction
    p1_orig = arr1[0, win1[0]:win1[1], win1[2]:win1[3]]
    p2_orig = arr2[0, win2[0]:win2[1], win2[2]:win2[3]]
    print(f'Original overlap: img1 mean={p1_orig.mean():.1f}, img2 mean={p2_orig.mean():.1f}')
    print(f'  img1 std={p1_orig.std():.1f}, img2 std={p2_orig.std():.1f}')

    # After BAGRN
    p1_bg = bg[0][0, win1[0]:win1[1], win1[2]:win1[3]]
    p2_bg = bg[1][0, win2[0]:win2[1], win2[2]:win2[3]]
    print(f'After BAGRN: img1 mean={p1_bg.mean():.1f}, img2 mean={p2_bg.mean():.1f}')
    print(f'  img1 std={p1_bg.std():.1f}, img2 std={p2_bg.std():.1f}')

    # After VOLRN
    p1_vr = vr[0][0, win1[0]:win1[1], win1[2]:win1[3]]
    p2_vr = vr[1][0, win2[0]:win2[1], win2[2]:win2[3]]
    print(f'After VOLRN: img1 mean={p1_vr.mean():.1f}, img2 mean={p2_vr.mean():.1f}')
    print(f'  img1 std={p1_vr.std():.1f}, img2 std={p2_vr.std():.1f}')

    # Check if VOLRN reduced local variation
    # Compute per-column means in overlap
    col_means_orig_1 = p1_orig.mean(axis=0)
    col_means_orig_2 = p2_orig.mean(axis=0)
    col_means_bg_1 = p1_bg.mean(axis=0)
    col_means_bg_2 = p2_bg.mean(axis=0)
    col_means_vr_1 = p1_vr.mean(axis=0)
    col_means_vr_2 = p2_vr.mean(axis=0)

    print(f'\nColumn-wise mean difference (img1 - img2):')
    diff_orig = col_means_orig_1 - col_means_orig_2
    diff_bg = col_means_bg_1 - col_means_bg_2
    diff_vr = col_means_vr_1 - col_means_vr_2
    print(f'  Original: mean_diff={diff_orig.mean():.1f}, std_diff={diff_orig.std():.1f}')
    print(f'  After BAGRN: mean_diff={diff_bg.mean():.1f}, std_diff={diff_bg.std():.1f}')
    print(f'  After VOLRN: mean_diff={diff_vr.mean():.1f}, std_diff={diff_vr.std():.1f}')

    # Check VOLRN correction
    vr1 = vr[0][0, win1[0]:win1[1], win1[2]:win1[3]]
    vr2 = vr[1][0, win2[0]:win2[1], win2[2]:win2[3]]
    print(f'After VOLRN overlap stats:')
    print(f'  img1: mean={vr1.mean():.1f}, std={vr1.std():.1f}')
    print(f'  img2: mean={vr2.mean():.1f}, std={vr2.std():.1f}')
    print(f'  mean diff: {abs(vr1.mean()-vr2.mean()):.1f}')

    # Block coefficients
    if len(coeffs) > 0:
        print(f'\nBlock coefficients (a, b):')
        for k in range(min(len(coeffs), 10)):
            print(f'  Block {k}: a={coeffs[k,0]:.4f}, b={coeffs[k,1]:.1f}')

    # Check block-level stats
    from src.volrn import image_blocking
    blocks, pairs = image_blocking(
        bg, transforms, bounds_list, nodata_list, 128, bands)
    print(f'\nBlock-level stats (BAGRN output):')
    for blk in blocks[:8]:
        print(f'  Block {blk.block_id} (img{blk.image_idx}): '
              f'mu={blk.mu[0]:.1f}, sigma={blk.sigma[0]:.1f}, '
              f'grid=({blk.grid_m},{blk.grid_n})')
    print(f'  ... ({len(blocks)} total blocks)')
    print(f'  Pairs: {len(pairs)}')
    for p in pairs[:4]:
        bi, bj = p.block_id_i, p.block_id_j
        print(f'  Pair ({bi},{bj}): img{blocks[bi].image_idx} mu={blocks[bi].mu[0]:.1f} '
              f'vs img{blocks[bj].image_idx} mu={blocks[bj].mu[0]:.1f}')

    # Check B matrix
    from src.volrn import _build_volrn_system
    B_mat, A_mat, b_vec, mu_sc = _build_volrn_system(blocks, pairs, 0)
    x_id = np.ones(2 * len(blocks))
    x_id[1::2] = 0.0
    Bx = B_mat @ x_id
    Ax_b = A_mat @ x_id - b_vec
    print(f'\n  B matrix: shape={B_mat.shape}, nnz={B_mat.nnz}')
    print(f'  ||Bx_id||: {np.linalg.norm(Bx):.4f}')
    print(f'  ||Ax_b_id||: {np.linalg.norm(Ax_b):.6f}')
    print(f'  Bx range: [{Bx.min():.4f}, {Bx.max():.4f}]')
    print(f'  Bx values: {Bx[:6]}')

    # Comparison table
    print(f'\n{"="*60}')
    print(f'  Synthetic Test Results')
    print(f'{"="*60}')
    print(f'{"Metric":<12} {"Baseline":>12} {"BAGRN":>12} {"VOLRN":>12} {"VOLRN-BAGRN":>14}')
    print(f'{"-"*62}')
    for k in baseline:
        b = baseline[k]
        bg = bg_m[k]
        vr_val = vr_m[k]
        diff = vr_val - bg
        print(f'{k:<12} {b:>12.4f} {bg:>12.4f} {vr_val:>12.4f} {diff:>+14.4f}')


if __name__ == '__main__':
    run_synthetic()
