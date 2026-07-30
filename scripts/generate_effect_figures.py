"""Generate effect figures (效果图) for 师兄: original vs BAGRN comparison."""
import os, sys, glob, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import read_geotiff
from src.overlap import get_overlap_window

INPUT_BASE = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\input'
OUTPUT_BASE = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\effect_figures'
os.makedirs(OUTPUT_BASE, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BANDS = ['B5', 'B8', 'B14']
SENSORS = ['DZ01V', 'DZ01S']


def squeeze(arr):
    while arr.ndim > 2:
        arr = arr[0]
    return arr

def load_images(band, sensor):
    pattern = os.path.join(INPUT_BASE, '**', f'*{sensor}*{band}*.TIF')
    files = sorted(glob.glob(pattern, recursive=True))
    files = [f for f in files if '_PAN' not in f.upper()]
    images = []
    for f in files:
        arr, tr, crs, nd = read_geotiff(f)
        images.append((squeeze(arr), tr, nd, os.path.basename(f)))
    return images


def bounds_from_arr(arr, tr):
    """Compute geographic bounds: (left, bottom, right, top)."""
    rows, cols = arr.shape
    left = tr.c
    right = tr.c + tr.a * cols
    top = tr.f
    bottom = tr.f + tr.e * rows  # e is negative
    return (left, bottom, right, top)


def extract_overlap(arr_i, tr_i, arr_j, tr_j):
    bi = bounds_from_arr(arr_i, tr_i)
    bj = bounds_from_arr(arr_j, tr_j)
    windows = get_overlap_window(bi, tr_i, bj, tr_j)
    if windows is None:
        return None
    wi, wj = windows
    patch_i = arr_i[wi[2]:wi[3], wi[0]:wi[1]]
    patch_j = arr_j[wj[2]:wj[3], wj[0]:wj[1]]
    # Crop both to same size
    h = min(patch_i.shape[0], patch_j.shape[0])
    w = min(patch_i.shape[1], patch_j.shape[1])
    patch_i = patch_i[:h, :w]
    patch_j = patch_j[:h, :w]
    return patch_i, patch_j, wi, wj


def fig1_overlap_comparison(band, sensor, out_dir):
    """Figure 1: overlap region comparison between two consecutive images."""
    images = load_images(band, sensor)
    if len(images) < 2:
        return
    arr_i, tr_i, nd_i, name_i = images[0]
    arr_j, tr_j, nd_j, name_j = images[1]

    result = extract_overlap(arr_i, tr_i, arr_j, tr_j)
    if result is None:
        return
    pi, pj, wi, wj = result

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # Raw overlap patches
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(pi, cmap='gray', vmin=0, vmax=3000)
    ax1.set_title(f'Image A (DN)', fontsize=11)
    ax1.set_xlabel('Column'); ax1.set_ylabel('Row')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(pj, cmap='gray', vmin=0, vmax=3000)
    ax2.set_title(f'Image B (DN)', fontsize=11)
    ax2.set_xlabel('Column'); ax2.set_ylabel('Row')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Difference map
    ax3 = fig.add_subplot(gs[0, 2])
    diff = pi.astype(np.float64) - pj.astype(np.float64)
    max_diff = max(np.abs(diff.min()), np.abs(diff.max()))
    im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    ax3.set_title(f'A - B (mean={np.nanmean(diff):.1f})', fontsize=11)
    ax3.set_xlabel('Column'); ax3.set_ylabel('Row')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Histograms
    ax4 = fig.add_subplot(gs[1, 0])
    pi_flat = pi.ravel()
    pj_flat = pj.ravel()
    mask = (pi_flat > 0) & (pj_flat > 0)
    if mask.sum() > 0:
        pmin = min(pi_flat[mask].min(), pj_flat[mask].min())
        pmax = max(pi_flat[mask].max(), pj_flat[mask].max())
        bins = np.linspace(pmin, pmax, 100)
        ax4.hist(pi_flat[mask], bins=bins, alpha=0.6, label='Image A', density=True, color='blue')
        ax4.hist(pj_flat[mask], bins=bins, alpha=0.6, label='Image B', density=True, color='red')
        ax4.set_title('Pixel Distribution in Overlap', fontsize=11)
        ax4.set_xlabel('DN Value'); ax4.set_ylabel('Density')
        ax4.legend()
        stats_text = f'A: μ={pi_flat[mask].mean():.1f}, σ={pi_flat[mask].std():.1f}\n'
        stats_text += f'B: μ={pj_flat[mask].mean():.1f}, σ={pj_flat[mask].std():.1f}'
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=9,
                 va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Full image overview
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(arr_i, cmap='gray', vmin=0, vmax=3000, alpha=0.7)
    h_i, w_i = arr_i.shape
    ax5.plot([wi[0], wi[1], wi[1], wi[0], wi[0]],
             [wi[2], wi[2], wi[3], wi[3], wi[2]], 'r-', linewidth=2)
    ax5.set_title(f'Image A with overlap box', fontsize=11)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(arr_j, cmap='gray', vmin=0, vmax=3000, alpha=0.7)
    ax6.plot([wj[0], wj[1], wj[1], wj[0], wj[0]],
             [wj[2], wj[2], wj[3], wj[3], wj[2]], 'r-', linewidth=2)
    ax6.set_title(f'Image B with overlap box', fontsize=11)

    short_i = name_i[:25] + '...' if len(name_i) > 25 else name_i
    short_j = name_j[:25] + '...' if len(name_j) > 25 else name_j
    fig.suptitle(f'{band} {sensor} — Overlap Comparison\n'
                 f'A: {short_i}\nB: {short_j}', fontsize=13, fontweight='bold')

    out_path = os.path.join(out_dir, f'{band}_{sensor}_overlap.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig2_mosaic_comparison(band, sensor, out_dir):
    """Figure 2: side-by-side mosaic before/after BAGRN."""
    # Find the baseline output
    base_dir = os.path.join(r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\mosaic_baseline',
                            f'{band}_{sensor}')
    orig_file = os.path.join(base_dir, f'mosaic_{band}_{sensor}_original.tif')
    bagrn_file = os.path.join(base_dir, f'mosaic_{band}_{sensor}_bagrn.tif')

    if not os.path.exists(orig_file) or not os.path.exists(bagrn_file):
        print(f"  Mosaic files not found for {band}_{sensor}, skipping")
        return

    orig_arr, _, _, orig_nd = read_geotiff(orig_file)
    bagrn_arr, _, _, bagrn_nd = read_geotiff(bagrn_file)
    orig_arr = squeeze(orig_arr)
    bagrn_arr = squeeze(bagrn_arr)

    vmin = max(0, np.percentile(bagrn_arr[bagrn_arr > 0], 2))
    vmax = np.percentile(bagrn_arr[bagrn_arr > 0], 98)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(orig_arr, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original (no normalization)', fontsize=12)
    axes[1].imshow(bagrn_arr, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title('After BAGRN', fontsize=12)
    for ax in axes:
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
    fig.suptitle(f'{band} {sensor} — Mosaic Comparison ({orig_arr.shape[0]}×{orig_arr.shape[1]})',
                 fontsize=14, fontweight='bold')

    out_path = os.path.join(out_dir, f'{band}_{sensor}_mosaic.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig3_cutline_comparison(band, sensor, out_dir):
    """Figure 3: horizontal cutline through mosaic showing DN jump reduction."""
    base_dir = os.path.join(r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\mosaic_baseline',
                            f'{band}_{sensor}')
    orig_file = os.path.join(base_dir, f'mosaic_{band}_{sensor}_original.tif')
    bagrn_file = os.path.join(base_dir, f'mosaic_{band}_{sensor}_bagrn.tif')

    if not os.path.exists(orig_file) or not os.path.exists(bagrn_file):
        return

    orig_arr, _, _, _ = read_geotiff(orig_file)
    bagrn_arr, _, _, _ = read_geotiff(bagrn_file)

    if len(orig_arr.shape) == 3:
        orig_arr = orig_arr[0]
        bagrn_arr = bagrn_arr[0]

    h, w = orig_arr.shape
    row_idx = h // 2
    orig_line = orig_arr[row_idx, :].astype(np.float64)
    bagrn_line = bagrn_arr[row_idx, :].astype(np.float64)
    mask = (orig_line > 0) & (bagrn_line > 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    cols = np.arange(w)
    orig_valid = np.where(mask, orig_line, np.nan)
    bagrn_valid = np.where(mask, bagrn_line, np.nan)

    ax1.plot(cols, orig_valid, color='gray', linewidth=0.5, alpha=0.8)
    ax1.set_ylabel('DN Value')
    ax1.set_title(f'Original Mosaic — Cutline at row {row_idx}', fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(cols, bagrn_valid, color='tab:blue', linewidth=0.5, alpha=0.8)
    ax2.set_ylabel('DN Value')
    ax2.set_xlabel('Column')
    ax2.set_title(f'BAGRN Mosaic — Cutline at row {row_idx}', fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'{band} {sensor} — Cutline Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'{band}_{sensor}_cutline.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig4_pairwise_diff(band, sensor, out_dir):
    """Figure 4: pairwise difference heatmaps before/after BAGRN."""
    images = load_images(band, sensor)
    if len(images) < 2:
        return

    pairs = [(0, 1), (0, 2)]
    fig, axes = plt.subplots(2, len(pairs), figsize=(8 * len(pairs), 8))
    if len(pairs) == 1:
        axes = axes.reshape(2, 1)

    for col, (i, j) in enumerate(pairs):
        if j >= len(images):
            continue
        ai, tri, ndi, ni = images[i]
        aj, trj, ndj, nj = images[j]
        result = extract_overlap(ai, tri, aj, trj)
        if result is None:
            continue
        pi, pj, wi, wj = result
        diff = pi.astype(np.float64) - pj.astype(np.float64)
        max_diff = max(np.abs(diff.min()), np.abs(diff.max()), 1)

        axes[0, col].imshow(diff, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
        axes[0, col].set_title(f'A-B (mean={np.nanmean(diff):.1f})', fontsize=10)
        axes[0, col].set_xlabel('Column'); axes[0, col].set_ylabel('Row')
        plt.colorbar(axes[0, col].images[0], ax=axes[0, col], fraction=0.046)

        # After BAGRN (use the baseline dir)
        bagrn_base = os.path.join(r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\mosaic_baseline',
                                   f'{band}_{sensor}')
        files = sorted(glob.glob(os.path.join(bagrn_base, '*_bagrn.tif')))
        files = [f for f in files if 'mosaic' not in f]
        if i < len(files) and j < len(files):
            bi, _, _, _ = read_geotiff(files[i])
            bj, _, _, _ = read_geotiff(files[j])
            bi = squeeze(bi)
            bj = squeeze(bj)
            br = extract_overlap(bi, tri, bj, trj)
            if br is not None:
                bpi, bpj, _, _ = br
                bdiff = bpi.astype(np.float64) - bpj.astype(np.float64)
                axes[1, col].imshow(bdiff, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
                axes[1, col].set_title(f'BAGRN A-B (mean={np.nanmean(bdiff):.1f})', fontsize=10)
                axes[1, col].set_xlabel('Column'); axes[1, col].set_ylabel('Row')
                plt.colorbar(axes[1, col].images[0], ax=axes[1, col], fraction=0.046)

    axes[0, 0].set_ylabel('Original', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('BAGRN', fontsize=12, fontweight='bold')
    fig.suptitle(f'{band} {sensor} — Pairwise Difference', fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'{band}_{sensor}_pairwise.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    for band in BANDS:
        for sensor in SENSORS:
            print(f"\n{'='*50}")
            print(f"  {band} {sensor}")
            print(f"{'='*50}")
            out_dir = os.path.join(OUTPUT_BASE, f'{band}_{sensor}')
            os.makedirs(out_dir, exist_ok=True)
            try:
                fig1_overlap_comparison(band, sensor, out_dir)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  Fig1 error: {e}")
            try:
                fig2_mosaic_comparison(band, sensor, out_dir)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  Fig2 error: {e}")
            try:
                fig4_pairwise_diff(band, sensor, out_dir)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  Fig4 error: {e}")

    print(f"\n\nOutput: {OUTPUT_BASE}")
    for d in sorted(os.listdir(OUTPUT_BASE)):
        dp = os.path.join(OUTPUT_BASE, d)
        if os.path.isdir(dp):
            pngs = [f for f in sorted(os.listdir(dp)) if f.endswith('.png')]
            print(f"  {d}/ ({len(pngs)} figures)")
            for f in pngs:
                size = os.path.getsize(os.path.join(dp, f))
                print(f"    {f} ({size/1024:.0f} KB)")
