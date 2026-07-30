"""Generate effect figures: Original vs BAGRN vs VOLRN comparison."""
import os, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.io_utils import read_geotiff

OUTPUT_BASE = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\final_v2'
FIG_DIR = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\effect_figures_v2'
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
BANDS = ['B5', 'B8', 'B14']


def squeeze(arr):
    while arr.ndim > 2:
        arr = arr[0]
    return arr


for band in BANDS:
    print(f"\n  {band}...")
    orig_file = os.path.join(OUTPUT_BASE, band, f'mosaic_{band}_original.tif')
    bagrn_file = os.path.join(OUTPUT_BASE, band, f'mosaic_{band}_bagrn.tif')
    volrn_file = os.path.join(OUTPUT_BASE, band, f'mosaic_{band}_volrn.tif')

    if not all(os.path.exists(f) for f in [orig_file, bagrn_file, volrn_file]):
        print(f"  Missing files for {band}, skipping")
        continue

    orig = squeeze(read_geotiff(orig_file)[0])
    bagrn = squeeze(read_geotiff(bagrn_file)[0])
    volrn = squeeze(read_geotiff(volrn_file)[0])

    vmin = max(0, np.percentile(bagrn[bagrn > 0], 2))
    vmax = np.percentile(bagrn[bagrn > 0], 98)

    # Figure 1: Side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    titles = ['Original', 'BAGRN (Global)', 'VOLRN (Local)']
    for ax, data, title in zip(axes, [orig, bagrn, volrn], titles):
        ax.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
    fig.suptitle(f'DZ01V {band} — Mosaic Comparison ({orig.shape[0]}×{orig.shape[1]})',
                 fontsize=16, fontweight='bold')
    out = os.path.join(FIG_DIR, f'{band}_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

    # Figure 2: Zoom into overlap region (center of image)
    h, w = orig.shape
    cy, cx = h // 2, w // 2
    zoom_h, zoom_w = min(500, h // 4), min(500, w // 4)
    r1, r2 = cy - zoom_h // 2, cy + zoom_h // 2
    c1, c2 = cx - zoom_w // 2, cx + zoom_w // 2

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, data, title in zip(axes, [orig, bagrn, volrn], titles):
        patch = data[r1:r2, c1:c2]
        ax.imshow(patch, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight='bold')
    fig.suptitle(f'{band} — Center Zoom (overlap region)', fontsize=14, fontweight='bold')
    out = os.path.join(FIG_DIR, f'{band}_zoom.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

    # Figure 3: Cutline comparison
    row_idx = h // 2
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for ax, data, title, color in zip(
            axes, [orig, bagrn, volrn], titles, ['gray', 'tab:blue', 'tab:green']):
        line = data[row_idx, :].astype(np.float64)
        mask = line > 0
        cols = np.arange(w)
        ax.plot(cols[mask], line[mask], color=color, linewidth=0.5, alpha=0.8)
        ax.set_ylabel('DN')
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Column')
    fig.suptitle(f'{band} — Cutline at row {row_idx}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f'{band}_cutline.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

    # Figure 4: Histogram comparison in overlap area
    fig, ax = plt.subplots(figsize=(10, 6))
    for data, label, color in zip([orig, bagrn, volrn], titles, ['gray', 'tab:blue', 'tab:green']):
        flat = data[r1:r2, c1:c2].ravel()
        flat = flat[flat > 0]
        if len(flat) > 0:
            ax.hist(flat, bins=100, alpha=0.5, label=label, density=True, color=color)
    ax.set_title(f'{band} — Pixel Distribution (Center Region)', fontsize=12, fontweight='bold')
    ax.set_xlabel('DN Value')
    ax.set_ylabel('Density')
    ax.legend()
    out = os.path.join(FIG_DIR, f'{band}_histogram.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

print(f"\nAll figures saved to: {FIG_DIR}")
