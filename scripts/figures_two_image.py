"""Generate effect figures for two-image pipeline results."""
import os, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.io_utils import read_geotiff

OUTPUT = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\two_image'
FIG_DIR = r'D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce\data\output\effect_figures_two'
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
    orig_file = os.path.join(OUTPUT, band, f'mosaic_{band}_original.tif')
    bagrn_file = os.path.join(OUTPUT, band, f'mosaic_{band}_bagrn.tif')
    volrn_file = os.path.join(OUTPUT, band, f'mosaic_{band}_volrn.tif')

    if not all(os.path.exists(f) for f in [orig_file, bagrn_file, volrn_file]):
        print(f"  Missing files, skipping")
        continue

    orig = squeeze(read_geotiff(orig_file)[0])
    bagrn = squeeze(read_geotiff(bagrn_file)[0])
    volrn = squeeze(read_geotiff(volrn_file)[0])

    valid = bagrn[bagrn > 0]
    vmin = max(0, np.percentile(valid, 2))
    vmax = np.percentile(valid, 98)

    # Figure 1: 3-panel comparison
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, data, title in zip(axes, [orig, bagrn, volrn],
                                ['Original', 'BAGRN (Global)', 'VOLRN (Local)']):
        ax.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14, fontweight='bold')
    fig.suptitle(f'DZ01V {band} — 2-Image Mosaic ({orig.shape[0]}x{orig.shape[1]})',
                 fontsize=16, fontweight='bold')
    out = os.path.join(FIG_DIR, f'{band}_3panel.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

    # Figure 2: Zoom into overlap region
    h, w = orig.shape
    cy, cx = h // 2, w // 2
    zh, zw = min(400, h // 4), min(400, w // 4)
    r1, r2 = cy - zh // 2, cy + zh // 2
    c1, c2 = cx - zw // 2, cx + zw // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, data, title in zip(axes, [orig, bagrn, volrn],
                                ['Original', 'BAGRN', 'VOLRN']):
        ax.imshow(data[r1:r2, c1:c2], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight='bold')
    fig.suptitle(f'{band} — Overlap Region Zoom', fontsize=14, fontweight='bold')
    out = os.path.join(FIG_DIR, f'{band}_zoom.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

    # Figure 3: Cutline
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    cols = np.arange(w)
    for ax, data, title, color in zip(
            axes, [orig, bagrn, volrn],
            ['Original', 'BAGRN', 'VOLRN'],
            ['gray', 'tab:blue', 'tab:green']):
        line = data[h // 2, :].astype(np.float64)
        mask = line > 0
        ax.plot(cols[mask], line[mask], color=color, linewidth=0.5)
        ax.set_ylabel('DN')
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Column')
    fig.suptitle(f'{band} — Cutline at row {h//2}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f'{band}_cutline.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

    # Figure 4: Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    for data, label, color in zip([orig, bagrn, volrn],
                                   ['Original', 'BAGRN', 'VOLRN'],
                                   ['gray', 'tab:blue', 'tab:green']):
        flat = data[r1:r2, c1:c2].ravel()
        flat = flat[flat > 0]
        if len(flat) > 0:
            ax.hist(flat, bins=100, alpha=0.5, label=label, density=True, color=color)
    ax.set_title(f'{band} — Pixel Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    out = os.path.join(FIG_DIR, f'{band}_histogram.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

print(f"\nAll done: {FIG_DIR}")
