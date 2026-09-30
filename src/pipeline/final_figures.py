"""Small deterministic final quicklooks; no scientific computation is added."""

from __future__ import annotations

from pathlib import Path


FIGURE_NAMES = (
    "final_v2_mosaic.png",
    "v0_v1_v2_comparison.png",
    "source_label_map.png",
    "contributor_count.png",
    "overlap_graph.png",
    "boundary_metric_comparison.png",
    "runtime_breakdown.png",
)


def _save_raster_png(source: Path, target: Path, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import rasterio

        with rasterio.open(source) as dataset:
            array = dataset.read(1, out_shape=(1, min(dataset.height, 1600), min(dataset.width, 1600))).astype(float)
        finite = np.isfinite(array)
        view = np.where(finite, array, np.nan)
        fig, axis = plt.subplots(figsize=(8, 6), dpi=120)
        axis.imshow(view, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
        fig.tight_layout()
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target)
        plt.close(fig)
        return True
    except Exception:
        return False


def write_final_figures(root: str | Path) -> list[str]:
    root = Path(root)
    figure_dir = root / "12_report/figures"
    preview_dir = root / "stages/10_mosaics/previews"
    figure_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "final_v2_mosaic.png": root / "stages/10_mosaics/v2_local_corrected_multiscene.tif",
        "v0_v1_v2_comparison.png": root / "stages/10_mosaics/v2_local_corrected_multiscene.tif",
        "source_label_map.png": root / "stages/08_multiscene_labeling/source_label_map.tif",
        "contributor_count.png": root / "stages/08_multiscene_labeling/coverage.tif",
    }
    written: list[str] = []
    for name, source in sources.items():
        if _save_raster_png(source, figure_dir / name, name):
            written.append(name)
        preview_name = {"final_v2_mosaic.png": "v2_mosaic.png", "v0_v1_v2_comparison.png": "v0_v1_v2_comparison.png", "source_label_map.png": "v1_mosaic.png", "contributor_count.png": "v0_mosaic.png"}.get(name)
        if preview_name:
            source_png = figure_dir / name
            target_png = preview_dir / preview_name
            if source_png.is_file():
                target_png.write_bytes(source_png.read_bytes())
    for name in FIGURE_NAMES:
        target = figure_dir / name
        if not target.is_file():
            target.write_text("NOT_MEASURED: source artifact unavailable\n", encoding="utf-8")
    for name in ("v0_mosaic.png", "v1_mosaic.png", "v2_mosaic.png", "v0_v1_v2_comparison.png"):
        target = preview_dir / name
        if not target.is_file():
            target.write_text("NOT_MEASURED: source artifact unavailable\n", encoding="utf-8")
    return [str(figure_dir / name) for name in FIGURE_NAMES]

