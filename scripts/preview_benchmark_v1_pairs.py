"""Create small UNVERIFIED paired B9 context previews; never exports source rasters."""

from contextlib import ExitStack
from pathlib import Path
import csv

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window

from build_benchmark_v1_inputs import overlap_window


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/benchmark/v1/previews"


def stretch(data, mask):
    good = data[mask > 0]
    if good.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    lo, hi = np.percentile(good, [2, 98])
    if hi <= lo:
        hi = lo + 1
    scaled = np.clip((data.astype(np.float32) - lo) * (255 / (hi - lo)), 0, 255).astype(np.uint8)
    scaled[mask == 0] = 0
    return scaled


def main():
    with (ROOT / "data/output/final_pipeline/b9_13scene_formal_v2/scene_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        manifests = list(csv.DictReader(stream))
    by_index = {int(m["scene_index"]): m for m in manifests}
    OUT.mkdir(parents=True, exist_ok=True)
    for i, j in [(0, 1), (2, 3), (8, 10)]:
        with ExitStack() as stack:
            a = stack.enter_context(rasterio.open(by_index[i]["b9_path"]))
            b = stack.enter_context(rasterio.open(by_index[j]["b9_path"]))
            wa, wb = overlap_window(a, b)
            width, height = min(384, int(wa.width)), min(384, int(wa.height))
            cx, cy = (int(wa.width) - width) // 2, (int(wa.height) - height) // 2
            panels = []
            for ds, w in ((a, wa), (b, wb)):
                patch = Window(w.col_off + cx, w.row_off + cy, width, height)
                pixels = ds.read(1, window=patch)
                mask = ds.read_masks(1, window=patch)
                panels.append(Image.fromarray(stretch(pixels, mask), mode="L").convert("RGB"))
            canvas = Image.new("RGB", (width * 2, height + 40), "white")
            canvas.paste(panels[0], (0, 40))
            canvas.paste(panels[1], (width, 40))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f"{i:02d} vs {j:02d} B9 | UNVERIFIED | no ground truth", fill="black")
            canvas.save(OUT / f"pair_{i:02d}_{j:02d}_unverified.png")


if __name__ == "__main__":
    main()
